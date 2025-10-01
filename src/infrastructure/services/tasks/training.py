"""Celery tasks that run the model training pipeline."""

import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, List
from uuid import UUID

from src.domain.entities.model import DenseLayerConfig, ModelType, RNNLayerConfig
from src.domain.entities.time_series import HistoricDataPoint
from src.domain.entities.training_job import TrainingStatus
from src.infrastructure.services.celery_config import celery_app
from src.infrastructure.services.tasks.base import CallbackTask, logger
from src.infrastructure.settings import get_settings


@celery_app.task(bind=True, base=CallbackTask, name="train_model_task")
def train_model_task(
    self,
    training_job_id: str,
    model_config: Dict[str, Any],
    collected_data: List[Dict[str, Any]],
    window_size: int,
) -> Dict[str, Any]:
    """Train a machine learning model with collected data."""

    model_repo = None
    training_job_repo = None

    def set_model_status(status: "ModelStatus", allow_downgrade: bool = True) -> None:
        if not existing_job or not existing_job.model_id or model_repo is None:
            return
        model_id = existing_job.model_id
        if model_id is None:
            return
        model_record = asyncio.run(model_repo.find_by_id(model_id))
        if not model_record:
            return
        if (
            status != ModelStatus.TRAINED
            and model_record.has_trained_artifacts()
            and not allow_downgrade
        ):
            return

        model_record.status = status
        model_record.update_timestamp()
        asyncio.run(model_repo.update(model_record))

    try:
        logger.info(
            "Starting model training",
            training_job_id=training_job_id,
            data_points=len(collected_data),
            window_size=window_size,
        )

        from src.domain.entities.model import Model
        from src.infrastructure.database.mongo_database import MongoDatabase
        from src.infrastructure.repositories.gridfs_model_artifacts_repository import (
            GridFSModelArtifactsRepository,
        )
        from src.infrastructure.repositories.model_repository import ModelRepository
        from src.infrastructure.repositories.training_job_repository import (
            TrainingJobRepository,
        )

        settings = get_settings()

        database = MongoDatabase(
            mongo_uri=settings.database.mongo_uri,
            db_name=settings.database.database_name,
        )
        training_job_repo = TrainingJobRepository(database)
        model_repo = ModelRepository(database)

        artifacts_repo = GridFSModelArtifactsRepository(
            mongo_client=database.client,
            database_name=settings.database.database_name,
        )

        from src.application.use_cases.model_training_use_case import (
            ModelTrainingUseCase,
        )
        from src.domain.entities.model import ModelStatus

        model_training = ModelTrainingUseCase(artifacts_repository=artifacts_repo)

        existing_job = asyncio.run(training_job_repo.get_by_id(UUID(training_job_id)))

        if existing_job and existing_job.status in (
            TrainingStatus.CANCELLED,
            TrainingStatus.CANCEL_REQUESTED,
        ):
            logger.info(
                "Training job already cancelled before model training",
                training_job_id=training_job_id,
            )
            set_model_status(ModelStatus.DRAFT, allow_downgrade=False)
            return {
                "training_job_id": training_job_id,
                "status": "cancelled",
                "message": "Job was cancelled before training started.",
            }

        asyncio.run(
            training_job_repo.update_task_refs(
                UUID(training_job_id),
                task_refs={"training_task_id": self.request.id},
            )
        )

        asyncio.run(
            training_job_repo.update_training_job_status(
                UUID(training_job_id),
                TrainingStatus.TRAINING,
                training_start=datetime.now(timezone.utc),
            )
        )

        model = Model()
        model.id = UUID(model_config["id"])
        model.name = model_config["name"]
        model.model_type = ModelType(model_config["model_type"])

        rnn_layers_payload = model_config.get("rnn_layers", [])
        if rnn_layers_payload:
            model.rnn_layers = [
                RNNLayerConfig(
                    units=int(layer.get("units", 0)),
                    dropout=float(layer.get("dropout", 0.1)),
                    recurrent_dropout=float(layer.get("recurrent_dropout", 0.0)),
                )
                for layer in rnn_layers_payload
            ]
        else:
            legacy_units = model_config.get("rnn_units", [64])
            legacy_dropout = float(model_config.get("rnn_dropout", 0.1))
            model.rnn_layers = [
                RNNLayerConfig(units=int(units), dropout=legacy_dropout)
                for units in legacy_units
            ]

        dense_layers_payload = model_config.get("dense_layers", [])
        if dense_layers_payload:
            model.dense_layers = [
                DenseLayerConfig(
                    units=int(layer.get("units", 0)),
                    dropout=float(layer.get("dropout", 0.1)),
                    activation=layer.get("activation", "relu"),
                )
                for layer in dense_layers_payload
            ]
        else:
            legacy_dense_units = model_config.get("dense_units", [32])
            legacy_dense_dropout = float(model_config.get("dense_dropout", 0.1))
            model.dense_layers = [
                DenseLayerConfig(units=int(units), dropout=legacy_dense_dropout)
                for units in legacy_dense_units
            ]

        model.learning_rate = float(model_config["learning_rate"])
        model.batch_size = int(model_config["batch_size"])
        model.epochs = int(model_config["epochs"])
        model.early_stopping_patience = model_config.get("early_stopping_patience")
        model.feature = str(model_config["feature"])
        model.validation_ratio = float(model_config.get("validation_ratio", 0.15))
        model.test_ratio = float(model_config.get("test_ratio", 0.15))
        model.entity_type = str(model_config.get("entity_type", ""))
        model.entity_id = str(model_config.get("entity_id", ""))

        data_points: List[HistoricDataPoint] = []
        for item in collected_data:
            timestamp_value = item.get("timestamp")
            if isinstance(timestamp_value, str):
                timestamp_dt = datetime.fromisoformat(timestamp_value)
            elif isinstance(timestamp_value, datetime):
                timestamp_dt = timestamp_value
            else:
                logger.warning(
                    "Skipping data point with invalid timestamp",
                    raw_timestamp=timestamp_value,
                )
                continue

            value_raw = item.get("value")
            if value_raw is None:
                logger.warning("Skipping data point with missing value")
                continue
            try:
                value = float(value_raw)
            except (TypeError, ValueError):
                logger.warning(
                    "Skipping data point with non-numeric value",
                    raw_value=value_raw,
                )
                continue

            data_points.append(HistoricDataPoint(timestamp=timestamp_dt, value=value))

        if len(data_points) < window_size + 10:
            error_msg = (
                f"Insufficient data: {len(data_points)} points available, "
                f"need at least {window_size + 10} for window_size={window_size}"
            )
            logger.warning(
                "Insufficient data after processing",
                available_points=len(data_points),
                required_minimum=window_size + 10,
                window_size=window_size,
            )
            asyncio.run(
                training_job_repo.update_training_job_status(
                    UUID(training_job_id),
                    TrainingStatus.FAILED,
                    training_end=datetime.now(timezone.utc),
                )
            )
            set_model_status(ModelStatus.DRAFT, allow_downgrade=False)
            return {
                "training_job_id": training_job_id,
                "status": "failed",
                "error": error_msg,
            }

        (
            metrics,
            model_artifact_id,
            x_scaler_artifact_id,
            y_scaler_artifact_id,
            metadata_artifact_id,
        ) = asyncio.run(
            model_training.execute(
                model_config=model,
                collected_data=data_points,
                window_size=window_size,
                training_job_id=training_job_id,
            )
        )

        job = asyncio.run(training_job_repo.get_by_id(UUID(training_job_id)))
        if job and job.status in (
            TrainingStatus.CANCELLED,
            TrainingStatus.CANCEL_REQUESTED,
        ):
            logger.info(
                "Job was cancelled, skipping final update",
                training_job_id=training_job_id,
            )
            return {
                "training_job_id": training_job_id,
                "status": "cancelled",
                "message": "Job was cancelled.",
            }

        asyncio.run(
            training_job_repo.complete_training_job(
                UUID(training_job_id),
                metrics=metrics,
                model_artifact_id=model_artifact_id,
                x_scaler_artifact_id=x_scaler_artifact_id,
                y_scaler_artifact_id=y_scaler_artifact_id,
                metadata_artifact_id=metadata_artifact_id,
            )
        )
        asyncio.run(
            training_job_repo.update_task_refs(UUID(training_job_id), clear=True)
        )
        if existing_job and existing_job.model_id:
            model_record = asyncio.run(model_repo.find_by_id(existing_job.model_id))
            if model_record:
                model_record.status = ModelStatus.TRAINED
                model_record.has_successful_training = True
                model_record.update_timestamp()
                asyncio.run(model_repo.update(model_record))

        logger.info(
            "training.completed",
            training_job_id=training_job_id,
            metrics=metrics.__dict__,
        )

        return {
            "training_job_id": training_job_id,
            "status": "completed",
            "metrics": metrics.__dict__,
            "model_artifact_id": model_artifact_id,
            "x_scaler_artifact_id": x_scaler_artifact_id,
            "y_scaler_artifact_id": y_scaler_artifact_id,
            "metadata_artifact_id": metadata_artifact_id,
        }

    except Exception as exc:  # pragma: no cover - defensive logging
        from src.domain.entities.model import ModelStatus

        set_model_status(ModelStatus.DRAFT, allow_downgrade=False)
        logger.error(
            "training.failed",
            training_job_id=training_job_id,
            error=str(exc),
            exc_info=exc,
        )

        try:
            from src.infrastructure.database.mongo_database import MongoDatabase
            from src.infrastructure.repositories.training_job_repository import (
                TrainingJobRepository,
            )

            settings = get_settings()

            database = MongoDatabase(
                mongo_uri=settings.database.mongo_uri,
                db_name=settings.database.database_name,
            )
            training_job_repo = TrainingJobRepository(database)

            asyncio.run(
                training_job_repo.fail_training_job(
                    UUID(training_job_id),
                    error=str(exc),
                    error_details={
                        "task_id": self.request.id,
                        "retries": self.request.retries,
                    },
                )
            )
            asyncio.run(
                training_job_repo.update_task_refs(UUID(training_job_id), clear=True)
            )
        except Exception as update_error:  # pragma: no cover
            logger.error(
                "training.status_update_failed",
                training_job_id=training_job_id,
                error=str(update_error),
                exc_info=update_error,
            )

        if self.request.retries < self.max_retries:
            logger.info(
                "training.retry_scheduled",
                training_job_id=training_job_id,
                attempt=self.request.retries + 1,
                delay_seconds=300,
            )
            raise self.retry(countdown=300, exc=exc)

        raise exc
