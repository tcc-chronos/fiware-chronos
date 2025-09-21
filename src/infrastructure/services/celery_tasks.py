"""
Infrastructure Services - Celery Tasks

This module contains Celery tasks for asynchronous data collection and model training.

Key Features:
- Scalable data collection: Handles large datasets by parallelizing STH-Comet requests
- Automatic data reordering: Ensures chronological order for time series training
- Robust error handling: Exponential backoff and comprehensive logging
- Centralized configuration: Uses settings from config.py
"""

import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID

import structlog
from celery import Task

from src.domain.entities.training_job import DataCollectionStatus, TrainingStatus
from src.infrastructure.services.celery_config import celery_app

logger = structlog.get_logger(__name__)


def _log_data_collection_summary(
    total_requested: int,
    total_collected: int,
    chunks: int,
    date_range: Optional[str] = None,
) -> None:
    """Helper function to log data collection summary."""
    efficiency = (total_collected / total_requested * 100) if total_requested > 0 else 0

    logger.info(
        "📊 Data Collection Summary",
        requested=total_requested,
        collected=total_collected,
        efficiency_percent=f"{efficiency:.1f}%",
        parallel_chunks=chunks,
        date_range=date_range or "N/A",
    )


class CallbackTask(Task):
    """Base task class with callback support."""

    def on_success(self, retval, task_id, args, kwargs):
        """Called on task success."""
        logger.info(
            "celery.task.succeeded",
            task_id=task_id,
            result=retval,
        )

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Called on task failure."""
        logger.error(
            "celery.task.failed",
            task_id=task_id,
            error=str(exc),
            traceback=einfo.traceback,
            exc_info=exc,
        )


@celery_app.task(bind=True, base=CallbackTask, name="collect_data_chunk")
def collect_data_chunk(
    self,
    job_id: str,
    training_job_id: str,
    entity_type: str,
    entity_id: str,
    attribute: str,
    h_limit: int,
    h_offset: int,
    fiware_service: str = "smart",
    fiware_servicepath: str = "/",
) -> Dict[str, Any]:
    """
    Collect a chunk of data from STH-Comet using hLimit and hOffset.

    Args:
        job_id: Data collection job ID
        training_job_id: Training job ID
        entity_type: FIWARE entity type
        entity_id: FIWARE entity ID
        attribute: Attribute to collect
        h_limit: Number of data points to collect (max 100 per STH-Comet limitation)
        h_offset: Historical offset from total count
        fiware_service: FIWARE service
        fiware_servicepath: FIWARE service path

    Returns:
        Dictionary with collection results
    """
    try:
        logger.info(
            "Starting data collection chunk",
            job_id=job_id,
            training_job_id=training_job_id,
            entity_type=entity_type,
            entity_id=entity_id,
            attribute=attribute,
            h_limit=h_limit,
            h_offset=h_offset,
        )

        # Import here to avoid circular imports
        from src.infrastructure.database.mongo_database import MongoDatabase
        from src.infrastructure.gateways.sth_comet_gateway import STHCometGateway
        from src.infrastructure.repositories.training_job_repository import (
            TrainingJobRepository,
        )
        from src.main.config import get_settings

        # Get centralized settings
        settings = get_settings()

        # Initialize dependencies with centralized configuration
        database = MongoDatabase(
            mongo_uri=settings.database.mongo_uri,
            db_name=settings.database.database_name,
        )
        training_job_repo = TrainingJobRepository(database)
        sth_gateway = STHCometGateway(settings.fiware.sth_url)

        training_job = asyncio.run(training_job_repo.get_by_id(UUID(training_job_id)))
        if training_job and training_job.status in (
            TrainingStatus.CANCELLED,
            TrainingStatus.CANCEL_REQUESTED,
        ):
            logger.info(
                "Skipping data collection chunk due to cancellation",
                training_job_id=training_job_id,
                job_id=job_id,
            )
            asyncio.run(
                training_job_repo.update_data_collection_job_status(
                    UUID(training_job_id),
                    UUID(job_id),
                    DataCollectionStatus.CANCELLED,
                    end_time=datetime.now(timezone.utc),
                    error="Cancelled by user request",
                )
            )
            return {
                "job_id": job_id,
                "training_job_id": training_job_id,
                "status": "cancelled",
                "data_points_collected": 0,
                "data_points": [],
                "h_offset": h_offset,
                "chunk_info": {
                    "requested_h_limit": h_limit,
                    "actual_collected": 0,
                    "h_offset": h_offset,
                },
                "message": "Chunk cancelled before execution.",
            }

        # Update job status to in progress
        asyncio.run(
            training_job_repo.update_data_collection_job_status(
                UUID(training_job_id),
                UUID(job_id),
                DataCollectionStatus.IN_PROGRESS,
                start_time=datetime.now(timezone.utc),
            )
        )

        # Collect data
        collected_data = asyncio.run(
            sth_gateway.collect_data(
                entity_type=entity_type,
                entity_id=entity_id,
                attribute=attribute,
                h_limit=h_limit,
                h_offset=h_offset,
                fiware_service=fiware_service,
                fiware_servicepath=fiware_servicepath,
            )
        )

        # Convert to serializable format with precise timestamp handling
        # Use "value" as the key to ensure consistency with preprocessing
        data_points = [
            {
                "timestamp": point.timestamp.isoformat(),
                "value": point.value,
                "h_offset": h_offset,
            }
            for point in collected_data
        ]

        # Evitar atualização se job foi cancelado
        job = asyncio.run(training_job_repo.get_by_id(UUID(training_job_id)))
        if job and job.status in (
            TrainingStatus.CANCELLED,
            TrainingStatus.CANCEL_REQUESTED,
        ):
            logger.info(
                "Job was cancelled, skipping final update",
                training_job_id=training_job_id,
                job_id=job_id,
            )
            asyncio.run(
                training_job_repo.update_data_collection_job_status(
                    UUID(training_job_id),
                    UUID(job_id),
                    DataCollectionStatus.CANCELLED,
                    end_time=datetime.now(timezone.utc),
                    error="Cancelled by user request",
                )
            )
            return {
                "job_id": job_id,
                "training_job_id": training_job_id,
                "status": "cancelled",
                "data_points_collected": 0,
                "data_points": [],
                "h_offset": h_offset,
                "chunk_info": {
                    "requested_h_limit": h_limit,
                    "actual_collected": 0,
                    "h_offset": h_offset,
                },
                "message": "Job was cancelled.",
            }
        else:
            asyncio.run(
                training_job_repo.update_data_collection_job_status(
                    UUID(training_job_id),
                    UUID(job_id),
                    DataCollectionStatus.COMPLETED,
                    end_time=datetime.now(timezone.utc),
                    data_points_collected=len(collected_data),
                )
            )

        logger.info(
            "Data collection chunk completed successfully",
            job_id=job_id,
            data_points_collected=len(collected_data),
            h_offset=h_offset,
            date_range=(
                f"{collected_data[0].timestamp.isoformat()} to "
                f"{collected_data[-1].timestamp.isoformat()}"
                if collected_data
                else "No data collected"
            ),
        )

        return {
            "job_id": job_id,
            "training_job_id": training_job_id,
            "status": "completed",
            "data_points_collected": len(collected_data),
            "data_points": data_points,
            "h_offset": h_offset,
            "chunk_info": {
                "requested_h_limit": h_limit,
                "actual_collected": len(collected_data),
                "h_offset": h_offset,
            },
        }

    except Exception as exc:
        logger.error(
            "Data collection chunk failed",
            job_id=job_id,
            training_job_id=training_job_id,
            error=str(exc),
            h_offset=h_offset,
            h_limit=h_limit,
            exc_info=True,
        )

        # Update job status to failed
        try:
            from src.infrastructure.database.mongo_database import MongoDatabase
            from src.infrastructure.repositories.training_job_repository import (
                TrainingJobRepository,
            )
            from src.main.config import get_settings

            settings = get_settings()
            database = MongoDatabase(
                mongo_uri=settings.database.mongo_uri,
                db_name=settings.database.database_name,
            )
            training_job_repo = TrainingJobRepository(database)

            asyncio.run(
                training_job_repo.update_data_collection_job_status(
                    UUID(training_job_id),
                    UUID(job_id),
                    DataCollectionStatus.FAILED,
                    end_time=datetime.now(timezone.utc),
                    error=str(exc),
                )
            )
        except Exception as update_error:
            logger.error(
                "celery.data_collection.status_update_failed",
                job_id=job_id,
                training_job_id=training_job_id,
                error=str(update_error),
                exc_info=update_error,
            )

        # Retry the task with exponential backoff
        if self.request.retries < self.max_retries:
            retry_delay = min(60 * (2**self.request.retries), 300)  # Max 5 minutes
            logger.info(
                "celery.data_collection.retry_scheduled",
                job_id=job_id,
                attempt=self.request.retries + 1,
                delay_seconds=retry_delay,
            )
            raise self.retry(countdown=retry_delay, exc=exc)

        # Return error information for better debugging
        return {
            "job_id": job_id,
            "training_job_id": training_job_id,
            "status": "failed",
            "error": str(exc),
            "h_offset": h_offset,
            "h_limit": h_limit,
            "retries_exhausted": True,
        }


@celery_app.task(bind=True, base=CallbackTask, name="train_model_task")
def train_model_task(
    self,
    training_job_id: str,
    model_config: Dict[str, Any],
    collected_data: List[Dict[str, Any]],
    window_size: int,
) -> Dict[str, Any]:
    """
    Train a machine learning model with collected data.

    Args:
        training_job_id: Training job ID
        model_config: Model configuration dictionary
        collected_data: List of collected data points
        window_size: Sequence window size

    Returns:
        Dictionary with training results
    """
    try:
        logger.info(
            "Starting model training",
            training_job_id=training_job_id,
            data_points=len(collected_data),
            window_size=window_size,
        )

        # Import here to avoid circular imports
        from src.application.dtos.training_dto import CollectedDataDTO
        from src.domain.entities.model import Model, ModelType
        from src.infrastructure.database.mongo_database import MongoDatabase
        from src.infrastructure.repositories.gridfs_model_artifacts_repository import (
            GridFSModelArtifactsRepository,
        )
        from src.infrastructure.repositories.training_job_repository import (
            TrainingJobRepository,
        )
        from src.main.config import get_settings

        # Get centralized settings
        settings = get_settings()

        # Initialize dependencies
        database = MongoDatabase(
            mongo_uri=settings.database.mongo_uri,
            db_name=settings.database.database_name,
        )
        training_job_repo = TrainingJobRepository(database)

        # Initialize GridFS artifacts repository
        artifacts_repo = GridFSModelArtifactsRepository(
            mongo_client=database.client,
            database_name=settings.database.database_name,
        )

        # Initialize model training use case with GridFS repository
        from src.application.use_cases.model_training_use_case import (
            ModelTrainingUseCase,
        )

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

        # Update training job status
        asyncio.run(
            training_job_repo.update_training_job_status(
                UUID(training_job_id),
                TrainingStatus.TRAINING,
                training_start=datetime.now(timezone.utc),
            )
        )

        # Convert model config dict to Model entity
        model = Model()
        model.id = UUID(model_config["id"])
        model.name = model_config["name"]
        model.model_type = ModelType(model_config["model_type"])
        model.rnn_units = model_config["rnn_units"]
        model.dense_units = model_config["dense_units"]
        model.rnn_dropout = model_config["rnn_dropout"]
        model.dense_dropout = model_config["dense_dropout"]
        model.learning_rate = model_config["learning_rate"]
        model.batch_size = model_config["batch_size"]
        model.epochs = model_config["epochs"]
        model.early_stopping_patience = model_config.get("early_stopping_patience")
        model.feature = model_config["feature"]
        model.entity_type = model_config.get("entity_type")
        model.entity_id = model_config.get("entity_id")

        # Convert collected data to DTOs
        data_dtos = [
            CollectedDataDTO(
                timestamp=datetime.fromisoformat(item["timestamp"]),
                value=item["value"],  # Use consistent "value" key
            )
            for item in collected_data
        ]

        # Validate we have enough data after deduplication
        if len(data_dtos) < window_size + 10:
            error_msg = (
                f"Insufficient data: {len(data_dtos)} points available, "
                f"need at least {window_size + 10} for window_size={window_size}"
            )
            logger.warning(
                "Insufficient data after processing",
                available_points=len(data_dtos),
                required_minimum=window_size + 10,
                window_size=window_size,
            )
            # Update job status with specific error
            asyncio.run(
                training_job_repo.update_training_job_status(
                    UUID(training_job_id),
                    TrainingStatus.FAILED,
                    training_end=datetime.now(timezone.utc),
                )
            )
            return {
                "training_job_id": training_job_id,
                "status": "failed",
                "error": error_msg,
            }

        # Train model
        (
            metrics,
            model_artifact_id,
            x_scaler_artifact_id,
            y_scaler_artifact_id,
            metadata_artifact_id,
        ) = asyncio.run(
            model_training.execute(
                model_config=model, collected_data=data_dtos, window_size=window_size
            )
        )

        # Evitar atualização se job foi cancelado
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
        else:
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

        logger.info(
            "celery.training.completed",
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

    except Exception as exc:
        logger.error(
            "celery.training.failed",
            training_job_id=training_job_id,
            error=str(exc),
            exc_info=exc,
        )

        # Update job status to failed
        try:
            from src.infrastructure.database.mongo_database import MongoDatabase
            from src.infrastructure.repositories.training_job_repository import (
                TrainingJobRepository,
            )
            from src.main.config import settings

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
        except Exception as update_error:
            logger.error(
                "celery.training.status_update_failed",
                training_job_id=training_job_id,
                error=str(update_error),
                exc_info=update_error,
            )

        # Retry the task
        if self.request.retries < self.max_retries:
            logger.info(
                "celery.training.retry_scheduled",
                training_job_id=training_job_id,
                attempt=self.request.retries + 1,
                delay_seconds=300,
            )
            raise self.retry(
                countdown=300, exc=exc
            )  # 5 minutes retry delay for training

        raise exc


@celery_app.task(bind=True, base=CallbackTask, name="orchestrate_training")
def orchestrate_training(
    self, training_job_id: str, model_id: str, last_n: int
) -> Dict[str, Any]:
    """
    Orchestrate the complete training process: data collection + preprocessing.

    This function efficiently handles large data collection requests by:
    1. Splitting requests into parallel chunks (100 records each due to STH-Comet limit)
    2. Executing chunks in parallel using Celery
    3. Reordering collected data by timestamp
    4. Handling failures gracefully

    Args:
        training_job_id: Training job ID
        model_id: Model ID
        last_n: Total number of data points to collect (can be > 100)

    Returns:
        Dictionary with orchestration results
    """
    try:
        # Import here to avoid circular imports
        from celery import group

        from src.domain.entities.training_job import DataCollectionJob
        from src.infrastructure.database.mongo_database import MongoDatabase
        from src.infrastructure.repositories.model_repository import ModelRepository
        from src.infrastructure.repositories.training_job_repository import (
            TrainingJobRepository,
        )
        from src.main.config import get_settings

        # Get centralized settings
        settings = get_settings()

        # Initialize dependencies
        database = MongoDatabase(
            mongo_uri=settings.database.mongo_uri,
            db_name=settings.database.database_name,
        )
        training_job_repo = TrainingJobRepository(database)
        model_repo = ModelRepository(database)

        # Initialize STH gateway for getting total count
        from src.infrastructure.gateways.sth_comet_gateway import STHCometGateway

        sth_gateway = STHCometGateway(settings.fiware.sth_url)

        # Get model configuration
        model = asyncio.run(model_repo.find_by_id(UUID(model_id)))
        if not model:
            raise ValueError(f"Model {model_id} not found")

        # Use model's lookback_window
        window_size = model.lookback_window

        logger.info(
            "Starting training orchestration",
            training_job_id=training_job_id,
            model_id=model_id,
            window_size=window_size,
            last_n=last_n,
            entity_type=model.entity_type,
            entity_id=model.entity_id,
            feature=model.feature,
        )

        # Validate training parameters make sense
        min_required_data = (
            window_size + 20
        )  # Need enough for meaningful train/val/test splits
        if last_n < min_required_data:
            raise ValueError(
                f"Insufficient training data requested: {last_n} points. "
                f"Need at least {min_required_data} points  "
                f"for window_size={window_size} to create "
                f"meaningful train/validation/test splits."
            )

        # Update training job status
        asyncio.run(
            training_job_repo.update_training_job_status(
                UUID(training_job_id),
                TrainingStatus.COLLECTING_DATA,
                data_collection_start=datetime.now(timezone.utc),
            )
        )

        # Validate model has required fields
        if not model.entity_type or not model.entity_id:
            raise ValueError(f"Model {model_id} missing entity_type or entity_id")

        # First, get total count of available data
        total_count = asyncio.run(
            sth_gateway.get_total_count_from_header(
                entity_type=model.entity_type,
                entity_id=model.entity_id,
                attribute=model.feature,
                fiware_service="smart",
                fiware_servicepath="/",
            )
        )

        logger.info(
            "Got total available data count",
            total_count=total_count,
            requested_last_n=last_n,
            entity_type=model.entity_type,
            entity_id=model.entity_id,
            attribute=model.feature,
        )

        # Validate we have enough data
        if total_count < last_n:
            logger.warning(
                "Requested more data than available",
                requested=last_n,
                available=total_count,
            )
            # Adjust last_n to available data
            last_n = total_count

        if total_count == 0:
            raise ValueError(f"No data available for entity {model.entity_id}")

        # Calculate optimal data collection strategy
        # STH-Comet has a hard limit (configurable)
        max_per_request = settings.fiware.max_per_request
        collection_jobs = []
        remaining = last_n

        # Calculate starting offset (we want the most recent N points)
        # STH-Comet data is ordered from oldest to newest
        # To get most recent N points, we start from (total_count - last_n)
        current_h_offset = max(0, total_count - last_n)

        # Create collection jobs with proper offsets
        while remaining > 0:
            chunk_size = min(remaining, max_per_request)
            job = DataCollectionJob(h_offset=current_h_offset, last_n=chunk_size)
            collection_jobs.append(job)

            remaining -= chunk_size
            current_h_offset += chunk_size

        # Add collection jobs to training job for tracking
        for job in collection_jobs:
            asyncio.run(
                training_job_repo.add_data_collection_job(UUID(training_job_id), job)
            )

        asyncio.run(
            training_job_repo.update_task_refs(
                UUID(training_job_id),
                task_refs={"orchestration_task_id": self.request.id},
            )
        )

        logger.info(
            "Created data collection strategy",
            training_job_id=training_job_id,
            total_jobs=len(collection_jobs),
            total_requested=last_n,
            total_available=total_count,
            chunk_size=max_per_request,
            starting_offset=max(0, total_count - last_n),
            estimated_parallel_requests=len(collection_jobs),
        )

        # Create parallel data collection tasks
        data_collection_tasks = group(
            [
                collect_data_chunk.s(
                    job_id=str(job.id),
                    training_job_id=training_job_id,
                    entity_type=model.entity_type,
                    entity_id=model.entity_id,
                    attribute=model.feature,
                    h_limit=job.last_n,  # last_n stores the chunk size (h_limit)
                    h_offset=job.h_offset,
                ).set(queue="data_collection", task_id=str(job.id))
                for job in collection_jobs
            ]
        )

        # Execute data collection in parallel
        logger.info(
            "Starting parallel data collection",
            parallel_tasks=len(collection_jobs),
            max_per_task=max_per_request,
        )

        # Use chord to schedule after all data_collection_tasks finish
        from celery import chord

        process_task_id = f"{training_job_id}:process"

        chord_result = chord(
            data_collection_tasks,
            process_collected_data.s(
                training_job_id=training_job_id,
                model_config={
                    "id": str(model.id),
                    "name": model.name,
                    "model_type": model.model_type.value,
                    "rnn_units": model.rnn_units,
                    "dense_units": model.dense_units,
                    "rnn_dropout": model.rnn_dropout,
                    "dense_dropout": model.dense_dropout,
                    "learning_rate": model.learning_rate,
                    "batch_size": model.batch_size,
                    "epochs": model.epochs,
                    "early_stopping_patience": model.early_stopping_patience,
                    "feature": model.feature,
                    "entity_type": model.entity_type,
                    "entity_id": model.entity_id,
                },
                window_size=window_size,
                last_n=last_n,
            ),
        ).apply_async(queue="orchestration", task_id=process_task_id)

        chord_group_id = getattr(chord_result.parent, "id", None)
        task_ref_payload = {
            "processing_task_id": process_task_id,
            "chord_callback_id": chord_result.id,
        }
        if chord_group_id:
            task_ref_payload["chord_group_id"] = chord_group_id

        asyncio.run(
            training_job_repo.update_task_refs(
                UUID(training_job_id),
                task_refs=task_ref_payload,
            )
        )

        logger.info(
            "celery.orchestration.started",
            training_job_id=training_job_id,
            collection_tasks=len(collection_jobs),
            chord_id=chord_result.id,
        )

        return {
            "training_job_id": training_job_id,
            "status": "data_collection_started",
            "collection_tasks": len(collection_jobs),
            "chord_id": chord_result.id,
        }

    except Exception as exc:
        logger.error(
            "celery.orchestration.failed",
            training_job_id=training_job_id,
            error=str(exc),
            exc_info=exc,
        )

        # Update job status to failed
        try:
            from src.infrastructure.database.mongo_database import MongoDatabase
            from src.infrastructure.repositories.training_job_repository import (
                TrainingJobRepository,
            )
            from src.main.config import get_settings

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
                        "orchestration_error": True,
                        "task_id": self.request.id,
                        "model_id": model_id,
                        "requested_data_points": last_n,
                    },
                )
            )
        except Exception as update_error:
            logger.error(
                "celery.orchestration.status_update_failed",
                training_job_id=training_job_id,
                error=str(update_error),
                exc_info=update_error,
            )

        # Don't retry orchestration tasks as they are complex and expensive
        logger.error(
            "celery.orchestration.not_retrying",
            training_job_id=training_job_id,
            model_id=model_id,
            last_n=last_n,
        )
        raise exc


@celery_app.task(bind=True, base=CallbackTask, name="process_collected_data")
def process_collected_data(
    self,
    collection_results: List[Dict[str, Any]],
    training_job_id: str,
    model_config: Dict[str, Any],
    window_size: int,
    last_n: int,
) -> Dict[str, Any]:
    """
    Process collected data and start model training.

    This task processes the results from parallel data collection chunks,
    reorders data chronologically, and initiates model training.
    """
    training_job_repo = None
    try:
        from src.infrastructure.database.mongo_database import MongoDatabase
        from src.infrastructure.repositories.training_job_repository import (
            TrainingJobRepository,
        )
        from src.main.config import get_settings

        settings = get_settings()
        database = MongoDatabase(
            mongo_uri=settings.database.mongo_uri,
            db_name=settings.database.database_name,
        )
        training_job_repo = TrainingJobRepository(database)

        training_job = asyncio.run(training_job_repo.get_by_id(UUID(training_job_id)))
        if training_job and training_job.status in (
            TrainingStatus.CANCELLED,
            TrainingStatus.CANCEL_REQUESTED,
        ):
            logger.info(
                "Skipping data processing because training job was cancelled",
                training_job_id=training_job_id,
            )
            return {
                "training_job_id": training_job_id,
                "status": "cancelled",
                "message": "Job was cancelled before processing.",
            }

        asyncio.run(
            training_job_repo.update_task_refs(
                UUID(training_job_id),
                task_refs={"processing_task_id": self.request.id},
            )
        )

        logger.info(
            "Processing collected data results",
            training_job_id=training_job_id,
            total_chunks=len(collection_results),
        )

        # Process and validate results
        successful_chunks = []
        failed_chunks = []
        all_data_points = []

        for result in collection_results:
            if isinstance(result, dict) and result.get("status") == "completed":
                successful_chunks.append(result)
                all_data_points.extend(result["data_points"])
            else:
                failed_chunks.append(result)
                logger.warning(
                    "Data collection chunk failed",
                    job_id=(
                        result.get("job_id") if isinstance(result, dict) else "unknown"
                    ),
                    error=(
                        result.get("error") if isinstance(result, dict) else str(result)
                    ),
                )

        # Sort data points by timestamp
        if all_data_points:
            all_data_points.sort(key=lambda x: datetime.fromisoformat(x["timestamp"]))
            _log_data_collection_summary(
                total_requested=last_n,
                total_collected=len(all_data_points),
                chunks=len(successful_chunks),
                date_range=(
                    f"{all_data_points[0]['timestamp']} to "
                    f"{all_data_points[-1]['timestamp']}"
                ),
            )
        else:
            logger.warning(
                "No data collected from any chunk", training_job_id=training_job_id
            )

        # Validate data sufficiency
        if len(all_data_points) < window_size:
            raise ValueError(
                "Insufficient data for training: collected "
                f"{len(all_data_points)}, need at least {window_size}"
            )

        # Check if job was cancelled
        training_job = asyncio.run(training_job_repo.get_by_id(UUID(training_job_id)))
        if training_job and training_job.status in (
            TrainingStatus.CANCELLED,
            TrainingStatus.CANCEL_REQUESTED,
        ):
            logger.info(
                "Job was cancelled, skipping training",
                training_job_id=training_job_id,
            )
            return {
                "training_job_id": training_job_id,
                "status": "cancelled",
                "message": "Job was cancelled before training.",
            }

        # Update job status to preprocessing
        asyncio.run(
            training_job_repo.update_training_job_status(
                UUID(training_job_id),
                TrainingStatus.PREPROCESSING,
                data_collection_end=datetime.now(timezone.utc),
                preprocessing_start=datetime.now(timezone.utc),
                total_data_points_collected=len(all_data_points),
            )
        )

        # Start model training task
        logger.info(
            "celery.data_processing.training_dispatch",
            training_job_id=training_job_id,
        )
        training_task_id = f"{training_job_id}:train"
        training_task = (
            train_model_task.s(
                training_job_id=training_job_id,
                model_config=model_config,
                collected_data=all_data_points,
                window_size=window_size,
            )
            .set(queue="model_training")
            .apply_async(task_id=training_task_id)
        )

        asyncio.run(
            training_job_repo.update_task_refs(
                UUID(training_job_id),
                task_refs={"training_task_id": training_task.id},
            )
        )

        logger.info(
            "celery.data_processing.completed",
            training_job_id=training_job_id,
            total_data_points=len(all_data_points),
            training_task_id=training_task.id,
        )

        return {
            "training_job_id": training_job_id,
            "status": "training_started",
            "total_data_points_collected": len(all_data_points),
            "successful_chunks": len(successful_chunks),
            "failed_chunks": len(failed_chunks),
            "training_task_id": training_task.id,
        }

    except Exception as exc:
        logger.error(
            "celery.data_processing.failed",
            training_job_id=training_job_id,
            error=str(exc),
            exc_info=exc,
        )

        try:
            if training_job_repo is not None:
                asyncio.run(
                    training_job_repo.fail_training_job(
                        UUID(training_job_id),
                        error=str(exc),
                        error_details={
                            "processing_error": True,
                            "task_id": self.request.id,
                        },
                    )
                )
        except Exception as update_error:
            logger.error(
                "celery.data_processing.status_update_failed",
                training_job_id=training_job_id,
                error=str(update_error),
                exc_info=update_error,
            )

        raise exc


@celery_app.task(name="cleanup_training_tasks")
def cleanup_training_tasks(training_job_id: str) -> None:
    """Best-effort cleanup of lingering Celery tasks for a training job."""

    try:
        from src.infrastructure.database.mongo_database import MongoDatabase
        from src.infrastructure.repositories.training_job_repository import (
            TrainingJobRepository,
        )
        from src.main.config import get_settings

        settings = get_settings()
        database = MongoDatabase(
            mongo_uri=settings.database.mongo_uri,
            db_name=settings.database.database_name,
        )
        training_job_repo = TrainingJobRepository(database)

        training_job = asyncio.run(training_job_repo.get_by_id(UUID(training_job_id)))

        if not training_job:
            return

        if training_job.status not in {
            TrainingStatus.CANCELLED,
            TrainingStatus.COMPLETED,
            TrainingStatus.FAILED,
        }:
            return

        task_refs = training_job.task_refs or {}
        residual_ids = {
            str(task_id)
            for task_id in (
                task_refs.get("orchestration_task_id"),
                task_refs.get("chord_callback_id"),
                task_refs.get("chord_group_id"),
                task_refs.get("processing_task_id"),
                task_refs.get("training_task_id"),
            )
            if task_id
        }
        residual_ids.update(
            str(task_id)
            for task_id in task_refs.get("data_collection_task_ids", [])
            if task_id
        )

        if residual_ids:
            logger.info(
                "Cleaning up lingering tasks",
                training_job_id=training_job_id,
                task_count=len(residual_ids),
            )
            for task_id in residual_ids:
                try:
                    celery_app.control.revoke(task_id, terminate=False)
                except Exception as revoke_error:  # pragma: no cover - best effort only
                    logger.warning(
                        "Failed to revoke lingering task",
                        training_job_id=training_job_id,
                        task_id=task_id,
                        error=str(revoke_error),
                    )

        if task_refs:
            asyncio.run(
                training_job_repo.update_task_refs(UUID(training_job_id), clear=True)
            )

    except Exception as exc:  # pragma: no cover - cleanup is best effort
        logger.warning(
            "Cleanup routine failed",
            training_job_id=training_job_id,
            error=str(exc),
        )


# Ensure tasks are registered with the Celery app
__all__ = [
    "collect_data_chunk",
    "train_model_task",
    "orchestrate_training",
    "process_collected_data",
    "cleanup_training_tasks",
]
