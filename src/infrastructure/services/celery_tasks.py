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
        logger.info(f"Task {task_id} succeeded", task_id=task_id, result=retval)

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Called on task failure."""
        logger.error(
            f"Task {task_id} failed",
            task_id=task_id,
            error=str(exc),
            traceback=einfo.traceback,
        )


@celery_app.task(bind=True, base=CallbackTask, name="collect_data_chunk")
def collect_data_chunk(
    self,
    job_id: str,
    training_job_id: str,
    entity_type: str,
    entity_id: str,
    attribute: str,
    last_n: int,
    h_offset: int,
    fiware_service: str = "smart",
    fiware_servicepath: str = "/",
) -> Dict[str, Any]:
    """
    Collect a chunk of data from STH-Comet.

    Args:
        job_id: Data collection job ID
        training_job_id: Training job ID
        entity_type: FIWARE entity type
        entity_id: FIWARE entity ID
        attribute: Attribute to collect
        last_n: Number of data points to collect (max 100 per STH-Comet limitation)
        h_offset: Historical offset
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
            last_n=last_n,
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
                last_n=last_n,
                h_offset=h_offset,
                fiware_service=fiware_service,
                fiware_servicepath=fiware_servicepath,
            )
        )

        # Convert to serializable format with precise timestamp handling
        data_points = [
            {
                "timestamp": point.timestamp.isoformat(),
                "value": point.value,
                "h_offset": h_offset,  # Include offset for debugging
            }
            for point in collected_data
        ]

        # Update job status to completed
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
                "requested_last_n": last_n,
                "actual_collected": len(collected_data),
                "offset": h_offset,
            },
        }

    except Exception as exc:
        logger.error(
            "Data collection chunk failed",
            job_id=job_id,
            training_job_id=training_job_id,
            error=str(exc),
            h_offset=h_offset,
            last_n=last_n,
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
        except Exception as e:
            logger.error(f"Failed to update job status: {e}")

        # Retry the task with exponential backoff
        if self.request.retries < self.max_retries:
            retry_delay = min(60 * (2**self.request.retries), 300)  # Max 5 minutes
            logger.info(
                f"Retrying data collection chunk {job_id} "
                f"(attempt {self.request.retries + 1}) in {retry_delay}s"
            )
            raise self.retry(countdown=retry_delay, exc=exc)

        # Return error information for better debugging
        return {
            "job_id": job_id,
            "training_job_id": training_job_id,
            "status": "failed",
            "error": str(exc),
            "h_offset": h_offset,
            "last_n": last_n,
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
        from src.application.use_cases.model_training_use_case import (
            ModelTrainingUseCase,
        )
        from src.domain.entities.model import Model, ModelType
        from src.infrastructure.database.mongo_database import MongoDatabase
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
        model_training = ModelTrainingUseCase()

        # Update training job status
        asyncio.run(
            training_job_repo.update_training_job_status(
                UUID(training_job_id),
                TrainingStatus.TRAINING,
                training_start=datetime.now(timezone.utc),
            )
        )

        # Convert model config dict to Model entity
        model = Model(
            id=UUID(model_config["id"]),
            name=model_config["name"],
            model_type=ModelType(model_config["model_type"]),
            rnn_units=model_config["rnn_units"],
            dense_units=model_config["dense_units"],
            rnn_dropout=model_config["rnn_dropout"],
            dense_dropout=model_config["dense_dropout"],
            learning_rate=model_config["learning_rate"],
            batch_size=model_config["batch_size"],
            epochs=model_config["epochs"],
            early_stopping_patience=model_config.get("early_stopping_patience"),
            feature=model_config["feature"],
            entity_type=model_config.get("entity_type"),
            entity_id=model_config.get("entity_id"),
        )

        # Convert collected data to DTOs
        data_dtos = [
            CollectedDataDTO(
                timestamp=datetime.fromisoformat(item["timestamp"]), value=item["value"]
            )
            for item in collected_data
        ]

        # Train model
        metrics, model_path, x_scaler_path, y_scaler_path, metadata_path = asyncio.run(
            model_training.execute(
                model_config=model, collected_data=data_dtos, window_size=window_size
            )
        )

        # Update training job with results
        asyncio.run(
            training_job_repo.complete_training_job(
                UUID(training_job_id),
                metrics=metrics,
                model_artifact_path=model_path,
                x_scaler_path=x_scaler_path,
                y_scaler_path=y_scaler_path,
                metadata_path=metadata_path,
            )
        )

        logger.info(
            "Model training completed successfully",
            training_job_id=training_job_id,
            metrics=metrics.__dict__,
        )

        return {
            "training_job_id": training_job_id,
            "status": "completed",
            "metrics": metrics.__dict__,
            "model_path": model_path,
            "x_scaler_path": x_scaler_path,
            "y_scaler_path": y_scaler_path,
            "metadata_path": metadata_path,
        }

    except Exception as exc:
        logger.error(
            "Model training failed", training_job_id=training_job_id, error=str(exc)
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
        except Exception as e:
            logger.error(f"Failed to update training job status: {e}")

        # Retry the task
        if self.request.retries < self.max_retries:
            logger.info(
                f"Retrying model training {training_job_id} "
                f"(attempt {self.request.retries + 1})"
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

        # Update training job status
        asyncio.run(
            training_job_repo.update_training_job_status(
                UUID(training_job_id),
                TrainingStatus.COLLECTING_DATA,
                data_collection_start=datetime.now(timezone.utc),
            )
        )

        # Calculate optimal data collection strategy
        # STH-Comet has a hard limit (configurable)
        max_per_request = settings.fiware.max_per_request
        collection_jobs = []
        remaining = last_n
        h_offset = 0

        # Create collection jobs with better distribution
        while remaining > 0:
            chunk_size = min(remaining, max_per_request)
            job = DataCollectionJob(h_offset=h_offset, last_n=chunk_size)
            collection_jobs.append(job)

            remaining -= chunk_size
            h_offset += chunk_size

        # Add collection jobs to training job for tracking
        for job in collection_jobs:
            asyncio.run(
                training_job_repo.add_data_collection_job(UUID(training_job_id), job)
            )

        logger.info(
            "Created data collection strategy",
            training_job_id=training_job_id,
            total_jobs=len(collection_jobs),
            total_requested=last_n,
            chunk_size=max_per_request,
            estimated_parallel_requests=len(collection_jobs),
        )

        # Create parallel data collection tasks
        data_collection_tasks = group(
            [
                celery_app.tasks["collect_data_chunk"].s(
                    job_id=str(job.id),
                    training_job_id=training_job_id,
                    entity_type=model.entity_type,
                    entity_id=model.entity_id,
                    attribute=model.feature,
                    last_n=job.last_n,
                    h_offset=job.h_offset,
                )
                for job in collection_jobs
            ]
        )

        # Execute data collection in parallel
        logger.info(
            "Starting parallel data collection",
            parallel_tasks=len(collection_jobs),
            max_per_task=max_per_request,
        )

        collection_result = data_collection_tasks.apply_async()
        collection_results = collection_result.get()  # Wait for all tasks to complete

        # Process and validate results
        successful_chunks = []
        failed_chunks = []
        all_data_points = []

        for result in collection_results:
            if result["status"] == "completed":
                successful_chunks.append(result)
                chunk_data = result["data_points"]
                all_data_points.extend(chunk_data)
                logger.info(
                    "Chunk completed successfully",
                    job_id=result["job_id"],
                    data_points=result["data_points_collected"],
                    h_offset=result["h_offset"],
                )
            else:
                failed_chunks.append(result)
                logger.error(
                    "Chunk failed",
                    job_id=result.get("job_id"),
                    error=result.get("error"),
                )

        # Sort data points by timestamp to ensure proper chronological order
        # This is crucial for time series model training
        if all_data_points:
            logger.info("Reordering collected data by timestamp")
            all_data_points.sort(key=lambda x: x["timestamp"])

            # Log comprehensive summary
            date_range = (
                f"{all_data_points[0]['timestamp']} to "
                f"{all_data_points[-1]['timestamp']}"
            )
            _log_data_collection_summary(
                total_requested=last_n,
                total_collected=len(all_data_points),
                chunks=len(successful_chunks),
                date_range=date_range,
            )
        else:
            logger.warning(
                "No data points collected",
                training_job_id=training_job_id,
                failed_chunks=len(failed_chunks),
            )

        # Check if we have enough data to proceed
        if len(failed_chunks) > 0:
            logger.warning(
                f"Some data collection chunks failed "
                f"({len(failed_chunks)}/{len(collection_jobs)})"
            )

        if len(all_data_points) < window_size:
            raise ValueError(
                f"Insufficient data collected: got {len(all_data_points)}, "
                f"need at least {window_size} for window size"
            )

        # Update training job status
        asyncio.run(
            training_job_repo.update_training_job_status(
                UUID(training_job_id),
                TrainingStatus.PREPROCESSING,
                data_collection_end=datetime.now(timezone.utc),
                preprocessing_start=datetime.now(timezone.utc),
                total_data_points_collected=len(all_data_points),
            )
        )

        # Convert model to dictionary for serialization
        model_dict = {
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
        }

        # Start model training task
        logger.info("Starting model training with collected data")
        training_task = celery_app.tasks["train_model_task"].delay(
            training_job_id=training_job_id,
            model_config=model_dict,
            collected_data=all_data_points,
            window_size=window_size,
        )

        training_result = training_task.get()  # Wait for training to complete

        logger.info(
            "Training orchestration completed successfully",
            training_job_id=training_job_id,
            final_status=training_result["status"],
            total_data_points_used=len(all_data_points),
        )

        return {
            "training_job_id": training_job_id,
            "status": "completed",
            "data_collection_summary": {
                "requested_points": last_n,
                "collected_points": len(all_data_points),
                "collection_jobs": len(collection_jobs),
                "successful_chunks": len(successful_chunks),
                "failed_chunks": len(failed_chunks),
                "data_sorted": True,
            },
            "training_result": training_result,
        }

    except Exception as exc:
        logger.error(
            "Training orchestration failed",
            training_job_id=training_job_id,
            error=str(exc),
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
        except Exception as e:
            logger.error(f"Failed to update training job status: {e}")

        # Don't retry orchestration tasks as they are complex and expensive
        logger.error(
            "Training orchestration failed - not retrying due to complexity",
            training_job_id=training_job_id,
            model_id=model_id,
            last_n=last_n,
        )
        raise exc
