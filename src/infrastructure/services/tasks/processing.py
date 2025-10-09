"""Celery tasks that process collected data before training."""

import asyncio
from collections import defaultdict
from datetime import datetime, timezone
from statistics import mean, median
from typing import Any, Dict, List
from uuid import UUID

from src.domain.entities.model import ModelStatus
from src.domain.entities.training_job import DataCollectionStatus, TrainingStatus
from src.infrastructure.services.celery_config import celery_app
from src.infrastructure.services.tasks.base import (
    CallbackTask,
    log_data_collection_summary,
    logger,
)
from src.infrastructure.services.tasks.training import train_model_task


@celery_app.task(bind=True, base=CallbackTask, name="process_collected_data")
def process_collected_data(
    self,
    collection_results: List[Dict[str, Any]],
    training_job_id: str,
    model_config: Dict[str, Any],
    window_size: int,
    last_n: int,
) -> Dict[str, Any]:
    """Process collected data and start model training."""

    training_job_repo = None
    model_repo = None

    def restore_model_status() -> None:
        if training_job_repo is None or model_repo is None:
            return
        training_job = asyncio.run(training_job_repo.get_by_id(UUID(training_job_id)))
        if not training_job or not training_job.model_id:
            return
        model_record = asyncio.run(model_repo.find_by_id(training_job.model_id))
        if not model_record:
            return
        target_status = (
            ModelStatus.TRAINED
            if model_record.has_trained_artifacts()
            else ModelStatus.DRAFT
        )
        if model_record.status == target_status:
            return
        model_record.status = target_status
        model_record.update_timestamp()
        asyncio.run(model_repo.update(model_record))

    try:
        from src.infrastructure.database.mongo_database import MongoDatabase
        from src.infrastructure.repositories.model_repository import ModelRepository
        from src.infrastructure.repositories.training_job_repository import (
            TrainingJobRepository,
        )
        from src.infrastructure.settings import get_settings

        settings = get_settings()
        database = MongoDatabase(
            mongo_uri=settings.database.mongo_uri,
            db_name=settings.database.database_name,
        )
        training_job_repo = TrainingJobRepository(database)
        model_repo = ModelRepository(database)

        training_job = asyncio.run(training_job_repo.get_by_id(UUID(training_job_id)))
        if training_job and training_job.status in (
            TrainingStatus.CANCELLED,
            TrainingStatus.CANCEL_REQUESTED,
        ):
            logger.info(
                "Skipping data processing because training job was cancelled",
                training_job_id=training_job_id,
            )
            restore_model_status()
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
            "Processing collected data",
            training_job_id=training_job_id,
            total_chunks=len(collection_results),
        )

        successful_chunks = []
        failed_chunks = []

        for chunk in collection_results:
            if chunk.get("status") == "completed":
                successful_chunks.append(chunk)
            else:
                failed_chunks.append(chunk)

        if not successful_chunks:
            logger.error(
                "No successful data collection chunks",
                training_job_id=training_job_id,
                failed_chunks=len(failed_chunks),
            )
            restore_model_status()
            asyncio.run(
                training_job_repo.update_training_job_status(
                    UUID(training_job_id),
                    TrainingStatus.FAILED,
                    preprocessing_end=datetime.now(timezone.utc),
                )
            )
            asyncio.run(
                training_job_repo.fail_training_job(
                    UUID(training_job_id),
                    error="No data collected",
                    error_details={
                        "failed_chunks": [
                            chunk.get("error") for chunk in failed_chunks
                        ],
                    },
                )
            )
            return {
                "training_job_id": training_job_id,
                "status": "failed",
                "error": "No successful data collection chunks",
                "failed_chunks": failed_chunks,
            }

        all_data_points: List[Dict[str, Any]] = []
        for chunk in successful_chunks:
            data_points = chunk.get("data_points", [])
            if isinstance(data_points, list):
                all_data_points.extend(data_points)

        def extract_timestamp(entry: Dict[str, Any]) -> str:
            timestamp = entry.get("timestamp")
            if timestamp is None:
                return ""
            if isinstance(timestamp, str):
                return timestamp
            return str(timestamp)

        all_data_points.sort(key=extract_timestamp)

        unique_data = {}
        for data_point in all_data_points:
            timestamp = data_point.get("timestamp")
            if timestamp not in unique_data:
                unique_data[timestamp] = data_point

        aggregated_data = defaultdict(list)
        for data_point in all_data_points:
            timestamp = data_point.get("timestamp")
            value = data_point.get("value")
            if timestamp and value is not None:
                aggregated_data[timestamp].append(value)

        merged_data = []
        for timestamp, values in aggregated_data.items():
            merged_data.append({"timestamp": timestamp, "value": mean(values)})

        merged_data.sort(key=lambda x: x["timestamp"])

        sampling_interval_seconds = None
        if len(merged_data) >= 2:
            timestamp_deltas: List[float] = []
            previous_ts = None
            for entry in merged_data:
                ts_raw = entry.get("timestamp")
                if not ts_raw:
                    continue
                try:
                    current_ts = datetime.fromisoformat(str(ts_raw))
                except ValueError:
                    logger.debug(
                        "processing.invalid_timestamp",
                        training_job_id=training_job_id,
                        timestamp_value=ts_raw,
                    )
                    continue
                if previous_ts is not None:
                    delta_seconds = (current_ts - previous_ts).total_seconds()
                    if delta_seconds > 0:
                        timestamp_deltas.append(delta_seconds)
                previous_ts = current_ts
            if timestamp_deltas:
                sampling_interval_seconds = int(median(timestamp_deltas))

        total_collected = len(merged_data)
        if total_collected == 0:
            logger.error(
                "No data after preprocessing",
                training_job_id=training_job_id,
                total_chunks=len(successful_chunks),
            )
            restore_model_status()
            asyncio.run(
                training_job_repo.update_training_job_status(
                    UUID(training_job_id),
                    TrainingStatus.FAILED,
                    preprocessing_end=datetime.now(timezone.utc),
                )
            )
            asyncio.run(
                training_job_repo.fail_training_job(
                    UUID(training_job_id),
                    error="No data after preprocessing",
                )
            )
            return {
                "training_job_id": training_job_id,
                "status": "failed",
                "error": "No data after preprocessing",
            }

        asyncio.run(
            training_job_repo.update_training_job_status(
                UUID(training_job_id),
                TrainingStatus.COLLECTING_DATA,
                data_collection_end=datetime.now(timezone.utc),
                total_data_points_collected=total_collected,
            )
        )

        asyncio.run(
            training_job_repo.update_training_job_status(
                UUID(training_job_id),
                TrainingStatus.PREPROCESSING,
                data_collection_end=datetime.now(timezone.utc),
                preprocessing_start=datetime.now(timezone.utc),
                total_data_points_collected=total_collected,
            )
        )

        logger.info(
            "Data preprocessing completed",
            training_job_id=training_job_id,
            total_points=total_collected,
        )

        all_h_offsets = [
            int(chunk.get("h_offset", 0))
            for chunk in successful_chunks
            if chunk.get("h_offset") is not None
        ]
        min_offset = min(all_h_offsets) if all_h_offsets else 0
        max_offset = max(all_h_offsets) if all_h_offsets else 0

        date_range = None
        if merged_data:
            start_time = merged_data[0]["timestamp"]
            end_time = merged_data[-1]["timestamp"]
            date_range = f"{start_time} to {end_time}"

        log_data_collection_summary(
            total_requested=last_n,
            total_collected=total_collected,
            chunks=len(successful_chunks),
            date_range=date_range,
        )

        asyncio.run(
            training_job_repo.update_training_job_status(
                UUID(training_job_id),
                TrainingStatus.PREPROCESSING,
                preprocessing_end=datetime.now(timezone.utc),
            )
        )

        if sampling_interval_seconds:
            next_prediction = None
            if training_job and training_job.next_prediction_at is None:
                next_prediction = datetime.now(timezone.utc)
            asyncio.run(
                training_job_repo.update_sampling_metadata(
                    UUID(training_job_id),
                    sampling_interval_seconds=int(sampling_interval_seconds),
                    next_prediction_at=next_prediction,
                )
            )

        for chunk in failed_chunks:
            job_id = chunk.get("job_id")
            if not job_id:
                continue
            asyncio.run(
                training_job_repo.update_data_collection_job_status(
                    UUID(training_job_id),
                    UUID(job_id),
                    DataCollectionStatus.FAILED,
                    end_time=datetime.now(timezone.utc),
                    error=chunk.get("error"),
                )
            )

        logger.info(
            "celery.data_processing.training_dispatch",
            training_job_id=training_job_id,
        )
        training_task_id = f"{training_job_id}:train"
        training_task = (
            train_model_task.s(
                training_job_id=training_job_id,
                model_config=model_config,
                collected_data=merged_data,
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
            total_data_points=total_collected,
            training_task_id=training_task.id,
            min_offset=min_offset,
            max_offset=max_offset,
        )

        return {
            "training_job_id": training_job_id,
            "status": "training_started",
            "total_data_points_collected": total_collected,
            "successful_chunks": len(successful_chunks),
            "failed_chunks": len(failed_chunks),
            "training_task_id": training_task.id,
        }

    except Exception as exc:  # pragma: no cover - defensive logging
        logger.error(
            "celery.data_processing.failed",
            training_job_id=training_job_id,
            error=str(exc),
            exc_info=exc,
        )
        restore_model_status()

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
        except Exception as update_error:  # pragma: no cover
            logger.error(
                "celery.data_processing.status_update_failed",
                training_job_id=training_job_id,
                error=str(update_error),
                exc_info=update_error,
            )

        raise exc
