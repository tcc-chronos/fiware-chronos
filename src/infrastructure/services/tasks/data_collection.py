"""Celery tasks responsible for data collection."""

import asyncio
from datetime import datetime, timezone
from typing import Any, Dict
from uuid import UUID

from src.domain.entities.training_job import DataCollectionStatus, TrainingStatus
from src.infrastructure.services.celery_config import celery_app
from src.infrastructure.services.tasks.base import CallbackTask, logger


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
    """Collect a chunk of data from STH-Comet using hLimit and hOffset."""

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

        settings = get_settings()

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
            }

        asyncio.run(
            training_job_repo.update_data_collection_job_status(
                UUID(training_job_id),
                UUID(job_id),
                DataCollectionStatus.IN_PROGRESS,
                start_time=datetime.now(timezone.utc),
            )
        )

        data_points_dto = asyncio.run(
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

        data_points = [
            {
                "timestamp": dto.timestamp.isoformat(),
                "value": dto.value,
            }
            for dto in data_points_dto
        ]

        total_collected = len(data_points)

        asyncio.run(
            training_job_repo.update_data_collection_job_status(
                UUID(training_job_id),
                UUID(job_id),
                DataCollectionStatus.COMPLETED,
                end_time=datetime.now(timezone.utc),
                data_points_collected=total_collected,
            )
        )

        logger.info(
            "Completed data collection chunk",
            job_id=job_id,
            training_job_id=training_job_id,
            collected_points=total_collected,
            requested_points=h_limit,
            h_offset=h_offset,
        )

        return {
            "job_id": job_id,
            "training_job_id": training_job_id,
            "status": "completed",
            "data_points_collected": total_collected,
            "data_points": data_points,
            "h_offset": h_offset,
            "chunk_info": {
                "requested_h_limit": h_limit,
                "actual_collected": total_collected,
                "h_offset": h_offset,
            },
        }

    except Exception as exc:  # pragma: no cover - defensive logging
        logger.error(
            "celery.data_collection.failed",
            job_id=job_id,
            training_job_id=training_job_id,
            error=str(exc),
            exc_info=exc,
        )

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
        except Exception as update_error:  # pragma: no cover
            logger.error(
                "celery.data_collection.status_update_failed",
                job_id=job_id,
                training_job_id=training_job_id,
                error=str(update_error),
                exc_info=update_error,
            )

        if self.request.retries < self.max_retries:
            retry_delay = min(60 * (2**self.request.retries), 300)
            logger.info(
                "celery.data_collection.retry_scheduled",
                job_id=job_id,
                attempt=self.request.retries + 1,
                delay_seconds=retry_delay,
            )
            raise self.retry(countdown=retry_delay, exc=exc)

        return {
            "job_id": job_id,
            "training_job_id": training_job_id,
            "status": "failed",
            "error": str(exc),
            "h_offset": h_offset,
            "h_limit": h_limit,
            "retries_exhausted": True,
        }
