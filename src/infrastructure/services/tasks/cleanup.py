"""Celery tasks for cleaning up residual workers metadata."""

import asyncio
from uuid import UUID

from src.domain.entities.training_job import TrainingStatus
from src.infrastructure.services.celery_config import celery_app
from src.infrastructure.services.tasks.base import logger


@celery_app.task(name="cleanup_training_tasks")
def cleanup_training_tasks(training_job_id: str) -> None:
    """Best-effort cleanup of lingering Celery tasks for a training job."""

    try:
        from src.infrastructure.database.mongo_database import MongoDatabase
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
            key: task_id
            for key, task_id in task_refs.items()
            if task_id and key != "cleanup_task_id"
        }

        if not residual_ids:
            return

        logger.info(
            "Cleaning up residual task references",
            training_job_id=training_job_id,
            residual_tasks=residual_ids,
        )

        asyncio.run(
            training_job_repo.update_task_refs(
                UUID(training_job_id),
                clear=True,
            )
        )

    except Exception as exc:  # pragma: no cover - defensive logging
        logger.error(
            "celery.cleanup.failed",
            training_job_id=training_job_id,
            error=str(exc),
            exc_info=exc,
        )
