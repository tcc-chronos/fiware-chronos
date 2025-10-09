"""Celery tasks responsible for scheduling recurring forecasts."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from src.infrastructure.services.celery_config import celery_app
from src.infrastructure.services.tasks.base import CallbackTask, logger
from src.infrastructure.settings import get_settings

_DEFAULT_SCHEDULER_INTERVAL = 60


@celery_app.task(bind=True, base=CallbackTask, name="schedule_forecasts")
def schedule_forecasts(self) -> dict[str, Any]:
    """Scan for training jobs that require predictions and enqueue them."""

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

        now = datetime.now(timezone.utc)
        ready_jobs = asyncio.run(
            training_job_repo.get_prediction_ready_jobs(
                reference_time=now,
                limit=50,
            )
        )

        dispatched = 0
        skipped = 0
        next_interval: Optional[int] = None

        for job in ready_jobs:
            if not job.prediction_config.enabled:
                skipped += 1
                continue
            if not job.prediction_config.entity_id:
                logger.warning(
                    "forecast.scheduler.missing_entity",
                    training_job_id=str(job.id),
                )
                skipped += 1
                continue
            if (
                job.sampling_interval_seconds is None
                or job.sampling_interval_seconds <= 0
            ):
                logger.warning(
                    "forecast.scheduler.missing_interval",
                    training_job_id=str(job.id),
                )
                skipped += 1
                continue

            next_run = now + timedelta(seconds=job.sampling_interval_seconds)
            claimed = asyncio.run(
                training_job_repo.claim_prediction_schedule(
                    job.id,
                    expected_next_prediction_at=job.next_prediction_at,
                    next_prediction_at=next_run,
                )
            )
            if not claimed:
                logger.info(
                    "forecast.scheduler.claim_failed",
                    training_job_id=str(job.id),
                )
                skipped += 1
                continue

            celery_app.send_task(
                "execute_forecast",
                kwargs={
                    "training_job_id": str(job.id),
                    "model_id": str(job.model_id) if job.model_id else None,
                },
                queue="forecast_execution",
            )
            dispatched += 1

            if next_interval is None or job.sampling_interval_seconds < next_interval:
                next_interval = job.sampling_interval_seconds

        countdown = next_interval or _DEFAULT_SCHEDULER_INTERVAL
        countdown = max(int(countdown), 1)
        try:
            self.apply_async(countdown=countdown)
        except Exception as exc:  # pragma: no cover - best effort warning
            logger.warning("forecast.scheduler.reschedule_failed", error=str(exc))

        return {
            "dispatched": dispatched,
            "skipped": skipped,
            "timestamp": now.isoformat(),
            "next_iteration_in": countdown,
        }

    except Exception as exc:  # pragma: no cover - defensive logging
        logger.error("forecast.scheduler.failed", error=str(exc), exc_info=exc)
        raise
