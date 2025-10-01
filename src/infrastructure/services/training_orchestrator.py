"""Celery-backed implementation of the training orchestrator port."""

from __future__ import annotations

import asyncio
from typing import Optional, Sequence
from uuid import UUID

from src.domain.ports.training_orchestrator import ITrainingOrchestrator
from src.infrastructure.services.celery_config import celery_app
from src.shared import get_logger

logger = get_logger(__name__)


class CeleryTrainingOrchestrator(ITrainingOrchestrator):
    """Dispatch training jobs through Celery."""

    def __init__(self, queue_name: str = "orchestration") -> None:
        self._queue_name = queue_name

    async def dispatch_training_job(
        self,
        *,
        training_job_id: UUID,
        model_id: UUID,
        last_n: int,
    ) -> str:
        """Send the orchestration task to Celery asynchronously."""

        def _send_task() -> str:
            logger.info(
                "training_orchestrator.dispatch",
                training_job_id=str(training_job_id),
                model_id=str(model_id),
                last_n=last_n,
                queue=self._queue_name,
            )
            result = celery_app.send_task(
                "orchestrate_training",
                kwargs={
                    "training_job_id": str(training_job_id),
                    "model_id": str(model_id),
                    "last_n": last_n,
                },
                queue=self._queue_name,
            )
            return result.id

        task_id: Optional[str] = await asyncio.to_thread(_send_task)
        return task_id or ""

    async def revoke_tasks(self, task_ids: Sequence[str]) -> None:
        if not task_ids:
            return

        def _revoke() -> None:
            for task_id in task_ids:
                try:
                    celery_app.control.revoke(
                        task_id,
                        terminate=True,
                        signal="SIGTERM",
                    )
                except Exception as exc:  # pragma: no cover - best effort
                    logger.warning(
                        "training_orchestrator.revoke_failed",
                        task_id=task_id,
                        error=str(exc),
                    )

        await asyncio.to_thread(_revoke)

    async def schedule_cleanup(
        self, training_job_id: UUID, countdown_seconds: int = 60
    ) -> None:
        def _schedule() -> None:
            celery_app.send_task(
                "cleanup_training_tasks",
                args=[str(training_job_id)],
                queue=self._queue_name,
                countdown=countdown_seconds,
            )

        await asyncio.to_thread(_schedule)
