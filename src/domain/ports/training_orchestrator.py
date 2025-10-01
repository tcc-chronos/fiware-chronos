"""Domain port for training orchestration dispatch."""

from __future__ import annotations

from typing import Protocol, Sequence
from uuid import UUID


class ITrainingOrchestrator(Protocol):
    """Defines how training jobs are dispatched to background workers."""

    async def dispatch_training_job(
        self,
        *,
        training_job_id: UUID,
        model_id: UUID,
        last_n: int,
    ) -> str:
        """Queue a training job for asynchronous execution.

        Returns:
            Identifier of the dispatched task (if available).
        """
        ...

    async def revoke_tasks(self, task_ids: Sequence[str]) -> None:
        """Attempt to revoke the provided asynchronous tasks."""

    async def schedule_cleanup(
        self, training_job_id: UUID, countdown_seconds: int = 60
    ) -> None:
        """Schedule cleanup of orchestration artefacts after cancellation."""
