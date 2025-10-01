from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from src.infrastructure.services.training_orchestrator import CeleryTrainingOrchestrator


@pytest.fixture(autouse=True)
def patch_to_thread(monkeypatch):
    async def immediate(func, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(
        "src.infrastructure.services.training_orchestrator.asyncio.to_thread",
        immediate,
    )


@pytest.mark.asyncio
async def test_dispatch_training_job_sends_task(monkeypatch) -> None:
    orchestrator = CeleryTrainingOrchestrator(queue_name="custom")

    send_task = MagicMock(return_value=SimpleNamespace(id="task-123"))
    monkeypatch.setattr(
        "src.infrastructure.services.training_orchestrator.celery_app.send_task",
        send_task,
    )

    task_id = await orchestrator.dispatch_training_job(
        training_job_id=uuid4(), model_id=uuid4(), last_n=10
    )

    assert task_id == "task-123"
    send_task.assert_called_once()


@pytest.mark.asyncio
async def test_revoke_tasks_ignores_empty(monkeypatch) -> None:
    orchestrator = CeleryTrainingOrchestrator()
    revoke = MagicMock()
    monkeypatch.setattr(
        "src.infrastructure.services.training_orchestrator.celery_app.control.revoke",
        revoke,
    )

    await orchestrator.revoke_tasks([])
    revoke.assert_not_called()

    await orchestrator.revoke_tasks(["id-1", "id-2"])
    assert revoke.call_count == 2


@pytest.mark.asyncio
async def test_schedule_cleanup_sends_task(monkeypatch) -> None:
    orchestrator = CeleryTrainingOrchestrator()
    send_task = MagicMock()
    monkeypatch.setattr(
        "src.infrastructure.services.training_orchestrator.celery_app.send_task",
        send_task,
    )

    await orchestrator.schedule_cleanup(uuid4(), countdown_seconds=30)

    send_task.assert_called_once()
    kwargs = send_task.call_args.kwargs
    assert kwargs["countdown"] == 30
