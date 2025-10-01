from __future__ import annotations

import os
from unittest.mock import MagicMock

import pytest

from src.main.worker import create_worker, main


class _StubCeleryApp:
    def __init__(self) -> None:
        self.main = "chronos"
        self.worker_main = MagicMock()


@pytest.fixture(autouse=True)
def patch_create_celery(monkeypatch):
    monkeypatch.setattr(
        "src.infrastructure.services.celery_config.create_celery_app",
        lambda **kwargs: _StubCeleryApp(),
    )


def test_create_worker_sets_environment(monkeypatch) -> None:
    monkeypatch.delenv("CELERY_BROKER_URL", raising=False)
    monkeypatch.delenv("CELERY_RESULT_BACKEND", raising=False)

    worker_app = create_worker()

    assert worker_app.main == "chronos"
    assert os.environ["CELERY_BROKER_URL"].startswith("amqp://")
    assert os.environ["CELERY_RESULT_BACKEND"].startswith("redis://")


def test_main_invokes_worker(monkeypatch) -> None:
    stub_app = _StubCeleryApp()
    monkeypatch.setattr("src.main.worker.create_worker", lambda: stub_app)

    main()

    stub_app.worker_main.assert_called_once()
