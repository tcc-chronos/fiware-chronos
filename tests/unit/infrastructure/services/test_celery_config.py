from __future__ import annotations

from src.infrastructure.services.celery_config import create_celery_app


def test_create_celery_app_uses_env(monkeypatch) -> None:
    monkeypatch.setenv("CELERY_BROKER_URL", "amqp://env")
    monkeypatch.setenv("CELERY_RESULT_BACKEND", "redis://env")

    app = create_celery_app()

    assert app.conf.broker_url == "amqp://env"
    assert app.conf.result_backend == "redis://env"
    assert "collect_data_chunk" in app.conf.task_routes


def test_create_celery_app_with_explicit_params() -> None:
    app = create_celery_app(
        broker_url="amqp://explicit",
        backend_url="redis://explicit",
    )
    assert app.conf.broker_url == "amqp://explicit"
    assert app.conf.result_backend == "redis://explicit"
