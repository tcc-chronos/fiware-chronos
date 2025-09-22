"""
Infrastructure Services - Celery Configuration

This module contains the Celery configuration and task definitions
for asynchronous data collection and model training.
"""

import os
from typing import Optional

from celery import Celery


def create_celery_app(
    broker_url: Optional[str] = None,
    backend_url: Optional[str] = None,
) -> Celery:
    """
    Create and configure Celery application.

    Args:
        broker_url: Message broker URL (uses env var if not provided)
        backend_url: Result backend URL (uses env var if not provided)

    Returns:
        Configured Celery application
    """
    # Use provided URLs or fall back to environment variables with defaults
    effective_broker = broker_url or os.getenv(
        "CELERY_BROKER_URL", "amqp://chronos:chronos@rabbitmq:5672/chronos"
    )
    effective_backend = backend_url or os.getenv(
        "CELERY_RESULT_BACKEND", "redis://redis:6379/0"
    )

    # Create Celery application
    app = Celery(
        "chronos_worker",
        broker=effective_broker,
        backend=effective_backend,
        include=[
            "src.infrastructure.services.tasks.data_collection",
            "src.infrastructure.services.tasks.training",
            "src.infrastructure.services.tasks.processing",
            "src.infrastructure.services.tasks.orchestration",
            "src.infrastructure.services.tasks.cleanup",
        ],
    )

    # Configure Celery
    app.conf.update(
        # Task configuration
        task_serializer="json",
        accept_content=["json"],
        result_serializer="json",
        timezone="UTC",
        enable_utc=True,
        # Result backend configuration
        result_expires=3600,  # 1 hour
        # Task routing
        task_routes={
            "collect_data_chunk": {"queue": "data_collection"},
            "train_model_task": {"queue": "model_training"},
            "orchestrate_training": {"queue": "orchestration"},
        },
        # Worker configuration
        worker_prefetch_multiplier=1,
        task_acks_late=True,
        worker_max_tasks_per_child=100,
        # Task retry configuration
        task_default_retry_delay=60,  # 1 minute
        task_max_retries=3,
        # Beat configuration (for periodic tasks if needed)
        beat_schedule={},
    )

    return app


# Create default instance for backwards compatibility
celery_app = create_celery_app()
