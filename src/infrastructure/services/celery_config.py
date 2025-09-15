"""
Infrastructure Services - Celery Configuration

This module contains the Celery configuration and task definitions
for asynchronous data collection and model training.
"""

import os

from celery import Celery

# Create Celery application
celery_app = Celery(
    "chronos_worker",
    broker=os.getenv("CELERY_BROKER_URL", "redis://redis:6379/0"),
    backend=os.getenv("CELERY_RESULT_BACKEND", "redis://redis:6379/0"),
    include=["src.infrastructure.services.celery_tasks"],
)

# Configure Celery
celery_app.conf.update(
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
        "src.infrastructure.services.celery_tasks.collect_data_chunk": {
            "queue": "data_collection"
        },
        "src.infrastructure.services.celery_tasks.train_model_task": {
            "queue": "model_training"
        },
        "src.infrastructure.services.celery_tasks.orchestrate_training": {
            "queue": "orchestration"
        },
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
