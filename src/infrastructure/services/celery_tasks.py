"""Celery task registry.

This module keeps backward compatibility with the previous single-file
implementation by re-exporting the task objects that are now organised
across specialised modules.
"""

from src.infrastructure.services.tasks.base import (
    CallbackTask,
    _log_data_collection_summary,
    log_data_collection_summary,
    logger,
)
from src.infrastructure.services.tasks.cleanup import cleanup_training_tasks
from src.infrastructure.services.tasks.data_collection import collect_data_chunk
from src.infrastructure.services.tasks.orchestration import orchestrate_training
from src.infrastructure.services.tasks.processing import process_collected_data
from src.infrastructure.services.tasks.training import train_model_task

__all__ = [
    "CallbackTask",
    "cleanup_training_tasks",
    "collect_data_chunk",
    "_log_data_collection_summary",
    "log_data_collection_summary",
    "logger",
    "orchestrate_training",
    "process_collected_data",
    "train_model_task",
]
