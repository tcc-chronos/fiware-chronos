"""Shared Celery infrastructure components."""

from typing import Optional

import structlog
from celery import Task

logger = structlog.get_logger(__name__)


def log_data_collection_summary(
    total_requested: int,
    total_collected: int,
    chunks: int,
    date_range: Optional[str] = None,
) -> None:
    """Log a high-level summary of the data collection phase."""
    efficiency = (total_collected / total_requested * 100) if total_requested > 0 else 0

    logger.info(
        "📊 Data Collection Summary",
        requested=total_requested,
        collected=total_collected,
        efficiency_percent=f"{efficiency:.1f}%",
        parallel_chunks=chunks,
        date_range=date_range or "N/A",
    )


_log_data_collection_summary = log_data_collection_summary


class CallbackTask(Task):
    """Base task class that centralizes logging behaviour."""

    def on_success(self, retval, task_id, args, kwargs):
        logger.info("task.succeeded", task_id=task_id, result=retval)

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        logger.error(
            "task.failed",
            task_id=task_id,
            error=str(exc),
            traceback=einfo.traceback,
            exc_info=exc,
        )
