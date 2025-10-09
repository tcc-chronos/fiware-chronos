#!/usr/bin/env python3
"""
Worker Entry Point - Main Layer

This module serves as the entry point for the Celery worker.
Similar to app.py, it initializes the necessary components and starts the worker.
Both API and Worker are application entry points that belong to the Main layer.
"""

import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Now import project modules
from src.main.config import get_settings  # noqa: E402
from src.shared import (  # noqa: E402
    configure_logging,
    get_logger,
    update_logging_from_settings,
)

# Configure logging with basic settings first
configure_logging()

# Load settings
settings = get_settings()

# Update logging with complete settings
update_logging_from_settings(settings)

# Get structured logger
logger = get_logger(__name__)


def create_worker():
    """
    Configure and return the Celery worker.

    Similar to create_app() in app.py, this function configures
    the worker with proper settings and environment.
    """
    settings = get_settings()

    # Configure environment variables if not already set
    os.environ.setdefault("CELERY_BROKER_URL", settings.celery.broker_url)
    os.environ.setdefault("CELERY_RESULT_BACKEND", settings.celery.result_backend_url)

    # Create Celery app with centralized configuration
    from src.infrastructure.services.celery_config import create_celery_app

    worker_app = create_celery_app(
        broker_url=settings.celery.broker_url,
        backend_url=settings.celery.result_backend_url,
    )

    logger.info(
        "Configuring Celery worker",
        broker_url=settings.celery.broker_url,
        backend_url=settings.celery.result_backend_url,
        app_name=worker_app.main,
    )

    return worker_app


def main():
    """Main entry point for Celery worker."""

    logger.info("Starting Celery worker")

    # Get configured worker
    worker_app = create_worker()

    # Start worker with proper configuration
    worker_app.worker_main(
        [
            "worker",
            "--loglevel=info",
            "--queues=data_collection,model_training,orchestration,"
            "forecast_scheduling,forecast_execution",
            "--concurrency=2",  # Limit concurrency for ML workloads
            "--max-tasks-per-child=10",  # Restart workers for memory mgmt
        ]
    )


if __name__ == "__main__":
    main()
