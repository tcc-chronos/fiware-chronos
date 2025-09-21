"""
Logging Configuration - Shared Layer

This module provides utilities for configuring logging across the application.
"""

import logging
import os
import sys
from typing import Any, Dict, List, Optional

import structlog
from structlog.types import Processor

from src.shared.consts import EnumEnvironment

DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def _get_log_config_from_env() -> Dict[str, Optional[str]]:
    """
    Get logging configuration from environment variables.

    This is used for initial bootstrap configuration before
    the full settings system is available.

    Returns:
        Dict containing logging configuration.
    """
    return {
        "level": os.environ.get("LOG_LEVEL", "INFO"),
        "format": os.environ.get("LOG_FORMAT", DEFAULT_LOG_FORMAT),
        "file_path": os.environ.get("LOG_FILE_PATH"),
    }


def configure_logging(
    level: Optional[str] = None,
    format_string: Optional[str] = None,
    file_path: Optional[str] = None,
    environment: str = "development",
) -> None:
    """
    Configure Python's standard logging system.

    This function should be called at application startup, before the main
    config system is initialized, to ensure early logging capability.

    Args:
        level: Optional override for the log level.
        format_string: Optional override for log format.
        file_path: Optional override for log file path.
        environment: Application environment (development, production, etc.)
    """
    # Get config from environment variables or use provided values
    env_config = _get_log_config_from_env()

    # Use parameters, env vars, or defaults
    log_level = level or env_config["level"] or "INFO"
    log_file = file_path or env_config["file_path"]

    # Convert string level to logging level
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    # Clear any existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Configure handlers
    handlers: List[logging.Handler] = []

    # Always add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    handlers.append(console_handler)

    # Add file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        handlers.append(file_handler)

    env_value = environment.lower()
    is_production = env_value == EnumEnvironment.PRODUCTION

    timestamper = structlog.processors.TimeStamper(fmt="iso", utc=True)
    renderer: Processor
    if is_production:
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer()

    formatter = structlog.stdlib.ProcessorFormatter(
        processor=renderer,
        foreign_pre_chain=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            timestamper,
        ],
    )

    for handler in handlers:
        handler.setFormatter(formatter)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            timestamper,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Configure root logger
    root_logger.handlers = handlers
    root_logger.setLevel(numeric_level)

    # Log configuration complete
    logging.info(f"Logging configured with level: {log_level}")
    if log_file:
        logging.info(f"Logging to file: {log_file}")


def update_logging_from_settings(settings: Any) -> None:
    """
    Update logging configuration using the application settings.

    This should be called after the settings system is fully initialized.

    Args:
        settings: The application settings object from Pydantic.
    """
    try:
        # Extract logging settings
        log_level = (
            settings.logging.level.value
            if hasattr(settings.logging.level, "value")
            else settings.logging.level
        )
        log_format = settings.logging.format
        log_file = settings.logging.file_path
        environment = (
            settings.environment.value
            if hasattr(settings.environment, "value")
            else settings.environment
        )

        # Reconfigure logging with complete settings
        configure_logging(
            level=log_level,
            format_string=log_format,
            file_path=log_file,
            environment=environment,
        )

        logging.info("Logging configuration updated from application settings")
    except Exception as e:
        logging.error(f"Failed to update logging from settings: {e}")


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a structlog logger configured for the project."""
    return structlog.get_logger(name)
