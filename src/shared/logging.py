"""
Logging Configuration - Shared Layer

This module provides utilities for configuring logging across the application.
"""

import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional

from src.shared.consts import EnumEnvironment


class JsonFormatter(logging.Formatter):
    """JSON formatter for logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        # ISO format timestamp with timezone
        import datetime

        timestamp = datetime.datetime.fromtimestamp(
            record.created, datetime.timezone.utc
        ).isoformat()

        log_data = {
            "timestamp": timestamp,
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "app": "fiware-chronos",
        }

        # Add exception info if present
        if record.exc_info:
            log_data["exc_info"] = self.formatException(record.exc_info)

        # Add any extra attributes
        for key, value in record.__dict__.items():
            if key not in {
                "args",
                "asctime",
                "created",
                "exc_info",
                "exc_text",
                "filename",
                "funcName",
                "id",
                "levelname",
                "levelno",
                "lineno",
                "module",
                "msecs",
                "message",
                "msg",
                "name",
                "pathname",
                "process",
                "processName",
                "relativeCreated",
                "stack_info",
                "thread",
                "threadName",
            }:
                log_data[key] = value

        return json.dumps(log_data)


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
        "format": os.environ.get(
            "LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        ),
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
    log_format = (
        format_string
        or env_config["format"]
        or "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
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

    # Configure formatters based on environment
    if environment.lower() == EnumEnvironment.PRODUCTION:
        # Use JSON formatter in production environment
        json_formatter = JsonFormatter()
        # Apply JSON formatter to all handlers
        for handler in handlers:
            handler.setFormatter(json_formatter)
    else:
        # Use standard formatter for other environments
        std_formatter = logging.Formatter(log_format)
        # Apply standard formatter to all handlers
        for handler in handlers:
            handler.setFormatter(std_formatter)

    # Configure root logger
    root_logger.handlers = handlers
    root_logger.setLevel(numeric_level)

    # Log configuration complete
    logging.info(f"Logging configured with level: {log_level}")
    if log_file:
        logging.info(f"Logging to file: {log_file}")


class StructuredLogger:
    """
    Simple structured logger that adds context to logs.
    This provides a simple alternative to structlog with similar API.
    """

    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.default_context = {"logger": name}

    def _log(self, level: int, event: str, **kwargs: Any) -> None:
        """Log with context."""
        # Create copy of default context and update with kwargs
        context = self.default_context.copy()
        context.update(kwargs)
        context["event"] = event

        # Log with extra context
        self.logger.log(level, event, extra=context)

    def debug(self, event: str, **kwargs: Any) -> None:
        """Log debug message with context."""
        self._log(logging.DEBUG, event, **kwargs)

    def info(self, event: str, **kwargs: Any) -> None:
        """Log info message with context."""
        self._log(logging.INFO, event, **kwargs)

    def warning(self, event: str, **kwargs: Any) -> None:
        """Log warning message with context."""
        self._log(logging.WARNING, event, **kwargs)

    def error(self, event: str, **kwargs: Any) -> None:
        """Log error message with context."""
        self._log(logging.ERROR, event, **kwargs)

    def critical(self, event: str, **kwargs: Any) -> None:
        """Log critical message with context."""
        self._log(logging.CRITICAL, event, **kwargs)

    def bind(self, **kwargs: Any) -> "StructuredLogger":
        """Create a new logger with additional default context."""
        new_logger = StructuredLogger(self.logger.name)
        new_logger.default_context = self.default_context.copy()
        new_logger.default_context.update(kwargs)
        return new_logger


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


def get_logger(name: str) -> StructuredLogger:
    """Get a structured logger with the given name."""
    return StructuredLogger(name)
