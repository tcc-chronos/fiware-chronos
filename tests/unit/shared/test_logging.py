from __future__ import annotations

import logging
from dataclasses import dataclass

from src.shared.logging import (
    configure_logging,
    get_logger,
    update_logging_from_settings,
)


def test_configure_logging_sets_root_handlers(tmp_path) -> None:
    log_file = tmp_path / "chronos.log"
    configure_logging(level="DEBUG", file_path=str(log_file), environment="development")

    root = logging.getLogger()
    assert root.level == logging.DEBUG
    assert any(isinstance(handler, logging.FileHandler) for handler in root.handlers)

    logger = get_logger(__name__)
    logger.info("structured log test")


@dataclass
class _LoggingSettings:
    level: str = "WARNING"
    format: str = "%(message)s"
    file_path: str | None = None


@dataclass
class _Settings:
    logging: _LoggingSettings
    environment: str = "production"


def test_update_logging_from_settings_applies_configuration() -> None:
    settings = _Settings(logging=_LoggingSettings(level="ERROR"))

    update_logging_from_settings(settings)

    root_logger = logging.getLogger()
    assert root_logger.level == logging.ERROR
