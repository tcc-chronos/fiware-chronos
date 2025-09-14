"""
Shared module - Cross-cutting concerns / Shared Layer

This module provides shared utilities, constants, and enums that are used
across multiple layers of the application.

Its primary responsibilities include:
- Defining cross-layer constants (e.g., environment names, log levels)
- Centralizing reusable enums and global values
- Serving as a common place for definitions that do not belong
  exclusively to Domain, Application, or Infrastructure

Following Clean Architecture principles:
- Shared module contains only *cross-cutting concerns*
- It must not depend on Infrastructure or Frameworks
"""

from .consts import EnumEnvironment, EnumLogLevel
from .logging import configure_logging, get_logger, update_logging_from_settings

__all__ = [
    "EnumEnvironment",
    "EnumLogLevel",
    "configure_logging",
    "get_logger",
    "update_logging_from_settings",
]
