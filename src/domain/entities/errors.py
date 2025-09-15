"""
Domain Errors

This module defines custom error classes for domain-specific exceptions.
"""

from typing import Any, Dict, Optional


class DomainError(Exception):
    """Base class for domain errors."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.details = details or {}
        super().__init__(message)


class ModelNotFoundError(DomainError):
    """Raised when a model cannot be found."""

    def __init__(self, model_id: str, details: Optional[Dict[str, Any]] = None):
        message = f"Model with ID {model_id} not found"
        super().__init__(message, details)


class ModelValidationError(DomainError):
    """Raised when model validation fails."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, details)


class ModelOperationError(DomainError):
    """Raised when an operation on a model fails."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, details)
