"""
Repositories Package - Infrastructure Layer

This package contains concrete implementations of the repository
interfaces defined in the domain layer. These implementations
handle the details of data persistence.
"""

from .model_repository import ModelRepository

__all__ = ["ModelRepository"]
