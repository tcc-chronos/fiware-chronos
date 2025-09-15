"""
Repositories Package

This package contains interfaces defining repository contracts
for data access operations. Specific implementations are provided
by the infrastructure layer.
"""

from .model_repository import IModelRepository

__all__ = ["IModelRepository"]
