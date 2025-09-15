"""
Model Repository Interface

This module defines the interface for model repositories following
the repository pattern. It abstracts the data access operations
for model entities, decoupling them from specific implementations.
"""

from abc import ABC, abstractmethod
from typing import List, Optional
from uuid import UUID

from src.domain.entities.model import Model


class IModelRepository(ABC):
    """Interface for Model repository implementations."""

    @abstractmethod
    async def find_by_id(self, model_id: UUID) -> Optional[Model]:
        """
        Find a model by its ID.

        Args:
            model_id: The unique identifier of the model to find

        Returns:
            The model if found, None otherwise
        """
        pass

    @abstractmethod
    async def find_all(self, skip: int = 0, limit: int = 100) -> List[Model]:
        """
        Find all models with pagination.

        Args:
            skip: Number of records to skip (for pagination)
            limit: Maximum number of records to return

        Returns:
            List of models
        """
        pass

    @abstractmethod
    async def create(self, model: Model) -> Model:
        """
        Create a new model.

        Args:
            model: The model to create

        Returns:
            The created model with any generated fields populated
        """
        pass

    @abstractmethod
    async def update(self, model: Model) -> Model:
        """
        Update an existing model.

        Args:
            model: The model with updated fields

        Returns:
            The updated model

        Raises:
            ModelNotFoundError: If the model does not exist
        """
        pass

    @abstractmethod
    async def delete(self, model_id: UUID) -> None:
        """
        Delete a model by its ID.

        Args:
            model_id: The unique identifier of the model to delete

        Raises:
            ModelNotFoundError: If the model does not exist
        """
        pass
