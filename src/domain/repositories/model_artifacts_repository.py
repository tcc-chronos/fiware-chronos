"""
Model Artifacts Repository Interface

This module defines the interface for model artifacts storage following
the repository pattern. It abstracts the data access operations for
model artifacts (models, scalers, metadata), decoupling them from
specific storage implementations like filesystem or GridFS.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from uuid import UUID


class ModelArtifact:
    """Represents a model artifact with its metadata."""

    def __init__(
        self,
        artifact_id: str,
        artifact_type: str,
        content: bytes,
        metadata: Optional[Dict[str, Any]] = None,
        filename: Optional[str] = None,
    ):
        """
        Initialize a model artifact.

        Args:
            artifact_id: Unique identifier for the artifact
            artifact_type: Type of artifact (model, x_scaler, y_scaler, metadata)
            content: Binary content of the artifact
            metadata: Additional metadata for the artifact
            filename: Original filename of the artifact
        """
        self.artifact_id = artifact_id
        self.artifact_type = artifact_type
        self.content = content
        self.metadata = metadata or {}
        self.filename = filename


class IModelArtifactsRepository(ABC):
    """Interface for Model Artifacts repository implementations."""

    @abstractmethod
    async def save_artifact(
        self,
        model_id: UUID,
        artifact_type: str,
        content: bytes,
        metadata: Optional[Dict[str, Any]] = None,
        filename: Optional[str] = None,
    ) -> str:
        """
        Save a model artifact.

        Args:
            model_id: ID of the model this artifact belongs to
            artifact_type: Type of artifact (model, x_scaler, y_scaler, metadata)
            content: Binary content of the artifact
            metadata: Additional metadata for the artifact
            filename: Original filename of the artifact

        Returns:
            Unique identifier for the saved artifact
        """
        pass

    @abstractmethod
    async def get_artifact(
        self,
        model_id: UUID,
        artifact_type: str,
    ) -> Optional[ModelArtifact]:
        """
        Retrieve a model artifact by model ID and type.

        Args:
            model_id: ID of the model
            artifact_type: Type of artifact to retrieve

        Returns:
            ModelArtifact if found, None otherwise
        """
        pass

    @abstractmethod
    async def get_artifact_by_id(self, artifact_id: str) -> Optional[ModelArtifact]:
        """
        Retrieve a model artifact by its unique identifier.

        Args:
            artifact_id: Unique identifier of the artifact

        Returns:
            ModelArtifact if found, None otherwise
        """
        pass

    @abstractmethod
    async def delete_artifact(self, artifact_id: str) -> bool:
        """
        Delete a model artifact by its unique identifier.

        Args:
            artifact_id: Unique identifier of the artifact

        Returns:
            True if deleted successfully, False if not found
        """
        pass

    @abstractmethod
    async def delete_model_artifacts(self, model_id: UUID) -> int:
        """
        Delete all artifacts for a specific model.

        Args:
            model_id: ID of the model

        Returns:
            Number of artifacts deleted
        """
        pass

    @abstractmethod
    async def list_model_artifacts(self, model_id: UUID) -> Dict[str, str]:
        """
        List all artifacts for a specific model.

        Args:
            model_id: ID of the model

        Returns:
            Dictionary mapping artifact types to their IDs
        """
        pass
