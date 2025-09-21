"""
GridFS Model Artifacts Repository - Infrastructure Layer

This module implements the ModelArtifactsRepository interface using MongoDB GridFS
as the underlying storage system. GridFS is ideal for storing large binary files
like machine learning models and scalers.
"""

from typing import Any, Dict, Optional
from uuid import UUID

import gridfs
import structlog
from bson import ObjectId
from pymongo import MongoClient
from pymongo.database import Database

from src.domain.entities.errors import ModelOperationError
from src.domain.repositories.model_artifacts_repository import (
    IModelArtifactsRepository,
    ModelArtifact,
)

logger = structlog.get_logger(__name__)


class GridFSModelArtifactsRepository(IModelArtifactsRepository):
    """MongoDB GridFS implementation of the ModelArtifactsRepository."""

    def __init__(self, mongo_client: MongoClient, database_name: str):
        """
        Initialize the GridFS model artifacts repository.

        Args:
            mongo_client: MongoDB client connection
            database_name: Name of the database to use
        """
        self.db: Database = mongo_client[database_name]
        self.fs = gridfs.GridFS(self.db, collection="model_artifacts")

    async def save_artifact(
        self,
        model_id: UUID,
        artifact_type: str,
        content: bytes,
        metadata: Optional[Dict[str, Any]] = None,
        filename: Optional[str] = None,
    ) -> str:
        """
        Save a model artifact to GridFS.

        Args:
            model_id: ID of the model this artifact belongs to
            artifact_type: Type of artifact (model, x_scaler, y_scaler, metadata)
            content: Binary content of the artifact
            metadata: Additional metadata for the artifact
            filename: Original filename of the artifact

        Returns:
            GridFS file ID as string

        Raises:
            ModelOperationError: If save operation fails
        """
        try:
            # Prepare metadata for GridFS
            file_metadata = {
                "model_id": str(model_id),
                "artifact_type": artifact_type,
                "content_type": self._get_content_type(artifact_type),
                **(metadata or {}),
            }

            # Generate filename if not provided
            if not filename:
                filename = f"{model_id}_{artifact_type}"

            # Save to GridFS
            file_id = self.fs.put(
                content,
                filename=filename,
                metadata=file_metadata,
            )

            logger.info(
                "Model artifact saved to GridFS",
                model_id=str(model_id),
                artifact_type=artifact_type,
                file_id=str(file_id),
                size_bytes=len(content),
            )

            return str(file_id)

        except Exception as e:
            logger.error(
                "Failed to save model artifact to GridFS",
                model_id=str(model_id),
                artifact_type=artifact_type,
                error=str(e),
            )
            raise ModelOperationError(f"Failed to save artifact: {e}") from e

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
        try:
            # Find the most recent artifact by model_id and artifact_type
            cursor = (
                self.fs.find(
                    {
                        "metadata.model_id": str(model_id),
                        "metadata.artifact_type": artifact_type,
                    }
                )
                .sort("uploadDate", -1)
                .limit(1)
            )

            grid_out = next(cursor, None)

            if not grid_out:
                logger.debug(
                    "Model artifact not found",
                    model_id=str(model_id),
                    artifact_type=artifact_type,
                )
                return None

            # Read content
            content = grid_out.read()

            logger.info(
                "Model artifact retrieved from GridFS",
                model_id=str(model_id),
                artifact_type=artifact_type,
                file_id=str(grid_out._id),
                size_bytes=len(content),
            )

            return ModelArtifact(
                artifact_id=str(grid_out._id),
                artifact_type=(
                    grid_out.metadata.get("artifact_type", "unknown")
                    if grid_out.metadata
                    else "unknown"
                ),
                content=content,
                metadata=dict(grid_out.metadata) if grid_out.metadata else None,
                filename=grid_out.filename,
            )

        except Exception as e:
            logger.error(
                "Failed to retrieve model artifact from GridFS",
                model_id=str(model_id),
                artifact_type=artifact_type,
                error=str(e),
            )
            raise ModelOperationError(f"Failed to retrieve artifact: {e}") from e

    async def get_artifact_by_id(self, artifact_id: str) -> Optional[ModelArtifact]:
        """
        Retrieve a model artifact by its unique identifier.

        Args:
            artifact_id: Unique identifier of the artifact

        Returns:
            ModelArtifact if found, None otherwise
        """
        try:
            # Convert string ID to ObjectId
            object_id = ObjectId(artifact_id)

            # Find the artifact by ID
            grid_out = self.fs.get(object_id)

            if not grid_out:
                return None

            # Read content
            content = grid_out.read()

            return ModelArtifact(
                artifact_id=artifact_id,
                artifact_type=(
                    grid_out.metadata.get("artifact_type", "unknown")
                    if grid_out.metadata
                    else "unknown"
                ),
                content=content,
                metadata=dict(grid_out.metadata) if grid_out.metadata else None,
                filename=grid_out.filename,
            )

        except gridfs.NoFile:
            logger.debug("Artifact not found by ID", artifact_id=artifact_id)
            return None
        except Exception as e:
            logger.error(
                "Failed to retrieve artifact by ID",
                artifact_id=artifact_id,
                error=str(e),
            )
            raise ModelOperationError(f"Failed to retrieve artifact: {e}") from e

    async def delete_artifact(self, artifact_id: str) -> bool:
        """
        Delete a model artifact by its unique identifier.

        Args:
            artifact_id: Unique identifier of the artifact

        Returns:
            True if deleted successfully, False if not found
        """
        try:
            object_id = ObjectId(artifact_id)

            # Check if file exists
            if not self.fs.exists(object_id):
                return False

            # Delete the file
            self.fs.delete(object_id)

            logger.info("Model artifact deleted", artifact_id=artifact_id)
            return True

        except Exception as e:
            logger.error(
                "Failed to delete model artifact",
                artifact_id=artifact_id,
                error=str(e),
            )
            raise ModelOperationError(f"Failed to delete artifact: {e}") from e

    async def delete_model_artifacts(self, model_id: UUID) -> int:
        """
        Delete all artifacts for a specific model.

        Args:
            model_id: ID of the model

        Returns:
            Number of artifacts deleted
        """
        try:
            # Find all artifacts for the model
            artifacts = self.fs.find({"metadata.model_id": str(model_id)})

            deleted_count = 0
            for artifact in artifacts:
                self.fs.delete(artifact._id)
                deleted_count += 1

            logger.info(
                "Model artifacts deleted",
                model_id=str(model_id),
                count=deleted_count,
            )

            return deleted_count

        except Exception as e:
            logger.error(
                "Failed to delete model artifacts",
                model_id=str(model_id),
                error=str(e),
            )
            raise ModelOperationError(f"Failed to delete artifacts: {e}") from e

    async def list_model_artifacts(self, model_id: UUID) -> Dict[str, str]:
        """
        List all artifacts for a specific model.

        Args:
            model_id: ID of the model

        Returns:
            Dictionary mapping artifact types to their IDs
        """
        try:
            artifacts = self.fs.find({"metadata.model_id": str(model_id)}).sort(
                "uploadDate", -1
            )

            result: Dict[str, str] = {}
            for artifact in artifacts:
                artifact_type = artifact.metadata.get("artifact_type", "unknown")
                if artifact_type in result:
                    continue  # keep the most recent artifact per type
                result[artifact_type] = str(artifact._id)

            logger.debug(
                "Listed model artifacts",
                model_id=str(model_id),
                artifacts=list(result.keys()),
            )

            return result

        except Exception as e:
            logger.error(
                "Failed to list model artifacts",
                model_id=str(model_id),
                error=str(e),
            )
            raise ModelOperationError(f"Failed to list artifacts: {e}") from e

    async def _delete_existing_artifact(
        self, model_id: UUID, artifact_type: str
    ) -> None:
        """Delete existing artifact of the same type for the model."""
        existing = self.fs.find_one(
            {
                "metadata.model_id": str(model_id),
                "metadata.artifact_type": artifact_type,
            }
        )

        if existing:
            self.fs.delete(existing._id)
            logger.debug(
                "Deleted existing artifact",
                model_id=str(model_id),
                artifact_type=artifact_type,
                old_file_id=str(existing._id),
            )

    def _get_content_type(self, artifact_type: str) -> str:
        """Get MIME type based on artifact type."""
        content_types = {
            "model": "application/octet-stream",  # Keras model
            "x_scaler": "application/pickle",  # Scikit-learn scaler
            "y_scaler": "application/pickle",  # Scikit-learn scaler
            "metadata": "application/json",  # JSON metadata
        }
        return content_types.get(artifact_type, "application/octet-stream")
