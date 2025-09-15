"""
MongoDB Model Repository - Infrastructure Layer

This module implements the ModelRepository interface using MongoDB
as the underlying data store.
"""

from typing import Any, Dict, List, Optional
from uuid import UUID

import pymongo

from src.domain.entities.errors import ModelNotFoundError, ModelOperationError
from src.domain.entities.model import Model, ModelStatus, ModelType, Training
from src.domain.repositories.model_repository import IModelRepository
from src.infrastructure.database import MongoDatabase


class ModelRepository(IModelRepository):
    """MongoDB implementation of the ModelRepository."""

    COLLECTION_NAME = "models"

    def __init__(self, mongo_database: MongoDatabase):
        """
        Initialize the MongoDB model repository.

        Args:
            mongo_database: MongoDB database client
        """
        self.db = mongo_database

        # Create indexes
        self._create_indexes()

    def _create_indexes(self) -> None:
        """Create necessary indexes on the collection."""
        # Index on ID for faster lookups
        self.db.create_index(self.COLLECTION_NAME, "id", unique=True)

        # Index on creation date for sorting
        self.db.create_index(self.COLLECTION_NAME, "created_at")

        # Index on model type for filtering
        self.db.create_index(self.COLLECTION_NAME, "model_type")

        # Index on status for filtering
        self.db.create_index(self.COLLECTION_NAME, "status")

    def _to_document(self, model: Model) -> Dict[str, Any]:
        """Convert a Model entity to a MongoDB document."""
        return {
            "id": str(model.id),
            "name": model.name,
            "description": model.description,
            "model_type": model.model_type.value,
            "status": model.status.value,
            "rnn_dropout": model.rnn_dropout,
            "dense_dropout": model.dense_dropout,
            "batch_size": model.batch_size,
            "epochs": model.epochs,
            "learning_rate": model.learning_rate,
            "validation_split": model.validation_split,
            "lookback_window": model.lookback_window,
            "forecast_horizon": model.forecast_horizon,
            "feature": model.feature,
            "rnn_units": model.rnn_units,
            "dense_units": model.dense_units,
            "early_stopping_patience": model.early_stopping_patience,
            "entity_type": model.entity_type,
            "entity_id": model.entity_id,
            "created_at": model.created_at,
            "updated_at": model.updated_at,
            "metadata": model.metadata,
            "trainings": [self._training_to_document(t) for t in model.trainings],
        }

    def _training_to_document(self, training: Training) -> Dict[str, Any]:
        """Convert a Training entity to a MongoDB document."""
        return {
            "id": str(training.id),
            "start_time": training.start_time,
            "end_time": training.end_time,
            "metrics": training.metrics,
            "dataset_info": training.dataset_info,
            "status": training.status,
            "error": training.error,
        }

    def _to_entity(self, document: Dict[str, Any]) -> Model:
        """Convert a MongoDB document to a Model entity."""
        return Model(
            id=UUID(document["id"]),
            name=document["name"],
            description=document.get("description"),
            model_type=ModelType(document["model_type"]),
            status=ModelStatus(document["status"]),
            rnn_dropout=document["rnn_dropout"],
            dense_dropout=document["dense_dropout"],
            batch_size=document["batch_size"],
            epochs=document["epochs"],
            learning_rate=document["learning_rate"],
            validation_split=document["validation_split"],
            lookback_window=document["lookback_window"],
            forecast_horizon=document["forecast_horizon"],
            feature=document.get("feature", "value"),
            rnn_units=document.get("rnn_units", [64]),
            dense_units=document.get("dense_units", [32]),
            early_stopping_patience=document.get("early_stopping_patience"),
            entity_type=document.get("entity_type"),
            entity_id=document.get("entity_id"),
            created_at=document["created_at"],
            updated_at=document["updated_at"],
            metadata=document.get("metadata", {}),
            trainings=[
                self._document_to_training(t) for t in document.get("trainings", [])
            ],
        )

    def _document_to_training(self, document: Dict[str, Any]) -> Training:
        """Convert a MongoDB document to a Training entity."""
        return Training(
            id=UUID(document["id"]),
            start_time=document["start_time"],
            end_time=document.get("end_time"),
            metrics=document.get("metrics", {}),
            dataset_info=document.get("dataset_info", {}),
            status=document["status"],
            error=document.get("error"),
        )

    async def find_by_id(self, model_id: UUID) -> Optional[Model]:
        """
        Find a model by its ID.

        Args:
            model_id: The unique identifier of the model to find

        Returns:
            The model if found, None otherwise
        """
        document = await self.db.find_one(self.COLLECTION_NAME, {"id": str(model_id)})
        if document is None:
            return None
        return self._to_entity(document)

    async def find_all(self, skip: int = 0, limit: int = 100) -> List[Model]:
        """
        Find all models with pagination.

        Args:
            skip: Number of records to skip (for pagination)
            limit: Maximum number of records to return

        Returns:
            List of models
        """
        documents = await self.db.find_many(
            self.COLLECTION_NAME,
            {},
            sort_by="created_at",
            sort_direction=pymongo.DESCENDING,
            skip=skip,
            limit=limit,
        )

        models = []
        for document in documents:
            models.append(self._to_entity(document))

        return models

    async def create(self, model: Model) -> Model:
        """
        Create a new model.

        Args:
            model: The model to create

        Returns:
            The created model with any generated fields populated

        Raises:
            ModelOperationError: If the model creation fails
        """
        try:
            document = self._to_document(model)
            await self.db.insert_one(self.COLLECTION_NAME, document)
            return model
        except Exception as e:
            raise ModelOperationError(f"Failed to create model: {str(e)}")

    async def update(self, model: Model) -> Model:
        """
        Update an existing model.

        Args:
            model: The model with updated fields

        Returns:
            The updated model

        Raises:
            ModelNotFoundError: If the model does not exist
            ModelOperationError: If the model update fails
        """
        try:
            document = self._to_document(model)
            await self.db.replace_one(
                self.COLLECTION_NAME, {"id": str(model.id)}, document
            )
            return model
        except Exception as e:
            if "Document not found" in str(e):
                raise ModelNotFoundError(str(model.id))
            raise ModelOperationError(f"Failed to update model: {str(e)}")

    async def delete(self, model_id: UUID) -> None:
        """
        Delete a model by its ID.

        Args:
            model_id: The unique identifier of the model to delete

        Raises:
            ModelNotFoundError: If the model does not exist
            ModelOperationError: If the model deletion fails
        """
        try:
            await self.db.delete_one(self.COLLECTION_NAME, {"id": str(model_id)})
        except Exception as e:
            if "Document not found" in str(e):
                raise ModelNotFoundError(str(model_id))
            raise ModelOperationError(f"Failed to delete model: {str(e)}")
