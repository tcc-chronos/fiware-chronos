"""
MongoDB Model Repository - Infrastructure Layer

This module implements the ModelRepository interface using MongoDB
as the underlying data store.
"""

from typing import Any, Dict, List, Optional
from uuid import UUID

import pymongo

from src.domain.entities.errors import ModelNotFoundError, ModelOperationError
from src.domain.entities.model import (
    DenseLayerConfig,
    Model,
    ModelStatus,
    ModelType,
    RNNLayerConfig,
)
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

    def _create_indexes(self) -> None:
        """Create necessary indexes on the collection."""
        pass

    def _to_document(self, model: Model) -> Dict[str, Any]:
        """Convert a Model entity to a MongoDB document."""
        return {
            "id": str(model.id),
            "name": model.name,
            "description": model.description,
            "model_type": model.model_type.value,
            "status": model.status.value,
            "batch_size": model.batch_size,
            "epochs": model.epochs,
            "learning_rate": model.learning_rate,
            "validation_split": model.validation_split,
            "lookback_window": model.lookback_window,
            "forecast_horizon": model.forecast_horizon,
            "feature": model.feature,
            "rnn_layers": [
                {
                    "units": layer.units,
                    "dropout": layer.dropout,
                    "recurrent_dropout": layer.recurrent_dropout,
                }
                for layer in model.rnn_layers
            ],
            "dense_layers": [
                {
                    "units": layer.units,
                    "dropout": layer.dropout,
                    "activation": layer.activation,
                }
                for layer in model.dense_layers
            ],
            "early_stopping_patience": model.early_stopping_patience,
            "entity_type": model.entity_type,
            "entity_id": model.entity_id,
            "created_at": model.created_at,
            "updated_at": model.updated_at,
            "has_successful_training": model.has_successful_training,
        }

    def _to_entity(self, document: Dict[str, Any]) -> Model:
        """Convert a MongoDB document to a Model entity."""
        status_value = document.get("status", ModelStatus.DRAFT.value)
        try:
            status = ModelStatus(status_value)
        except ValueError:
            status = (
                ModelStatus.TRAINED
                if status_value in {"ready", "trained"}
                else ModelStatus.DRAFT
            )

        rnn_layers_payload = document.get("rnn_layers") or []
        if rnn_layers_payload:
            rnn_layers = [
                RNNLayerConfig(
                    units=int(layer.get("units", 0)),
                    dropout=float(layer.get("dropout", 0.1)),
                    recurrent_dropout=float(layer.get("recurrent_dropout", 0.0)),
                )
                for layer in rnn_layers_payload
                if layer and layer.get("units") is not None
            ]
        else:
            legacy_units = document.get("rnn_units")
            legacy_dropout = float(document.get("rnn_dropout", 0.1))
            if legacy_units:
                rnn_layers = [
                    RNNLayerConfig(units=int(units), dropout=legacy_dropout)
                    for units in legacy_units
                    if units is not None
                ]
            else:
                rnn_layers = []

        if not rnn_layers:
            rnn_layers = [RNNLayerConfig(units=64)]

        dense_layers_payload = document.get("dense_layers") or []
        if dense_layers_payload:
            dense_layers = [
                DenseLayerConfig(
                    units=int(layer.get("units", 0)),
                    dropout=float(layer.get("dropout", 0.1)),
                    activation=layer.get("activation", "relu"),
                )
                for layer in dense_layers_payload
                if layer and layer.get("units") is not None
            ]
        else:
            legacy_units = document.get("dense_units")
            legacy_dropout = float(document.get("dense_dropout", 0.1))
            if legacy_units:
                dense_layers = [
                    DenseLayerConfig(units=int(units), dropout=legacy_dropout)
                    for units in legacy_units
                    if units is not None
                ]
            else:
                dense_layers = []

        if not rnn_layers:
            rnn_layers = [RNNLayerConfig(units=64)]

        return Model(
            id=UUID(document["id"]),
            name=document["name"],
            description=document.get("description"),
            model_type=ModelType(document["model_type"]),
            status=status,
            batch_size=document["batch_size"],
            epochs=document["epochs"],
            learning_rate=document["learning_rate"],
            validation_split=document["validation_split"],
            lookback_window=document["lookback_window"],
            forecast_horizon=document["forecast_horizon"],
            feature=document.get("feature") or "",
            rnn_layers=rnn_layers,
            dense_layers=dense_layers,
            early_stopping_patience=document.get("early_stopping_patience"),
            entity_type=document.get("entity_type") or "",
            entity_id=document.get("entity_id") or "",
            created_at=document["created_at"],
            updated_at=document["updated_at"],
            has_successful_training=document.get("has_successful_training", False),
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

    async def find_all(
        self,
        skip: int = 0,
        limit: int = 100,
        model_type: Optional[str] = None,
        status: Optional[str] = None,
        entity_id: Optional[str] = None,
        feature: Optional[str] = None,
    ) -> List[Model]:
        """
        Find all models with pagination and filtering options.

        Args:
            skip: Number of records to skip (for pagination)
            limit: Maximum number of records to return
            model_type: Filter by model type (e.g., 'lstm', 'gru')
            status: Filter by model status (e.g., 'draft', 'trained')
            entity_id: Filter by FIWARE entity ID
            feature: Filter by feature name

        Returns:
            List of models matching the criteria
        """
        # Build query with optional filters
        query = {}

        if model_type is not None:
            query["model_type"] = model_type

        if status is not None:
            query["status"] = status

        if entity_id is not None:
            query["entity_id"] = entity_id

        if feature is not None:
            query["feature"] = feature

        documents = await self.db.find_many(
            self.COLLECTION_NAME,
            query,
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
