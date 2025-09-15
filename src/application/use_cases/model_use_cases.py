"""
Model Use Cases - Application Layer

This module defines use cases for model operations.
It orchestrates the flow of data to and from the entities
and implements the business rules of the application.
"""

from typing import List, Optional
from uuid import UUID

from dependency_injector.wiring import Provide, inject

from src.domain.entities.errors import ModelNotFoundError
from src.domain.entities.model import Model
from src.domain.repositories.model_repository import IModelRepository

from ..dtos.model_dto import (
    ModelCreateDTO,
    ModelDetailResponseDTO,
    ModelResponseDTO,
    ModelUpdateDTO,
    TrainingMetricsDTO,
)


class GetModelsUseCase:
    """Use case for retrieving models."""

    @inject
    def __init__(
        self, model_repository: IModelRepository = Provide["model_repository"]
    ):
        self.model_repository = model_repository

    async def execute(
        self,
        skip: int = 0,
        limit: int = 100,
        model_type: Optional[str] = None,
        status: Optional[str] = None,
        entity_id: Optional[str] = None,
        feature: Optional[str] = None,
    ) -> List[ModelResponseDTO]:
        """
        Retrieve a list of models with pagination and filtering.

        Args:
            skip: Number of records to skip
            limit: Maximum number of records to return
            model_type: Filter by model type (e.g., 'lstm', 'gru')
            status: Filter by model status (e.g., 'draft', 'trained')
            entity_id: Filter by FIWARE entity ID
            feature: Filter by feature name

        Returns:
            List of model response DTOs matching the criteria
        """
        models = await self.model_repository.find_all(
            skip=skip,
            limit=limit,
            model_type=model_type,
            status=status,
            entity_id=entity_id,
            feature=feature,
        )
        return [self._to_response_dto(model) for model in models]

    def _to_response_dto(self, model: Model) -> ModelResponseDTO:
        """Convert a domain model to a response DTO."""
        return ModelResponseDTO(
            id=model.id,
            name=model.name,
            description=model.description,
            model_type=model.model_type,
            status=model.status,
            rnn_dropout=model.rnn_dropout,
            dense_dropout=model.dense_dropout,
            batch_size=model.batch_size,
            epochs=model.epochs,
            learning_rate=model.learning_rate,
            validation_split=model.validation_split,
            lookback_window=model.lookback_window,
            forecast_horizon=model.forecast_horizon,
            feature=model.feature,
            rnn_units=model.rnn_units,
            dense_units=model.dense_units,
            early_stopping_patience=model.early_stopping_patience,
            entity_type=model.entity_type,
            entity_id=model.entity_id,
            created_at=model.created_at,
            updated_at=model.updated_at,
            metadata=model.metadata,
        )


class GetModelByIdUseCase:
    """Use case for retrieving a model by ID."""

    @inject
    def __init__(
        self, model_repository: IModelRepository = Provide["model_repository"]
    ):
        self.model_repository = model_repository

    async def execute(self, model_id: UUID) -> ModelDetailResponseDTO:
        """
        Retrieve a model by its ID.

        Args:
            model_id: The unique identifier of the model

        Returns:
            Detailed model response DTO

        Raises:
            ModelNotFoundError: If the model does not exist
        """
        model = await self.model_repository.find_by_id(model_id)
        if not model:
            raise ModelNotFoundError(str(model_id))

        return self._to_detail_response_dto(model)

    def _to_detail_response_dto(self, model: Model) -> ModelDetailResponseDTO:
        """Convert a domain model to a detailed response DTO."""
        trainings = [
            TrainingMetricsDTO(
                id=training.id,
                start_time=training.start_time,
                end_time=training.end_time,
                metrics=training.metrics,
                status=training.status,
            )
            for training in model.trainings
        ]

        return ModelDetailResponseDTO(
            id=model.id,
            name=model.name,
            description=model.description,
            model_type=model.model_type,
            status=model.status,
            rnn_dropout=model.rnn_dropout,
            dense_dropout=model.dense_dropout,
            batch_size=model.batch_size,
            epochs=model.epochs,
            learning_rate=model.learning_rate,
            validation_split=model.validation_split,
            lookback_window=model.lookback_window,
            forecast_horizon=model.forecast_horizon,
            feature=model.feature,
            rnn_units=model.rnn_units,
            dense_units=model.dense_units,
            early_stopping_patience=model.early_stopping_patience,
            entity_type=model.entity_type,
            entity_id=model.entity_id,
            created_at=model.created_at,
            updated_at=model.updated_at,
            metadata=model.metadata,
            trainings=trainings,
        )


class CreateModelUseCase:
    """Use case for creating a new model."""

    @inject
    def __init__(
        self, model_repository: IModelRepository = Provide["model_repository"]
    ):
        self.model_repository = model_repository

    async def execute(self, model_dto: ModelCreateDTO) -> ModelResponseDTO:
        """
        Create a new model.

        Args:
            model_dto: The model create DTO

        Returns:
            The created model as a response DTO
        """
        name = model_dto.name
        if name is None:
            feature = model_dto.feature
            model_type = model_dto.model_type.value
            name = f"{model_type} - {feature}"

        description = model_dto.description
        if description is None:
            feature = model_dto.feature
            model_type = model_dto.model_type.value
            description = f"{model_type} model for {feature} forecasting"

        # Create the domain model from the DTO
        model = Model(
            name=name,
            description=description,
            model_type=model_dto.model_type,
            rnn_dropout=model_dto.rnn_dropout,
            dense_dropout=model_dto.dense_dropout,
            batch_size=model_dto.batch_size,
            epochs=model_dto.epochs,
            learning_rate=model_dto.learning_rate,
            validation_split=model_dto.validation_split,
            lookback_window=model_dto.lookback_window,
            forecast_horizon=model_dto.forecast_horizon,
            feature=model_dto.feature,
            rnn_units=model_dto.rnn_units,
            dense_units=model_dto.dense_units,
            early_stopping_patience=model_dto.early_stopping_patience,
            entity_type=model_dto.entity_type,
            entity_id=model_dto.entity_id,
            metadata=model_dto.metadata,
        )

        # Save the model via repository
        created_model = await self.model_repository.create(model)

        # Convert to response DTO
        return ModelResponseDTO(
            id=created_model.id,
            name=created_model.name,
            description=created_model.description,
            model_type=created_model.model_type,
            status=created_model.status,
            rnn_dropout=created_model.rnn_dropout,
            dense_dropout=created_model.dense_dropout,
            batch_size=created_model.batch_size,
            epochs=created_model.epochs,
            learning_rate=created_model.learning_rate,
            validation_split=created_model.validation_split,
            lookback_window=created_model.lookback_window,
            forecast_horizon=created_model.forecast_horizon,
            feature=created_model.feature,
            rnn_units=created_model.rnn_units,
            dense_units=created_model.dense_units,
            early_stopping_patience=created_model.early_stopping_patience,
            entity_type=created_model.entity_type,
            entity_id=created_model.entity_id,
            created_at=created_model.created_at,
            updated_at=created_model.updated_at,
            metadata=created_model.metadata,
        )


class UpdateModelUseCase:
    """Use case for updating an existing model."""

    @inject
    def __init__(
        self, model_repository: IModelRepository = Provide["model_repository"]
    ):
        self.model_repository = model_repository

    async def execute(
        self, model_id: UUID, model_dto: ModelUpdateDTO
    ) -> ModelResponseDTO:
        """
        Update an existing model.

        Args:
            model_id: The unique identifier of the model to update
            model_dto: The model update DTO with fields to update

        Returns:
            The updated model as a response DTO

        Raises:
            ModelNotFoundError: If the model does not exist
        """
        # Find the existing model
        model = await self.model_repository.find_by_id(model_id)
        if not model:
            raise ModelNotFoundError(str(model_id))

        # Update fields that are provided in the DTO
        if model_dto.name is not None:
            model.name = model_dto.name

        if model_dto.description is not None:
            model.description = model_dto.description

        if model_dto.rnn_dropout is not None:
            model.rnn_dropout = model_dto.rnn_dropout

        if model_dto.dense_dropout is not None:
            model.dense_dropout = model_dto.dense_dropout

        if model_dto.batch_size is not None:
            model.batch_size = model_dto.batch_size

        if model_dto.epochs is not None:
            model.epochs = model_dto.epochs

        if model_dto.learning_rate is not None:
            model.learning_rate = model_dto.learning_rate

        if model_dto.validation_split is not None:
            model.validation_split = model_dto.validation_split

        if model_dto.lookback_window is not None:
            model.lookback_window = model_dto.lookback_window

        if model_dto.forecast_horizon is not None:
            model.forecast_horizon = model_dto.forecast_horizon

        if model_dto.feature is not None:
            model.feature = model_dto.feature
            if model_dto.name is None and model.name and "-" in model.name:
                model.name = f"{model.model_type.value} - {model.feature}"
            if (
                model_dto.description is None
                and model.description
                and "forecasting" in model.description
            ):
                model.description = (
                    f"{model.model_type.value} model for {model.feature} forecasting"
                )

        if model_dto.rnn_units is not None:
            model.rnn_units = model_dto.rnn_units

        if model_dto.dense_units is not None:
            model.dense_units = model_dto.dense_units

        if model_dto.early_stopping_patience is not None:
            model.early_stopping_patience = model_dto.early_stopping_patience

        if model_dto.entity_type is not None:
            model.entity_type = model_dto.entity_type

        if model_dto.entity_id is not None:
            model.entity_id = model_dto.entity_id

        if model_dto.metadata is not None:
            # Merge the existing metadata with the new metadata
            model.metadata.update(model_dto.metadata)

        # Update timestamp
        model.update_timestamp()

        # Save the updated model
        updated_model = await self.model_repository.update(model)

        # Convert to response DTO
        return ModelResponseDTO(
            id=updated_model.id,
            name=updated_model.name,
            description=updated_model.description,
            model_type=updated_model.model_type,
            status=updated_model.status,
            rnn_dropout=updated_model.rnn_dropout,
            dense_dropout=updated_model.dense_dropout,
            batch_size=updated_model.batch_size,
            epochs=updated_model.epochs,
            learning_rate=updated_model.learning_rate,
            validation_split=updated_model.validation_split,
            lookback_window=updated_model.lookback_window,
            forecast_horizon=updated_model.forecast_horizon,
            feature=updated_model.feature,
            rnn_units=updated_model.rnn_units,
            dense_units=updated_model.dense_units,
            early_stopping_patience=updated_model.early_stopping_patience,
            entity_type=updated_model.entity_type,
            entity_id=updated_model.entity_id,
            created_at=updated_model.created_at,
            updated_at=updated_model.updated_at,
            metadata=updated_model.metadata,
        )


class DeleteModelUseCase:
    """Use case for deleting a model."""

    @inject
    def __init__(
        self, model_repository: IModelRepository = Provide["model_repository"]
    ):
        self.model_repository = model_repository

    async def execute(self, model_id: UUID) -> None:
        """
        Delete a model by its ID.

        Args:
            model_id: The unique identifier of the model to delete

        Raises:
            ModelNotFoundError: If the model does not exist
        """
        # Check if the model exists first
        model = await self.model_repository.find_by_id(model_id)
        if not model:
            raise ModelNotFoundError(str(model_id))

        # Delete the model
        await self.model_repository.delete(model_id)
