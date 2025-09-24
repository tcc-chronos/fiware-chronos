"""
Model Use Cases - Application Layer

This module defines use cases for model operations.
It orchestrates the flow of data to and from the entities
and implements the business rules of the application.
"""

from typing import List, Optional
from uuid import UUID

from dependency_injector.wiring import Provide, inject

from src.domain.entities.errors import ModelNotFoundError, ModelOperationError
from src.domain.entities.model import (
    DenseLayerConfig,
    Model,
    ModelStatus,
    ModelType,
    RNNLayerConfig,
)
from src.domain.entities.training_job import TrainingMetrics
from src.domain.repositories.model_artifacts_repository import IModelArtifactsRepository
from src.domain.repositories.model_repository import IModelRepository
from src.domain.repositories.training_job_repository import ITrainingJobRepository
from src.domain.services import validate_model_configuration

from ..dtos.model_dto import (
    DenseLayerDTO,
    ModelCreateDTO,
    ModelDetailResponseDTO,
    ModelResponseDTO,
    ModelTrainingSummaryDTO,
    ModelTypeOptionDTO,
    ModelUpdateDTO,
    RNNLayerDTO,
)
from ..dtos.training_dto import TrainingMetricsDTO


def _to_rnn_layer_config(dto: RNNLayerDTO) -> RNNLayerConfig:
    return RNNLayerConfig(
        units=dto.units,
        dropout=dto.dropout,
        recurrent_dropout=dto.recurrent_dropout,
    )


def _to_dense_layer_config(dto: DenseLayerDTO) -> DenseLayerConfig:
    return DenseLayerConfig(
        units=dto.units,
        dropout=dto.dropout,
        activation=dto.activation,
    )


def _to_rnn_layer_dto(config: RNNLayerConfig) -> RNNLayerDTO:
    return RNNLayerDTO(
        units=config.units,
        dropout=config.dropout,
        recurrent_dropout=config.recurrent_dropout,
    )


def _to_dense_layer_dto(config: DenseLayerConfig) -> DenseLayerDTO:
    return DenseLayerDTO(
        units=config.units,
        dropout=config.dropout,
        activation=config.activation,
    )


def _to_training_metrics_dto(
    metrics: Optional[TrainingMetrics],
) -> Optional[TrainingMetricsDTO]:
    if metrics is None:
        return None

    return TrainingMetricsDTO(**metrics.__dict__)


class GetModelsUseCase:
    """Use case for retrieving models."""

    @inject
    def __init__(
        self,
        model_repository: IModelRepository = Provide["model_repository"],
        training_job_repository: ITrainingJobRepository = Provide[
            "training_job_repository"
        ],
    ):
        self.model_repository = model_repository
        self.training_job_repository = training_job_repository

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
        responses: List[ModelResponseDTO] = []
        for model in models:
            trainings = await self._get_training_summaries(model.id)
            responses.append(self._to_response_dto(model, trainings))
        return responses

    async def _get_training_summaries(
        self, model_id: UUID
    ) -> List[ModelTrainingSummaryDTO]:
        training_jobs = await self.training_job_repository.get_by_model_id(model_id)
        summaries: List[ModelTrainingSummaryDTO] = []
        for job in sorted(training_jobs, key=lambda j: j.created_at, reverse=True):
            summaries.append(
                ModelTrainingSummaryDTO(
                    id=job.id,
                    status=job.status,
                    start_time=job.start_time,
                    end_time=job.end_time,
                    error=job.error,
                    data_collection_progress=job.get_data_collection_progress(),
                    total_data_points_requested=job.total_data_points_requested,
                    total_data_points_collected=job.total_data_points_collected,
                    created_at=job.created_at,
                    updated_at=job.updated_at,
                    metrics=_to_training_metrics_dto(job.metrics),
                )
            )
        return summaries

    def _to_response_dto(
        self, model: Model, trainings: List[ModelTrainingSummaryDTO]
    ) -> ModelResponseDTO:
        """Convert a domain model to a response DTO."""
        return ModelResponseDTO(
            id=model.id,
            name=model.name,
            description=model.description,
            model_type=model.model_type,
            status=model.status,
            batch_size=model.batch_size,
            epochs=model.epochs,
            learning_rate=model.learning_rate,
            validation_ratio=model.validation_ratio,
            test_ratio=model.test_ratio,
            lookback_window=model.lookback_window,
            forecast_horizon=model.forecast_horizon,
            feature=model.feature,
            rnn_layers=[_to_rnn_layer_dto(layer) for layer in model.rnn_layers],
            dense_layers=[_to_dense_layer_dto(layer) for layer in model.dense_layers],
            early_stopping_patience=model.early_stopping_patience,
            entity_type=model.entity_type,
            entity_id=model.entity_id,
            created_at=model.created_at,
            updated_at=model.updated_at,
            trainings=trainings,
        )


class GetModelByIdUseCase:
    """Use case for retrieving a model by ID."""

    @inject
    def __init__(
        self,
        model_repository: IModelRepository = Provide["model_repository"],
        training_job_repository: ITrainingJobRepository = Provide[
            "training_job_repository"
        ],
    ):
        self.model_repository = model_repository
        self.training_job_repository = training_job_repository

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

        trainings = await self.training_job_repository.get_by_model_id(model_id)
        training_summary = [
            ModelTrainingSummaryDTO(
                id=job.id,
                status=job.status,
                start_time=job.start_time,
                end_time=job.end_time,
                error=job.error,
                data_collection_progress=job.get_data_collection_progress(),
                total_data_points_requested=job.total_data_points_requested,
                total_data_points_collected=job.total_data_points_collected,
                created_at=job.created_at,
                updated_at=job.updated_at,
                metrics=_to_training_metrics_dto(job.metrics),
            )
            for job in sorted(trainings, key=lambda j: j.created_at, reverse=True)
        ]

        return ModelDetailResponseDTO(
            id=model.id,
            name=model.name,
            description=model.description,
            model_type=model.model_type,
            status=model.status,
            batch_size=model.batch_size,
            epochs=model.epochs,
            learning_rate=model.learning_rate,
            validation_ratio=model.validation_ratio,
            test_ratio=model.test_ratio,
            lookback_window=model.lookback_window,
            forecast_horizon=model.forecast_horizon,
            feature=model.feature,
            rnn_layers=[_to_rnn_layer_dto(layer) for layer in model.rnn_layers],
            dense_layers=[_to_dense_layer_dto(layer) for layer in model.dense_layers],
            early_stopping_patience=model.early_stopping_patience,
            entity_type=model.entity_type,
            entity_id=model.entity_id,
            created_at=model.created_at,
            updated_at=model.updated_at,
            trainings=training_summary,
        )


class CreateModelUseCase:
    """Use case for creating a new model."""

    @inject
    def __init__(
        self,
        model_repository: IModelRepository = Provide["model_repository"],
        training_job_repository: ITrainingJobRepository = Provide[
            "training_job_repository"
        ],
    ):
        self.model_repository = model_repository
        self.training_job_repository = training_job_repository

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
            model_type_label = model_dto.model_type.value.upper()
            name = f"{model_type_label} - {feature}"

        description = model_dto.description
        if description is None:
            feature = model_dto.feature
            model_type_label = model_dto.model_type.value.upper()
            description = f"{model_type_label} model for {feature} forecasting"

        # Create the domain model from the DTO
        model = Model(
            name=name,
            description=description,
            model_type=model_dto.model_type,
            batch_size=model_dto.batch_size,
            epochs=model_dto.epochs,
            learning_rate=model_dto.learning_rate,
            validation_ratio=model_dto.validation_ratio,
            test_ratio=model_dto.test_ratio,
            lookback_window=model_dto.lookback_window,
            forecast_horizon=model_dto.forecast_horizon,
            feature=model_dto.feature,
            rnn_layers=[_to_rnn_layer_config(layer) for layer in model_dto.rnn_layers],
            dense_layers=[
                _to_dense_layer_config(layer) for layer in model_dto.dense_layers
            ],
            early_stopping_patience=model_dto.early_stopping_patience,
            entity_type=model_dto.entity_type,
            entity_id=model_dto.entity_id,
        )

        validate_model_configuration(model)

        # Save the model via repository
        created_model = await self.model_repository.create(model)

        # Convert to response DTO
        return ModelResponseDTO(
            id=created_model.id,
            name=created_model.name,
            description=created_model.description,
            model_type=created_model.model_type,
            status=created_model.status,
            batch_size=created_model.batch_size,
            epochs=created_model.epochs,
            learning_rate=created_model.learning_rate,
            validation_ratio=created_model.validation_ratio,
            test_ratio=created_model.test_ratio,
            lookback_window=created_model.lookback_window,
            forecast_horizon=created_model.forecast_horizon,
            feature=created_model.feature,
            rnn_layers=[_to_rnn_layer_dto(layer) for layer in created_model.rnn_layers],
            dense_layers=[
                _to_dense_layer_dto(layer) for layer in created_model.dense_layers
            ],
            early_stopping_patience=created_model.early_stopping_patience,
            entity_type=created_model.entity_type,
            entity_id=created_model.entity_id,
            created_at=created_model.created_at,
            updated_at=created_model.updated_at,
            trainings=[],
        )


class UpdateModelUseCase:
    """Use case for updating an existing model."""

    @inject
    def __init__(
        self,
        model_repository: IModelRepository = Provide["model_repository"],
        training_job_repository: ITrainingJobRepository = Provide[
            "training_job_repository"
        ],
    ):
        self.model_repository = model_repository
        self.training_job_repository = training_job_repository

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

        if model.status != ModelStatus.DRAFT:
            raise ModelOperationError(
                f"Model {model_id} cannot be edited since it is "
                f"already in status {model.status.value}"
            )

        original_name = model.name or ""
        original_description = model.description or ""
        original_model_type = model.model_type
        original_feature = model.feature

        # Update fields that are provided in the DTO
        if model_dto.name is not None:
            model.name = model_dto.name

        if model_dto.description is not None:
            model.description = model_dto.description

        if model_dto.model_type is not None:
            model.model_type = model_dto.model_type

        if model_dto.batch_size is not None:
            model.batch_size = model_dto.batch_size

        if model_dto.epochs is not None:
            model.epochs = model_dto.epochs

        if model_dto.learning_rate is not None:
            model.learning_rate = model_dto.learning_rate

        if model_dto.validation_ratio is not None:
            model.validation_ratio = model_dto.validation_ratio

        if model_dto.test_ratio is not None:
            model.test_ratio = model_dto.test_ratio

        if model_dto.lookback_window is not None:
            model.lookback_window = model_dto.lookback_window

        if model_dto.forecast_horizon is not None:
            model.forecast_horizon = model_dto.forecast_horizon

        if model_dto.feature is not None:
            model.feature = model_dto.feature

        if model_dto.rnn_layers is not None:
            model.rnn_layers = [
                _to_rnn_layer_config(layer) for layer in model_dto.rnn_layers
            ]

        if model_dto.dense_layers is not None:
            model.dense_layers = [
                _to_dense_layer_config(layer) for layer in model_dto.dense_layers
            ]

        if model_dto.early_stopping_patience is not None:
            model.early_stopping_patience = model_dto.early_stopping_patience

        if model_dto.entity_type is not None:
            model.entity_type = model_dto.entity_type

        if model_dto.entity_id is not None:
            model.entity_id = model_dto.entity_id

        # Update derived defaults when users did not provide overrides
        default_name_before = (
            f"{original_model_type.value.upper()} - {original_feature}"
        )
        legacy_default_name = f"{original_model_type.value} - {original_feature}"
        default_description_before = (
            f"{original_model_type.value.upper()} model for "
            f"{original_feature} forecasting"
        )
        legacy_default_description = (
            f"{original_model_type.value} model for {original_feature} forecasting"
        )

        validate_model_configuration(model)

        if (
            model_dto.name is None
            and original_name
            and original_name in {default_name_before, legacy_default_name}
        ):
            model.name = f"{model.model_type.value.upper()} - {model.feature}"

        if (
            model_dto.description is None
            and original_description
            and original_description
            in {default_description_before, legacy_default_description}
        ):
            model.description = (
                f"{model.model_type.value.upper()} model for "
                f"{model.feature} forecasting"
            )

        # Update timestamp
        model.update_timestamp()

        # Save the updated model
        updated_model = await self.model_repository.update(model)

        training_jobs = await self.training_job_repository.get_by_model_id(model_id)
        training_summary = [
            ModelTrainingSummaryDTO(
                id=job.id,
                status=job.status,
                start_time=job.start_time,
                end_time=job.end_time,
                error=job.error,
                data_collection_progress=job.get_data_collection_progress(),
                total_data_points_requested=job.total_data_points_requested,
                total_data_points_collected=job.total_data_points_collected,
                created_at=job.created_at,
                updated_at=job.updated_at,
            )
            for job in sorted(training_jobs, key=lambda j: j.created_at, reverse=True)
        ]

        # Convert to response DTO
        return ModelResponseDTO(
            id=updated_model.id,
            name=updated_model.name,
            description=updated_model.description,
            model_type=updated_model.model_type,
            status=updated_model.status,
            batch_size=updated_model.batch_size,
            epochs=updated_model.epochs,
            learning_rate=updated_model.learning_rate,
            validation_ratio=updated_model.validation_ratio,
            test_ratio=updated_model.test_ratio,
            lookback_window=updated_model.lookback_window,
            forecast_horizon=updated_model.forecast_horizon,
            feature=updated_model.feature,
            rnn_layers=[_to_rnn_layer_dto(layer) for layer in updated_model.rnn_layers],
            dense_layers=[
                _to_dense_layer_dto(layer) for layer in updated_model.dense_layers
            ],
            early_stopping_patience=updated_model.early_stopping_patience,
            entity_type=updated_model.entity_type,
            entity_id=updated_model.entity_id,
            created_at=updated_model.created_at,
            updated_at=updated_model.updated_at,
            trainings=training_summary,
        )


class DeleteModelUseCase:
    """Use case for deleting a model."""

    @inject
    def __init__(
        self,
        model_repository: IModelRepository = Provide["model_repository"],
        training_job_repository: ITrainingJobRepository = Provide[
            "training_job_repository"
        ],
        artifacts_repository: IModelArtifactsRepository = Provide[
            "model_artifacts_repository"
        ],
    ):
        self.model_repository = model_repository
        self.training_job_repository = training_job_repository
        self.artifacts_repository = artifacts_repository

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

        # Delete associated training jobs and artifacts before removing the model
        training_jobs = await self.training_job_repository.get_by_model_id(model_id)
        for training_job in training_jobs:
            await self.training_job_repository.delete(training_job.id)

        await self.artifacts_repository.delete_model_artifacts(model_id)

        # Delete the model
        await self.model_repository.delete(model_id)


class GetModelTypesUseCase:
    """Use case for listing available model types."""

    async def execute(self) -> List[ModelTypeOptionDTO]:
        """Return the model types supported by the platform."""

        options: List[ModelTypeOptionDTO] = []
        for model_type in ModelType:
            options.append(
                ModelTypeOptionDTO(
                    value=model_type,
                    label=model_type.value.upper(),
                )
            )
        return options
