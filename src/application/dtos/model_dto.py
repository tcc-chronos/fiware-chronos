"""
Model DTOs - Application Layer

This module defines Data Transfer Objects (DTOs) for the model entities.
These DTOs are used to transfer data between the application layer and
the presentation layer (API).
"""

from datetime import datetime
from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel, Field, field_validator, model_validator

from src.domain.entities.model import ModelStatus, ModelType
from src.domain.entities.training_job import TrainingStatus


class ModelTrainingSummaryDTO(BaseModel):
    """Summary information about a training job for a model."""

    id: UUID
    status: TrainingStatus
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error: Optional[str] = None
    data_collection_progress: float = 0.0
    total_data_points_requested: int = 0
    total_data_points_collected: int = 0
    created_at: datetime
    updated_at: datetime

    model_config = {
        "json_schema_extra": {
            "example": {
                "id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                "status": "completed",
                "start_time": "2025-09-21T16:00:00Z",
                "end_time": "2025-09-21T16:30:00Z",
                "error": None,
                "data_collection_progress": 100.0,
                "total_data_points_requested": 10000,
                "total_data_points_collected": 10000,
                "created_at": "2025-09-21T16:00:00Z",
                "updated_at": "2025-09-21T16:30:00Z",
            }
        }
    }


class ModelTypeOptionDTO(BaseModel):
    """DTO representing an available model type option."""

    value: ModelType
    label: str

    model_config = {
        "json_schema_extra": {
            "example": {
                "value": "lstm",
                "label": "LSTM",
            }
        }
    }


class RNNLayerDTO(BaseModel):
    """DTO describing a recurrent layer configuration."""

    units: int = Field(..., description="Number of neurons for the RNN layer", gt=0)
    dropout: float = Field(
        0.1,
        description="Dropout rate applied after the RNN layer",
        ge=0.0,
        lt=1.0,
    )
    recurrent_dropout: float = Field(
        0.0,
        description="Dropout rate applied to recurrent connections",
        ge=0.0,
        lt=1.0,
    )


class DenseLayerDTO(BaseModel):
    """DTO describing a dense layer configuration."""

    units: int = Field(..., description="Number of neurons for the dense layer", gt=0)
    dropout: float = Field(
        0.1,
        description="Dropout rate applied after the dense layer",
        ge=0.0,
        lt=1.0,
    )
    activation: str = Field(
        "relu",
        description="Activation function for the dense layer",
        min_length=1,
    )


class ModelCreateDTO(BaseModel):
    """DTO for creating a new model."""

    name: Optional[str] = Field(
        None, description="Name of the model", min_length=1, max_length=100
    )
    description: Optional[str] = Field(None, description="Description of the model")
    model_type: ModelType = Field(..., description="Type of model architecture")

    # Hyperparameters
    batch_size: int = Field(
        default=32, description="Batch size for training", ge=1, le=1024
    )
    epochs: int = Field(
        default=100, description="Number of epochs for training", ge=1, le=1000
    )
    learning_rate: float = Field(
        default=0.001, description="Learning rate", gt=0.0, le=1.0
    )
    validation_split: float = Field(
        default=0.2, description="Validation split ratio", ge=0.0, lt=1.0
    )
    rnn_layers: List[RNNLayerDTO] = Field(
        ...,
        description="Configuration for each RNN layer (ordered from input to output)",
        min_length=1,
    )
    dense_layers: List[DenseLayerDTO] = Field(
        default_factory=list,
        description="Configuration for each dense layer after the recurrent stack",
    )
    early_stopping_patience: Optional[int] = None

    # Input/Output configuration
    lookback_window: int = Field(
        default=24, description="Size of lookback window", ge=1, le=8760
    )
    forecast_horizon: int = Field(
        default=1, description="Size of forecast horizon", ge=1, le=8760
    )
    feature: str = Field(
        ..., description="Feature name to use from STH Comet", min_length=1
    )

    # FIWARE STH Comet configuration
    entity_type: str = Field(..., description="Entity type in FIWARE", min_length=1)
    entity_id: str = Field(..., description="Entity ID in FIWARE", min_length=1)

    @field_validator("name")
    @classmethod
    def generate_name_if_none(cls, v, info):
        """Generate a name if none is provided."""
        if v is not None:
            return v

        values = info.data
        feature = values.get("feature")
        model_type = values.get("model_type")

        if feature is None or model_type is None:
            return v

        return f"{model_type.value.upper()} - {feature}"

    @field_validator("description")
    @classmethod
    def generate_description_if_none(cls, v, info):
        """Generate a description if none is provided."""
        if v is not None:
            return v

        values = info.data
        feature = values.get("feature")
        model_type = values.get("model_type")

        if feature is None or model_type is None:
            return v

        return f"{model_type.value.upper()} model for {feature} forecasting"

    @model_validator(mode="after")
    def calculate_early_stopping_patience(self) -> "ModelCreateDTO":
        """Calculate early_stopping_patience if not provided."""
        # If specific value is provided, use it
        if self.early_stopping_patience is not None:
            return self

        # Smart default: around 10% of epochs, minimum 5, maximum 20
        epochs = self.epochs
        self.early_stopping_patience = min(max(int(epochs * 0.1), 5), 20)
        return self

    model_config = {
        "json_schema_extra": {
            "example": {
                "name": "LSTM - temperature",
                "description": "LSTM model for temperature forecasting",
                "model_type": "lstm",
                "batch_size": 32,
                "epochs": 100,
                "learning_rate": 0.001,
                "lookback_window": 24,
                "forecast_horizon": 6,
                "feature": "temperature",
                "rnn_layers": [
                    {"units": 128, "dropout": 0.1, "recurrent_dropout": 0.0},
                    {"units": 64, "dropout": 0.2, "recurrent_dropout": 0.05},
                ],
                "dense_layers": [
                    {"units": 64, "dropout": 0.1, "activation": "relu"},
                    {"units": 32, "dropout": 0.1, "activation": "relu"},
                ],
                "early_stopping_patience": 10,
                "entity_type": "Sensor",
                "entity_id": "urn:ngsi-ld:Chronos:ESP32:001",
            }
        }
    }


class ModelUpdateDTO(BaseModel):
    """DTO for updating an existing model."""

    name: Optional[str] = Field(
        None, description="Name of the model", min_length=1, max_length=100
    )
    description: Optional[str] = Field(None, description="Description of the model")
    model_type: Optional[ModelType] = Field(
        None, description="Type of model architecture"
    )

    # Hyperparameters
    batch_size: Optional[int] = Field(
        None, description="Batch size for training", ge=1, le=1024
    )
    epochs: Optional[int] = Field(
        None, description="Number of epochs for training", ge=1, le=1000
    )
    learning_rate: Optional[float] = Field(
        None, description="Learning rate", gt=0.0, le=1.0
    )
    validation_split: Optional[float] = Field(
        None, description="Validation split ratio", ge=0.0, lt=1.0
    )
    rnn_layers: Optional[List[RNNLayerDTO]] = Field(
        None,
        description="Configuration for each RNN layer (ordered from input to output)",
        min_length=1,
    )
    dense_layers: Optional[List[DenseLayerDTO]] = Field(
        None, description="Configuration for each dense layer"
    )
    early_stopping_patience: Optional[int] = None

    # Input/Output configuration
    lookback_window: Optional[int] = Field(
        None, description="Size of lookback window", ge=1, le=8760
    )
    forecast_horizon: Optional[int] = Field(
        None, description="Size of forecast horizon", ge=1, le=8760
    )
    feature: Optional[str] = Field(
        None, description="Feature name to use from STH Comet", min_length=1
    )

    # FIWARE STH Comet configuration
    entity_type: Optional[str] = Field(
        None, description="Entity type in FIWARE", min_length=1
    )
    entity_id: Optional[str] = Field(
        None, description="Entity ID in FIWARE", min_length=1
    )

    @model_validator(mode="after")
    def calculate_early_stopping_patience(self) -> "ModelUpdateDTO":
        """Calculate early_stopping_patience if not provided."""
        # If specific value is provided, use it
        if self.early_stopping_patience is not None:
            return self

        # Only calculate if epochs is provided
        if self.epochs is None:
            return self

        # Smart default: around 10% of epochs, minimum 5, maximum 20
        epochs = self.epochs
        self.early_stopping_patience = min(max(int(epochs * 0.1), 5), 20)
        return self

    model_config = {
        "json_schema_extra": {
            "example": {
                "name": "Updated Temperature Model",
                "description": "Modelo atualizado para previsão de temperatura",
                "rnn_layers": [
                    {"units": 128, "dropout": 0.15, "recurrent_dropout": 0.05},
                    {"units": 64, "dropout": 0.1, "recurrent_dropout": 0.0},
                ],
                "dense_layers": [
                    {"units": 64, "dropout": 0.1, "activation": "relu"},
                    {"units": 32, "dropout": 0.1, "activation": "relu"},
                ],
                "epochs": 150,
                "batch_size": 64,
                "learning_rate": 0.0005,
                "feature": "temperatura",
            }
        }
    }


class ModelResponseDTO(BaseModel):
    """DTO for model response."""

    id: UUID
    name: str
    description: Optional[str] = None
    model_type: ModelType
    status: ModelStatus

    # Hyperparameters
    batch_size: int
    epochs: int
    learning_rate: float
    validation_split: float
    rnn_layers: List[RNNLayerDTO]
    dense_layers: List[DenseLayerDTO]
    early_stopping_patience: Optional[int] = None

    # Input/Output configuration
    lookback_window: int
    forecast_horizon: int
    feature: str

    # FIWARE STH Comet configuration
    entity_type: Optional[str] = None
    entity_id: Optional[str] = None

    created_at: datetime
    updated_at: datetime
    trainings: List[ModelTrainingSummaryDTO] = Field(default_factory=list)

    model_config = {
        "json_schema_extra": {
            "example": {
                "id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                "name": "LSTM - temperature",
                "description": "LSTM model for temperature forecasting",
                "model_type": "lstm",
                "status": "draft",
                "batch_size": 32,
                "epochs": 100,
                "learning_rate": 0.001,
                "validation_split": 0.2,
                "rnn_layers": [
                    {"units": 128, "dropout": 0.1, "recurrent_dropout": 0.0},
                    {"units": 64, "dropout": 0.2, "recurrent_dropout": 0.05},
                ],
                "dense_layers": [
                    {"units": 64, "dropout": 0.1, "activation": "relu"},
                    {"units": 32, "dropout": 0.1, "activation": "relu"},
                ],
                "early_stopping_patience": 10,
                "lookback_window": 24,
                "forecast_horizon": 6,
                "feature": "temperature",
                "entity_type": "Sensor",
                "entity_id": "urn:ngsi-ld:Chronos:ESP32:001",
                "created_at": "2025-09-14T10:00:00Z",
                "updated_at": "2025-09-14T10:00:00Z",
                "trainings": [
                    {
                        "id": "4fa85f64-5717-4562-b3fc-2c963f66afa7",
                        "status": "completed",
                        "start_time": "2025-09-14T11:00:00Z",
                        "end_time": "2025-09-14T12:30:00Z",
                        "error": None,
                        "data_collection_progress": 100.0,
                        "total_data_points_requested": 10000,
                        "total_data_points_collected": 10000,
                        "created_at": "2025-09-14T11:00:00Z",
                        "updated_at": "2025-09-14T12:30:00Z",
                    }
                ],
            }
        }
    }


class ModelDetailResponseDTO(ModelResponseDTO):
    """DTO for detailed model response."""

    model_config = ModelResponseDTO.model_config
