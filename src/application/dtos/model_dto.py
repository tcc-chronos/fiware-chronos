"""
Model DTOs - Application Layer

This module defines Data Transfer Objects (DTOs) for the model entities.
These DTOs are used to transfer data between the application layer and
the presentation layer (API).
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field, field_validator, model_validator

from src.domain.entities.model import ModelStatus, ModelType


class TrainingMetricsDTO(BaseModel):
    """DTO for model training metrics."""

    id: UUID
    start_time: datetime
    end_time: Optional[datetime] = None
    metrics: Dict[str, float] = Field(default_factory=dict)
    status: str

    model_config = {
        "json_schema_extra": {
            "example": {
                "id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                "start_time": "2025-09-14T10:00:00Z",
                "end_time": "2025-09-14T11:30:00Z",
                "metrics": {
                    "val_loss": 0.123,
                    "val_mae": 0.345,
                    "loss": 0.111,
                    "mae": 0.222,
                },
                "status": "completed",
            }
        }
    }


class ModelCreateDTO(BaseModel):
    """DTO for creating a new model."""

    name: Optional[str] = Field(
        None, description="Name of the model", min_length=1, max_length=100
    )
    description: Optional[str] = Field(None, description="Description of the model")
    model_type: ModelType = Field(
        default=ModelType.LSTM, description="Type of model architecture"
    )

    # Hyperparameters
    dropout: float = Field(default=0.2, description="Dropout rate", ge=0.0, le=0.9)
    recurrent_dropout: float = Field(
        default=0.0, description="Recurrent dropout rate", ge=0.0, le=0.9
    )
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
    rnn_units: List[int] = Field(
        ..., description="List of units for each RNN layer", min_length=1
    )
    dense_units: List[int] = Field(
        default_factory=lambda: [64, 32],
        description="List of units for each Dense layer",
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
        default="value", description="Feature name to use from STH Comet"
    )

    # FIWARE STH Comet configuration
    entity_type: Optional[str] = Field(None, description="Entity type in FIWARE")
    entity_id: Optional[str] = Field(None, description="Entity ID in FIWARE")

    # Additional metadata
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )

    @field_validator("name")
    @classmethod
    def generate_name_if_none(cls, v, info):
        """Generate a name if none is provided."""
        if v is not None:
            return v

        values = info.data
        feature = values.get("feature", "value")
        model_type = values.get("model_type", ModelType.LSTM)

        return f"{model_type} - {feature}"

    @field_validator("description")
    @classmethod
    def generate_description_if_none(cls, v, info):
        """Generate a description if none is provided."""
        if v is not None:
            return v

        values = info.data
        feature = values.get("feature", "value")
        model_type = values.get("model_type", ModelType.LSTM)

        return f"{model_type} model for {feature} forecasting"

    @field_validator("rnn_units")
    @classmethod
    def validate_rnn_units(cls, v):
        """Validate that rnn_units has at least one positive value."""
        if not v or any(unit <= 0 for unit in v):
            raise ValueError("rnn_units must contain at least one positive value")
        return v

    @field_validator("dense_units")
    @classmethod
    def validate_dense_units(cls, v):
        """Validate that dense_units contains only positive values."""
        if any(unit <= 0 for unit in v):
            raise ValueError("dense_units must contain only positive values")
        return v

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
                "name": "Temperature Forecasting Model",
                "description": "LSTM model for temperature forecasting",
                "model_type": "lstm",
                "dropout": 0.2,
                "batch_size": 32,
                "epochs": 100,
                "learning_rate": 0.001,
                "lookback_window": 24,
                "forecast_horizon": 6,
                "feature": "temperature",
                "rnn_units": [128, 64],
                "dense_units": [64, 32],
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

    # Hyperparameters
    dropout: Optional[float] = Field(None, description="Dropout rate", ge=0.0, le=0.9)
    recurrent_dropout: Optional[float] = Field(
        None, description="Recurrent dropout rate", ge=0.0, le=0.9
    )
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
    rnn_units: Optional[List[int]] = Field(
        None, description="List of units for each RNN layer"
    )
    dense_units: Optional[List[int]] = Field(
        None, description="List of units for each Dense layer"
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
        None, description="Feature name to use from STH Comet"
    )

    # FIWARE STH Comet configuration
    entity_type: Optional[str] = Field(None, description="Entity type in FIWARE")
    entity_id: Optional[str] = Field(None, description="Entity ID in FIWARE")

    # Additional metadata
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

    @field_validator("rnn_units")
    @classmethod
    def validate_rnn_units(cls, v):
        """Validate that rnn_units has only positive values."""
        if v and any(unit <= 0 for unit in v):
            raise ValueError("rnn_units must contain only positive values")
        return v

    @field_validator("dense_units")
    @classmethod
    def validate_dense_units(cls, v):
        """Validate that dense_units contains only positive values."""
        if v and any(unit <= 0 for unit in v):
            raise ValueError("dense_units must contain only positive values")
        return v

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
                "rnn_units": [128, 64],
                "dense_units": [64, 32],
                "epochs": 150,
                "batch_size": 64,
                "dropout": 0.3,
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
    dropout: float
    recurrent_dropout: float
    batch_size: int
    epochs: int
    learning_rate: float
    validation_split: float
    rnn_units: List[int]
    dense_units: List[int]
    early_stopping_patience: Optional[int] = None

    # Input/Output configuration
    lookback_window: int
    forecast_horizon: int
    feature: str

    # FIWARE STH Comet configuration
    entity_type: Optional[str] = None
    entity_id: Optional[str] = None

    # Metadata
    created_at: datetime
    updated_at: datetime
    metadata: Dict[str, Any]

    model_config = {
        "json_schema_extra": {
            "example": {
                "id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                "name": "Temperature Forecasting Model",
                "description": "LSTM model for temperature forecasting",
                "model_type": "lstm",
                "status": "draft",
                "dropout": 0.2,
                "recurrent_dropout": 0.0,
                "batch_size": 32,
                "epochs": 100,
                "learning_rate": 0.001,
                "validation_split": 0.2,
                "rnn_units": [128, 64],
                "dense_units": [64, 32],
                "early_stopping_patience": 10,
                "lookback_window": 24,
                "forecast_horizon": 6,
                "feature": "temperature",
                "entity_type": "Sensor",
                "entity_id": "urn:ngsi-ld:Chronos:ESP32:001",
                "created_at": "2025-09-14T10:00:00Z",
                "updated_at": "2025-09-14T10:00:00Z",
                "metadata": {},
            }
        }
    }


class ModelDetailResponseDTO(ModelResponseDTO):
    """DTO for detailed model response with training metrics."""

    trainings: List[TrainingMetricsDTO] = Field(default_factory=list)

    model_config = {
        "json_schema_extra": {
            "example": {
                "id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                "name": "Temperature Forecasting Model",
                "description": "LSTM model for temperature forecasting",
                "model_type": "lstm",
                "status": "trained",
                "dropout": 0.2,
                "recurrent_dropout": 0.0,
                "batch_size": 32,
                "epochs": 100,
                "learning_rate": 0.001,
                "validation_split": 0.2,
                "rnn_units": [128, 64],
                "dense_units": [64, 32],
                "early_stopping_patience": 10,
                "lookback_window": 24,
                "forecast_horizon": 6,
                "feature": "temperature",
                "entity_type": "Sensor",
                "entity_id": "urn:ngsi-ld:Chronos:ESP32:001",
                "created_at": "2025-09-14T10:00:00Z",
                "updated_at": "2025-09-14T12:30:00Z",
                "metadata": {},
                "trainings": [
                    {
                        "id": "4fa85f64-5717-4562-b3fc-2c963f66afa7",
                        "start_time": "2025-09-14T11:00:00Z",
                        "end_time": "2025-09-14T12:30:00Z",
                        "metrics": {"val_loss": 0.123, "val_mae": 0.345},
                        "status": "completed",
                    }
                ],
            }
        }
    }
