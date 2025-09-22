"""
Domain Entities - Model

This module defines the core domain entities related to deep learning models.
These entities encapsulate the business rules and logic of the models,
without dependencies on external frameworks or infrastructure.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional
from uuid import UUID, uuid4


class ModelType(str, Enum):
    """Type of deep learning model architecture."""

    LSTM = "lstm"
    GRU = "gru"


class ModelStatus(str, Enum):
    """Status of the model."""

    DRAFT = "draft"
    TRAINING = "training"
    TRAINED = "trained"


@dataclass
class RNNLayerConfig:
    """Configuration for an individual recurrent layer."""

    units: int
    dropout: float = 0.1
    recurrent_dropout: float = 0.0


@dataclass
class DenseLayerConfig:
    """Configuration for an individual dense layer."""

    units: int
    dropout: float = 0.1
    activation: str = "relu"


@dataclass
class Model:
    """Represents a deep learning model configuration for time series forecasting."""

    id: UUID = field(default_factory=uuid4)
    name: str = ""
    description: Optional[str] = None
    model_type: ModelType = ModelType.LSTM
    status: ModelStatus = ModelStatus.DRAFT

    # Hyperparameters
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 0.001
    validation_split: float = 0.2
    rnn_layers: List[RNNLayerConfig] = field(
        default_factory=lambda: [RNNLayerConfig(units=64)]
    )
    dense_layers: List[DenseLayerConfig] = field(default_factory=list)
    early_stopping_patience: Optional[int] = None

    # Input/Output configuration
    lookback_window: int = 24
    forecast_horizon: int = 1
    feature: str = ""

    # FIWARE STH Comet configuration
    entity_type: str = ""
    entity_id: str = ""

    # Other attributes
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    has_successful_training: bool = False

    def update_timestamp(self) -> None:
        """Update the 'updated_at' timestamp to current time."""
        self.updated_at = datetime.now(timezone.utc)

    def has_trained_artifacts(self) -> bool:
        """Indicate whether the model has ever completed a successful training."""
        return self.has_successful_training

    def clear_artifacts(self) -> None:
        """Reset training flag when all trainings are removed."""
        self.has_successful_training = False
        self.update_timestamp()
