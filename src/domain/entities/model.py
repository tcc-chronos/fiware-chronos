"""
Domain Entities - Model

This module defines the core domain entities related to deep learning models.
These entities encapsulate the business rules and logic of the models,
without dependencies on external frameworks or infrastructure.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4


class ModelType(str, Enum):
    """Type of deep learning model architecture."""

    LSTM = "lstm"
    GRU = "gru"


class ModelStatus(str, Enum):
    """Status of the model."""

    DRAFT = "draft"
    READY = "ready"
    TRAINING = "training"
    TRAINED = "trained"
    ERROR = "error"


@dataclass
class Training:
    """Represents a training instance of a model."""

    id: UUID = field(default_factory=uuid4)
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    end_time: Optional[datetime] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    dataset_info: Dict[str, Any] = field(default_factory=dict)
    status: str = "pending"
    error: Optional[str] = None


@dataclass
class Model:
    """Represents a deep learning model configuration for time series forecasting."""

    id: UUID = field(default_factory=uuid4)
    name: str = ""
    description: Optional[str] = None
    model_type: ModelType = ModelType.LSTM
    status: ModelStatus = ModelStatus.DRAFT

    # Hyperparameters
    rnn_dropout: float = 0.0
    dense_dropout: float = 0.2
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 0.001
    validation_split: float = 0.2
    rnn_units: List[int] = field(default_factory=lambda: [64])
    dense_units: List[int] = field(default_factory=lambda: [32])
    early_stopping_patience: Optional[int] = None

    # Input/Output configuration
    lookback_window: int = 24
    forecast_horizon: int = 1
    feature: str = "value"

    # FIWARE STH Comet configuration
    entity_type: Optional[str] = None
    entity_id: Optional[str] = None

    # Model artifacts (GridFS file IDs)
    model_artifact_id: Optional[str] = None
    x_scaler_artifact_id: Optional[str] = None
    y_scaler_artifact_id: Optional[str] = None
    metadata_artifact_id: Optional[str] = None

    # Other attributes
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)
    trainings: List[Training] = field(default_factory=list)

    def update_timestamp(self) -> None:
        """Update the 'updated_at' timestamp to current time."""
        self.updated_at = datetime.now(timezone.utc)

    def add_training(self, training: Training) -> None:
        """Add a new training instance to this model."""
        self.trainings.append(training)
        self.update_timestamp()

    def get_latest_training(self) -> Optional[Training]:
        """Get the most recent training instance, if any."""
        if not self.trainings:
            return None
        return sorted(self.trainings, key=lambda t: t.start_time, reverse=True)[0]

    def get_best_training(
        self, metric: str = "val_loss", higher_is_better: bool = False
    ) -> Optional[Training]:
        """
        Get the best training instance based on a specific metric.

        Args:
            metric: The metric name to compare (default: val_loss)
            higher_is_better: Whether higher metric values are better (default: False)

        Returns:
            The best training instance or None if no trainings exist
        """
        if not self.trainings:
            return None

        trainings_with_metric = [t for t in self.trainings if metric in t.metrics]
        if not trainings_with_metric:
            return None

        if higher_is_better:
            return max(trainings_with_metric, key=lambda t: t.metrics[metric])
        else:
            return min(trainings_with_metric, key=lambda t: t.metrics[metric])

    def set_artifact_ids(
        self,
        model_artifact_id: Optional[str] = None,
        x_scaler_artifact_id: Optional[str] = None,
        y_scaler_artifact_id: Optional[str] = None,
        metadata_artifact_id: Optional[str] = None,
    ) -> None:
        """Set the GridFS artifact IDs for the model components."""
        if model_artifact_id is not None:
            self.model_artifact_id = model_artifact_id
        if x_scaler_artifact_id is not None:
            self.x_scaler_artifact_id = x_scaler_artifact_id
        if y_scaler_artifact_id is not None:
            self.y_scaler_artifact_id = y_scaler_artifact_id
        if metadata_artifact_id is not None:
            self.metadata_artifact_id = metadata_artifact_id
        self.update_timestamp()

    def has_trained_artifacts(self) -> bool:
        """Check if the model has all required trained artifacts."""
        return all(
            [
                self.model_artifact_id,
                self.x_scaler_artifact_id,
                self.y_scaler_artifact_id,
                self.metadata_artifact_id,
            ]
        )

    def clear_artifacts(self) -> None:
        """Clear all artifact IDs (useful when retraining)."""
        self.model_artifact_id = None
        self.x_scaler_artifact_id = None
        self.y_scaler_artifact_id = None
        self.metadata_artifact_id = None
        self.update_timestamp()
