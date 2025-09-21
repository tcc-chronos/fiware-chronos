"""
Domain Entities - Training Job

This module defines the core domain entities related to training jobs.
These entities encapsulate the business rules and logic of the training process,
without dependencies on external frameworks or infrastructure.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4


class TrainingStatus(str, Enum):
    """Status of a training job."""

    PENDING = "pending"
    CANCEL_REQUESTED = "cancel_requested"
    COLLECTING_DATA = "collecting_data"
    PREPROCESSING = "preprocessing"
    TRAINING = "training"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class DataCollectionStatus(str, Enum):
    """Status of data collection phase."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class DataCollectionJob:
    """Represents a data collection job for a specific chunk of data."""

    id: UUID = field(default_factory=uuid4)
    h_offset: int = 0
    last_n: int = 100
    status: DataCollectionStatus = DataCollectionStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error: Optional[str] = None
    data_points_collected: int = 0


@dataclass
class TrainingMetrics:
    """Training metrics and evaluation results."""

    mse: Optional[float] = None
    mae: Optional[float] = None
    rmse: Optional[float] = None
    mape: Optional[float] = None
    r2: Optional[float] = None
    mae_pct: Optional[float] = None
    rmse_pct: Optional[float] = None
    best_train_loss: Optional[float] = None
    best_val_loss: Optional[float] = None
    best_epoch: Optional[int] = None


@dataclass
class TrainingJob:
    """Represents a training job for a deep learning model."""

    id: UUID = field(default_factory=uuid4)
    model_id: Optional[UUID] = None
    status: TrainingStatus = TrainingStatus.PENDING

    # Training configuration
    last_n: int = 1000

    # Data collection
    data_collection_jobs: List[DataCollectionJob] = field(default_factory=list)
    total_data_points_requested: int = 0
    total_data_points_collected: int = 0
    task_refs: Dict[str, Any] = field(default_factory=dict)

    # Training process
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    data_collection_start: Optional[datetime] = None
    data_collection_end: Optional[datetime] = None
    preprocessing_start: Optional[datetime] = None
    preprocessing_end: Optional[datetime] = None
    training_start: Optional[datetime] = None
    training_end: Optional[datetime] = None

    # Results
    metrics: Optional[TrainingMetrics] = None
    model_artifact_id: Optional[str] = None
    x_scaler_artifact_id: Optional[str] = None
    y_scaler_artifact_id: Optional[str] = None
    metadata_artifact_id: Optional[str] = None

    # Error handling
    error: Optional[str] = None
    error_details: Optional[Dict[str, Any]] = None

    # Audit
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def update_timestamp(self) -> None:
        """Update the 'updated_at' timestamp to current time."""
        self.updated_at = datetime.now(timezone.utc)

    def add_data_collection_job(self, job: DataCollectionJob) -> None:
        """Add a data collection job to this training job."""
        self.data_collection_jobs.append(job)
        self.update_timestamp()

    def get_data_collection_progress(self) -> float:
        try:
            requested = int(self.total_data_points_requested or 0)
            collected = int(self.total_data_points_collected or 0)
            if requested == 0:
                return 0.0
            return (collected / requested) * 100.0
        except Exception:
            return 0.0

    def mark_data_collection_complete(self) -> None:
        """Mark data collection phase as complete."""
        self.data_collection_end = datetime.now(timezone.utc)
        self.update_timestamp()

    def mark_preprocessing_complete(self) -> None:
        """Mark preprocessing phase as complete."""
        self.preprocessing_end = datetime.now(timezone.utc)
        self.update_timestamp()

    def mark_training_complete(self, metrics: TrainingMetrics) -> None:
        """Mark training phase as complete with metrics."""
        self.training_end = datetime.now(timezone.utc)
        self.metrics = metrics
        self.status = TrainingStatus.COMPLETED
        self.end_time = datetime.now(timezone.utc)
        self.update_timestamp()

    def mark_failed(
        self, error: str, error_details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Mark training job as failed."""
        self.status = TrainingStatus.FAILED
        self.error = error
        self.error_details = error_details
        self.end_time = datetime.now(timezone.utc)
        self.update_timestamp()

    def get_total_duration(self) -> Optional[float]:
        """Get total duration in seconds, if completed."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None

    def get_training_duration(self) -> Optional[float]:
        """Get training phase duration in seconds, if completed."""
        if self.training_start and self.training_end:
            return (self.training_end - self.training_start).total_seconds()
        return None
