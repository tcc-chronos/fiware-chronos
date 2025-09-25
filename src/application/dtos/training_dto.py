"""
Application DTOs - Training

This module contains Data Transfer Objects (DTOs) for training operations.
DTOs are used to transfer data between layers and define the API contracts.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field

from src.domain.entities.training_job import DataCollectionStatus, TrainingStatus


class TrainingRequestDTO(BaseModel):
    """DTO for training request."""

    last_n: int = Field(
        default=1000,
        ge=0,
        le=1000000,
        description="Number of most recent data points to collect for training",
    )


class DataCollectionJobDTO(BaseModel):
    """DTO for data collection job status."""

    id: UUID
    h_offset: int
    last_n: int
    status: DataCollectionStatus
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error: Optional[str] = None
    data_points_collected: int


class TrainingMetricsDTO(BaseModel):
    """DTO for training metrics."""

    mse: Optional[float] = None
    mae: Optional[float] = None
    rmse: Optional[float] = None
    theil_u: Optional[float] = None
    mape: Optional[float] = None
    r2: Optional[float] = None
    mae_pct: Optional[float] = None
    rmse_pct: Optional[float] = None
    best_train_loss: Optional[float] = None
    best_val_loss: Optional[float] = None
    best_epoch: Optional[int] = None


class TrainingJobDTO(BaseModel):
    """DTO for training job status and details."""

    id: UUID
    model_id: Optional[UUID]
    status: TrainingStatus

    # Configuration
    last_n: int

    # Progress tracking
    data_collection_jobs: List["DataCollectionJobDTO"]
    total_data_points_requested: int
    total_data_points_collected: int
    data_collection_progress: float

    # Timing
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    data_collection_start: Optional[datetime] = None
    data_collection_end: Optional[datetime] = None
    preprocessing_start: Optional[datetime] = None
    preprocessing_end: Optional[datetime] = None
    training_start: Optional[datetime] = None
    training_end: Optional[datetime] = None

    # Results
    metrics: Optional[TrainingMetricsDTO] = None
    model_artifact_id: Optional[str] = None

    # Error handling
    error: Optional[str] = None
    error_details: Optional[Dict[str, Any]] = None

    # Audit
    created_at: datetime
    updated_at: datetime

    # Computed fields
    total_duration_seconds: Optional[float] = None
    training_duration_seconds: Optional[float] = None


class TrainingJobSummaryDTO(BaseModel):
    """DTO for training job summary (for listing purposes)."""

    id: UUID
    model_id: Optional[UUID]
    status: TrainingStatus
    data_collection_progress: float
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error: Optional[str] = None
    created_at: datetime


class StartTrainingResponseDTO(BaseModel):
    """DTO for training start response."""

    training_job_id: UUID
    message: str = "Training job started successfully"
    status: TrainingStatus = TrainingStatus.PENDING


class STHCometDataPointDTO(BaseModel):
    """DTO for STH-Comet data point."""

    attrName: str
    attrType: str
    attrValue: str
    recvTime: datetime


class STHCometResponseDTO(BaseModel):
    """DTO for STH-Comet response."""

    contextResponses: List[Dict]


class CollectedDataDTO(BaseModel):
    """DTO for collected and processed data."""

    timestamp: datetime
    value: float


class DataCollectionSummaryDTO(BaseModel):
    """DTO for data collection summary."""

    total_requested: int
    total_collected: int
    collection_jobs: int
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_seconds: Optional[float] = None
