"""
Application DTOs - Prediction

Data Transfer Objects for prediction responses produced by the
model prediction use case.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class HistoricalPointDTO(BaseModel):
    """Represents a point from the context window used for prediction."""

    timestamp: datetime
    value: float


class ForecastPointDTO(BaseModel):
    """Represents an individual forecasted value."""

    step: int = Field(ge=1, description="Relative step of the forecast (1-indexed)")
    value: float = Field(description="Predicted value for the step")
    timestamp: Optional[datetime] = Field(
        default=None,
        description=(
            "Expected timestamp for the prediction if the sampling interval could "
            "be inferred."
        ),
    )


class PredictionResponseDTO(BaseModel):
    """DTO returned by the prediction endpoint."""

    model_id: UUID
    training_job_id: UUID
    lookback_window: int
    forecast_horizon: int
    generated_at: datetime
    context_window: List[HistoricalPointDTO]
    predictions: List[ForecastPointDTO]
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PredictionMetadataDTO(BaseModel):
    """Optional metadata providing additional information about the prediction."""

    entity_type: str
    entity_id: str
    feature: str
    model_status: Optional[str] = None
    training_metrics: Optional[Dict[str, Any]] = None
    model_info: Dict[str, Any] = Field(default_factory=dict)


class PredictionToggleRequestDTO(BaseModel):
    """Payload used to enable or disable recurring predictions."""

    enabled: bool = Field(description="Flag indicating whether predictions should run")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional metadata to persist alongside the configuration",
    )


class PredictionToggleResponseDTO(BaseModel):
    """Response summarising the prediction toggle state."""

    model_id: UUID
    training_job_id: UUID
    enabled: bool
    entity_id: Optional[str]
    next_prediction_at: Optional[datetime]
    sampling_interval_seconds: Optional[int]


class PredictionHistoryRequestDTO(BaseModel):
    """Parameters for retrieving stored predictions from STH-Comet."""

    start: Optional[datetime] = Field(
        default=None, description="Start timestamp filter (inclusive)"
    )
    end: Optional[datetime] = Field(
        default=None, description="End timestamp filter (inclusive)"
    )
    limit: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Maximum number of prediction records to return",
    )


class PredictionHistoryPointDTO(BaseModel):
    """Represents a persisted prediction value."""

    timestamp: datetime
    value: float


class PredictionHistoryResponseDTO(BaseModel):
    """Response containing historical predictions for a training job."""

    entity_id: str
    feature: str
    points: List[PredictionHistoryPointDTO]
