"""Domain entities for Orion prediction publishing."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List


@dataclass(slots=True)
class ForecastSeriesPoint:
    """Represents one point in the forecast horizon."""

    step: int
    value: float
    target_timestamp: datetime


@dataclass(slots=True)
class PredictionRecord:
    """Represents a prediction ready to be persisted in Orion."""

    entity_id: str
    entity_type: str
    source_entity: str
    model_id: str
    training_id: str
    horizon: int
    feature: str
    generated_at: datetime
    series: List[ForecastSeriesPoint] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
