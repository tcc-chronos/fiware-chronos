"""Domain entities for time-series / historic data."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(slots=True)
class HistoricDataPoint:
    """Represents a single historic data point collected from STH-Comet."""

    timestamp: datetime
    value: float
