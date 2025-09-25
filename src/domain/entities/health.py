"""
Health domain entities.

This module defines value objects for representing service health
and application observability information across the system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


class ServiceStatus(str, Enum):
    """High-level availability for a dependency or the system."""

    UP = "up"
    DEGRADED = "degraded"
    DOWN = "down"
    UNKNOWN = "unknown"


@dataclass(slots=True)
class DependencyStatus:
    """Health status for a single external dependency."""

    name: str
    status: ServiceStatus
    message: Optional[str] = None
    checked_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    latency_ms: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SystemHealth:
    """Aggregated health for the application."""

    status: ServiceStatus
    dependencies: List[DependencyStatus] = field(default_factory=list)


@dataclass(slots=True)
class ApplicationInfo:
    """Operational metadata surfaced by the /info endpoint."""

    name: str
    description: str
    version: str
    environment: str
    git_commit: str
    build_time: str
    started_at: datetime
    uptime_seconds: float
    status: ServiceStatus
    dependencies: List[DependencyStatus] = field(default_factory=list)
    extras: Dict[str, Any] = field(default_factory=dict)
