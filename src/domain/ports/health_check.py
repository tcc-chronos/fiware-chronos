"""Domain service abstraction for health checks."""

from __future__ import annotations

from typing import Protocol

from src.domain.entities.health import SystemHealth


class IHealthCheckService(Protocol):
    """Interface for retrieving system health information."""

    async def evaluate(self) -> SystemHealth:
        """Collect and aggregate health information for dependencies."""
        ...
