"""DTOs for system health and application info responses."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from src.domain.entities.health import (
    ApplicationInfo,
    DependencyStatus,
    ServiceStatus,
    SystemHealth,
)


class DependencyStatusDTO(BaseModel):
    """Serializable representation of a dependency health check."""

    name: str = Field(description="Dependency identifier")
    status: ServiceStatus = Field(description="Aggregated status for the dependency")
    message: Optional[str] = Field(
        default=None, description="Human readable status note"
    )
    checked_at: datetime = Field(description="Timestamp of the last check")
    latency_ms: Optional[float] = Field(
        default=None, description="Latency in milliseconds"
    )
    details: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metrics"
    )

    @classmethod
    def from_domain(cls, status: DependencyStatus) -> "DependencyStatusDTO":
        return cls(
            name=status.name,
            status=status.status,
            message=status.message,
            checked_at=status.checked_at,
            latency_ms=status.latency_ms,
            details=status.details,
        )

    model_config = {
        "json_schema_extra": {
            "example": {
                "name": "mongo",
                "status": "up",
                "message": "MongoDB ping successful",
                "checked_at": "2024-09-09T12:00:00Z",
                "latency_ms": 12.5,
                "details": {"database": "chronos_db"},
            }
        }
    }


class SystemHealthDTO(BaseModel):
    """DTO representing the /health response payload."""

    status: ServiceStatus = Field(description="Overall system status")
    dependencies: List[DependencyStatusDTO] = Field(
        default_factory=list, description="Detailed dependency information"
    )

    @classmethod
    def from_domain(cls, health: SystemHealth) -> "SystemHealthDTO":
        return cls(
            status=health.status,
            dependencies=[
                DependencyStatusDTO.from_domain(dep) for dep in health.dependencies
            ],
        )

    model_config = {
        "json_schema_extra": {
            "example": {
                "status": "up",
                "dependencies": [
                    {
                        "name": "mongo",
                        "status": "up",
                        "message": "MongoDB ping successful",
                        "checked_at": "2024-09-09T12:00:00Z",
                        "latency_ms": 12.5,
                        "details": {"database": "chronos_db"},
                    }
                ],
            }
        }
    }


class ApplicationInfoDTO(BaseModel):
    """DTO representing metadata returned by /info."""

    name: str = Field(description="Application name")
    description: str = Field(description="Application description")
    version: str = Field(description="Application version")
    environment: str = Field(description="Current deployment environment")
    git_commit: str = Field(description="Git commit hash")
    build_time: str = Field(description="Build timestamp")
    started_at: datetime = Field(description="Application start timestamp")
    uptime_seconds: float = Field(description="Uptime in seconds")
    status: ServiceStatus = Field(description="Overall system status")
    dependencies: List[DependencyStatusDTO] = Field(
        default_factory=list, description="Dependency status snapshot"
    )
    extras: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata and diagnostic information",
    )

    @classmethod
    def from_domain(cls, info: ApplicationInfo) -> "ApplicationInfoDTO":
        return cls(
            name=info.name,
            description=info.description,
            version=info.version,
            environment=info.environment,
            git_commit=info.git_commit,
            build_time=info.build_time,
            started_at=info.started_at,
            uptime_seconds=info.uptime_seconds,
            status=info.status,
            dependencies=[
                DependencyStatusDTO.from_domain(dep) for dep in info.dependencies
            ],
            extras=info.extras,
        )

    model_config = {
        "json_schema_extra": {
            "example": {
                "name": "Fiware Chronos GE",
                "description": "Generic Enabler for training models",
                "version": "1.0.0",
                "environment": "development",
                "git_commit": "abcdef1",
                "build_time": "2024-09-09T11:30:00Z",
                "started_at": "2024-09-09T12:00:00Z",
                "uptime_seconds": 3600.5,
                "status": "up",
                "dependencies": [
                    {
                        "name": "rabbitmq",
                        "status": "up",
                        "message": "RabbitMQ connection successful",
                        "checked_at": "2024-09-09T12:00:05Z",
                        "latency_ms": 18.3,
                        "details": {},
                    }
                ],
                "extras": {
                    "celery": {
                        "broker": "amqp://rabbitmq:5672/chronos",
                        "result_backend": "redis://redis:6379/0",
                    },
                    "fiware": {
                        "iot_agent_url": "http://localhost:4041",
                        "orion_url": "http://localhost:1026",
                        "sth_url": "http://localhost:8666",
                    },
                },
            }
        }
    }
