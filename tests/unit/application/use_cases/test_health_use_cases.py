from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import pytest

from src.application.models import SystemInfo
from src.application.use_cases.health_use_cases import (
    GetApplicationInfoUseCase,
    GetHealthStatusUseCase,
)
from src.domain.entities.health import DependencyStatus, ServiceStatus, SystemHealth


@dataclass
class _StubHealthService:
    health: SystemHealth

    async def evaluate(self) -> SystemHealth:
        return self.health


@pytest.mark.asyncio
async def test_get_health_status_use_case_returns_dto() -> None:
    dependencies = [
        DependencyStatus(name="mongo", status=ServiceStatus.UP),
        DependencyStatus(name="redis", status=ServiceStatus.DOWN),
    ]
    health = SystemHealth(status=ServiceStatus.DOWN, dependencies=dependencies)

    use_case = GetHealthStatusUseCase(health_check_service=_StubHealthService(health))

    dto = await use_case.execute()

    assert dto.status is ServiceStatus.DOWN
    assert len(dto.dependencies) == 2
    assert dto.dependencies[1].status is ServiceStatus.DOWN


@pytest.mark.asyncio
async def test_get_application_info_use_case_sanitizes_urls() -> None:
    dependencies = [DependencyStatus(name="mongo", status=ServiceStatus.UP)]
    health = SystemHealth(status=ServiceStatus.UP, dependencies=dependencies)

    system_info = SystemInfo(
        title="Chronos",
        description="Chronos GE",
        version="1.2.3",
        environment="development",
        git_commit="abc1234",
        build_time="2024-09-09T10:00:00Z",
        celery_broker_url="amqp://chronos:secret@rabbitmq:5672/chronos",
        celery_result_backend_url="redis://:redispass@redis:6379/0",
        fiware_orion_url="http://orion:1026",
        fiware_iot_agent_url="http://iot-agent:4041",
        fiware_sth_url="http://sth:8666",
    )

    started_at = datetime.now(timezone.utc) - timedelta(seconds=120)

    use_case = GetApplicationInfoUseCase(
        health_check_service=_StubHealthService(health),
        system_info=system_info,
    )

    dto = await use_case.execute(started_at)

    assert dto.status is ServiceStatus.UP
    assert dto.name == "Chronos"
    assert abs(dto.uptime_seconds - 120) < 2
    assert dto.extras["celery"]["broker"] == "amqp://rabbitmq:5672/chronos"
    assert dto.extras["celery"]["result_backend"] == "redis://redis:6379/0"
    assert dto.extras["fiware"]["orion_url"] == "http://orion:1026"
    assert dto.dependencies[0].name == "mongo"
