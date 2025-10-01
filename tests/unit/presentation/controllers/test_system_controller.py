from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from fastapi import Request

from src.application.models import SystemInfo
from src.application.use_cases.health_use_cases import (
    GetApplicationInfoUseCase,
    GetHealthStatusUseCase,
)
from src.domain.entities.health import DependencyStatus, ServiceStatus, SystemHealth
from src.presentation.controllers.system_controller import health, info


class _HealthService:
    def __init__(self, status: ServiceStatus):
        self._health = SystemHealth(
            status=status,
            dependencies=[DependencyStatus(name="mongo", status=status)],
        )

    async def evaluate(self) -> SystemHealth:
        return self._health


@pytest.mark.asyncio
async def test_health_endpoint_returns_status():
    dto = await health(
        get_health_status_use_case=GetHealthStatusUseCase(
            _HealthService(ServiceStatus.UP)
        )
    )
    assert dto.status is ServiceStatus.UP
    assert dto.dependencies[0].name == "mongo"


@pytest.mark.asyncio
async def test_info_endpoint_returns_application_info():
    health_service = _HealthService(ServiceStatus.UP)
    system_info = SystemInfo(
        title="Chronos",
        description="desc",
        version="1.0",
        environment="dev",
        git_commit="abc",
        build_time="now",
        celery_broker_url="amqp://",
        celery_result_backend_url="redis://",
        fiware_orion_url="http://orion",
        fiware_iot_agent_url="http://iot",
        fiware_sth_url="http://sth",
    )
    info_use_case = GetApplicationInfoUseCase(health_service, system_info)

    scope = {
        "type": "http",
        "method": "GET",
        "path": "/info",
        "headers": [],
        "query_string": b"",
        "server": ("test", 80),
        "app": SimpleNamespace(
            state=SimpleNamespace(started_at=datetime.now(timezone.utc))
        ),
    }
    request = Request(scope)

    dto = await info(request=request, get_application_info_use_case=info_use_case)
    assert dto.name == "Chronos"
    assert dto.status is ServiceStatus.UP
