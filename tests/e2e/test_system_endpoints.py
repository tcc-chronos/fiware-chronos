from __future__ import annotations

import pytest
from dependency_injector import providers
from fastapi.testclient import TestClient

from src.application.models import SystemInfo
from src.application.use_cases.health_use_cases import (
    GetApplicationInfoUseCase,
    GetHealthStatusUseCase,
)
from src.domain.entities.health import DependencyStatus, ServiceStatus, SystemHealth
from src.main.app import create_app
from src.main.container import get_container


class _StubMongo:
    async def create_indexes(self):
        return None

    def close(self):
        return None


class _HealthCheckService:
    def __init__(self, status: ServiceStatus):
        self._health = SystemHealth(
            status=status,
            dependencies=[DependencyStatus(name="mongo", status=status)],
        )

    async def evaluate(self) -> SystemHealth:
        return self._health


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setattr(
        "src.main.container.MongoDatabase", lambda *args, **kwargs: _StubMongo()
    )
    app = create_app()
    container = get_container()

    health_provider = _HealthCheckService(ServiceStatus.UP)
    health_use_case = GetHealthStatusUseCase(health_provider)
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

    container.get_health_status_use_case.override(providers.Object(health_use_case))
    container.get_application_info_use_case.override(
        providers.Factory(
            GetApplicationInfoUseCase,
            health_check_service=health_provider,
            system_info=system_info,
        )
    )

    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "up"


def test_info_endpoint(client):
    response = client.get("/info")
    assert response.status_code == 200
    body = response.json()
    assert body["name"] == "Chronos"
