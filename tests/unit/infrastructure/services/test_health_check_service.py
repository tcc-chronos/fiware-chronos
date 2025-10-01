from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock

import pytest

from src.domain.entities.health import DependencyStatus, ServiceStatus
from src.infrastructure.database.mongo_database import MongoDatabase
from src.infrastructure.services.health_check_service import HealthCheckService


@dataclass
class _StubMongoClient:
    class _Admin:
        @staticmethod
        def command(cmd: str) -> None:
            if cmd != "ping":
                raise ValueError("Unexpected command")

    @property
    def admin(self) -> "_StubMongoClient._Admin":
        return self._Admin()


@dataclass
class _StubMongoDatabase:
    name: str = "chronos"

    def __post_init__(self) -> None:
        self.client = _StubMongoClient()
        self.db = SimpleNamespace(name=self.name)

    async def create_indexes(self) -> None:  # pragma: no cover - not used here
        pass

    def close(self) -> None:
        pass


def _make_service() -> HealthCheckService:
    return HealthCheckService(
        mongo_database=cast(MongoDatabase, _StubMongoDatabase()),
        broker_url="amqp://guest@localhost/",
        redis_url="redis://localhost/0",
        orion_url="http://orion",
        iot_agent_url="http://iot",
        sth_comet_url="http://sth",
    )


def test_aggregate_status_priority() -> None:
    service = _make_service()
    statuses = [
        DependencyStatus(name="mongo", status=ServiceStatus.UP),
        DependencyStatus(name="redis", status=ServiceStatus.DEGRADED),
        DependencyStatus(name="broker", status=ServiceStatus.DOWN),
    ]
    assert service._aggregate_status(statuses) is ServiceStatus.DOWN


@pytest.mark.asyncio
async def test_normalize_url_adds_slash() -> None:
    service = _make_service()
    result = service._normalize_url("http://example.com", "/health")
    assert result == "http://example.com/health"


@pytest.mark.asyncio
async def test_evaluate_collects_dependency_statuses(monkeypatch) -> None:
    service = _make_service()

    mongo_status = DependencyStatus(name="mongo", status=ServiceStatus.UP)
    rabbit_status = DependencyStatus(name="rabbitmq", status=ServiceStatus.UNKNOWN)
    redis_status = DependencyStatus(name="redis", status=ServiceStatus.DEGRADED)
    http_statuses = [
        DependencyStatus(name="iot_agent", status=ServiceStatus.UP),
        DependencyStatus(name="orion", status=ServiceStatus.UP),
        DependencyStatus(name="sth_comet", status=ServiceStatus.DEGRADED),
    ]

    monkeypatch.setattr(service, "_check_mongo", AsyncMock(return_value=mongo_status))
    monkeypatch.setattr(
        service, "_check_rabbitmq", AsyncMock(return_value=rabbit_status)
    )
    monkeypatch.setattr(service, "_check_redis", AsyncMock(return_value=redis_status))
    monkeypatch.setattr(
        service,
        "_check_http_service",
        AsyncMock(side_effect=http_statuses),
    )

    health = await service.evaluate()

    assert len(health.dependencies) == 6
    assert health.status is ServiceStatus.DEGRADED


@pytest.mark.asyncio
async def test_check_mongo_handles_failure() -> None:
    class _FailingMongo:
        def __init__(self) -> None:
            self.client = SimpleNamespace(
                admin=SimpleNamespace(
                    command=lambda cmd: (_ for _ in ()).throw(
                        RuntimeError("mongo error")
                    )
                )
            )
            self.db = SimpleNamespace(name="chronos")

        async def create_indexes(self):  # pragma: no cover - unused
            return None

        def close(self):
            return None

    service = HealthCheckService(
        mongo_database=cast(MongoDatabase, _FailingMongo()),
        broker_url="",
        redis_url="",
        orion_url="",
        iot_agent_url="",
        sth_comet_url="",
    )

    status = await service._check_mongo()
    assert status.status is ServiceStatus.DOWN


@pytest.mark.asyncio
async def test_check_rabbitmq_success(monkeypatch) -> None:
    service = _make_service()

    class _Connection:
        def close(self) -> None:
            return None

    monkeypatch.setattr(
        "src.infrastructure.services.health_check_service.pika.BlockingConnection",
        lambda *args, **kwargs: _Connection(),
    )
    monkeypatch.setattr(
        "src.infrastructure.services.health_check_service.pika.URLParameters",
        lambda url: url,
    )

    status = await service._check_rabbitmq()
    assert status.status is ServiceStatus.UP


@pytest.mark.asyncio
async def test_check_rabbitmq_failure(monkeypatch) -> None:
    service = _make_service()

    def _raise(*args, **kwargs):
        raise RuntimeError("fail")

    monkeypatch.setattr(
        "src.infrastructure.services.health_check_service.pika.BlockingConnection",
        _raise,
    )

    status = await service._check_rabbitmq()
    assert status.status is ServiceStatus.DOWN


@pytest.mark.asyncio
async def test_check_redis_success(monkeypatch) -> None:
    service = _make_service()

    class _RedisClient:
        async def ping(self):
            return True

        async def close(self):
            return None

    monkeypatch.setattr(
        "src.infrastructure.services.health_check_service.aioredis.from_url",
        lambda *args, **kwargs: _RedisClient(),
    )

    status = await service._check_redis()
    assert status.status is ServiceStatus.UP


@pytest.mark.asyncio
async def test_check_http_service_attempts_multiple_paths(monkeypatch) -> None:
    service = _make_service()

    class _Response:
        def __init__(self, status_code: int):
            self.status_code = status_code

        def json(self):  # pragma: no cover - not used
            return {}

    responses = iter([_Response(500), _Response(200)])

    class _Client:
        def __init__(self):
            self.calls: list[int] = []

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def get(self, url, headers=None):
            self.calls.append(self.calls.__len__())
            return next(responses)

    monkeypatch.setattr("httpx.AsyncClient", lambda timeout: _Client())

    status = await service._check_http_service(
        name="iot", base_url="http://iot", paths=("/fail", "/ok")
    )
    assert status.status is ServiceStatus.UP
