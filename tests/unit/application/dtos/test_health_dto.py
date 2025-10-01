from __future__ import annotations

from datetime import datetime, timezone

from src.application.dtos.health_dto import (
    ApplicationInfoDTO,
    DependencyStatusDTO,
    SystemHealthDTO,
)
from src.domain.entities.health import (
    ApplicationInfo,
    DependencyStatus,
    ServiceStatus,
    SystemHealth,
)


def test_dependency_status_dto_from_domain() -> None:
    domain = DependencyStatus(name="mongo", status=ServiceStatus.UP)
    dto = DependencyStatusDTO.from_domain(domain)
    assert dto.name == "mongo"
    assert dto.status is ServiceStatus.UP


def test_system_health_dto_from_domain() -> None:
    domain = SystemHealth(status=ServiceStatus.UP, dependencies=[])
    dto = SystemHealthDTO.from_domain(domain)
    assert dto.status is ServiceStatus.UP
    assert dto.dependencies == []


def test_application_info_dto_from_domain() -> None:
    now = datetime.now(timezone.utc)
    info = ApplicationInfo(
        name="Chronos",
        description="desc",
        version="1.0",
        environment="development",
        git_commit="abc",
        build_time="2024-09-01",
        started_at=now,
        uptime_seconds=42.0,
        status=ServiceStatus.UP,
        dependencies=[DependencyStatus(name="mongo", status=ServiceStatus.UP)],
        extras={"foo": "bar"},
    )

    dto = ApplicationInfoDTO.from_domain(info)
    assert dto.name == "Chronos"
    assert dto.status is ServiceStatus.UP
    assert dto.extras == {"foo": "bar"}
