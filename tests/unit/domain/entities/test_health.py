from __future__ import annotations

from datetime import timezone

from src.domain.entities.health import DependencyStatus, ServiceStatus, SystemHealth


def test_dependency_status_defaults() -> None:
    status = DependencyStatus(name="mongo", status=ServiceStatus.UP)
    assert status.checked_at.tzinfo == timezone.utc
    assert status.details == {}


def test_system_health_container() -> None:
    dependency = DependencyStatus(name="redis", status=ServiceStatus.DOWN)
    health = SystemHealth(status=ServiceStatus.DEGRADED, dependencies=[dependency])
    assert health.dependencies[0] is dependency
    assert health.status is ServiceStatus.DEGRADED
