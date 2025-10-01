"""Use cases for health and application info endpoints."""

from datetime import datetime, timezone
from typing import Optional
from urllib.parse import urlsplit, urlunsplit

from src.application.dtos.health_dto import ApplicationInfoDTO, SystemHealthDTO
from src.application.models import SystemInfo
from src.domain.entities.health import ApplicationInfo
from src.domain.ports.health_check import IHealthCheckService


class GetHealthStatusUseCase:
    """Use case responsible for returning health status."""

    def __init__(self, health_check_service: IHealthCheckService) -> None:
        self._health_check_service = health_check_service

    async def execute(self) -> SystemHealthDTO:
        system_health = await self._health_check_service.evaluate()
        return SystemHealthDTO.from_domain(system_health)


class GetApplicationInfoUseCase:
    """Use case responsible for returning application info."""

    def __init__(
        self,
        health_check_service: IHealthCheckService,
        system_info: SystemInfo,
    ) -> None:
        self._health_check_service = health_check_service
        self._info = system_info

    async def execute(self, started_at: Optional[datetime]) -> ApplicationInfoDTO:
        system_health = await self._health_check_service.evaluate()

        now = datetime.now(timezone.utc)
        started = started_at or now
        uptime_seconds = max(0.0, (now - started).total_seconds())

        extras = {
            "environment": self._info.environment,
            "celery": {
                "broker": self._redact_url(self._info.celery_broker_url),
                "result_backend": self._redact_url(
                    self._info.celery_result_backend_url
                ),
            },
            "fiware": {
                "orion_url": self._info.fiware_orion_url,
                "iot_agent_url": self._info.fiware_iot_agent_url,
                "sth_url": self._info.fiware_sth_url,
            },
        }

        info = ApplicationInfo(
            name=self._info.title,
            description=self._info.description,
            version=self._info.version,
            environment=self._info.environment,
            git_commit=self._info.git_commit,
            build_time=self._info.build_time,
            started_at=started,
            uptime_seconds=uptime_seconds,
            status=system_health.status,
            dependencies=system_health.dependencies,
            extras=extras,
        )

        return ApplicationInfoDTO.from_domain(info)

    def _redact_url(self, url: str) -> str:
        if not url:
            return url

        parsed = urlsplit(url)
        if parsed.username or parsed.password:
            hostname = parsed.hostname or ""
            port_part = f":{parsed.port}" if parsed.port else ""
            netloc = f"{hostname}{port_part}"
            return urlunsplit(
                (parsed.scheme, netloc, parsed.path, parsed.query, parsed.fragment)
            )

        return url
