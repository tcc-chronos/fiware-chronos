"""Infrastructure implementation for system health checks."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from time import perf_counter
from typing import Iterable, List
from urllib.parse import urljoin

import httpx
import pika
import redis.asyncio as aioredis

from src.domain.entities.health import DependencyStatus, ServiceStatus, SystemHealth
from src.domain.ports.health_check import IHealthCheckService
from src.infrastructure.database.mongo_database import MongoDatabase


class HealthCheckService(IHealthCheckService):
    """Collect health information for external dependencies."""

    def __init__(
        self,
        mongo_database: MongoDatabase,
        broker_url: str,
        redis_url: str,
        orion_url: str,
        iot_agent_url: str,
        sth_comet_url: str,
        *,
        http_timeout: float = 5.0,
        socket_timeout: float = 5.0,
    ) -> None:
        self._mongo_database = mongo_database
        self._broker_url = broker_url
        self._redis_url = redis_url
        self._orion_url = orion_url
        self._iot_agent_url = iot_agent_url
        self._sth_comet_url = sth_comet_url
        self._http_timeout = http_timeout
        self._socket_timeout = socket_timeout

    async def evaluate(self) -> SystemHealth:
        """Run checks concurrently and aggregate system health."""

        checks = {
            "mongo": asyncio.create_task(self._check_mongo()),
            "rabbitmq": asyncio.create_task(self._check_rabbitmq()),
            "redis": asyncio.create_task(self._check_redis()),
            "iot_agent": asyncio.create_task(
                self._check_http_service(
                    name="iot_agent",
                    base_url=self._iot_agent_url,
                    paths=("/iot/about", "/iot", "/"),
                )
            ),
            "orion": asyncio.create_task(
                self._check_http_service(
                    name="orion",
                    base_url=self._orion_url,
                    paths=("/version", "/ngsi-ld/ex/v1/version", "/"),
                )
            ),
            "sth_comet": asyncio.create_task(
                self._check_http_service(
                    name="sth_comet",
                    base_url=self._sth_comet_url,
                    paths=("/version", "/version", "/STH/v1", "/"),
                )
            ),
        }

        dependency_statuses: List[DependencyStatus] = []

        for name, task in checks.items():
            try:
                dependency_statuses.append(await task)
            except Exception as exc:  # pragma: no cover - defensive fallback
                dependency_statuses.append(
                    DependencyStatus(
                        name=name,
                        status=ServiceStatus.DOWN,
                        message=str(exc),
                    )
                )

        overall_status = self._aggregate_status(dependency_statuses)
        return SystemHealth(status=overall_status, dependencies=dependency_statuses)

    def _aggregate_status(self, statuses: Iterable[DependencyStatus]) -> ServiceStatus:
        has_unknown = False
        has_degraded = False

        for status in statuses:
            if status.status == ServiceStatus.DOWN:
                return ServiceStatus.DOWN
            if status.status == ServiceStatus.DEGRADED:
                has_degraded = True
            if status.status == ServiceStatus.UNKNOWN:
                has_unknown = True

        if has_degraded:
            return ServiceStatus.DEGRADED
        if has_unknown:
            return ServiceStatus.UNKNOWN
        return ServiceStatus.UP

    async def _check_mongo(self) -> DependencyStatus:
        if not self._mongo_database:
            return DependencyStatus(
                name="mongo",
                status=ServiceStatus.UNKNOWN,
                message="Mongo database client not configured.",
            )

        start = perf_counter()
        try:
            await asyncio.to_thread(self._mongo_database.client.admin.command, "ping")
            latency_ms = (perf_counter() - start) * 1000
            return DependencyStatus(
                name="mongo",
                status=ServiceStatus.UP,
                message="MongoDB ping successful",
                latency_ms=latency_ms,
                details={"database": self._mongo_database.db.name},
            )
        except Exception as exc:
            latency_ms = (perf_counter() - start) * 1000
            return DependencyStatus(
                name="mongo",
                status=ServiceStatus.DOWN,
                message=f"MongoDB ping failed: {exc}",
                latency_ms=latency_ms,
            )

    async def _check_rabbitmq(self) -> DependencyStatus:
        if not self._broker_url:
            return DependencyStatus(
                name="rabbitmq",
                status=ServiceStatus.UNKNOWN,
                message="RabbitMQ broker URL not configured.",
            )

        start = perf_counter()

        def _ping() -> None:
            connection = pika.BlockingConnection(pika.URLParameters(self._broker_url))
            connection.close()

        try:
            await asyncio.to_thread(_ping)
            latency_ms = (perf_counter() - start) * 1000
            return DependencyStatus(
                name="rabbitmq",
                status=ServiceStatus.UP,
                message="RabbitMQ connection successful",
                latency_ms=latency_ms,
            )
        except Exception as exc:
            latency_ms = (perf_counter() - start) * 1000
            return DependencyStatus(
                name="rabbitmq",
                status=ServiceStatus.DOWN,
                message=f"RabbitMQ connection failed: {exc}",
                latency_ms=latency_ms,
            )

    async def _check_redis(self) -> DependencyStatus:
        if not self._redis_url:
            return DependencyStatus(
                name="redis",
                status=ServiceStatus.UNKNOWN,
                message="Redis URL not configured.",
            )

        start = perf_counter()
        client = aioredis.from_url(
            self._redis_url,
            socket_connect_timeout=self._socket_timeout,
            socket_timeout=self._socket_timeout,
        )
        try:
            await client.ping()
            latency_ms = (perf_counter() - start) * 1000
            return DependencyStatus(
                name="redis",
                status=ServiceStatus.UP,
                message="Redis ping successful",
                latency_ms=latency_ms,
            )
        except Exception as exc:
            latency_ms = (perf_counter() - start) * 1000
            return DependencyStatus(
                name="redis",
                status=ServiceStatus.DOWN,
                message=f"Redis ping failed: {exc}",
                latency_ms=latency_ms,
            )
        finally:
            await client.close()

    async def _check_http_service(
        self,
        *,
        name: str,
        base_url: str,
        paths: Iterable[str],
    ) -> DependencyStatus:
        if not base_url:
            return DependencyStatus(
                name=name,
                status=ServiceStatus.UNKNOWN,
                message="Service URL not configured.",
            )

        attempts_log: List[dict] = []
        last_result: DependencyStatus | None = None

        for path in paths:
            result = await self._hit_http_endpoint(
                name=name, base_url=base_url, path=path
            )
            attempts_log.append(
                {
                    "path": path,
                    "status": result.status.value,
                    "message": result.message,
                    "checked_at": datetime.now(timezone.utc).isoformat(),
                }
            )

            if result.status != ServiceStatus.DOWN:
                result.details.setdefault("attempts", attempts_log)
                return result

            last_result = result

        if last_result is None:
            return DependencyStatus(
                name=name,
                status=ServiceStatus.UNKNOWN,
                message="Unable to evaluate service health",
            )

        last_result.details.setdefault("attempts", attempts_log)
        return last_result

    async def _hit_http_endpoint(
        self,
        *,
        name: str,
        base_url: str,
        path: str,
    ) -> DependencyStatus:
        url = self._normalize_url(base_url, path)
        start = perf_counter()

        try:
            async with httpx.AsyncClient(timeout=self._http_timeout) as client:
                response = await client.get(url)

            latency_ms = (perf_counter() - start) * 1000
            status_code = response.status_code

            if status_code >= 500:
                status = ServiceStatus.DOWN
            elif status_code >= 400:
                status = ServiceStatus.DEGRADED
            else:
                status = ServiceStatus.UP

            return DependencyStatus(
                name=name,
                status=status,
                message=f"HTTP {status_code}",
                latency_ms=latency_ms,
                details={"url": url, "status_code": status_code},
            )

        except httpx.RequestError as exc:
            latency_ms = (perf_counter() - start) * 1000
            return DependencyStatus(
                name=name,
                status=ServiceStatus.DOWN,
                message=f"HTTP request failed: {exc}",
                latency_ms=latency_ms,
                details={"url": url},
            )

    def _normalize_url(self, base_url: str, path: str) -> str:
        if not path:
            return base_url
        base = base_url if base_url.endswith("/") else f"{base_url}/"
        relative = path.lstrip("/")
        return urljoin(base, relative)
