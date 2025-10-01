"""Lightweight settings structures consumed by the application layer."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SystemInfo:
    """Subset of configuration required by system-related use cases."""

    title: str
    description: str
    version: str
    environment: str
    git_commit: str
    build_time: str
    celery_broker_url: str
    celery_result_backend_url: str
    fiware_orion_url: str
    fiware_iot_agent_url: str
    fiware_sth_url: str
