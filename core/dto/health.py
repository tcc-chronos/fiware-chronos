from typing import Optional

from pydantic import BaseModel


class HealthResult(BaseModel):
    ok: bool
    error: Optional[str] = None


class HealthDependencies(BaseModel):
    mongo: HealthResult
    broker: HealthResult
    redis: HealthResult


class HealthResponse(BaseModel):
    status: str
    uptime_s: Optional[float]
    deps: HealthDependencies
    version: str
    timestamp: str
