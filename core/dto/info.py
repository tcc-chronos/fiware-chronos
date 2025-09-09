from typing import Optional

from pydantic import BaseModel


class RuntimeResponse(BaseModel):
    python: str
    fastapi: str
    uvicorn: str


class BuildResponse(BaseModel):
    gitCommit: Optional[str] = None
    buildTime: Optional[str] = None


class InfoResponse(BaseModel):
    name: str
    version: str
    apiPort: int
    startedAt: Optional[str] = None
    uptime_s: Optional[float] = None
    build: BuildResponse
    runtime: RuntimeResponse
