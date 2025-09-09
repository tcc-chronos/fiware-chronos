import sys
from datetime import datetime, timezone

from fastapi import APIRouter

from config.settings import settings
from core.dto.info import BuildResponse, InfoResponse, RuntimeResponse
from pkg.utils.versions import pkg_version

router = APIRouter()


@router.get("/info", response_model=InfoResponse)
def info():
    started_at_iso = None
    uptime_s = None
    try:
        from apps.api.main import app

        started = getattr(app.state, "started_at", None)
        if started:
            started_at_iso = started.isoformat()
            uptime_s = (datetime.now(timezone.utc) - started).total_seconds()
    except Exception:
        pass

    response = InfoResponse(
        name=settings.APP_NAME,
        version=settings.APP_VERSION,
        apiPort=settings.API_PORT,
        startedAt=started_at_iso,
        uptime_s=uptime_s,
        build=BuildResponse(
            gitCommit=settings.GIT_COMMIT,
            buildTime=settings.BUILD_TIME,
        ),
        runtime=RuntimeResponse(
            python=sys.version.split()[0],
            fastapi=pkg_version("fastapi"),
            uvicorn=pkg_version("uvicorn"),
        ),
    )
    return response
