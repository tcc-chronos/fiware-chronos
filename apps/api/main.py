from datetime import datetime, timezone

from fastapi import FastAPI

from apps.api.routers import health, info
from config.logging import setup_logging
from config.settings import settings


def create_app() -> FastAPI:
    setup_logging()

    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    app.state.started_at = datetime.now(timezone.utc)

    app.include_router(health.router, tags=["meta"])
    app.include_router(info.router, tags=["meta"])

    return app


app = create_app()
