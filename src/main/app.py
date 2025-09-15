"""
Main Application - Main Layer

This module serves as the entry point for the FastAPI application.
It initializes the container, creates the FastAPI app, and includes
the API routers.
"""

from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.main.config import get_settings
from src.main.container import app_lifespan, init_container
from src.presentation.controllers import models_router
from src.shared import configure_logging, get_logger, update_logging_from_settings

# Configure logging with basic settings first - before configuration is loaded
# This ensures we have logging during the configuration loading process
configure_logging()

# Load settings
settings = get_settings()

# Update logging with complete settings
update_logging_from_settings(settings)

# Get structured logger
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan management.

    This context manager is called when the application starts up,
    and when it shuts down. It uses the container's app_lifespan
    to properly manage application resources.
    """
    # Set application startup time
    app.state.started_at = datetime.now(timezone.utc)
    logger.info("Application starting up")

    # Use container's lifecycle management
    async with app_lifespan() as container:
        app.state.container = container
        yield

    logger.info("Application shutting down")


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    Returns:
        FastAPI: The configured FastAPI application
    """
    settings = get_settings()

    # Initialize dependency injection container
    init_container(settings)

    # Create FastAPI app
    app = FastAPI(
        title=settings.ge.title,
        description=settings.ge.description,
        version=settings.ge.version,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # Set up CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(models_router)

    return app


app = create_app()
