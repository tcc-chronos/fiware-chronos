"""
Dependency container injection module - Main Layer

This module implements the dependency injection container
to simplify the management and lifecycle of dependencies
in the application.
"""

import logging

from dependency_injector import containers, providers

from .config import AppSettings

# from contextlib import asynccontextmanager


logger = logging.getLogger(__name__)


class AppContainer(containers.DeclarativeContainer):
    """Composition Root using dependecy-injector."""

    wiring_config = containers.WiringConfiguration(
        packages=["..presentation", "..application"]
    )

    # Settings
    config = providers.Configuration()

    # Infrastructure

    # Application (use cases)

    # Presentation


# -------------------------
# Global Container Instance
# -------------------------
_app_container: AppContainer | None = None


def init_container(settings: AppSettings) -> AppContainer:
    """Initialize global container with application settings."""

    global _app_container

    container = AppContainer()
    container.config.from_pydantic(settings)

    _app_container = container
    return container


def get_container() -> AppContainer:
    """Get the initialized global container."""

    if _app_container is None:
        raise RuntimeError("Container not initialized. Call init_container() first.")
    return _app_container


# @asynccontextmanager
# async def app_lifespan():
#     """
#     Centralized lifecycle management for external resources.
#     """
#     container = get_container()

#     db_config: DatabaseConfig = container.database_config()
#     redis_client = container.redis_client()
#     broker_client = container.broker_client()

#     try:
#         # Startup
#         await db_config.connect()
#         await db_config.create_indexes()

#         if hasattr(redis_client, "connect"):
#             await redis_client.connect()

#         if hasattr(broker_client, "start"):
#             await broker_client.start()

#         logger.info("Application container initialized successfully")
#         yield container

#     finally:
#         # Shutdown
#         await db_config.disconnect()

#         if hasattr(redis_client, "disconnect"):
#             await redis_client.disconnect()

#         if hasattr(broker_client, "stop"):
#             await broker_client.stop()

#         container.shutdown_resources()
#         logger.info("Application container shutdown successfully")
