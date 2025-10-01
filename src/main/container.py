"""
Dependency container injection module - Main Layer

This module implements the dependency injection container
to simplify the management and lifecycle of dependencies
in the application.
"""

from contextlib import asynccontextmanager

from dependency_injector import containers, providers

from src.application.models import SystemInfo
from src.application.use_cases.device_use_cases import GetDevicesUseCase
from src.application.use_cases.health_use_cases import (
    GetApplicationInfoUseCase,
    GetHealthStatusUseCase,
)
from src.application.use_cases.model_prediction_use_case import ModelPredictionUseCase
from src.application.use_cases.model_training_use_case import ModelTrainingUseCase
from src.application.use_cases.model_use_cases import (
    CreateModelUseCase,
    DeleteModelUseCase,
    GetModelByIdUseCase,
    GetModelsUseCase,
    GetModelTypesUseCase,
    UpdateModelUseCase,
)
from src.application.use_cases.training_management_use_case import (
    TrainingManagementUseCase,
)
from src.infrastructure.database import MongoDatabase
from src.infrastructure.gateways.iot_agent_gateway import IoTAgentGateway
from src.infrastructure.gateways.sth_comet_gateway import STHCometGateway
from src.infrastructure.repositories.gridfs_model_artifacts_repository import (
    GridFSModelArtifactsRepository,
)
from src.infrastructure.repositories.model_repository import ModelRepository
from src.infrastructure.repositories.training_job_repository import (
    TrainingJobRepository,
)
from src.infrastructure.services.health_check_service import HealthCheckService
from src.infrastructure.services.training_orchestrator import CeleryTrainingOrchestrator
from src.shared import get_logger

from .config import AppSettings

logger = get_logger(__name__)


class AppContainer(containers.DeclarativeContainer):
    """Composition Root using dependecy-injector."""

    wiring_config = containers.WiringConfiguration(
        packages=["..presentation", "..application"]
    )

    # Settings
    config = providers.Configuration()
    # Infrastructure
    mongo_database = providers.Singleton(
        MongoDatabase,
        mongo_uri=config.database.mongo_uri,
        db_name=config.database.database_name,
    )

    model_repository = providers.Singleton(
        ModelRepository,
        mongo_database=mongo_database,
    )

    training_orchestrator = providers.Singleton(
        CeleryTrainingOrchestrator,
    )

    training_job_repository = providers.Singleton(
        TrainingJobRepository,
        database=mongo_database,
    )

    # GridFS repository for model artifacts
    model_artifacts_repository = providers.Singleton(
        GridFSModelArtifactsRepository,
        mongo_client=providers.Callable(lambda db: db.client, mongo_database),
        database_name=config.database.database_name,
    )

    # Gateways
    iot_agent_gateway = providers.Singleton(
        IoTAgentGateway,
        iot_agent_url=config.fiware.iot_agent_url,
    )

    sth_comet_gateway = providers.Singleton(
        STHCometGateway,
        base_url=config.fiware.sth_url,
    )

    # Application (use cases)
    get_models_use_case = providers.Factory(
        GetModelsUseCase,
        model_repository=model_repository,
        training_job_repository=training_job_repository,
    )

    get_model_types_use_case = providers.Factory(GetModelTypesUseCase)

    get_model_by_id_use_case = providers.Factory(
        GetModelByIdUseCase,
        model_repository=model_repository,
        training_job_repository=training_job_repository,
    )

    create_model_use_case = providers.Factory(
        CreateModelUseCase,
        model_repository=model_repository,
    )

    update_model_use_case = providers.Factory(
        UpdateModelUseCase,
        model_repository=model_repository,
        training_job_repository=training_job_repository,
    )

    delete_model_use_case = providers.Factory(
        DeleteModelUseCase,
        model_repository=model_repository,
        training_job_repository=training_job_repository,
        artifacts_repository=model_artifacts_repository,
    )

    get_devices_use_case = providers.Factory(
        GetDevicesUseCase,
        iot_agent_gateway=iot_agent_gateway,
    )

    health_check_service = providers.Singleton(
        HealthCheckService,
        mongo_database=mongo_database,
        broker_url=config.celery.broker_url,
        redis_url=config.celery.result_backend_url,
        orion_url=config.fiware.orion_url,
        iot_agent_url=config.fiware.iot_agent_url,
        sth_comet_url=config.fiware.sth_url,
    )

    system_info = providers.Singleton(
        SystemInfo,
        title=config.ge.title,
        description=config.ge.description,
        version=config.ge.version,
        environment=providers.Callable(
            lambda env: env.value if hasattr(env, "value") else str(env),
            config.environment,
        ),
        git_commit=config.ge.git_commit,
        build_time=config.ge.build_time,
        celery_broker_url=config.celery.broker_url,
        celery_result_backend_url=config.celery.result_backend_url,
        fiware_orion_url=config.fiware.orion_url,
        fiware_iot_agent_url=config.fiware.iot_agent_url,
        fiware_sth_url=config.fiware.sth_url,
    )

    get_health_status_use_case = providers.Factory(
        GetHealthStatusUseCase,
        health_check_service=health_check_service,
    )

    get_application_info_use_case = providers.Factory(
        GetApplicationInfoUseCase,
        health_check_service=health_check_service,
        system_info=system_info,
    )

    training_management_use_case = providers.Factory(
        TrainingManagementUseCase,
        training_job_repository=training_job_repository,
        model_repository=model_repository,
        artifacts_repository=model_artifacts_repository,
        sth_gateway=sth_comet_gateway,
        training_orchestrator=training_orchestrator,
        fiware_service=providers.Object("smart"),
        fiware_service_path=providers.Object("/"),
    )

    model_training_use_case = providers.Factory(
        ModelTrainingUseCase,
        artifacts_repository=model_artifacts_repository,
    )

    model_prediction_use_case = providers.Factory(
        ModelPredictionUseCase,
        model_repository=model_repository,
        training_job_repository=training_job_repository,
        artifacts_repository=model_artifacts_repository,
        sth_gateway=sth_comet_gateway,
        iot_agent_gateway=iot_agent_gateway,
        fiware_service=providers.Object("smart"),
        fiware_service_path=providers.Object("/"),
    )

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
        raise RuntimeError("Container has not been initialized yet")

    return _app_container


@asynccontextmanager
async def app_lifespan():
    """
    Centralized lifecycle management for external resources.

    This async context manager can be used in the FastAPI lifespan
    to properly initialize and clean up resources. It follows
    the Clean Architecture principles by managing infrastructure
    components through the DI container.
    """
    container = get_container()

    # Ensure these resources exist in the container
    mongo_database = container.mongo_database()

    # Redis and message broker - for future use
    # redis_client = None
    # if hasattr(container, "redis_client"):
    #     redis_client = container.redis_client()
    #
    # broker_client = None
    # if hasattr(container, "broker_client"):
    #     broker_client = container.broker_client()

    try:
        # Startup - MongoDB is already connected in __init__
        logger.info("container.mongo.ensure_connection")
        # Create necessary indexes
        await mongo_database.create_indexes()

        # Future Redis/Broker initialization
        # if redis_client and hasattr(redis_client, "connect"):
        #     await redis_client.connect()
        # if broker_client and hasattr(broker_client, "start"):
        #     await broker_client.start()

        logger.info("container.resources.initialized")
        yield container

    finally:
        # Shutdown - Clean up resources
        logger.info("container.mongo.close")
        mongo_database.close()

        # Future Redis/Broker cleanup
        # if redis_client and hasattr(redis_client, "disconnect"):
        #     await redis_client.disconnect()
        # if broker_client and hasattr(broker_client, "stop"):
        #     await broker_client.stop()

        logger.info("container.resources.shutdown")
