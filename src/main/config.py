"""
Application Settings - Main Layer

Use Pydantic Settings for configuration management.
This module handles configuration settings provided using
environment variables, .env files and default values.
"""

from typing import Optional

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.shared import EnumEnvironment, EnumLogLevel
from src.shared.env import load_secret_file_variables  # noqa: F401


class DatabaseSettings(BaseSettings):
    """Database configuration settings."""

    mongo_uri: str = Field(
        default="mongodb://localhost:27017/chronos_db",
        description="MongoDB connection URI",
    )
    database_name: str = Field(
        default="chronos_db", description="Name of the MongoDB database"
    )

    model_config = SettingsConfigDict(
        env_prefix="DB_", case_sensitive=False, extra="ignore"
    )


class GESettings(BaseSettings):
    """Generic Enabler configuration settings."""

    title: str = Field(default="Fiware Chronos GE", description="GE title")
    description: str = Field(
        default="Generic Enabler for training and deploying "
        "deep learning models with FIWARE",
        description="GE description",
    )
    version: str = Field(default="1.0.0", description="GE version")
    git_commit: str = Field(
        default="unknown",
        description="Git commit hash",
        validation_alias=AliasChoices("GE_GIT_COMMIT", "GIT_COMMIT"),
    )
    build_time: str = Field(
        default="unknown",
        description="Build timestamp",
        validation_alias=AliasChoices("GE_BUILD_TIME", "BUILD_TIME"),
    )
    debug: bool = Field(default=False, description="Enable debug mode")
    port: int = Field(default=8000, description="Port to bind the server")
    reload: bool = Field(
        default=False, description="Enable auto-reload for development"
    )

    model_config = SettingsConfigDict(
        env_prefix="GE_", case_sensitive=False, extra="ignore"
    )


class CelerySettings(BaseSettings):
    """Celery configuration settings."""

    broker_url: str = Field(
        default="amqp://chronos:chronos@rabbitmq:5672/chronos",
        description="Message broker URL",
        alias="CELERY_BROKER_URL",
    )
    result_backend_url: str = Field(
        default="redis://redis:6379/0",
        description="Result backend URL",
        alias="CELERY_RESULT_BACKEND",
    )

    model_config = SettingsConfigDict(
        env_prefix="CELERY_", case_sensitive=False, extra="ignore"
    )


class FiwareSettings(BaseSettings):
    """Fiware configuration settings."""

    orion_url: str = Field(
        default="http://localhost:1026", description="Orion Context Broker URL"
    )
    sth_url: str = Field(default="http://localhost:8666", description="STH Comet URL")
    iot_agent_url: str = Field(
        default="http://localhost:4041", description="IoT Agent URL"
    )
    max_per_request: int = Field(
        default=100,
        description="Maximum records per STH-Comet request",
        alias="STH_MAX_PER_REQUEST",
    )
    service: str = Field(
        default="smart",
        description="Default FIWARE service header value",
    )
    service_path: str = Field(
        default="/",
        description="Default FIWARE service path header",
    )
    forecast_service_group: str = Field(
        default="Forecast",
        description="Dedicated IoT Agent service group for forecast entities",
    )
    forecast_service_apikey: str = Field(
        default="CHRONOS_FORECAST",
        description="API key used for the forecast service group",
    )
    forecast_service_resource: str = Field(
        default="/forecast",
        description="Resource path associated with the forecast service group",
    )
    forecast_entity_type: str = Field(
        default="Prediction",
        description="Default entity type used for forecast entities in Orion",
    )
    forecast_device_transport: str = Field(
        default="MQTT",
        description="Transport protocol used when registering forecast devices",
    )
    forecast_device_protocol: str = Field(
        default="PDI-IoTA-UltraLight",
        description="IoT Agent protocol used when registering forecast devices",
    )

    model_config = SettingsConfigDict(
        env_prefix="FIWARE_", case_sensitive=False, extra="ignore"
    )


class LoggingSettings(BaseSettings):
    """Logging configuration settings."""

    level: EnumLogLevel = Field(default=EnumLogLevel.INFO, description="Logging level")
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Logging format",
    )
    file_path: Optional[str] = Field(
        default=None, description="Log file path (if None, logs to console)"
    )

    model_config = SettingsConfigDict(
        env_prefix="LOG_", case_sensitive=False, extra="ignore"
    )


class AppSettings(BaseSettings):
    """Main application settings, aggregating all sub-settings."""

    environment: EnumEnvironment = Field(
        default=EnumEnvironment.DEVELOPMENT, description="Application environment"
    )

    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    ge: GESettings = Field(default_factory=GESettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    celery: CelerySettings = Field(default_factory=CelerySettings)
    fiware: FiwareSettings = Field(default_factory=FiwareSettings)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        env_nested_delimiter="__",
        extra="ignore",
    )


def get_settings() -> AppSettings:
    """
    Get application settings instance Factory.

    Used to be mocked in tests, allowing different settings based on enviroment.
    """
    return AppSettings()


settings = get_settings()
