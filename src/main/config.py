"""
Application Settings - Main Layer

Use Pydantic Settings for configuration management.
This module handles configuration settings provided using
environment variables, .env files and default values.
"""

from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.shared import EnumEnvironment, EnumLogLevel


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
    git_commit: str = Field(default="unknown", description="Git commit hash")
    build_time: str = Field(default="unknown", description="Build timestamp")
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
        default="amqp://chronos:chronos@localhost:5672/chronos",
        description="Message broker URL",
    )
    redis_url: str = Field(default="redis://localhost:6379/0", description="Redis URL")

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
