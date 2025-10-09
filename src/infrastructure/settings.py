"""Infrastructure-level configuration helpers for background workers."""

from __future__ import annotations

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class _DatabaseSettings(BaseSettings):
    mongo_uri: str = Field(
        default="mongodb://localhost:27017/chronos_db",
        validation_alias=AliasChoices("DB_MONGO_URI", "MONGO_URI"),
        description="MongoDB connection URI",
    )
    database_name: str = Field(
        default="chronos_db",
        validation_alias=AliasChoices("DB_DATABASE_NAME", "DATABASE_NAME"),
        description="MongoDB database name",
    )

    model_config = SettingsConfigDict(case_sensitive=False, extra="ignore")


class _FiwareSettings(BaseSettings):
    sth_url: str = Field(
        default="http://localhost:8666",
        validation_alias=AliasChoices("FIWARE_STH_URL", "STH_URL"),
        description="STH-Comet base URL",
    )
    max_per_request: int = Field(
        default=100,
        validation_alias=AliasChoices(
            "FIWARE_STH_MAX_PER_REQUEST", "STH_MAX_PER_REQUEST"
        ),
        description="Maximum records per STH-Comet request",
    )
    service: str = Field(
        default="smart",
        validation_alias=AliasChoices("FIWARE_SERVICE", "STH_SERVICE"),
        description="Default FIWARE service header",
    )
    service_path: str = Field(
        default="/",
        validation_alias=AliasChoices("FIWARE_SERVICE_PATH", "STH_SERVICE_PATH"),
        description="Default FIWARE service path header",
    )
    orion_url: str = Field(
        default="http://localhost:1026",
        validation_alias=AliasChoices("FIWARE_ORION_URL", "ORION_URL"),
        description="Orion Context Broker URL",
    )
    iot_agent_url: str = Field(
        default="http://localhost:4041",
        validation_alias=AliasChoices("FIWARE_IOT_AGENT_URL", "IOT_AGENT_URL"),
        description="IoT Agent URL",
    )
    forecast_service_group: str = Field(
        default="Forecast",
        validation_alias=AliasChoices(
            "FIWARE_FORECAST_SERVICE_GROUP", "FORECAST_SERVICE_GROUP"
        ),
        description="Service group reserved for predictions",
    )
    forecast_service_apikey: str = Field(
        default="CHRONOS_FORECAST",
        validation_alias=AliasChoices(
            "FIWARE_FORECAST_SERVICE_APIKEY", "FORECAST_SERVICE_APIKEY"
        ),
        description="API key for the forecast service group",
    )
    forecast_service_resource: str = Field(
        default="/forecast",
        validation_alias=AliasChoices(
            "FIWARE_FORECAST_SERVICE_RESOURCE", "FORECAST_SERVICE_RESOURCE"
        ),
        description="Resource path for the forecast service group",
    )
    forecast_entity_type: str = Field(
        default="Prediction",
        validation_alias=AliasChoices(
            "FIWARE_FORECAST_ENTITY_TYPE", "FORECAST_ENTITY_TYPE"
        ),
        description="Default Orion entity type for predictions",
    )

    model_config = SettingsConfigDict(case_sensitive=False, extra="ignore")


class InfrastructureSettings(BaseSettings):
    database: _DatabaseSettings = Field(default_factory=_DatabaseSettings)
    fiware: _FiwareSettings = Field(default_factory=_FiwareSettings)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore",
    )


_settings: InfrastructureSettings | None = None


def get_settings() -> InfrastructureSettings:
    """Lazy-load infrastructure settings for Celery workers."""
    global _settings
    if _settings is None:
        _settings = InfrastructureSettings()
    return _settings
