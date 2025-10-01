from __future__ import annotations

from src.main.config import AppSettings, get_settings
from src.shared.consts import EnumEnvironment


def test_get_settings_loads_defaults(monkeypatch) -> None:
    monkeypatch.delenv("DB_MONGO_URI", raising=False)
    settings = get_settings()
    assert settings.database.mongo_uri.startswith("mongodb://")
    assert settings.environment == EnumEnvironment.DEVELOPMENT


def test_settings_respect_environment_variables(monkeypatch) -> None:
    monkeypatch.setenv("DB_MONGO_URI", "mongodb://test")
    monkeypatch.setenv("GE_TITLE", "Testing")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")

    settings = AppSettings()

    assert settings.database.mongo_uri == "mongodb://test"
    assert settings.ge.title == "Testing"
    assert settings.logging.level.value == "DEBUG"
