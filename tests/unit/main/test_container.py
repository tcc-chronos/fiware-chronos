from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest
from dependency_injector import providers

from src.main.config import AppSettings
from src.main.container import app_lifespan, get_container, init_container


@dataclass
class _StubMongoDatabase:
    ensured_indexes: bool = False
    closed: bool = False

    async def create_indexes(self) -> None:
        self.ensured_indexes = True

    def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_init_and_get_container(monkeypatch) -> None:
    settings = AppSettings()
    container = init_container(settings)
    assert hasattr(container, "mongo_database")
    assert get_container() is container

    stub_db = _StubMongoDatabase()
    container.mongo_database.override(providers.Object(stub_db))

    async with app_lifespan():
        pass


@pytest.mark.asyncio
async def test_app_lifespan_manages_resources(monkeypatch) -> None:
    container = init_container(AppSettings())
    stub_db = _StubMongoDatabase()
    container.mongo_database.override(providers.Object(stub_db))

    async with app_lifespan():
        await asyncio.sleep(0)

    assert stub_db.ensured_indexes is True
    assert stub_db.closed is True


def test_get_container_without_init_raises(monkeypatch) -> None:
    monkeypatch.setattr("src.main.container._app_container", None)
    with pytest.raises(RuntimeError):
        get_container()
