from __future__ import annotations

from typing import Dict, Iterable, cast

import pytest

from src.infrastructure.database.mongo_database import MongoDatabase
from tests.conftest import FakeCollection


class _StubMongoClient:
    def __init__(self, uri: str) -> None:
        self.uri = uri
        self.databases: Dict[str, _StubDatabase] = {}

    def __getitem__(self, name: str) -> "_StubDatabase":
        return self.databases.setdefault(name, _StubDatabase())

    def close(self) -> None:
        pass


class _StubDatabase:
    def __init__(self) -> None:
        self.collections: Dict[str, FakeCollection] = {}
        self.created_indexes: list[tuple] = []

    def __getitem__(self, name: str) -> FakeCollection:
        return self.collections.setdefault(name, FakeCollection())

    def list_collection_names(self) -> Iterable[str]:  # pragma: no cover
        return self.collections.keys()

    @property
    def name(self) -> str:
        return "test_db"


@pytest.fixture(autouse=True)
def patch_mongo_client(monkeypatch) -> None:
    monkeypatch.setattr(
        "src.infrastructure.database.mongo_database.MongoClient",
        _StubMongoClient,
    )


@pytest.mark.asyncio
async def test_insert_and_find_document() -> None:
    database = MongoDatabase("mongodb://localhost:27017", "chronos")

    document = {"id": "123", "value": 42}
    await database.insert_one("models", document)

    result = await database.find_one("models", {"id": "123"})
    assert result == document


@pytest.mark.asyncio
async def test_replace_document_success() -> None:
    database = MongoDatabase("mongodb://localhost:27017", "chronos")
    await database.insert_one("models", {"id": "1", "value": 1})

    new_doc = {"id": "1", "value": 2}
    replaced = await database.replace_one("models", {"id": "1"}, new_doc)
    assert replaced["value"] == 2


@pytest.mark.asyncio
async def test_delete_document_success() -> None:
    database = MongoDatabase("mongodb://localhost:27017", "chronos")
    await database.insert_one("models", {"id": "1"})

    await database.delete_one("models", {"id": "1"})
    assert await database.find_one("models", {"id": "1"}) is None


@pytest.mark.asyncio
async def test_find_many_supports_filters() -> None:
    database = MongoDatabase("mongodb://localhost:27017", "chronos")
    await database.insert_one("models", {"id": "1", "status": "draft"})
    await database.insert_one("models", {"id": "2", "status": "trained"})

    results = await database.find_many("models", {"status": "draft"})
    assert len(results) == 1
    assert results[0]["id"] == "1"


@pytest.mark.asyncio
async def test_create_indexes_drops_and_creates() -> None:
    database = MongoDatabase("mongodb://localhost:27017", "chronos")
    collection = cast(FakeCollection, database.db["models"])

    await database.create_indexes()

    assert "created_at_idx" in {entry[1] for entry in collection.created_indexes}
