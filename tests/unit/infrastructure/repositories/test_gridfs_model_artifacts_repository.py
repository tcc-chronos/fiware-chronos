from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, cast
from uuid import uuid4

import gridfs
import pytest
from bson import ObjectId
from pymongo import MongoClient

from src.infrastructure.repositories.gridfs_model_artifacts_repository import (
    GridFSModelArtifactsRepository,
)


class _GridOut:
    def __init__(self, _id: ObjectId, content: bytes, metadata: dict, filename: str):
        self._id = _id
        self._content = content
        self.metadata = metadata
        self.filename = filename
        self.uploadDate = datetime.now(timezone.utc)

    def read(self) -> bytes:
        return self._content


class _GridFS:
    def __init__(self) -> None:
        self.files: Dict[ObjectId, _GridOut] = {}

    def put(self, content: bytes, filename: str, metadata: dict) -> ObjectId:
        object_id = ObjectId()
        self.files[object_id] = _GridOut(object_id, content, metadata, filename)
        return object_id

    def find(self, query: dict):
        results = [f for f in self.files.values() if self._matches(f, query)]
        return _Cursor(results)

    def find_one(self, query: dict):
        for file in self.files.values():
            if self._matches(file, query):
                return file
        return None

    def get(self, object_id: ObjectId) -> _GridOut:
        if object_id not in self.files:
            raise gridfs.NoFile
        return self.files[object_id]

    def exists(self, object_id: ObjectId) -> bool:
        return object_id in self.files

    def delete(self, object_id: ObjectId) -> None:
        self.files.pop(object_id, None)

    @staticmethod
    def _matches(file: _GridOut, query: dict) -> bool:
        for key, value in query.items():
            current = file.metadata
            for part in key.split("."):
                if part == "metadata":
                    continue
                current = current.get(part)
                if current is None:
                    break
            if current != value:
                return False
        return True


class _Cursor:
    def __init__(self, items: List[_GridOut]):
        self.items = items
        self._index = 0

    def sort(self, *args, **kwargs):
        return self

    def limit(self, amount: int):  # pragma: no cover - not used
        return self

    def __iter__(self):
        self._index = 0
        return self

    def __next__(self):
        if self._index >= len(self.items):
            raise StopIteration
        item = self.items[self._index]
        self._index += 1
        return item


class _MongoClient:
    def __init__(self):
        self.db: Dict[str, Dict[str, _GridFS]] = {}

    def __getitem__(self, name: str):
        return self.db.setdefault(name, {})


@pytest.fixture(autouse=True)
def patch_gridfs(monkeypatch):
    fs = _GridFS()
    monkeypatch.setattr(
        "src.infrastructure.repositories."
        "gridfs_model_artifacts_repository.gridfs.GridFS",
        lambda db, collection: fs,
    )
    return fs


@pytest.fixture()
def repository(patch_gridfs):
    client = cast(MongoClient, _MongoClient())
    return GridFSModelArtifactsRepository(mongo_client=client, database_name="chronos")


@pytest.mark.asyncio
async def test_save_and_get_artifact(repository):
    model_id = uuid4()
    artifact_id = await repository.save_artifact(model_id, "model", b"data")

    artifact = await repository.get_artifact(model_id, "model")
    assert artifact is not None
    assert artifact.artifact_id == artifact_id
    assert artifact.content == b"data"


@pytest.mark.asyncio
async def test_get_artifact_by_id(repository):
    model_id = uuid4()
    artifact_id = await repository.save_artifact(model_id, "metadata", b"{}")

    artifact = await repository.get_artifact_by_id(artifact_id)
    assert artifact is not None
    assert artifact.artifact_type == "metadata"


@pytest.mark.asyncio
async def test_delete_artifact(repository):
    model_id = uuid4()
    artifact_id = await repository.save_artifact(model_id, "model", b"binary")

    deleted = await repository.delete_artifact(artifact_id)
    assert deleted is True


@pytest.mark.asyncio
async def test_list_model_artifacts(repository):
    model_id = uuid4()
    await repository.save_artifact(model_id, "model", b"binary")
    await repository.save_artifact(model_id, "x_scaler", b"x")

    listing = await repository.list_model_artifacts(model_id)
    assert set(listing.keys()) == {"model", "x_scaler"}


@pytest.mark.asyncio
async def test_delete_model_artifacts(repository):
    model_id = uuid4()
    await repository.save_artifact(model_id, "model", b"binary")
    await repository.save_artifact(model_id, "y_scaler", b"y")

    deleted = await repository.delete_model_artifacts(model_id)
    assert deleted == 2
