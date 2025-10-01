from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, cast
from uuid import uuid4

import pytest

from src.domain.entities.training_job import (
    DataCollectionJob,
    TrainingJob,
    TrainingStatus,
)
from src.infrastructure.database.mongo_database import MongoDatabase
from src.infrastructure.repositories.training_job_repository import (
    TrainingJobRepository,
)


def _collection(repository: TrainingJobRepository) -> _Collection:
    return cast(_Database, repository.database).collection


class _Collection:
    def __init__(self) -> None:
        self.documents: Dict[str, dict] = {}
        self.inserted: List[dict] = []

    def insert_one(self, document: dict):
        key = document.get("id")
        if not isinstance(key, str):
            raise ValueError("document must contain string 'id'")
        self.documents[key] = document
        self.inserted.append(document)

        return type("Result", (), {"inserted_id": key})

    def find_one(self, query: dict):
        key = query.get("id")
        if not isinstance(key, str):
            return None
        return self.documents.get(key)

    def find(self, query: dict):
        results = [
            doc
            for doc in self.documents.values()
            if doc.get("model_id") == query.get("model_id")
        ]

        class Cursor:
            def __init__(self, docs: list[dict]):
                self.docs = docs

            def sort(self, *args, **kwargs):
                return self

            def __iter__(self):
                return iter(self.docs)

        return Cursor(results)

    def replace_one(self, query: dict, document: dict):
        key = query.get("id")
        if isinstance(key, str) and key in self.documents:
            self.documents[key] = document
            return type("Result", (), {"modified_count": 1})

        return type("Result", (), {"modified_count": 0})

    def delete_one(self, query: dict):
        key = query.get("id")
        if isinstance(key, str) and key in self.documents:
            del self.documents[key]
            deleted = True
        else:
            deleted = False
        return type("Result", (), {"deleted_count": 1 if deleted else 0})

    def update_one(self, query: dict, update: dict):
        key = query.get("id")
        if not isinstance(key, str):
            return type("Result", (), {"modified_count": 0})

        doc = self.documents.get(key)
        if not doc:
            return type("Result", (), {"modified_count": 0})

        if "$set" in update:
            for field, value in update["$set"].items():
                doc[field] = value

        return type("Result", (), {"modified_count": 1})


class _Database:
    def __init__(self) -> None:
        self.collection = _Collection()

    def get_collection(self, name: str) -> _Collection:
        return self.collection


@pytest.fixture()
def repository() -> TrainingJobRepository:
    db = _Database()
    return TrainingJobRepository(database=cast(MongoDatabase, db))


@pytest.mark.asyncio
async def test_create_and_get_training_job(repository: TrainingJobRepository) -> None:
    job = TrainingJob(
        id=uuid4(),
        model_id=uuid4(),
        status=TrainingStatus.PENDING,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )

    created = await repository.create(job)
    assert created.id == job.id

    fetched = await repository.get_by_id(job.id)
    assert fetched is not None
    assert fetched.id == job.id


@pytest.mark.asyncio
async def test_update_training_job_status(repository: TrainingJobRepository) -> None:
    job_id = uuid4()
    collection = _collection(repository)
    collection.documents[str(job_id)] = {
        "id": str(job_id),
        "model_id": str(uuid4()),
        "status": TrainingStatus.PENDING.value,
        "last_n": 50,
        "total_data_points_requested": 0,
        "total_data_points_collected": 0,
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
        "data_collection_jobs": [],
        "task_refs": {},
    }

    updated = await repository.update_training_job_status(
        job_id, TrainingStatus.COMPLETED
    )
    assert updated is True
    assert collection.documents[str(job_id)]["status"] == TrainingStatus.COMPLETED.value


@pytest.mark.asyncio
async def test_add_data_collection_job(repository: TrainingJobRepository) -> None:
    job_id = uuid4()
    collection = _collection(repository)
    collection.documents[str(job_id)] = {
        "id": str(job_id),
        "model_id": str(uuid4()),
        "status": TrainingStatus.PENDING.value,
        "last_n": 20,
        "total_data_points_requested": 0,
        "total_data_points_collected": 0,
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
        "data_collection_jobs": [],
        "task_refs": {"data_collection_task_ids": []},
    }

    job = DataCollectionJob(h_offset=0, last_n=10)
    added = await repository.add_data_collection_job(job_id, job)
    assert added is True


@pytest.mark.asyncio
async def test_fail_training_job_sets_error(repository: TrainingJobRepository) -> None:
    job_id = uuid4()
    collection = _collection(repository)
    collection.documents[str(job_id)] = {
        "id": str(job_id),
        "model_id": str(uuid4()),
        "status": TrainingStatus.PENDING.value,
        "last_n": 10,
        "total_data_points_requested": 0,
        "total_data_points_collected": 0,
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
        "data_collection_jobs": [],
        "task_refs": {},
    }

    failed = await repository.fail_training_job(job_id, "boom")
    assert failed is True
    doc = collection.documents[str(job_id)]
    assert doc["status"] == TrainingStatus.FAILED.value
    assert doc["error"] == "boom"
