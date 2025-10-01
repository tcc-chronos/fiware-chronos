from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterator, List, Sequence
from uuid import uuid4

import pytest

from src.domain.entities.model import (
    DenseLayerConfig,
    Model,
    ModelStatus,
    ModelType,
    RNNLayerConfig,
)
from src.domain.entities.training_job import (
    DataCollectionJob,
    DataCollectionStatus,
    TrainingJob,
    TrainingMetrics,
    TrainingStatus,
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture()
def sample_model() -> Model:
    return Model(
        id=uuid4(),
        name="Test Model",
        description="Unit test model",
        model_type=ModelType.LSTM,
        status=ModelStatus.DRAFT,
        batch_size=64,
        epochs=50,
        learning_rate=0.01,
        validation_ratio=0.1,
        test_ratio=0.1,
        lookback_window=24,
        forecast_horizon=6,
        feature="temperature",
        entity_type="Sensor",
        entity_id="urn:ngsi-ld:Sensor:001",
        rnn_layers=[RNNLayerConfig(units=128, dropout=0.2)],
        dense_layers=[DenseLayerConfig(units=64, dropout=0.1)],
        early_stopping_patience=5,
    )


@pytest.fixture()
def sample_training_metrics() -> TrainingMetrics:
    return TrainingMetrics(
        mse=0.1,
        mae=0.05,
        rmse=0.2,
        theil_u=0.01,
        mape=1.5,
        r2=0.9,
        mae_pct=0.5,
        rmse_pct=0.8,
        best_train_loss=0.2,
        best_val_loss=0.3,
        best_epoch=40,
    )


@pytest.fixture()
def sample_training_job(sample_model: Model) -> TrainingJob:
    job = TrainingJob()
    job.model_id = sample_model.id
    job.status = TrainingStatus.PENDING
    job.last_n = 500
    job.total_data_points_requested = 500
    job.total_data_points_collected = 0
    now = datetime.now(timezone.utc)
    job.created_at = now
    job.updated_at = now
    job.data_collection_jobs.append(
        DataCollectionJob(
            id=uuid4(),
            h_offset=0,
            last_n=250,
            status=DataCollectionStatus.PENDING,
        )
    )
    job.start_time = datetime.now(timezone.utc)
    return job


class FakeCursor:
    def __init__(self, documents: Sequence[Dict[str, Any]]):
        self._documents = list(documents)
        self._skip = 0
        self._limit: int | None = None

    def sort(self, *args: Any, **kwargs: Any) -> "FakeCursor":
        return self

    def skip(self, amount: int) -> "FakeCursor":
        self._skip = amount
        return self

    def limit(self, amount: int) -> "FakeCursor":
        self._limit = amount
        return self

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        docs = self._documents[self._skip :]
        if self._limit is not None:
            docs = docs[: self._limit]
        return iter(docs)


class FakeCollection:
    def __init__(self) -> None:
        self.documents: Dict[str, Dict[str, Any]] = {}
        self.inserts: List[Dict[str, Any]] = []
        self.last_query: Dict[str, Any] | None = None
        self.dropped_indexes: List[str] = []
        self.created_indexes: List[tuple[Any, ...]] = []

    def find_one(self, query: Dict[str, Any]) -> Dict[str, Any] | None:
        self.last_query = query
        key = query.get("id")
        if not isinstance(key, str):
            return None
        return self.documents.get(key)

    def find(self, query: Dict[str, Any]) -> FakeCursor:
        self.last_query = query
        results = [doc for doc in self.documents.values() if self._matches(doc, query)]
        return FakeCursor(results)

    def insert_one(self, document: Dict[str, Any]) -> Any:
        self.inserts.append(document)
        self.documents[document["id"]] = document
        return SimpleNamespace(acknowledged=True, inserted_id=document["id"])

    def replace_one(self, query: Dict[str, Any], document: Dict[str, Any]) -> Any:
        key = query.get("id")
        if not isinstance(key, str) or key not in self.documents:
            return SimpleNamespace(matched_count=0, acknowledged=False)
        self.documents[key] = document
        return SimpleNamespace(matched_count=1, acknowledged=True)

    def delete_one(self, query: Dict[str, Any]) -> Any:
        key = query.get("id")
        if isinstance(key, str) and key in self.documents:
            del self.documents[key]
            return SimpleNamespace(deleted_count=1, acknowledged=True)
        return SimpleNamespace(deleted_count=0, acknowledged=False)

    def drop_index(self, index_name: str) -> None:
        self.dropped_indexes.append(index_name)

    def create_index(self, keys: Any, name: str | None = None, **kwargs: Any) -> Any:
        self.created_indexes.append((keys, name, kwargs))
        return name or keys

    @staticmethod
    def _matches(document: Dict[str, Any], query: Dict[str, Any]) -> bool:
        for key, value in query.items():
            if document.get(key) != value:
                return False
        return True


class FakeMongoDatabase:
    def __init__(self) -> None:
        self.collections: Dict[str, FakeCollection] = {}

    def get_collection(self, name: str) -> FakeCollection:
        return self.collections.setdefault(name, FakeCollection())

    async def find_one(self, collection_name: str, query: Dict[str, Any]) -> Any:
        return self.get_collection(collection_name).find_one(query)

    async def find_many(
        self,
        collection_name: str,
        query: Dict[str, Any],
        sort_by: str | None = None,
        sort_direction: int = 1,
        skip: int = 0,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        cursor = self.get_collection(collection_name).find(query)
        cursor.sort(sort_by, sort_direction)
        cursor.skip(skip)
        cursor.limit(limit)
        return list(cursor)

    async def insert_one(self, collection_name: str, document: Dict[str, Any]) -> Any:
        result = self.get_collection(collection_name).insert_one(document)
        if not getattr(result, "acknowledged", True):
            raise Exception(f"Failed to insert document in {collection_name}")
        return document

    async def replace_one(
        self, collection_name: str, query: Dict[str, Any], document: Dict[str, Any]
    ) -> Any:
        result = self.get_collection(collection_name).replace_one(query, document)
        if getattr(result, "matched_count", 0) == 0:
            raise Exception(f"Document not found in {collection_name}")
        if not getattr(result, "acknowledged", True):
            raise Exception(f"Failed to replace document in {collection_name}")
        return document

    async def delete_one(self, collection_name: str, query: Dict[str, Any]) -> Any:
        result = self.get_collection(collection_name).delete_one(query)
        if getattr(result, "deleted_count", 0) == 0:
            raise Exception(f"Document not found in {collection_name}")
        if not getattr(result, "acknowledged", True):
            raise Exception(f"Failed to delete document in {collection_name}")
        return None

    async def create_indexes(self) -> None:  # pragma: no cover - stub for tests
        pass

    def close(self) -> None:
        pass


@pytest.fixture()
def fake_mongo_database() -> FakeMongoDatabase:
    return FakeMongoDatabase()


@pytest.fixture()
def dummy_now() -> datetime:
    return datetime.now(timezone.utc)
