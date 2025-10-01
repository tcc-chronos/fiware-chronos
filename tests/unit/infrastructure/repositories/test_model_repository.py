from __future__ import annotations

from datetime import datetime, timezone
from typing import cast
from uuid import UUID, uuid4

import pytest

from src.domain.entities.errors import ModelNotFoundError
from src.domain.entities.model import ModelStatus
from src.infrastructure.database.mongo_database import MongoDatabase
from src.infrastructure.repositories.model_repository import ModelRepository
from tests.conftest import FakeMongoDatabase


@pytest.mark.asyncio
async def test_create_and_find_model(
    sample_model, fake_mongo_database: FakeMongoDatabase
) -> None:
    repository = ModelRepository(
        mongo_database=cast(MongoDatabase, fake_mongo_database)
    )

    created = await repository.create(sample_model)
    assert created.id == sample_model.id

    found = await repository.find_by_id(sample_model.id)
    assert found is not None
    assert found.id == sample_model.id


@pytest.mark.asyncio
async def test_find_all_applies_filters(
    sample_model, fake_mongo_database: FakeMongoDatabase
) -> None:
    repository = ModelRepository(
        mongo_database=cast(MongoDatabase, fake_mongo_database)
    )
    await repository.create(sample_model)

    results = await repository.find_all(model_type=sample_model.model_type.value)
    assert len(results) == 1

    results = await repository.find_all(model_type="gru")
    assert results == []


@pytest.mark.asyncio
async def test_to_entity_handles_legacy_fields(
    fake_mongo_database: FakeMongoDatabase,
) -> None:
    repository = ModelRepository(
        mongo_database=cast(MongoDatabase, fake_mongo_database)
    )
    document = {
        "id": str(uuid4()),
        "name": "Legacy",
        "description": "legacy",
        "model_type": "lstm",
        "status": "ready",
        "batch_size": 32,
        "epochs": 10,
        "learning_rate": 0.1,
        "validation_split": 0.2,
        "test_ratio": 0.1,
        "lookback_window": 5,
        "forecast_horizon": 1,
        "feature": "temp",
        "rnn_units": [64, 32],
        "rnn_dropout": 0.1,
        "dense_units": [16],
        "dense_dropout": 0.1,
        "entity_type": "Sensor",
        "entity_id": "urn:1",
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
        "has_successful_training": True,
    }

    fake_mongo_database.get_collection("models").documents[document["id"]] = document

    model = await repository.find_by_id(UUID(document["id"]))
    assert model is not None
    assert model.status is ModelStatus.TRAINED
    assert len(model.rnn_layers) == 2


@pytest.mark.asyncio
async def test_update_raises_when_missing(
    fake_mongo_database: FakeMongoDatabase, sample_model
) -> None:
    repository = ModelRepository(
        mongo_database=cast(MongoDatabase, fake_mongo_database)
    )
    with pytest.raises(ModelNotFoundError):
        await repository.update(sample_model)


@pytest.mark.asyncio
async def test_delete_raises_when_missing(
    fake_mongo_database: FakeMongoDatabase, sample_model
) -> None:
    repository = ModelRepository(
        mongo_database=cast(MongoDatabase, fake_mongo_database)
    )
    with pytest.raises(ModelNotFoundError):
        await repository.delete(sample_model.id)
