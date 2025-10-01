from __future__ import annotations

from typing import cast

import pytest

from src.application.dtos.model_dto import ModelCreateDTO, RNNLayerDTO
from src.application.use_cases.model_use_cases import (
    CreateModelUseCase,
    GetModelsUseCase,
)
from src.domain.entities.model import ModelType
from src.infrastructure.database.mongo_database import MongoDatabase
from src.infrastructure.repositories.model_repository import ModelRepository
from src.infrastructure.repositories.training_job_repository import (
    TrainingJobRepository,
)
from tests.conftest import FakeMongoDatabase


@pytest.mark.asyncio
async def test_create_and_list_model_integration():
    database = cast(MongoDatabase, FakeMongoDatabase())
    model_repo = ModelRepository(database)
    training_repo = TrainingJobRepository(database)

    create_use_case = CreateModelUseCase(
        model_repository=model_repo, training_job_repository=training_repo
    )
    list_use_case = GetModelsUseCase(
        model_repository=model_repo, training_job_repository=training_repo
    )

    dto = ModelCreateDTO(
        name="model-test",
        description="model-description",
        model_type=ModelType.LSTM,
        rnn_layers=[RNNLayerDTO(units=32, dropout=0.1, recurrent_dropout=0)],
        dense_layers=[],
        feature="humidity",
        entity_type="Sensor",
        entity_id="urn:ngsi-ld:Sensor:001",
    )

    created = await create_use_case.execute(dto)
    assert created.feature == "humidity"

    models = await list_use_case.execute()
    assert any(model.id == created.id for model in models)
