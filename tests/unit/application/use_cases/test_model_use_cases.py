from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional
from uuid import UUID, uuid4

import pytest

from src.application.dtos.model_dto import ModelCreateDTO, ModelUpdateDTO
from src.application.use_cases.model_use_cases import (
    CreateModelUseCase,
    DeleteModelUseCase,
    GetModelByIdUseCase,
    GetModelsUseCase,
    GetModelTypesUseCase,
    UpdateModelUseCase,
)
from src.domain.entities.model import (
    DenseLayerConfig,
    Model,
    ModelStatus,
    ModelType,
    RNNLayerConfig,
)
from src.domain.entities.training_job import (
    TrainingJob,
    TrainingMetrics,
    TrainingStatus,
)
from src.domain.repositories.model_artifacts_repository import IModelArtifactsRepository
from src.domain.repositories.model_repository import IModelRepository
from src.domain.repositories.training_job_repository import ITrainingJobRepository


@dataclass
class _InMemoryModelRepository(IModelRepository):
    items: Dict[UUID, Model] = field(default_factory=dict)

    async def find_by_id(self, model_id: UUID) -> Optional[Model]:
        return self.items.get(model_id)

    async def find_all(
        self,
        skip: int = 0,
        limit: int = 100,
        model_type: Optional[str] = None,
        status: Optional[str] = None,
        entity_id: Optional[str] = None,
        feature: Optional[str] = None,
    ) -> List[Model]:
        models = list(self.items.values())
        filtered: List[Model] = []
        for model in models[skip : skip + limit]:
            if model_type and model.model_type.value != model_type:
                continue
            if status and model.status.value != status:
                continue
            if entity_id and model.entity_id != entity_id:
                continue
            if feature and model.feature != feature:
                continue
            filtered.append(model)
        return filtered

    async def create(self, model: Model) -> Model:
        self.items[model.id] = model
        return model

    async def update(self, model: Model) -> Model:
        if model.id not in self.items:
            raise Exception("Document not found in models")
        self.items[model.id] = model
        return model

    async def delete(self, model_id: UUID) -> None:
        if model_id not in self.items:
            raise Exception("Document not found in models")
        del self.items[model_id]


@dataclass
class _InMemoryTrainingJobRepository(ITrainingJobRepository):
    jobs: Dict[UUID, List[TrainingJob]] = field(
        default_factory=lambda: defaultdict(list)
    )

    async def create(self, training_job: TrainingJob) -> TrainingJob:
        model_id = training_job.model_id
        if model_id is None:
            raise ValueError("training_job.model_id must be set")
        self.jobs.setdefault(model_id, []).append(training_job)
        return training_job

    async def get_by_model_id(self, model_id: UUID) -> List[TrainingJob]:
        return list(self.jobs.get(model_id, []))

    async def get_by_id(self, training_job_id: UUID) -> Optional[TrainingJob]:
        for job_list in self.jobs.values():
            for job in job_list:
                if job.id == training_job_id:
                    return job
        return None

    async def update(self, training_job: TrainingJob) -> TrainingJob:
        return training_job

    async def delete(self, training_job_id: UUID) -> bool:
        for model_id, job_list in list(self.jobs.items()):
            self.jobs[model_id] = [job for job in job_list if job.id != training_job_id]
        return True

    async def add_data_collection_job(
        self, *args, **kwargs
    ):  # pragma: no cover - not exercised
        return True

    async def update_data_collection_job_status(
        self, *args, **kwargs
    ):  # pragma: no cover - not exercised
        return True

    async def update_training_job_status(
        self, *args, **kwargs
    ):  # pragma: no cover - not exercised
        return True

    async def complete_training_job(
        self, *args, **kwargs
    ):  # pragma: no cover - not exercised
        return True

    async def update_task_refs(
        self, *args, **kwargs
    ):  # pragma: no cover - not exercised
        return True

    async def fail_training_job(
        self, *args, **kwargs
    ):  # pragma: no cover - not exercised
        return True


class _FakeArtifactsRepository(IModelArtifactsRepository):
    def __init__(self) -> None:
        self.deleted: List[UUID] = []

    async def save_artifact(
        self,
        model_id: UUID,
        artifact_type: str,
        content: bytes,
        metadata: Optional[Dict[str, str]] = None,
        filename: Optional[str] = None,
    ) -> str:  # pragma: no cover
        return "artifact-id"

    async def get_artifact(
        self, model_id: UUID, artifact_type: str
    ):  # pragma: no cover
        return None

    async def get_artifact_by_id(self, artifact_id: str):  # pragma: no cover
        return None

    async def delete_artifact(self, artifact_id: str) -> bool:  # pragma: no cover
        return True

    async def delete_model_artifacts(self, model_id: UUID) -> int:
        self.deleted.append(model_id)
        return 0

    async def list_model_artifacts(self, model_id: UUID):  # pragma: no cover
        return {}


def _make_model(name: str = "Model") -> Model:
    return Model(
        id=uuid4(),
        name=name,
        description=f"{name} description",
        model_type=ModelType.LSTM,
        status=ModelStatus.DRAFT,
        batch_size=32,
        epochs=50,
        learning_rate=0.01,
        validation_ratio=0.1,
        test_ratio=0.1,
        lookback_window=24,
        forecast_horizon=12,
        feature="temperature",
        entity_type="Sensor",
        entity_id="urn:ngsi-ld:Sensor:001",
        rnn_layers=[RNNLayerConfig(units=64, dropout=0.1)],
        dense_layers=[DenseLayerConfig(units=32, dropout=0.1)],
        early_stopping_patience=5,
    )


def _make_training_job(model_id: UUID) -> TrainingJob:
    now = datetime.now(timezone.utc)
    return TrainingJob(
        id=uuid4(),
        model_id=model_id,
        status=TrainingStatus.COMPLETED,
        total_data_points_requested=100,
        total_data_points_collected=100,
        created_at=now,
        updated_at=now,
        start_time=now,
        end_time=now,
        metrics=TrainingMetrics(mae=0.1, mse=0.2),
    )


@pytest.fixture()
def model_repo() -> _InMemoryModelRepository:
    repo = _InMemoryModelRepository()
    model = _make_model()
    repo.items[model.id] = model
    return repo


@pytest.fixture()
def training_repo(
    model_repo: _InMemoryModelRepository,
) -> _InMemoryTrainingJobRepository:
    repo = _InMemoryTrainingJobRepository()
    model_id = next(iter(model_repo.items.keys()))
    repo.jobs.setdefault(model_id, []).append(_make_training_job(model_id))
    return repo


@pytest.mark.asyncio
async def test_get_models_use_case_returns_training_summaries(
    model_repo: _InMemoryModelRepository, training_repo: _InMemoryTrainingJobRepository
) -> None:
    use_case = GetModelsUseCase(
        model_repository=model_repo,
        training_job_repository=training_repo,
    )
    models = await use_case.execute()
    assert len(models) == 1
    assert models[0].trainings
    if models[0].trainings[0].metrics:
        assert models[0].trainings[0].metrics.mae == pytest.approx(0.1)


@pytest.mark.asyncio
async def test_get_model_by_id_use_case_returns_details(
    model_repo: _InMemoryModelRepository, training_repo: _InMemoryTrainingJobRepository
) -> None:
    model_id = next(iter(model_repo.items.keys()))
    use_case = GetModelByIdUseCase(
        model_repository=model_repo,
        training_job_repository=training_repo,
    )
    model = await use_case.execute(model_id=model_id)
    assert model.id == model_id
    assert model.trainings[0].status is TrainingStatus.COMPLETED


@pytest.mark.asyncio
async def test_create_model_use_case_generates_defaults(
    model_repo: _InMemoryModelRepository,
    training_repo: _InMemoryTrainingJobRepository,
) -> None:
    use_case = CreateModelUseCase(
        model_repository=model_repo,
        training_job_repository=training_repo,
    )
    dto = ModelCreateDTO.model_validate(
        {
            "model_type": ModelType.GRU,
            "rnn_layers": [{"units": 128, "dropout": 0.1, "recurrent_dropout": 0.0}],
            "dense_layers": [],
            "feature": "humidity",
            "entity_type": "Sensor",
            "entity_id": "urn:ngsi-ld:Sensor:002",
        }
    )
    result = await use_case.execute(dto)
    assert result.name == "GRU - humidity"
    assert result.description is not None and result.description.startswith("GRU model")
    assert model_repo.items[result.id].feature == "humidity"


@pytest.mark.asyncio
async def test_update_model_use_case_overrides_defaults(
    model_repo: _InMemoryModelRepository,
    training_repo: _InMemoryTrainingJobRepository,
) -> None:
    model_id = next(iter(model_repo.items.keys()))
    repo_model = model_repo.items[model_id]
    repo_model.name = "LSTM - temperature"
    repo_model.description = "lstm model for temperature forecasting"

    use_case = UpdateModelUseCase(
        model_repository=model_repo,
        training_job_repository=training_repo,
    )
    dto = ModelUpdateDTO.model_validate(
        {"feature": "humidity", "model_type": ModelType.GRU}
    )
    updated = await use_case.execute(model_id=model_id, model_dto=dto)
    assert updated.feature == "humidity"
    assert updated.name == "GRU - humidity"


@pytest.mark.asyncio
async def test_delete_model_use_case_removes_dependencies(
    model_repo: _InMemoryModelRepository,
    training_repo: _InMemoryTrainingJobRepository,
) -> None:
    model_id = next(iter(model_repo.items.keys()))
    artifacts = _FakeArtifactsRepository()
    use_case = DeleteModelUseCase(
        model_repository=model_repo,
        training_job_repository=training_repo,
        artifacts_repository=artifacts,
    )

    await use_case.execute(model_id=model_id)

    assert model_id not in model_repo.items
    assert artifacts.deleted == [model_id]


@pytest.mark.asyncio
async def test_delete_model_use_case_raises_for_missing_model(
    model_repo: _InMemoryModelRepository,
    training_repo: _InMemoryTrainingJobRepository,
) -> None:
    artifacts = _FakeArtifactsRepository()
    use_case = DeleteModelUseCase(
        model_repository=model_repo,
        training_job_repository=training_repo,
        artifacts_repository=artifacts,
    )

    with pytest.raises(Exception):
        await use_case.execute(model_id=uuid4())


@pytest.mark.asyncio
async def test_get_model_types_use_case_lists_enum() -> None:
    use_case = GetModelTypesUseCase()
    result = await use_case.execute()
    labels = {item.label for item in result}
    assert {"LSTM", "GRU"} <= labels
