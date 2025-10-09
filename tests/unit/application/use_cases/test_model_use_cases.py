from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

import pytest

from src.application.dtos.model_dto import (
    DenseLayerDTO,
    ModelCreateDTO,
    ModelUpdateDTO,
    RNNLayerDTO,
)
from src.application.use_cases.model_use_cases import (
    CreateModelUseCase,
    DeleteModelUseCase,
    GetModelByIdUseCase,
    GetModelsUseCase,
    GetModelTypesUseCase,
    UpdateModelUseCase,
    _to_dense_layer_config,
    _to_dense_layer_dto,
    _to_rnn_layer_config,
    _to_rnn_layer_dto,
)
from src.domain.entities.errors import ModelNotFoundError, ModelOperationError
from src.domain.entities.iot import (
    DeviceAttribute,
    IoTDevice,
    IoTDeviceCollection,
    IoTServiceGroup,
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
from src.domain.gateways.iot_agent_gateway import IIoTAgentGateway
from src.domain.gateways.orion_gateway import IOrionGateway
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
    prediction_schedule: Dict[UUID, datetime] = field(default_factory=dict)

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

    async def update_sampling_metadata(
        self,
        training_job_id: UUID,
        *,
        sampling_interval_seconds: Optional[int],
        next_prediction_at: Optional[datetime] = None,
    ) -> bool:
        job = await self.get_by_id(training_job_id)
        if not job:
            return False
        job.sampling_interval_seconds = sampling_interval_seconds
        job.next_prediction_at = next_prediction_at
        if next_prediction_at:
            self.prediction_schedule[training_job_id] = next_prediction_at
        return True

    async def claim_prediction_schedule(
        self,
        training_job_id: UUID,
        *,
        expected_next_prediction_at: Optional[datetime],
        next_prediction_at: datetime,
    ) -> bool:
        current = self.prediction_schedule.get(training_job_id)
        if expected_next_prediction_at and current != expected_next_prediction_at:
            return False
        job = await self.get_by_id(training_job_id)
        if not job:
            return False
        job.next_prediction_at = next_prediction_at
        self.prediction_schedule[training_job_id] = next_prediction_at
        return True

    async def update_prediction_schedule(
        self,
        training_job_id: UUID,
        *,
        next_prediction_at: datetime,
    ) -> bool:
        job = await self.get_by_id(training_job_id)
        if not job:
            return False
        job.next_prediction_at = next_prediction_at
        self.prediction_schedule[training_job_id] = next_prediction_at
        return True

    async def get_prediction_ready_jobs(
        self,
        *,
        reference_time: datetime,
        limit: int = 50,
    ) -> List[TrainingJob]:
        ready: List[TrainingJob] = []
        for model_jobs in self.jobs.values():
            for job in model_jobs:
                if (
                    job.next_prediction_at is not None
                    and job.next_prediction_at <= reference_time
                ):
                    ready.append(job)
        ready.sort(key=lambda job: job.next_prediction_at or datetime.max)
        return ready[:limit]


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


def test_layer_conversion_helpers() -> None:
    dto = RNNLayerDTO(units=128, dropout=0.2, recurrent_dropout=0.1)
    config = _to_rnn_layer_config(dto)
    round_trip = _to_rnn_layer_dto(config)
    assert round_trip.units == dto.units

    dense_dto = DenseLayerDTO(units=64, dropout=0.3, activation="relu")
    dense_config = _to_dense_layer_config(dense_dto)
    dense_round_trip = _to_dense_layer_dto(dense_config)
    assert dense_round_trip.activation == "relu"


class _IoTGatewayStub(IIoTAgentGateway):
    def __init__(self) -> None:
        self.deleted_devices: List[str] = []

    async def get_devices(
        self, service: str = "smart", service_path: str = "/"
    ) -> IoTDeviceCollection:  # pragma: no cover - not used
        return IoTDeviceCollection(count=0, devices=[])

    async def get_service_groups(
        self, service: str = "smart", service_path: str = "/"
    ) -> List[IoTServiceGroup]:  # pragma: no cover - not used
        return []

    async def ensure_service_group(
        self,
        *,
        service: str,
        service_path: str,
        apikey: str,
        entity_type: str,
        resource: str,
        cbroker: str,
    ) -> IoTServiceGroup:  # pragma: no cover - not used
        return IoTServiceGroup(
            apikey=apikey,
            cbroker=cbroker,
            entity_type=entity_type,
            resource=resource,
            service=service,
            service_path=service_path,
        )

    async def ensure_device(
        self,
        *,
        device_id: str,
        entity_name: str,
        entity_type: str,
        attributes: List[DeviceAttribute],
        transport: str,
        protocol: str,
        service: str,
        service_path: str,
    ) -> IoTDevice:  # pragma: no cover - not used
        return IoTDevice(
            device_id=device_id,
            service=service,
            service_path=service_path,
            entity_name=entity_name,
            entity_type=entity_type,
            transport=transport,
            protocol=protocol,
            attributes=attributes,
        )

    async def delete_device(
        self,
        device_id: str,
        *,
        service: str,
        service_path: str,
    ) -> None:
        self.deleted_devices.append(device_id)


class _OrionGatewayStub(IOrionGateway):
    def __init__(self, *, raise_on_delete: bool = False) -> None:
        self.deleted_entities: List[str] = []
        self.raise_on_delete = raise_on_delete

    async def ensure_entity(
        self,
        *,
        entity_id: str,
        entity_type: str,
        payload: Dict[str, Any],
        service: str,
        service_path: str,
    ) -> None:  # pragma: no cover - not used
        return None

    async def upsert_prediction(
        self,
        prediction,
        *,
        service: str,
        service_path: str,
    ) -> None:  # pragma: no cover - not used
        return None

    async def create_subscription(
        self,
        *,
        entity_id: str,
        entity_type: str,
        attrs: List[str],
        notification_url: str,
        service: str,
        service_path: str,
        attrs_format: str = "legacy",
    ) -> str:  # pragma: no cover - not used
        return "subscription-id"

    async def delete_subscription(
        self,
        subscription_id: str,
        *,
        service: str,
        service_path: str,
    ) -> None:  # pragma: no cover - not used
        return None

    async def delete_entity(
        self,
        entity_id: str,
        *,
        service: str,
        service_path: str,
    ) -> None:
        if self.raise_on_delete:
            raise RuntimeError("boom")
        self.deleted_entities.append(entity_id)


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
async def test_create_model_use_case_handles_missing_name_description(
    model_repo: _InMemoryModelRepository,
    training_repo: _InMemoryTrainingJobRepository,
) -> None:
    use_case = CreateModelUseCase(
        model_repository=model_repo,
        training_job_repository=training_repo,
    )
    dto = ModelCreateDTO.model_validate(
        {
            "model_type": ModelType.LSTM,
            "rnn_layers": [{"units": 32}],
            "dense_layers": [],
            "feature": "pressure",
            "entity_type": "Sensor",
            "entity_id": "urn:ngsi-ld:Sensor:003",
        }
    )
    dto.name = None
    dto.description = None

    result = await use_case.execute(dto)

    assert result.name.startswith("LSTM")
    assert "pressure" in (result.description or "")


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


@pytest.mark.asyncio
async def test_update_model_use_case_updates_all_fields(
    model_repo: _InMemoryModelRepository,
    training_repo: _InMemoryTrainingJobRepository,
) -> None:
    model_id = next(iter(model_repo.items.keys()))
    repo_model = model_repo.items[model_id]
    repo_model.status = ModelStatus.DRAFT

    use_case = UpdateModelUseCase(
        model_repository=model_repo,
        training_job_repository=training_repo,
    )
    dto = ModelUpdateDTO.model_validate(
        {
            "name": "Custom",
            "description": "Custom desc",
            "batch_size": 16,
            "epochs": 10,
            "learning_rate": 0.02,
            "validation_ratio": 0.2,
            "test_ratio": 0.1,
            "lookback_window": 12,
            "forecast_horizon": 3,
            "feature": "humidity",
            "rnn_layers": [{"units": 128}],
            "dense_layers": [{"units": 64}],
            "early_stopping_patience": 7,
            "entity_type": "NewSensor",
            "entity_id": "urn:ngsi-ld:Sensor:999",
        }
    )

    updated = await use_case.execute(model_id=model_id, model_dto=dto)

    assert updated.name == "Custom"
    assert updated.batch_size == 16
    assert updated.entity_type == "NewSensor"


@pytest.mark.asyncio
async def test_update_model_use_case_raises_for_unknown_model(
    training_repo: _InMemoryTrainingJobRepository,
) -> None:
    use_case = UpdateModelUseCase(
        model_repository=_InMemoryModelRepository(),
        training_job_repository=training_repo,
    )
    dto = ModelUpdateDTO.model_validate({"name": "new"})

    with pytest.raises(ModelNotFoundError):
        await use_case.execute(model_id=uuid4(), model_dto=dto)


@pytest.mark.asyncio
async def test_delete_model_use_case_handles_iot_errors(
    model_repo: _InMemoryModelRepository,
    training_repo: _InMemoryTrainingJobRepository,
) -> None:
    model_id = next(iter(model_repo.items.keys()))
    job = training_repo.jobs[model_id][0]
    job.prediction_config.metadata = {"device_id": "device-1"}

    class _FailingIoT(_IoTGatewayStub):
        async def delete_device(
            self, device_id: str, *, service: str, service_path: str
        ) -> None:
            raise RuntimeError("failure")

    use_case = DeleteModelUseCase(
        model_repository=model_repo,
        training_job_repository=training_repo,
        artifacts_repository=_FakeArtifactsRepository(),
        iot_agent_gateway=_FailingIoT(),
        orion_gateway=_OrionGatewayStub(),
        fiware_service="smart",
        fiware_service_path="/",
    )

    with pytest.raises(ModelOperationError):
        await use_case.execute(model_id=model_id)


@pytest.mark.asyncio
async def test_update_model_use_case_requires_draft_status(
    model_repo: _InMemoryModelRepository,
    training_repo: _InMemoryTrainingJobRepository,
) -> None:
    model_id = next(iter(model_repo.items.keys()))
    model_repo.items[model_id].status = ModelStatus.TRAINED
    use_case = UpdateModelUseCase(
        model_repository=model_repo,
        training_job_repository=training_repo,
    )
    dto = ModelUpdateDTO.model_validate({"name": "Should fail"})

    with pytest.raises(ModelOperationError):
        await use_case.execute(model_id=model_id, model_dto=dto)


@pytest.mark.asyncio
async def test_delete_model_use_case_cleans_prediction_resources(
    model_repo: _InMemoryModelRepository,
    training_repo: _InMemoryTrainingJobRepository,
) -> None:
    model_id = next(iter(model_repo.items.keys()))
    job = training_repo.jobs[model_id][0]
    job.prediction_config.entity_id = "urn:prediction:entity"
    job.prediction_config.metadata = {"device_id": "device-1"}
    artifacts = _FakeArtifactsRepository()
    iot_gateway = _IoTGatewayStub()
    orion_gateway = _OrionGatewayStub()

    use_case = DeleteModelUseCase(
        model_repository=model_repo,
        training_job_repository=training_repo,
        artifacts_repository=artifacts,
        iot_agent_gateway=iot_gateway,
        orion_gateway=orion_gateway,
        fiware_service="smart",
        fiware_service_path="/",
    )

    await use_case.execute(model_id=model_id)

    assert "urn:prediction:entity" in orion_gateway.deleted_entities
    assert "device-1" in iot_gateway.deleted_devices


@pytest.mark.asyncio
async def test_delete_model_use_case_wraps_gateway_errors(
    model_repo: _InMemoryModelRepository,
    training_repo: _InMemoryTrainingJobRepository,
) -> None:
    model_id = next(iter(model_repo.items.keys()))
    job = training_repo.jobs[model_id][0]
    job.prediction_config.entity_id = "urn:prediction:entity"
    artifacts = _FakeArtifactsRepository()
    orion_gateway = _OrionGatewayStub(raise_on_delete=True)

    use_case = DeleteModelUseCase(
        model_repository=model_repo,
        training_job_repository=training_repo,
        artifacts_repository=artifacts,
        iot_agent_gateway=_IoTGatewayStub(),
        orion_gateway=orion_gateway,
        fiware_service="smart",
        fiware_service_path="/",
    )

    with pytest.raises(ModelOperationError):
        await use_case.execute(model_id=model_id)
