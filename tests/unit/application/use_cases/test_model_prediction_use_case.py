from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

import numpy as np
import pytest

from src.application.use_cases.model_prediction_use_case import (
    ModelPredictionDependencyError,
    ModelPredictionError,
    ModelPredictionNotFoundError,
    ModelPredictionUseCase,
)
from src.domain.entities.iot import (
    DeviceAttribute,
    IoTDevice,
    IoTDeviceCollection,
    IoTServiceGroup,
)
from src.domain.entities.model import Model, ModelStatus, ModelType
from src.domain.entities.time_series import HistoricDataPoint
from src.domain.entities.training_job import (
    TrainingJob,
    TrainingMetrics,
    TrainingStatus,
)
from src.domain.gateways.iot_agent_gateway import IIoTAgentGateway
from src.domain.gateways.sth_comet_gateway import ISTHCometGateway
from src.domain.repositories.model_artifacts_repository import (
    IModelArtifactsRepository,
    ModelArtifact,
)
from src.domain.repositories.model_repository import IModelRepository
from src.domain.repositories.training_job_repository import ITrainingJobRepository


class IdentityScaler:
    def transform(self, data: np.ndarray) -> np.ndarray:
        return np.asarray(data, dtype=np.float32)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        return np.asarray(data, dtype=np.float32)


class ConstantModel:
    def __init__(self, value: float):
        self._value = value

    def predict(self, _: Any, verbose: int = 0) -> np.ndarray:  # noqa: D401
        return np.array([[self._value]], dtype=np.float32)


class StubModelRepository(IModelRepository):
    def __init__(self, model: Optional[Model]):
        self._model = model

    async def find_by_id(self, model_id: UUID) -> Optional[Model]:
        if self._model and self._model.id == model_id:
            return self._model
        return None

    async def find_all(self, *args, **kwargs):  # pragma: no cover - unused
        return []

    async def create(self, model: Model) -> Model:  # pragma: no cover - unused
        raise NotImplementedError

    async def update(self, model: Model) -> Model:  # pragma: no cover - unused
        self._model = model
        return model

    async def delete(self, model_id: UUID) -> None:  # pragma: no cover - unused
        raise NotImplementedError


class StubTrainingJobRepository(ITrainingJobRepository):
    def __init__(self, training_job: Optional[TrainingJob]):
        self._training_job = training_job
        self._sampling_metadata: dict[UUID, dict[str, Optional[datetime]]] = {}
        self._prediction_schedule: dict[UUID, datetime] = {}

    async def create(self, training_job: TrainingJob):  # pragma: no cover - unused
        raise NotImplementedError

    async def get_by_id(self, training_job_id: UUID) -> Optional[TrainingJob]:
        if self._training_job and self._training_job.id == training_job_id:
            return self._training_job
        return None

    async def get_by_model_id(self, model_id: UUID):  # pragma: no cover - unused
        return []

    async def update(self, training_job: TrainingJob):  # pragma: no cover - unused
        raise NotImplementedError

    async def delete(self, training_job_id: UUID) -> bool:  # pragma: no cover
        raise NotImplementedError

    async def add_data_collection_job(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError

    async def update_data_collection_job_status(
        self, *args, **kwargs
    ):  # pragma: no cover
        raise NotImplementedError

    async def update_training_job_status(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError

    async def complete_training_job(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError

    async def fail_training_job(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError

    async def update_task_refs(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError

    async def update_sampling_metadata(
        self,
        training_job_id: UUID,
        *,
        sampling_interval_seconds: Optional[int],
        next_prediction_at: Optional[datetime] = None,
    ) -> bool:
        self._sampling_metadata[training_job_id] = {
            "sampling_interval_seconds": sampling_interval_seconds,
            "next_prediction_at": next_prediction_at,
        }
        if self._training_job and self._training_job.id == training_job_id:
            self._training_job.sampling_interval_seconds = sampling_interval_seconds
            self._training_job.next_prediction_at = next_prediction_at
        if next_prediction_at:
            self._prediction_schedule[training_job_id] = next_prediction_at
        return True

    async def claim_prediction_schedule(
        self,
        training_job_id: UUID,
        *,
        expected_next_prediction_at: Optional[datetime],
        next_prediction_at: datetime,
    ) -> bool:
        current = self._prediction_schedule.get(training_job_id)
        if expected_next_prediction_at and current != expected_next_prediction_at:
            return False
        self._prediction_schedule[training_job_id] = next_prediction_at
        if self._training_job and self._training_job.id == training_job_id:
            self._training_job.next_prediction_at = next_prediction_at
        return True

    async def update_prediction_schedule(
        self,
        training_job_id: UUID,
        *,
        next_prediction_at: datetime,
    ) -> bool:
        self._prediction_schedule[training_job_id] = next_prediction_at
        if self._training_job and self._training_job.id == training_job_id:
            self._training_job.next_prediction_at = next_prediction_at
        return True

    async def get_prediction_ready_jobs(
        self,
        *,
        reference_time: datetime,
        limit: int = 50,
    ) -> List[TrainingJob]:
        ready_ids = [
            job_id
            for job_id, scheduled in self._prediction_schedule.items()
            if scheduled <= reference_time
        ][:limit]
        if not ready_ids or not self._training_job:
            return []
        if self._training_job.id in ready_ids:
            return [self._training_job]
        return []


class StubArtifactsRepository(IModelArtifactsRepository):
    def __init__(self, artifacts: Dict[str, ModelArtifact]):
        self._artifacts = artifacts

    async def save_artifact(self, *args, **kwargs):  # pragma: no cover - unused
        raise NotImplementedError

    async def get_artifact(self, *args, **kwargs):  # pragma: no cover - unused
        raise NotImplementedError

    async def get_artifact_by_id(self, artifact_id: str) -> Optional[ModelArtifact]:
        return self._artifacts.get(artifact_id)

    async def delete_artifact(self, artifact_id: str):  # pragma: no cover - unused
        raise NotImplementedError

    async def delete_model_artifacts(self, model_id: UUID):  # pragma: no cover - unused
        raise NotImplementedError

    async def list_model_artifacts(self, model_id: UUID):  # pragma: no cover - unused
        raise NotImplementedError


class StubSTHGateway(ISTHCometGateway):
    def __init__(self, points: List[HistoricDataPoint]):
        self._points = sorted(points, key=lambda point: point.timestamp)

    async def collect_data(
        self,
        entity_type: str,
        entity_id: str,
        attribute: str,
        h_limit: int,
        h_offset: int,
        fiware_service: str = "smart",
        fiware_servicepath: str = "/",
        **kwargs,
    ) -> List[HistoricDataPoint]:
        start = max(0, h_offset)
        end = start + max(0, h_limit)
        return self._points[start:end]

    async def get_total_count_from_header(
        self, *args, **kwargs
    ) -> int:  # pragma: no cover
        return len(self._points)


class PartialDataGateway(ISTHCometGateway):
    def __init__(self, points: List[HistoricDataPoint]):
        self._points = sorted(points, key=lambda p: p.timestamp)
        self.calls = 0

    async def get_total_count_from_header(self, *args, **kwargs) -> int:
        return len(self._points)

    async def collect_data(
        self,
        entity_type: str,
        entity_id: str,
        attribute: str,
        h_limit: int,
        h_offset: int,
        fiware_service: str = "smart",
        fiware_servicepath: str = "/",
        **kwargs,
    ) -> List[HistoricDataPoint]:
        self.calls += 1
        if self.calls == 1:
            start = max(0, h_offset)
            end = min(len(self._points), start + max(1, h_limit // 2))
            return self._points[start:end]
        return []


class FailingSTHGateway(ISTHCometGateway):
    def __init__(self, *, raise_on_header: bool = False):
        self.raise_on_header = raise_on_header

    async def get_total_count_from_header(self, *args, **kwargs) -> int:
        if self.raise_on_header:
            raise RuntimeError("header failure")
        return 10

    async def collect_data(self, *args, **kwargs) -> List[HistoricDataPoint]:
        raise RuntimeError("collect failure")


class StubIoTAgentGateway(IIoTAgentGateway):
    def __init__(self, *, has_entity: bool, feature: str):
        self._has_entity = has_entity
        self._feature = feature
        self._service_groups: dict[tuple[str, str], IoTServiceGroup] = {}
        self._devices: dict[str, IoTDevice] = {}

    async def get_devices(self, *args, **kwargs) -> IoTDeviceCollection:
        if not self._has_entity:
            return IoTDeviceCollection(count=0, devices=[])

        device = IoTDevice(
            device_id="sensor-1",
            service="smart",
            service_path="/",
            entity_name="sensor-1",
            entity_type="Sensor",
            transport="mqtt",
            protocol="MQTT",
            attributes=[
                DeviceAttribute(object_id="obj", name=self._feature, type="Number")
            ],
        )
        return IoTDeviceCollection(count=1, devices=[device])

    async def get_service_groups(
        self, service: str = "smart", service_path: str = "/"
    ) -> List[IoTServiceGroup]:
        group = self._service_groups.get((service, service_path))
        return [group] if group else []

    async def ensure_service_group(
        self,
        *,
        service: str,
        service_path: str,
        apikey: str,
        entity_type: str,
        resource: str,
        cbroker: str,
    ) -> IoTServiceGroup:
        group = IoTServiceGroup(
            apikey=apikey,
            cbroker=cbroker,
            entity_type=entity_type,
            resource=resource,
            service=service,
            service_path=service_path,
        )
        self._service_groups[(service, service_path)] = group
        return group

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
    ) -> IoTDevice:
        device = IoTDevice(
            device_id=device_id,
            service=service,
            service_path=service_path,
            entity_name=entity_name,
            entity_type=entity_type,
            transport=transport,
            protocol=protocol,
            attributes=attributes,
        )
        self._devices[device_id] = device
        return device

    async def delete_device(
        self,
        device_id: str,
        *,
        service: str,
        service_path: str,
    ) -> None:
        self._devices.pop(device_id, None)


@pytest.mark.asyncio
async def test_prediction_use_case_returns_forecast():
    model = Model(
        id=uuid4(),
        name="Test model",
        model_type=ModelType.LSTM,
        status=ModelStatus.TRAINED,
        entity_type="Sensor",
        entity_id="sensor-1",
        feature="value",
        lookback_window=3,
        forecast_horizon=2,
    )

    training_job = TrainingJob(
        id=uuid4(),
        model_id=model.id,
        status=TrainingStatus.COMPLETED,
        last_n=100,
        model_artifact_id="model-art",
        x_scaler_artifact_id="x-art",
        y_scaler_artifact_id="y-art",
        metadata_artifact_id="meta-art",
        metrics=TrainingMetrics(mse=0.1, mae=0.2),
    )

    metadata_payload = {
        "window_size": 3,
        "feature_columns": ["value"],
        "model_config": {"forecast_horizon": 2},
    }
    artifacts = {
        "model-art": ModelArtifact("model-art", "model", b"unused"),
        "x-art": ModelArtifact("x-art", "x_scaler", b"unused"),
        "y-art": ModelArtifact("y-art", "y_scaler", b"unused"),
        "meta-art": ModelArtifact(
            "meta-art", "metadata", json.dumps(metadata_payload).encode("utf-8")
        ),
    }

    base_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
    points = [
        HistoricDataPoint(timestamp=base_time + timedelta(minutes=i), value=float(i))
        for i in range(6)
    ]

    use_case = ModelPredictionUseCase(
        model_repository=StubModelRepository(model),
        training_job_repository=StubTrainingJobRepository(training_job),
        artifacts_repository=StubArtifactsRepository(artifacts),
        sth_gateway=StubSTHGateway(points),
        iot_agent_gateway=StubIoTAgentGateway(has_entity=True, feature="value"),
        model_loader=lambda _: ConstantModel(10.0),
        scaler_loader=lambda _: IdentityScaler(),
    )

    response = await use_case.execute(model.id, training_job.id)

    assert response.lookback_window == 3
    assert response.forecast_horizon == 2
    assert len(response.context_window) == 3
    assert response.context_window[-1].value == points[-1].value
    assert len(response.predictions) == 2
    assert all(pred.value == pytest.approx(10.0) for pred in response.predictions)

    expected_first_ts = points[-1].timestamp + timedelta(minutes=1)
    assert response.predictions[0].timestamp == expected_first_ts

    assert response.metadata["training_job_metadata_id"] == "meta-art"
    assert response.metadata["stored_metadata"]["window_size"] == 3
    assert response.metadata["training_metrics"]["mse"] == pytest.approx(0.1)
    assert response.generated_at.tzinfo is not None


@pytest.mark.asyncio
async def test_prediction_use_case_raises_when_model_missing():
    training_job = TrainingJob(
        id=uuid4(),
        model_id=uuid4(),
        status=TrainingStatus.COMPLETED,
        model_artifact_id="model-art",
        x_scaler_artifact_id="x-art",
        y_scaler_artifact_id="y-art",
        metadata_artifact_id="meta-art",
    )

    use_case = ModelPredictionUseCase(
        model_repository=StubModelRepository(None),
        training_job_repository=StubTrainingJobRepository(training_job),
        artifacts_repository=StubArtifactsRepository({}),
        sth_gateway=StubSTHGateway([]),
        iot_agent_gateway=StubIoTAgentGateway(has_entity=True, feature="value"),
        model_loader=lambda _: ConstantModel(10.0),
        scaler_loader=lambda _: IdentityScaler(),
    )

    with pytest.raises(ModelPredictionNotFoundError):
        await use_case.execute(uuid4(), training_job.id)


@pytest.mark.asyncio
async def test_prediction_use_case_raises_when_entity_missing():
    model = Model(
        id=uuid4(),
        name="Test model",
        model_type=ModelType.LSTM,
        status=ModelStatus.TRAINED,
        entity_type="Sensor",
        entity_id="sensor-1",
        feature="value",
        lookback_window=3,
        forecast_horizon=2,
    )

    training_job = TrainingJob(
        id=uuid4(),
        model_id=model.id,
        status=TrainingStatus.COMPLETED,
        model_artifact_id="model-art",
        x_scaler_artifact_id="x-art",
        y_scaler_artifact_id="y-art",
        metadata_artifact_id="meta-art",
    )

    metadata_payload = {
        "window_size": 3,
        "feature_columns": ["value"],
        "model_config": {"forecast_horizon": 2},
    }
    artifacts = {
        "model-art": ModelArtifact("model-art", "model", b"unused"),
        "x-art": ModelArtifact("x-art", "x_scaler", b"unused"),
        "y-art": ModelArtifact("y-art", "y_scaler", b"unused"),
        "meta-art": ModelArtifact(
            "meta-art", "metadata", json.dumps(metadata_payload).encode("utf-8")
        ),
    }

    use_case = ModelPredictionUseCase(
        model_repository=StubModelRepository(model),
        training_job_repository=StubTrainingJobRepository(training_job),
        artifacts_repository=StubArtifactsRepository(artifacts),
        sth_gateway=StubSTHGateway([]),
        iot_agent_gateway=StubIoTAgentGateway(has_entity=False, feature="value"),
        model_loader=lambda _: ConstantModel(10.0),
        scaler_loader=lambda _: IdentityScaler(),
    )

    with pytest.raises(ModelPredictionNotFoundError):
        await use_case.execute(model.id, training_job.id)


@pytest.mark.asyncio
async def test_collect_recent_points_raises_when_insufficient_total():
    lookback = 5
    model = Model(
        id=uuid4(),
        name="Short history",
        model_type=ModelType.LSTM,
        status=ModelStatus.TRAINED,
        entity_type="Sensor",
        entity_id="sensor-1",
        feature="value",
        lookback_window=lookback,
        forecast_horizon=1,
    )

    points = [
        HistoricDataPoint(
            timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=i),
            value=float(i),
        )
        for i in range(2)
    ]

    gateway = StubSTHGateway(points)
    use_case = ModelPredictionUseCase(
        model_repository=StubModelRepository(model),
        training_job_repository=StubTrainingJobRepository(None),
        artifacts_repository=StubArtifactsRepository({}),
        sth_gateway=gateway,
        iot_agent_gateway=StubIoTAgentGateway(has_entity=True, feature="value"),
        model_loader=lambda _: ConstantModel(1.0),
        scaler_loader=lambda _: IdentityScaler(),
    )

    with pytest.raises(ModelPredictionError):
        await use_case._collect_recent_points(model, lookback)


@pytest.mark.asyncio
async def test_collect_recent_points_raises_when_partial_chunks():
    lookback = 4
    model = Model(
        id=uuid4(),
        name="Partial chunks",
        model_type=ModelType.LSTM,
        status=ModelStatus.TRAINED,
        entity_type="Sensor",
        entity_id="sensor-1",
        feature="value",
        lookback_window=lookback,
        forecast_horizon=1,
    )

    points = [
        HistoricDataPoint(
            timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=i),
            value=float(i),
        )
        for i in range(10)
    ]

    gateway = PartialDataGateway(points)
    use_case = ModelPredictionUseCase(
        model_repository=StubModelRepository(model),
        training_job_repository=StubTrainingJobRepository(None),
        artifacts_repository=StubArtifactsRepository({}),
        sth_gateway=gateway,
        iot_agent_gateway=StubIoTAgentGateway(has_entity=True, feature="value"),
        model_loader=lambda _: ConstantModel(1.0),
        scaler_loader=lambda _: IdentityScaler(),
    )

    with pytest.raises(ModelPredictionError):
        await use_case._collect_recent_points(model, lookback)


@pytest.mark.asyncio
async def test_prediction_use_case_requires_model_entity_configuration():
    model = Model(
        id=uuid4(),
        name="Incomplete model",
        model_type=ModelType.LSTM,
        status=ModelStatus.TRAINED,
        entity_type="",
        entity_id="",
        feature="value",
    )
    training_job = TrainingJob(
        id=uuid4(),
        model_id=model.id,
        status=TrainingStatus.COMPLETED,
        model_artifact_id="model-art",
        x_scaler_artifact_id="x-art",
        y_scaler_artifact_id="y-art",
        metadata_artifact_id="meta-art",
    )
    artifacts = {
        "model-art": ModelArtifact("model-art", "model", b"unused"),
        "x-art": ModelArtifact("x-art", "x_scaler", b"unused"),
        "y-art": ModelArtifact("y-art", "y_scaler", b"unused"),
        "meta-art": ModelArtifact("meta-art", "metadata", b"{}"),
    }

    use_case = ModelPredictionUseCase(
        model_repository=StubModelRepository(model),
        training_job_repository=StubTrainingJobRepository(training_job),
        artifacts_repository=StubArtifactsRepository(artifacts),
        sth_gateway=StubSTHGateway([]),
        iot_agent_gateway=StubIoTAgentGateway(has_entity=True, feature="value"),
        model_loader=lambda _: ConstantModel(1.0),
        scaler_loader=lambda _: IdentityScaler(),
    )

    with pytest.raises(ModelPredictionError):
        await use_case.execute(model.id, training_job.id)


@pytest.mark.asyncio
async def test_prediction_use_case_requires_training_job_artifacts():
    model = Model(
        id=uuid4(),
        name="Missing artifacts",
        model_type=ModelType.LSTM,
        status=ModelStatus.TRAINED,
        entity_type="Sensor",
        entity_id="sensor-1",
        feature="value",
    )
    training_job = TrainingJob(
        id=uuid4(),
        model_id=model.id,
        status=TrainingStatus.COMPLETED,
        model_artifact_id=None,
        x_scaler_artifact_id=None,
        y_scaler_artifact_id=None,
        metadata_artifact_id=None,
    )

    use_case = ModelPredictionUseCase(
        model_repository=StubModelRepository(model),
        training_job_repository=StubTrainingJobRepository(training_job),
        artifacts_repository=StubArtifactsRepository({}),
        sth_gateway=StubSTHGateway([]),
        iot_agent_gateway=StubIoTAgentGateway(has_entity=True, feature="value"),
        model_loader=lambda _: ConstantModel(1.0),
        scaler_loader=lambda _: IdentityScaler(),
    )

    with pytest.raises(ModelPredictionError):
        await use_case.execute(model.id, training_job.id)


@pytest.mark.asyncio
async def test_prediction_use_case_fails_when_metadata_invalid_json():
    model = Model(
        id=uuid4(),
        name="Invalid metadata",
        model_type=ModelType.LSTM,
        status=ModelStatus.TRAINED,
        entity_type="Sensor",
        entity_id="sensor-1",
        feature="value",
    )
    training_job = TrainingJob(
        id=uuid4(),
        model_id=model.id,
        status=TrainingStatus.COMPLETED,
        model_artifact_id="model-art",
        x_scaler_artifact_id="x-art",
        y_scaler_artifact_id="y-art",
        metadata_artifact_id="meta-art",
    )
    artifacts = {
        "model-art": ModelArtifact("model-art", "model", b"unused"),
        "x-art": ModelArtifact("x-art", "x_scaler", b"unused"),
        "y-art": ModelArtifact("y-art", "y_scaler", b"unused"),
        "meta-art": ModelArtifact("meta-art", "metadata", b"not-json"),
    }

    use_case = ModelPredictionUseCase(
        model_repository=StubModelRepository(model),
        training_job_repository=StubTrainingJobRepository(training_job),
        artifacts_repository=StubArtifactsRepository(artifacts),
        sth_gateway=StubSTHGateway([]),
        iot_agent_gateway=StubIoTAgentGateway(has_entity=True, feature="value"),
        model_loader=lambda _: ConstantModel(1.0),
        scaler_loader=lambda _: IdentityScaler(),
    )

    with pytest.raises(ModelPredictionError):
        await use_case.execute(model.id, training_job.id)


@pytest.mark.asyncio
async def test_prediction_use_case_rejects_metadata_feature_columns():
    model = Model(
        id=uuid4(),
        name="Multi feature",
        model_type=ModelType.LSTM,
        status=ModelStatus.TRAINED,
        entity_type="Sensor",
        entity_id="sensor-1",
        feature="value",
    )
    training_job = TrainingJob(
        id=uuid4(),
        model_id=model.id,
        status=TrainingStatus.COMPLETED,
        model_artifact_id="model-art",
        x_scaler_artifact_id="x-art",
        y_scaler_artifact_id="y-art",
        metadata_artifact_id="meta-art",
    )
    metadata_payload = {"feature_columns": ["value", "other"]}
    artifacts = {
        "model-art": ModelArtifact("model-art", "model", b"unused"),
        "x-art": ModelArtifact("x-art", "x_scaler", b"unused"),
        "y-art": ModelArtifact("y-art", "y_scaler", b"unused"),
        "meta-art": ModelArtifact(
            "meta-art", "metadata", json.dumps(metadata_payload).encode("utf-8")
        ),
    }

    use_case = ModelPredictionUseCase(
        model_repository=StubModelRepository(model),
        training_job_repository=StubTrainingJobRepository(training_job),
        artifacts_repository=StubArtifactsRepository(artifacts),
        sth_gateway=StubSTHGateway([]),
        iot_agent_gateway=StubIoTAgentGateway(has_entity=True, feature="value"),
        model_loader=lambda _: ConstantModel(1.0),
        scaler_loader=lambda _: IdentityScaler(),
    )

    with pytest.raises(ModelPredictionError):
        await use_case.execute(model.id, training_job.id)


@pytest.mark.asyncio
async def test_prediction_use_case_validates_feature_exposure() -> None:
    model = Model(
        id=uuid4(),
        name="Feature mismatch",
        model_type=ModelType.LSTM,
        status=ModelStatus.TRAINED,
        entity_type="Sensor",
        entity_id="sensor-1",
        feature="value",
    )
    training_job = TrainingJob(
        id=uuid4(),
        model_id=model.id,
        status=TrainingStatus.COMPLETED,
        model_artifact_id="model-art",
        x_scaler_artifact_id="x-art",
        y_scaler_artifact_id="y-art",
        metadata_artifact_id="meta-art",
    )
    artifacts = {
        "model-art": ModelArtifact("model-art", "model", b"unused"),
        "x-art": ModelArtifact("x-art", "x_scaler", b"unused"),
        "y-art": ModelArtifact("y-art", "y_scaler", b"unused"),
        "meta-art": ModelArtifact("meta-art", "metadata", b"{}"),
    }

    use_case = ModelPredictionUseCase(
        model_repository=StubModelRepository(model),
        training_job_repository=StubTrainingJobRepository(training_job),
        artifacts_repository=StubArtifactsRepository(artifacts),
        sth_gateway=StubSTHGateway([]),
        iot_agent_gateway=StubIoTAgentGateway(has_entity=True, feature="other"),
        model_loader=lambda _: ConstantModel(1.0),
        scaler_loader=lambda _: IdentityScaler(),
    )

    with pytest.raises(ModelPredictionError):
        await use_case.execute(model.id, training_job.id)


@pytest.mark.asyncio
async def test_collect_recent_points_handles_gateway_exceptions() -> None:
    model = Model(
        id=uuid4(),
        name="Gateway error",
        model_type=ModelType.LSTM,
        status=ModelStatus.TRAINED,
        entity_type="Sensor",
        entity_id="sensor-1",
        feature="value",
    )
    use_case = ModelPredictionUseCase(
        model_repository=StubModelRepository(model),
        training_job_repository=StubTrainingJobRepository(None),
        artifacts_repository=StubArtifactsRepository({}),
        sth_gateway=FailingSTHGateway(raise_on_header=True),
        iot_agent_gateway=StubIoTAgentGateway(has_entity=True, feature="value"),
        model_loader=lambda _: ConstantModel(1.0),
        scaler_loader=lambda _: IdentityScaler(),
    )

    with pytest.raises(ModelPredictionDependencyError):
        await use_case._collect_recent_points(model, lookback_window=3)


@pytest.mark.asyncio
async def test_collect_recent_points_breaks_on_empty_chunks() -> None:
    model = Model(
        id=uuid4(),
        name="Empty chunks",
        model_type=ModelType.LSTM,
        status=ModelStatus.TRAINED,
        entity_type="Sensor",
        entity_id="sensor-1",
        feature="value",
        lookback_window=3,
    )

    class _Gateway(ISTHCometGateway):
        async def get_total_count_from_header(self, *args, **kwargs) -> int:
            return 5

        async def collect_data(self, *args, **kwargs) -> List[HistoricDataPoint]:
            return []

    use_case = ModelPredictionUseCase(
        model_repository=StubModelRepository(model),
        training_job_repository=StubTrainingJobRepository(None),
        artifacts_repository=StubArtifactsRepository({}),
        sth_gateway=_Gateway(),
        iot_agent_gateway=StubIoTAgentGateway(has_entity=True, feature="value"),
        model_loader=lambda _: ConstantModel(1.0),
        scaler_loader=lambda _: IdentityScaler(),
    )

    with pytest.raises(ModelPredictionError):
        await use_case._collect_recent_points(model, lookback_window=3)


@pytest.mark.asyncio
async def test_collect_recent_points_handles_collection_failures() -> None:
    model = Model(
        id=uuid4(),
        name="Collect failure",
        model_type=ModelType.LSTM,
        status=ModelStatus.TRAINED,
        entity_type="Sensor",
        entity_id="sensor-1",
        feature="value",
    )
    gateway = FailingSTHGateway(raise_on_header=False)
    use_case = ModelPredictionUseCase(
        model_repository=StubModelRepository(model),
        training_job_repository=StubTrainingJobRepository(None),
        artifacts_repository=StubArtifactsRepository({}),
        sth_gateway=gateway,
        iot_agent_gateway=StubIoTAgentGateway(has_entity=True, feature="value"),
        model_loader=lambda _: ConstantModel(1.0),
        scaler_loader=lambda _: IdentityScaler(),
    )

    with pytest.raises(ModelPredictionDependencyError):
        await use_case._collect_recent_points(model, lookback_window=3)


@pytest.mark.asyncio
async def test_collect_recent_points_requires_positive_total() -> None:
    model = Model(
        id=uuid4(),
        name="No data",
        model_type=ModelType.LSTM,
        status=ModelStatus.TRAINED,
        entity_type="Sensor",
        entity_id="sensor-1",
        feature="value",
    )

    class _Gateway(ISTHCometGateway):
        async def get_total_count_from_header(self, *args, **kwargs) -> int:
            return 0

        async def collect_data(self, *args, **kwargs) -> List[HistoricDataPoint]:
            return []

    use_case = ModelPredictionUseCase(
        model_repository=StubModelRepository(model),
        training_job_repository=StubTrainingJobRepository(None),
        artifacts_repository=StubArtifactsRepository({}),
        sth_gateway=_Gateway(),
        iot_agent_gateway=StubIoTAgentGateway(has_entity=True, feature="value"),
        model_loader=lambda _: ConstantModel(1.0),
        scaler_loader=lambda _: IdentityScaler(),
    )

    with pytest.raises(ModelPredictionError):
        await use_case._collect_recent_points(model, lookback_window=3)


@pytest.mark.asyncio
async def test_get_training_job_raises_when_missing() -> None:
    model = Model(
        id=uuid4(),
        name="Model",
        model_type=ModelType.LSTM,
        status=ModelStatus.TRAINED,
        entity_type="Sensor",
        entity_id="sensor-1",
        feature="value",
    )
    use_case = ModelPredictionUseCase(
        model_repository=StubModelRepository(model),
        training_job_repository=StubTrainingJobRepository(None),
        artifacts_repository=StubArtifactsRepository({}),
        sth_gateway=StubSTHGateway([]),
        iot_agent_gateway=StubIoTAgentGateway(has_entity=True, feature="value"),
        model_loader=lambda _: ConstantModel(1.0),
        scaler_loader=lambda _: IdentityScaler(),
    )

    with pytest.raises(ModelPredictionNotFoundError):
        await use_case._get_training_job(uuid4(), model.id)


def test_validate_training_job_requires_completed_status() -> None:
    job = TrainingJob(
        id=uuid4(),
        model_id=uuid4(),
        status=TrainingStatus.TRAINING,
        model_artifact_id="m",
        x_scaler_artifact_id="x",
        y_scaler_artifact_id="y",
        metadata_artifact_id="meta",
    )
    use_case = ModelPredictionUseCase(
        model_repository=StubModelRepository(None),
        training_job_repository=StubTrainingJobRepository(job),
        artifacts_repository=StubArtifactsRepository({}),
        sth_gateway=StubSTHGateway([]),
        iot_agent_gateway=StubIoTAgentGateway(has_entity=True, feature="value"),
        model_loader=lambda _: ConstantModel(1.0),
        scaler_loader=lambda _: IdentityScaler(),
    )

    with pytest.raises(ModelPredictionError):
        use_case._validate_training_job(job)


def test_require_artifact_id_raises_for_missing() -> None:
    use_case = ModelPredictionUseCase(
        model_repository=StubModelRepository(None),
        training_job_repository=StubTrainingJobRepository(None),
        artifacts_repository=StubArtifactsRepository({}),
        sth_gateway=StubSTHGateway([]),
        iot_agent_gateway=StubIoTAgentGateway(has_entity=True, feature="value"),
        model_loader=lambda _: ConstantModel(1.0),
        scaler_loader=lambda _: IdentityScaler(),
    )

    with pytest.raises(ModelPredictionNotFoundError):
        use_case._require_artifact_id(None, "model")


def test_parse_metadata_invalid_json() -> None:
    use_case = ModelPredictionUseCase(
        model_repository=StubModelRepository(None),
        training_job_repository=StubTrainingJobRepository(None),
        artifacts_repository=StubArtifactsRepository({}),
        sth_gateway=StubSTHGateway([]),
        iot_agent_gateway=StubIoTAgentGateway(has_entity=True, feature="value"),
        model_loader=lambda _: ConstantModel(1.0),
        scaler_loader=lambda _: IdentityScaler(),
    )

    with pytest.raises(ModelPredictionError):
        use_case._parse_metadata(ModelArtifact("meta", "metadata", b"{invalid"))


def test_resolve_lookback_window_validations() -> None:
    use_case = ModelPredictionUseCase(
        model_repository=StubModelRepository(None),
        training_job_repository=StubTrainingJobRepository(None),
        artifacts_repository=StubArtifactsRepository({}),
        sth_gateway=StubSTHGateway([]),
        iot_agent_gateway=StubIoTAgentGateway(has_entity=True, feature="value"),
        model_loader=lambda _: ConstantModel(1.0),
        scaler_loader=lambda _: IdentityScaler(),
    )
    model = Model(
        id=uuid4(),
        name="Model",
        model_type=ModelType.LSTM,
        status=ModelStatus.TRAINED,
        entity_type="Sensor",
        entity_id="sensor-1",
        feature="value",
        lookback_window=5,
    )

    with pytest.raises(ModelPredictionError):
        use_case._resolve_lookback_window({"window_size": "bad"}, model)

    with pytest.raises(ModelPredictionError):
        use_case._resolve_lookback_window({"window_size": -1}, model)


def test_resolve_forecast_horizon_validations() -> None:
    use_case = ModelPredictionUseCase(
        model_repository=StubModelRepository(None),
        training_job_repository=StubTrainingJobRepository(None),
        artifacts_repository=StubArtifactsRepository({}),
        sth_gateway=StubSTHGateway([]),
        iot_agent_gateway=StubIoTAgentGateway(has_entity=True, feature="value"),
        model_loader=lambda _: ConstantModel(1.0),
        scaler_loader=lambda _: IdentityScaler(),
    )
    model = Model(
        id=uuid4(),
        name="Model",
        model_type=ModelType.LSTM,
        status=ModelStatus.TRAINED,
        entity_type="Sensor",
        entity_id="sensor-1",
        feature="value",
        forecast_horizon=1,
    )

    with pytest.raises(ModelPredictionError):
        use_case._resolve_forecast_horizon(
            {"model_config": {"forecast_horizon": "bad"}}, model
        )

    with pytest.raises(ModelPredictionError):
        use_case._resolve_forecast_horizon(
            {"model_config": {"forecast_horizon": -1}}, model
        )


@pytest.mark.asyncio
async def test_ensure_entity_active_wraps_gateway_errors() -> None:
    class RaisingGateway(IIoTAgentGateway):
        async def get_devices(self, *args, **kwargs):
            raise RuntimeError("boom")

        async def get_service_groups(self, *args, **kwargs):
            return []

        async def ensure_service_group(self, **kwargs):
            raise NotImplementedError

        async def ensure_device(self, **kwargs):
            raise NotImplementedError

        async def delete_device(
            self, device_id: str, *, service: str, service_path: str
        ):
            raise NotImplementedError

    model = Model(
        id=uuid4(),
        name="Model",
        model_type=ModelType.LSTM,
        status=ModelStatus.TRAINED,
        entity_type="Sensor",
        entity_id="sensor-1",
        feature="value",
    )
    use_case = ModelPredictionUseCase(
        model_repository=StubModelRepository(model),
        training_job_repository=StubTrainingJobRepository(None),
        artifacts_repository=StubArtifactsRepository({}),
        sth_gateway=StubSTHGateway([]),
        iot_agent_gateway=RaisingGateway(),
        model_loader=lambda _: ConstantModel(1.0),
        scaler_loader=lambda _: IdentityScaler(),
    )

    with pytest.raises(ModelPredictionDependencyError):
        await use_case._ensure_entity_active(model)


def test_load_trained_model_wraps_errors() -> None:
    use_case = ModelPredictionUseCase(
        model_repository=StubModelRepository(None),
        training_job_repository=StubTrainingJobRepository(None),
        artifacts_repository=StubArtifactsRepository({}),
        sth_gateway=StubSTHGateway([]),
        iot_agent_gateway=StubIoTAgentGateway(has_entity=True, feature="value"),
        model_loader=lambda _: (_ for _ in ()).throw(RuntimeError("fail")),
        scaler_loader=lambda _: IdentityScaler(),
    )

    artifact = ModelArtifact("model-art", "model", b"bytes")
    with pytest.raises(ModelPredictionError):
        use_case._load_trained_model(artifact)


def test_load_scaler_wraps_errors() -> None:
    use_case = ModelPredictionUseCase(
        model_repository=StubModelRepository(None),
        training_job_repository=StubTrainingJobRepository(None),
        artifacts_repository=StubArtifactsRepository({}),
        sth_gateway=StubSTHGateway([]),
        iot_agent_gateway=StubIoTAgentGateway(has_entity=True, feature="value"),
        model_loader=lambda _: ConstantModel(1.0),
        scaler_loader=lambda _: (_ for _ in ()).throw(RuntimeError("fail")),
    )

    artifact = ModelArtifact("scaler", "x_scaler", b"bytes")
    with pytest.raises(ModelPredictionError):
        use_case._load_scaler(artifact)


def test_infer_prediction_timestamps_edge_cases() -> None:
    use_case = ModelPredictionUseCase(
        model_repository=StubModelRepository(None),
        training_job_repository=StubTrainingJobRepository(None),
        artifacts_repository=StubArtifactsRepository({}),
        sth_gateway=StubSTHGateway([]),
        iot_agent_gateway=StubIoTAgentGateway(has_entity=True, feature="value"),
        model_loader=lambda _: ConstantModel(1.0),
        scaler_loader=lambda _: IdentityScaler(),
    )
    assert use_case._infer_prediction_timestamps([], horizon=0) == []

    single_point = [HistoricDataPoint(timestamp=datetime.now(timezone.utc), value=1.0)]
    assert use_case._infer_prediction_timestamps(single_point, horizon=2) == [
        None,
        None,
    ]

    repeated_points = [
        HistoricDataPoint(
            timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc), value=1.0
        ),
        HistoricDataPoint(
            timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc), value=2.0
        ),
    ]
    assert use_case._infer_prediction_timestamps(repeated_points, horizon=1) == [None]


def test_build_metadata_includes_metrics() -> None:
    use_case = ModelPredictionUseCase(
        model_repository=StubModelRepository(None),
        training_job_repository=StubTrainingJobRepository(None),
        artifacts_repository=StubArtifactsRepository({}),
        sth_gateway=StubSTHGateway([]),
        iot_agent_gateway=StubIoTAgentGateway(has_entity=True, feature="value"),
        model_loader=lambda _: ConstantModel(1.0),
        scaler_loader=lambda _: IdentityScaler(),
    )

    model = Model(
        id=uuid4(),
        name="Model",
        model_type=ModelType.LSTM,
        status=ModelStatus.TRAINED,
        entity_type="Sensor",
        entity_id="sensor-1",
        feature="value",
        epochs=5,
        batch_size=32,
        learning_rate=0.01,
    )
    training_job = TrainingJob(
        id=uuid4(),
        model_id=model.id,
        status=TrainingStatus.COMPLETED,
        model_artifact_id="model-art",
        x_scaler_artifact_id="x",
        y_scaler_artifact_id="y",
        metadata_artifact_id="meta",
        metrics=TrainingMetrics(mae=0.1, mse=0.2),
    )

    result = use_case._build_metadata(model, training_job, {"extra": 1})
    assert result["training_metrics"]["mae"] == pytest.approx(0.1)
    assert result["stored_metadata"] == {"extra": 1}


def test_load_model_from_bytes_uses_tempfile(monkeypatch) -> None:
    captured = {}

    def fake_load_model(path: str) -> str:
        captured["path"] = path
        with open(path, "rb") as fh:
            captured["content"] = fh.read()
        return "loaded"

    monkeypatch.setattr(
        "src.application.use_cases.model_prediction_use_case.load_model",
        fake_load_model,
    )

    use_case = ModelPredictionUseCase(
        model_repository=StubModelRepository(None),
        training_job_repository=StubTrainingJobRepository(None),
        artifacts_repository=StubArtifactsRepository({}),
        sth_gateway=StubSTHGateway([]),
        iot_agent_gateway=StubIoTAgentGateway(has_entity=True, feature="value"),
        model_loader=lambda _: ConstantModel(1.0),
        scaler_loader=lambda _: IdentityScaler(),
    )

    result = use_case._load_model_from_bytes(b"payload")
    assert result == "loaded"
    assert captured["content"] == b"payload"


def test_load_scaler_from_bytes(monkeypatch) -> None:
    def fake_joblib_load(buffer):
        data = buffer.read()
        buffer.close()
        return data.decode()

    monkeypatch.setattr(
        "src.application.use_cases.model_prediction_use_case.joblib_load",
        fake_joblib_load,
    )

    use_case = ModelPredictionUseCase(
        model_repository=StubModelRepository(None),
        training_job_repository=StubTrainingJobRepository(None),
        artifacts_repository=StubArtifactsRepository({}),
        sth_gateway=StubSTHGateway([]),
        iot_agent_gateway=StubIoTAgentGateway(has_entity=True, feature="value"),
        model_loader=lambda _: ConstantModel(1.0),
        scaler_loader=lambda _: IdentityScaler(),
    )

    result = use_case._load_scaler_from_bytes(b"123")
    assert result == "123"


@pytest.mark.asyncio
async def test_get_training_job_validates_model_relation() -> None:
    model = Model(
        id=uuid4(),
        name="Model",
        model_type=ModelType.LSTM,
        status=ModelStatus.TRAINED,
        entity_type="Sensor",
        entity_id="sensor-1",
        feature="value",
    )
    training_job = TrainingJob(
        id=uuid4(),
        model_id=uuid4(),
        status=TrainingStatus.COMPLETED,
        model_artifact_id="model-art",
        x_scaler_artifact_id="x-art",
        y_scaler_artifact_id="y-art",
        metadata_artifact_id="meta-art",
    )
    use_case = ModelPredictionUseCase(
        model_repository=StubModelRepository(model),
        training_job_repository=StubTrainingJobRepository(training_job),
        artifacts_repository=StubArtifactsRepository({}),
        sth_gateway=StubSTHGateway([]),
        iot_agent_gateway=StubIoTAgentGateway(has_entity=True, feature="value"),
        model_loader=lambda _: ConstantModel(1.0),
        scaler_loader=lambda _: IdentityScaler(),
    )

    with pytest.raises(ModelPredictionError):
        await use_case._get_training_job(training_job.id, model.id)


@pytest.mark.asyncio
async def test_get_required_artifacts_raises_when_missing() -> None:
    training_job = TrainingJob(
        id=uuid4(),
        model_id=uuid4(),
        status=TrainingStatus.COMPLETED,
        model_artifact_id="model-art",
        x_scaler_artifact_id="x-art",
        y_scaler_artifact_id="y-art",
        metadata_artifact_id="meta-art",
    )
    use_case = ModelPredictionUseCase(
        model_repository=StubModelRepository(None),
        training_job_repository=StubTrainingJobRepository(training_job),
        artifacts_repository=StubArtifactsRepository({}),
        sth_gateway=StubSTHGateway([]),
        iot_agent_gateway=StubIoTAgentGateway(has_entity=True, feature="value"),
        model_loader=lambda _: ConstantModel(1.0),
        scaler_loader=lambda _: IdentityScaler(),
    )

    with pytest.raises(ModelPredictionNotFoundError):
        await use_case._get_required_artifacts(training_job)
