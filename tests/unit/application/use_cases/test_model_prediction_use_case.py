from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

import numpy as np
import pytest

from src.application.use_cases.model_prediction_use_case import (
    ModelPredictionError,
    ModelPredictionNotFoundError,
    ModelPredictionUseCase,
)
from src.domain.entities.iot import DeviceAttribute, IoTDevice, IoTDeviceCollection
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


class StubIoTAgentGateway(IIoTAgentGateway):
    def __init__(self, *, has_entity: bool, feature: str):
        self._has_entity = has_entity
        self._feature = feature

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
