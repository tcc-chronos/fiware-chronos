from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

import pytest

from src.application.dtos.prediction_dto import (
    PredictionHistoryRequestDTO,
    PredictionToggleRequestDTO,
)
from src.application.use_cases.prediction_management_use_case import (
    GetPredictionHistoryUseCase,
    PredictionManagementError,
    PredictionNotReadyError,
    TogglePredictionUseCase,
)
from src.domain.entities.iot import (
    DeviceAttribute,
    IoTDevice,
    IoTDeviceCollection,
    IoTServiceGroup,
)
from src.domain.entities.model import Model, ModelStatus, ModelType
from src.domain.entities.time_series import HistoricDataPoint
from src.domain.entities.training_job import TrainingJob, TrainingStatus
from src.domain.gateways.iot_agent_gateway import IIoTAgentGateway
from src.domain.gateways.orion_gateway import IOrionGateway
from src.domain.gateways.sth_comet_gateway import ISTHCometGateway
from src.domain.repositories.model_repository import IModelRepository
from src.domain.repositories.training_job_repository import ITrainingJobRepository


class _ModelRepository(IModelRepository):
    def __init__(self, model: Model):
        self.model = model

    async def find_by_id(self, model_id: UUID) -> Optional[Model]:
        if not self.model:
            return None
        return self.model if self.model.id == model_id else None

    async def find_all(self, *args, **kwargs):  # pragma: no cover - unused
        return []

    async def create(self, model: Model) -> Model:  # pragma: no cover - unused
        self.model = model
        return model

    async def update(self, model: Model) -> Model:
        self.model = model
        return model

    async def delete(self, model_id: UUID) -> None:  # pragma: no cover - unused
        if self.model and self.model.id == model_id:
            self.model = None  # type: ignore[assignment]


class _TrainingJobRepository(ITrainingJobRepository):
    def __init__(self, job: TrainingJob):
        self.jobs: Dict[UUID, TrainingJob] = {job.id: job}
        self.prediction_schedule: Dict[UUID, datetime] = {}

    async def create(self, training_job: TrainingJob) -> TrainingJob:
        self.jobs[training_job.id] = training_job
        return training_job

    async def get_by_id(self, training_job_id: UUID) -> Optional[TrainingJob]:
        return self.jobs.get(training_job_id)

    async def get_by_model_id(self, model_id: UUID) -> List[TrainingJob]:
        return [job for job in self.jobs.values() if job.model_id == model_id]

    async def update(self, training_job: TrainingJob) -> TrainingJob:
        self.jobs[training_job.id] = training_job
        return training_job

    async def delete(self, training_job_id: UUID) -> bool:
        return self.jobs.pop(training_job_id, None) is not None

    async def add_data_collection_job(
        self, *args, **kwargs
    ):  # pragma: no cover - unused
        return True

    async def update_data_collection_job_status(
        self, *args, **kwargs
    ):  # pragma: no cover - unused
        return True

    async def update_training_job_status(
        self, *args, **kwargs
    ):  # pragma: no cover - unused
        return True

    async def complete_training_job(self, *args, **kwargs):  # pragma: no cover - unused
        return True

    async def fail_training_job(self, *args, **kwargs):  # pragma: no cover - unused
        return True

    async def update_task_refs(self, *args, **kwargs):  # pragma: no cover - unused
        return True

    async def update_sampling_metadata(
        self,
        training_job_id: UUID,
        *,
        sampling_interval_seconds: Optional[int],
        next_prediction_at: Optional[datetime] = None,
    ) -> bool:
        job = self.jobs.get(training_job_id)
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
        job = self.jobs.get(training_job_id)
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
        job = self.jobs.get(training_job_id)
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
        for job_id, scheduled in self.prediction_schedule.items():
            if scheduled <= reference_time:
                job = self.jobs.get(job_id)
                if job:
                    ready.append(job)
        ready.sort(key=lambda job: job.next_prediction_at or datetime.max)
        return ready[:limit]


class _IoTAgentGateway(IIoTAgentGateway):
    def __init__(self) -> None:
        self.service_groups: Dict[tuple[str, str], IoTServiceGroup] = {}
        self.devices: Dict[str, IoTDevice] = {}
        self.deleted_devices: List[str] = []

    async def get_devices(
        self, *args, **kwargs
    ) -> IoTDeviceCollection:  # pragma: no cover - unused
        return IoTDeviceCollection(
            count=len(self.devices), devices=list(self.devices.values())
        )

    async def get_service_groups(
        self, service: str = "smart", service_path: str = "/"
    ) -> List[IoTServiceGroup]:  # pragma: no cover - unused
        group = self.service_groups.get((service, service_path))
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
        self.service_groups[(service, service_path)] = group
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
        self.devices[device_id] = device
        return device

    async def delete_device(
        self,
        device_id: str,
        *,
        service: str,
        service_path: str,
    ) -> None:
        self.deleted_devices.append(device_id)
        self.devices.pop(device_id, None)


class _OrionGateway(IOrionGateway):
    def __init__(self) -> None:
        self.entities: Dict[str, Dict[str, Any]] = {}
        self.subscriptions: Dict[str, Dict[str, Any]] = {}
        self.deleted_subscriptions: List[str] = []
        self.deleted_entities: List[str] = []

    async def ensure_entity(
        self,
        *,
        entity_id: str,
        entity_type: str,
        payload: Dict[str, Any],
        service: str,
        service_path: str,
    ) -> None:
        self.entities[entity_id] = {
            "entity_type": entity_type,
            "payload": payload,
            "service": service,
            "service_path": service_path,
        }

    async def upsert_prediction(
        self,
        prediction,
        *,
        service: str,
        service_path: str,
    ) -> None:  # pragma: no cover - unused
        self.entities.setdefault(prediction.entity_id, {})[
            "last_prediction"
        ] = prediction

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
    ) -> str:
        subscription_id = f"sub-{len(self.subscriptions) + 1}"
        self.subscriptions[subscription_id] = {
            "entity_id": entity_id,
            "entity_type": entity_type,
            "attrs": attrs,
            "notification_url": notification_url,
            "service": service,
            "service_path": service_path,
            "attrs_format": attrs_format,
        }
        return subscription_id

    async def delete_subscription(
        self,
        subscription_id: str,
        *,
        service: str,
        service_path: str,
    ) -> None:
        self.deleted_subscriptions.append(subscription_id)
        self.subscriptions.pop(subscription_id, None)

    async def delete_entity(
        self,
        entity_id: str,
        *,
        service: str,
        service_path: str,
    ) -> None:
        self.deleted_entities.append(entity_id)
        self.entities.pop(entity_id, None)


class _STHGateway(ISTHCometGateway):
    def __init__(self, total: int, points: List[HistoricDataPoint]):
        self.total = total
        self.points = points

    async def collect_data(
        self,
        entity_type: str,
        entity_id: str,
        attribute: str,
        h_limit: int,
        h_offset: int,
        fiware_service: str = "smart",
        fiware_servicepath: str = "/",
    ) -> List[HistoricDataPoint]:
        start = max(0, h_offset)
        end = start + h_limit
        return self.points[start:end]

    async def get_total_count_from_header(
        self,
        entity_type: str,
        entity_id: str,
        attribute: str,
        fiware_service: str = "smart",
        fiware_servicepath: str = "/",
    ) -> int:
        if isinstance(self.total, Exception):
            raise self.total
        return self.total


def _make_model() -> Model:
    return Model(
        id=uuid4(),
        name="Chronos model",
        description="",
        model_type=ModelType.LSTM,
        status=ModelStatus.TRAINED,
        batch_size=32,
        epochs=50,
        learning_rate=0.01,
        validation_ratio=0.1,
        test_ratio=0.1,
        lookback_window=24,
        forecast_horizon=6,
        feature="temperature",
        entity_type="Sensor",
        entity_id="urn:ngsi-ld:Sensor:001",
    )


def _make_training_job(model: Model) -> TrainingJob:
    job = TrainingJob()
    job.model_id = model.id
    job.status = TrainingStatus.COMPLETED
    return job


@pytest.mark.asyncio
async def test_toggle_predictions_enable_success() -> None:
    model = _make_model()
    job = _make_training_job(model)
    repo = _ModelRepository(model)
    training_repo = _TrainingJobRepository(job)
    iot_gateway = _IoTAgentGateway()
    orion_gateway = _OrionGateway()

    use_case = TogglePredictionUseCase(
        model_repository=repo,
        training_job_repository=training_repo,
        iot_agent_gateway=iot_gateway,
        orion_gateway=orion_gateway,
        fiware_service="smart",
        fiware_service_path="/",
        forecast_service_group="forecast-group",
        forecast_service_apikey="apikey",
        forecast_service_resource="/forecast",
        forecast_entity_type="Forecast",
        orion_cb_url="http://orion",
        sth_notification_url="http://sth",
        forecast_device_transport="mqtt",
        forecast_device_protocol="PDI-IoTA-UltraLight",
    )

    response = await use_case.execute(
        model_id=model.id,
        training_job_id=job.id,
        request=PredictionToggleRequestDTO(enabled=True, metadata={"extra": "value"}),
    )

    refreshed = await training_repo.get_by_id(job.id)
    assert response.enabled is True
    assert refreshed is not None
    assert refreshed.prediction_config.enabled is True
    assert refreshed.prediction_config.entity_id is not None
    assert refreshed.prediction_config.subscription_id in orion_gateway.subscriptions
    assert refreshed.prediction_config.metadata["device_id"].startswith(
        "chronos-forecast-"
    )
    assert iot_gateway.service_groups[("smart", "/")].resource == "/forecast"
    assert (
        iot_gateway.devices[
            refreshed.prediction_config.metadata["device_id"]
        ].entity_name
        == refreshed.prediction_config.entity_id
    )
    assert (
        orion_gateway.entities[refreshed.prediction_config.entity_id]["payload"][
            "forecastSeries"
        ]["value"]["horizon"]
        == model.forecast_horizon
    )


@pytest.mark.asyncio
async def test_toggle_predictions_disable_requests_cleanup() -> None:
    model = _make_model()
    job = _make_training_job(model)
    job.prediction_config.enabled = True
    job.prediction_config.subscription_id = "sub-existing"
    job.prediction_config.entity_id = "urn:prediction:entity"
    job.prediction_config.metadata = {"device_id": "chronos-device"}

    repo = _ModelRepository(model)
    training_repo = _TrainingJobRepository(job)
    iot_gateway = _IoTAgentGateway()
    # Pre-populate device so delete path is exercised
    iot_gateway.devices["chronos-device"] = IoTDevice(
        device_id="chronos-device",
        service="smart",
        service_path="/",
        entity_name="urn:prediction:entity",
        entity_type="Forecast",
        transport="mqtt",
        protocol="PDI-IoTA-UltraLight",
        attributes=[
            DeviceAttribute(
                object_id="fs", name="forecastSeries", type="StructuredValue"
            )
        ],
    )
    orion_gateway = _OrionGateway()
    orion_gateway.subscriptions["sub-existing"] = {"entity_id": "urn:prediction:entity"}

    use_case = TogglePredictionUseCase(
        model_repository=repo,
        training_job_repository=training_repo,
        iot_agent_gateway=iot_gateway,
        orion_gateway=orion_gateway,
        fiware_service="smart",
        fiware_service_path="/",
        forecast_service_group="forecast-group",
        forecast_service_apikey="apikey",
        forecast_service_resource="/forecast",
        forecast_entity_type="Forecast",
        orion_cb_url="http://orion",
        sth_notification_url="http://sth",
        forecast_device_transport="mqtt",
        forecast_device_protocol="PDI-IoTA-UltraLight",
    )

    response = await use_case.execute(
        model_id=model.id,
        training_job_id=job.id,
        request=PredictionToggleRequestDTO(enabled=False),
    )

    refreshed = await training_repo.get_by_id(job.id)
    assert response.enabled is False
    assert refreshed is not None
    assert refreshed.prediction_config.enabled is False
    assert "sub-existing" in orion_gateway.deleted_subscriptions


@pytest.mark.asyncio
async def test_toggle_predictions_requires_completed_training() -> None:
    model = _make_model()
    job = _make_training_job(model)
    job.status = TrainingStatus.TRAINING

    use_case = TogglePredictionUseCase(
        model_repository=_ModelRepository(model),
        training_job_repository=_TrainingJobRepository(job),
        iot_agent_gateway=_IoTAgentGateway(),
        orion_gateway=_OrionGateway(),
        fiware_service="smart",
        fiware_service_path="/",
        forecast_service_group="forecast-group",
        forecast_service_apikey="apikey",
        forecast_service_resource="/forecast",
        forecast_entity_type="Forecast",
        orion_cb_url="http://orion",
        sth_notification_url="http://sth",
        forecast_device_transport="mqtt",
        forecast_device_protocol="PDI-IoTA-UltraLight",
    )

    with pytest.raises(PredictionNotReadyError):
        await use_case.execute(
            model_id=model.id,
            training_job_id=job.id,
            request=PredictionToggleRequestDTO(enabled=True),
        )


@pytest.mark.asyncio
async def test_get_prediction_history_returns_ordered_points() -> None:
    model = _make_model()
    job = _make_training_job(model)
    job.prediction_config.entity_id = "urn:prediction:entity"
    job.prediction_config.entity_type = "Forecast"

    now = datetime.now(timezone.utc)
    # Simulate multiple groups and future points
    points = (
        [
            HistoricDataPoint(
                timestamp=now - timedelta(minutes=idx),
                value=float(idx),
                group_timestamp=now,
            )
            for idx in range(2)
        ]
        + [
            HistoricDataPoint(
                timestamp=now - timedelta(minutes=10 + idx),
                value=50 + float(idx),
                group_timestamp=now - timedelta(minutes=10),
            )
            for idx in range(2)
        ]
        + [
            HistoricDataPoint(
                timestamp=now - timedelta(hours=1, minutes=idx),
                value=100 + float(idx),
            )
            for idx in range(3)
        ]
    )

    use_case = GetPredictionHistoryUseCase(
        model_repository=_ModelRepository(model),
        training_job_repository=_TrainingJobRepository(job),
        sth_gateway=_STHGateway(total=len(points), points=points),
        fiware_service="smart",
        fiware_service_path="/",
    )

    request = PredictionHistoryRequestDTO(limit=5)
    response = await use_case.execute(model.id, job.id, request)

    assert response.entity_id == job.prediction_config.entity_id
    assert len(response.points) == 5
    assert response.points[0].timestamp < response.points[-1].timestamp


@pytest.mark.asyncio
async def test_get_prediction_history_handles_empty_results() -> None:
    model = _make_model()
    job = _make_training_job(model)
    job.prediction_config.entity_id = "urn:prediction:entity"

    use_case = GetPredictionHistoryUseCase(
        model_repository=_ModelRepository(model),
        training_job_repository=_TrainingJobRepository(job),
        sth_gateway=_STHGateway(total=0, points=[]),
        fiware_service="smart",
        fiware_service_path="/",
    )

    response = await use_case.execute(
        model.id,
        job.id,
        PredictionHistoryRequestDTO(limit=5),
    )

    assert response.points == []


@pytest.mark.asyncio
async def test_get_prediction_history_wraps_gateway_errors() -> None:
    model = _make_model()
    job = _make_training_job(model)
    job.prediction_config.entity_id = "urn:prediction:entity"

    gateway = _STHGateway(total=RuntimeError("boom"), points=[])
    use_case = GetPredictionHistoryUseCase(
        model_repository=_ModelRepository(model),
        training_job_repository=_TrainingJobRepository(job),
        sth_gateway=gateway,
        fiware_service="smart",
        fiware_service_path="/",
    )

    with pytest.raises(PredictionManagementError):
        await use_case.execute(
            model.id,
            job.id,
            PredictionHistoryRequestDTO(limit=5),
        )


@pytest.mark.asyncio
async def test_toggle_predictions_raises_when_model_missing() -> None:
    model = _make_model()
    job = _make_training_job(model)
    use_case = TogglePredictionUseCase(
        model_repository=_ModelRepository(model),
        training_job_repository=_TrainingJobRepository(job),
        iot_agent_gateway=_IoTAgentGateway(),
        orion_gateway=_OrionGateway(),
        fiware_service="smart",
        fiware_service_path="/",
        forecast_service_group="forecast-group",
        forecast_service_apikey="apikey",
        forecast_service_resource="/forecast",
        forecast_entity_type="Forecast",
        orion_cb_url="http://orion",
        sth_notification_url="http://sth",
        forecast_device_transport="mqtt",
        forecast_device_protocol="PDI-IoTA-UltraLight",
    )

    with pytest.raises(PredictionManagementError):
        await use_case.execute(
            model_id=uuid4(),
            training_job_id=job.id,
            request=PredictionToggleRequestDTO(enabled=True),
        )


@pytest.mark.asyncio
async def test_toggle_predictions_requires_trained_model() -> None:
    model = _make_model()
    model.status = ModelStatus.DRAFT
    job = _make_training_job(model)
    use_case = TogglePredictionUseCase(
        model_repository=_ModelRepository(model),
        training_job_repository=_TrainingJobRepository(job),
        iot_agent_gateway=_IoTAgentGateway(),
        orion_gateway=_OrionGateway(),
        fiware_service="smart",
        fiware_service_path="/",
        forecast_service_group="forecast-group",
        forecast_service_apikey="apikey",
        forecast_service_resource="/forecast",
        forecast_entity_type="Forecast",
        orion_cb_url="http://orion",
        sth_notification_url="http://sth",
        forecast_device_transport="mqtt",
        forecast_device_protocol="PDI-IoTA-UltraLight",
    )

    with pytest.raises(PredictionManagementError):
        await use_case.execute(
            model_id=model.id,
            training_job_id=job.id,
            request=PredictionToggleRequestDTO(enabled=True),
        )


@pytest.mark.asyncio
async def test_toggle_predictions_validates_job_relationship() -> None:
    model = _make_model()
    job = _make_training_job(model)
    job.model_id = uuid4()
    use_case = TogglePredictionUseCase(
        model_repository=_ModelRepository(model),
        training_job_repository=_TrainingJobRepository(job),
        iot_agent_gateway=_IoTAgentGateway(),
        orion_gateway=_OrionGateway(),
        fiware_service="smart",
        fiware_service_path="/",
        forecast_service_group="forecast-group",
        forecast_service_apikey="apikey",
        forecast_service_resource="/forecast",
        forecast_entity_type="Forecast",
        orion_cb_url="http://orion",
        sth_notification_url="http://sth",
        forecast_device_transport="mqtt",
        forecast_device_protocol="PDI-IoTA-UltraLight",
    )

    with pytest.raises(PredictionManagementError):
        await use_case.execute(
            model_id=model.id,
            training_job_id=job.id,
            request=PredictionToggleRequestDTO(enabled=True),
        )


@pytest.mark.asyncio
async def test_get_prediction_history_requires_prediction_entity() -> None:
    model = _make_model()
    job = _make_training_job(model)
    job.prediction_config.entity_id = ""

    use_case = GetPredictionHistoryUseCase(
        model_repository=_ModelRepository(model),
        training_job_repository=_TrainingJobRepository(job),
        sth_gateway=_STHGateway(total=10, points=[]),
        fiware_service="smart",
        fiware_service_path="/",
    )

    with pytest.raises(PredictionManagementError):
        await use_case.execute(
            model.id,
            job.id,
            PredictionHistoryRequestDTO(limit=3),
        )


def test_determine_sth_limit_scales_with_horizon() -> None:
    model = _make_model()
    job = _make_training_job(model)
    use_case = GetPredictionHistoryUseCase(
        model_repository=_ModelRepository(model),
        training_job_repository=_TrainingJobRepository(job),
        sth_gateway=_STHGateway(total=100, points=[]),
        fiware_service="smart",
        fiware_service_path="/",
    )
    limit = use_case._determine_sth_limit(requested_limit=5, forecast_horizon=30)
    assert limit == 100


def test_normalize_timestamp_handles_naive_values() -> None:
    model = _make_model()
    job = _make_training_job(model)
    use_case = GetPredictionHistoryUseCase(
        model_repository=_ModelRepository(model),
        training_job_repository=_TrainingJobRepository(job),
        sth_gateway=_STHGateway(total=0, points=[]),
        fiware_service="smart",
        fiware_service_path="/",
    )
    naive = datetime(2024, 1, 1)
    normalized = use_case._normalize_timestamp(naive)
    assert normalized.tzinfo == timezone.utc


@pytest.mark.asyncio
async def test_get_prediction_history_uses_fallback_ordering() -> None:
    model = _make_model()
    job = _make_training_job(model)
    job.prediction_config.entity_id = "urn:prediction:entity"

    base = datetime.now(timezone.utc)
    points = [
        HistoricDataPoint(timestamp=base - timedelta(minutes=i), value=float(i))
        for i in range(6)
    ]

    use_case = GetPredictionHistoryUseCase(
        model_repository=_ModelRepository(model),
        training_job_repository=_TrainingJobRepository(job),
        sth_gateway=_STHGateway(total=len(points), points=points),
        fiware_service="smart",
        fiware_service_path="/",
    )

    response = await use_case.execute(
        model.id,
        job.id,
        PredictionHistoryRequestDTO(limit=3),
    )

    assert len(response.points) == 3
    assert response.points[-1].timestamp > response.points[0].timestamp


@pytest.mark.asyncio
async def test_toggle_predictions_requires_training_job() -> None:
    model = _make_model()
    job = _make_training_job(model)

    class _MissingTrainingJobRepo(_TrainingJobRepository):
        async def get_by_id(self, training_job_id: UUID) -> Optional[TrainingJob]:
            return None

    use_case = TogglePredictionUseCase(
        model_repository=_ModelRepository(model),
        training_job_repository=_MissingTrainingJobRepo(job),
        iot_agent_gateway=_IoTAgentGateway(),
        orion_gateway=_OrionGateway(),
        fiware_service="smart",
        fiware_service_path="/",
        forecast_service_group="forecast",
        forecast_service_apikey="apikey",
        forecast_service_resource="/forecast",
        forecast_entity_type="Forecast",
        orion_cb_url="http://orion",
        sth_notification_url="http://sth",
        forecast_device_transport="mqtt",
        forecast_device_protocol="protocol",
    )

    with pytest.raises(PredictionManagementError):
        await use_case.execute(
            model_id=model.id,
            training_job_id=uuid4(),
            request=PredictionToggleRequestDTO(enabled=True),
        )


@pytest.mark.asyncio
async def test_toggle_predictions_raises_when_refresh_missing() -> None:
    model = _make_model()
    job = _make_training_job(model)

    class _EvictingTrainingRepo(_TrainingJobRepository):
        async def update(self, training_job: TrainingJob) -> TrainingJob:
            await super().update(training_job)
            self.jobs.pop(training_job.id, None)
            return training_job

    repo = _EvictingTrainingRepo(job)
    use_case = TogglePredictionUseCase(
        model_repository=_ModelRepository(model),
        training_job_repository=repo,
        iot_agent_gateway=_IoTAgentGateway(),
        orion_gateway=_OrionGateway(),
        fiware_service="smart",
        fiware_service_path="/",
        forecast_service_group="forecast",
        forecast_service_apikey="apikey",
        forecast_service_resource="/forecast",
        forecast_entity_type="Forecast",
        orion_cb_url="http://orion",
        sth_notification_url="http://sth",
        forecast_device_transport="mqtt",
        forecast_device_protocol="protocol",
    )

    with pytest.raises(PredictionManagementError):
        await use_case.execute(
            model_id=model.id,
            training_job_id=job.id,
            request=PredictionToggleRequestDTO(enabled=True),
        )


@pytest.mark.asyncio
async def test_toggle_predictions_validates_model_feature_presence() -> None:
    model = _make_model()
    model.feature = ""
    job = _make_training_job(model)
    use_case = TogglePredictionUseCase(
        model_repository=_ModelRepository(model),
        training_job_repository=_TrainingJobRepository(job),
        iot_agent_gateway=_IoTAgentGateway(),
        orion_gateway=_OrionGateway(),
        fiware_service="smart",
        fiware_service_path="/",
        forecast_service_group="forecast-group",
        forecast_service_apikey="apikey",
        forecast_service_resource="/forecast",
        forecast_entity_type="Forecast",
        orion_cb_url="http://orion",
        sth_notification_url="http://sth",
        forecast_device_transport="mqtt",
        forecast_device_protocol="protocol",
    )

    with pytest.raises(PredictionManagementError):
        await use_case.execute(
            model_id=model.id,
            training_job_id=job.id,
            request=PredictionToggleRequestDTO(enabled=True),
        )


@pytest.mark.asyncio
async def test_get_prediction_history_applies_filters() -> None:
    model = _make_model()
    job = _make_training_job(model)
    job.prediction_config.entity_id = "urn:prediction:entity"

    base = datetime.now(timezone.utc)
    points = [
        HistoricDataPoint(timestamp=base - timedelta(minutes=10), value=1.0),
        HistoricDataPoint(timestamp=base, value=2.0),
        HistoricDataPoint(timestamp=base + timedelta(minutes=10), value=3.0),
    ]

    use_case = GetPredictionHistoryUseCase(
        model_repository=_ModelRepository(model),
        training_job_repository=_TrainingJobRepository(job),
        sth_gateway=_STHGateway(total=len(points), points=points),
        fiware_service="smart",
        fiware_service_path="/",
    )

    response = await use_case.execute(
        model.id,
        job.id,
        PredictionHistoryRequestDTO(
            limit=2,
            start=base - timedelta(minutes=5),
            end=base + timedelta(minutes=5),
        ),
    )

    assert len(response.points) == 1
    assert response.points[0].value == 2.0


@pytest.mark.asyncio
async def test_enable_predictions_creates_subscription_when_missing() -> None:
    model = _make_model()
    job = _make_training_job(model)
    training_repo = _TrainingJobRepository(job)
    iot_gateway = _IoTAgentGateway()
    orion_gateway = _OrionGateway()
    use_case = TogglePredictionUseCase(
        model_repository=_ModelRepository(model),
        training_job_repository=training_repo,
        iot_agent_gateway=iot_gateway,
        orion_gateway=orion_gateway,
        fiware_service="smart",
        fiware_service_path="/",
        forecast_service_group="group",
        forecast_service_apikey="apikey",
        forecast_service_resource="/forecast",
        forecast_entity_type="Forecast",
        orion_cb_url="http://orion",
        sth_notification_url="http://sth",
        forecast_device_transport="http",
        forecast_device_protocol="proto",
    )

    await use_case._enable_predictions(
        model_id=model.id,
        training_job_id=job.id,
        model=model,
        training_job=job,
        metadata={},
    )

    assert job.prediction_config.subscription_id in orion_gateway.subscriptions
    assert job.prediction_config.metadata["device_id"] in iot_gateway.devices


@pytest.mark.asyncio
async def test_get_prediction_history_requires_model() -> None:
    model = _make_model()
    job = _make_training_job(model)
    job.prediction_config.entity_id = "urn:prediction:entity"
    training_repo = _TrainingJobRepository(job)
    history = GetPredictionHistoryUseCase(
        model_repository=_ModelRepository(None),
        training_job_repository=training_repo,
        sth_gateway=_STHGateway(total=1, points=[]),
        fiware_service="smart",
        fiware_service_path="/",
    )

    with pytest.raises(PredictionManagementError):
        await history.execute(
            model_id=model.id,
            training_job_id=job.id,
            request=PredictionHistoryRequestDTO(limit=1),
        )


def test_format_datetime_normalizes_naive_input() -> None:
    model = _make_model()
    job = _make_training_job(model)
    use_case = TogglePredictionUseCase(
        model_repository=_ModelRepository(model),
        training_job_repository=_TrainingJobRepository(job),
        iot_agent_gateway=_IoTAgentGateway(),
        orion_gateway=_OrionGateway(),
        fiware_service="smart",
        fiware_service_path="/",
        forecast_service_group="group",
        forecast_service_apikey="apikey",
        forecast_service_resource="/forecast",
        forecast_entity_type="Forecast",
        orion_cb_url="http://orion",
        sth_notification_url="http://sth",
        forecast_device_transport="http",
        forecast_device_protocol="proto",
    )

    formatted = use_case._format_datetime(datetime(2024, 1, 1))
    assert formatted.endswith("Z")
