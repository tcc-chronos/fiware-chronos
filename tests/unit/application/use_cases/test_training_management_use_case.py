from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple
from uuid import UUID, uuid4

import pytest

from src.application.dtos.training_dto import TrainingRequestDTO
from src.application.use_cases.training_management_use_case import (
    TrainingManagementError,
    TrainingManagementUseCase,
)
from src.domain.entities.errors import ModelValidationError
from src.domain.entities.iot import (
    DeviceAttribute,
    IoTDevice,
    IoTDeviceCollection,
    IoTServiceGroup,
)
from src.domain.entities.model import Model, ModelStatus
from src.domain.entities.prediction import PredictionRecord
from src.domain.entities.time_series import HistoricDataPoint
from src.domain.entities.training_job import (
    DataCollectionJob,
    DataCollectionStatus,
    TrainingJob,
    TrainingMetrics,
    TrainingStatus,
)
from src.domain.gateways.iot_agent_gateway import IIoTAgentGateway
from src.domain.gateways.orion_gateway import IOrionGateway
from src.domain.gateways.sth_comet_gateway import ISTHCometGateway
from src.domain.ports.training_orchestrator import ITrainingOrchestrator
from src.domain.repositories.model_artifacts_repository import (
    IModelArtifactsRepository,
    ModelArtifact,
)
from src.domain.repositories.model_repository import IModelRepository
from src.domain.repositories.training_job_repository import ITrainingJobRepository


class _ModelRepository(IModelRepository):
    def __init__(self, model: Model):
        self.model = model

    async def find_by_id(self, model_id: UUID) -> Optional[Model]:
        if self.model.id != model_id:
            return None
        return self.model

    async def find_all(
        self,
        skip: int = 0,
        limit: int = 100,
        model_type: Optional[str] = None,
        status: Optional[str] = None,
        entity_id: Optional[str] = None,
        feature: Optional[str] = None,
    ) -> List[Model]:  # pragma: no cover - not exercised
        return []

    async def update(self, model: Model) -> Model:
        self.model = model
        return model

    async def create(self, model: Model) -> Model:  # pragma: no cover - unused
        raise NotImplementedError

    async def delete(self, model_id: UUID) -> None:  # pragma: no cover
        raise NotImplementedError


class _TrainingJobRepository(ITrainingJobRepository):
    def __init__(self) -> None:
        self.jobs: Dict[UUID, TrainingJob] = {}
        self.updated_status: List[TrainingStatus] = []
        self.task_refs: Dict[UUID, dict] = {}
        self.prediction_schedule: Dict[UUID, datetime] = {}

    async def create(self, training_job: TrainingJob) -> TrainingJob:
        training_job.id = training_job.id or uuid4()
        self.jobs[training_job.id] = training_job
        return training_job

    async def get_by_model_id(self, model_id: UUID) -> List[TrainingJob]:
        return [job for job in self.jobs.values() if job.model_id == model_id]

    async def get_by_id(self, training_job_id: UUID) -> Optional[TrainingJob]:
        return self.jobs.get(training_job_id)

    async def update(
        self, training_job: TrainingJob
    ) -> TrainingJob:  # pragma: no cover
        self.jobs[training_job.id] = training_job
        return training_job

    async def add_data_collection_job(
        self, training_job_id: UUID, job: DataCollectionJob
    ) -> bool:  # pragma: no cover
        training_job = self.jobs.get(training_job_id)
        if not training_job:
            return False
        training_job.data_collection_jobs.append(job)
        return True

    async def update_training_job_status(
        self,
        training_job_id: UUID,
        status: TrainingStatus,
        data_collection_start: Optional[datetime] = None,
        data_collection_end: Optional[datetime] = None,
        preprocessing_start: Optional[datetime] = None,
        preprocessing_end: Optional[datetime] = None,
        training_start: Optional[datetime] = None,
        training_end: Optional[datetime] = None,
        total_data_points_collected: Optional[int] = None,
        end_time: Optional[datetime] = None,
        **kwargs,
    ) -> bool:
        job = self.jobs.get(training_job_id)
        if not job:
            return False
        job.status = status
        job.updated_at = datetime.now(timezone.utc)
        self.updated_status.append(status)
        if data_collection_start:
            job.data_collection_start = data_collection_start
        if data_collection_end:
            job.data_collection_end = data_collection_end
        if preprocessing_start:
            job.preprocessing_start = preprocessing_start
        if preprocessing_end:
            job.preprocessing_end = preprocessing_end
        if training_start:
            job.training_start = training_start
        if training_end:
            job.training_end = training_end
        if total_data_points_collected is not None:
            job.total_data_points_collected = total_data_points_collected
        return True

    async def update_data_collection_job_status(
        self,
        training_job_id: UUID,
        job_id: UUID,
        status: DataCollectionStatus,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        error: Optional[str] = None,
        data_points_collected: Optional[int] = None,
        **kwargs,
    ) -> bool:
        job = self.jobs.get(training_job_id)
        if not job:
            return False
        for dc_job in job.data_collection_jobs:
            if dc_job.id == job_id:
                dc_job.status = status
                if error:
                    dc_job.error = error
                if data_points_collected is not None:
                    dc_job.data_points_collected = data_points_collected
                return True
        return False

    async def delete(self, training_job_id: UUID) -> bool:
        return self.jobs.pop(training_job_id, None) is not None

    async def update_task_refs(
        self,
        training_job_id: UUID,
        *,
        task_refs: Optional[dict] = None,
        add_data_collection_ids: Optional[List[str]] = None,
        clear: bool = False,
    ) -> bool:
        if clear:
            self.task_refs.pop(training_job_id, None)
            return True
        if task_refs:
            self.task_refs[training_job_id] = task_refs
        if add_data_collection_ids:
            self.task_refs.setdefault(training_job_id, {}).setdefault(
                "data_collection_task_ids",
                [],
            ).extend(add_data_collection_ids)
        return True

    async def fail_training_job(
        self,
        training_job_id: UUID,
        error: str,
        error_details: Optional[dict] = None,
    ) -> bool:
        job = self.jobs.get(training_job_id)
        if not job:
            return False
        job.status = TrainingStatus.FAILED
        job.error = error
        job.error_details = error_details
        return True

    async def complete_training_job(
        self,
        training_job_id: UUID,
        metrics: TrainingMetrics,
        model_artifact_id: str,
        x_scaler_artifact_id: str,
        y_scaler_artifact_id: str,
        metadata_artifact_id: str,
    ) -> bool:  # pragma: no cover
        job = self.jobs.get(training_job_id)
        if not job:
            return False
        job.status = TrainingStatus.COMPLETED
        job.metrics = metrics
        job.model_artifact_id = model_artifact_id
        job.x_scaler_artifact_id = x_scaler_artifact_id
        job.y_scaler_artifact_id = y_scaler_artifact_id
        job.metadata_artifact_id = metadata_artifact_id
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
        for job in self.jobs.values():
            if (
                job.next_prediction_at is not None
                and job.next_prediction_at <= reference_time
            ):
                ready.append(job)
        ready.sort(key=lambda job: job.next_prediction_at or datetime.max)
        return ready[:limit]


class _ArtifactsRepository(IModelArtifactsRepository):
    def __init__(
        self,
        artifacts: Optional[Dict[str, ModelArtifact]] = None,
    ) -> None:
        self.deleted: List[str] = []
        self.artifacts = artifacts or {}
        self.raise_on_get = False

    async def save_artifact(
        self,
        model_id: UUID,
        artifact_type: str,
        content: bytes,
        metadata: Optional[Dict[str, Any]] = None,
        filename: Optional[str] = None,
    ) -> str:  # pragma: no cover
        return "artifact-id"

    async def get_artifact(
        self, model_id: UUID, artifact_type: str
    ) -> Optional[ModelArtifact]:  # pragma: no cover
        return None

    async def get_artifact_by_id(self, artifact_id: str) -> Optional[ModelArtifact]:
        if self.raise_on_get:
            raise RuntimeError("boom")
        return self.artifacts.get(artifact_id)

    async def delete_artifact(self, artifact_id: str) -> bool:
        self.deleted.append(artifact_id)
        return True

    async def delete_model_artifacts(self, model_id: UUID) -> int:  # pragma: no cover
        return 0

    async def list_model_artifacts(
        self, model_id: UUID
    ) -> Dict[str, str]:  # pragma: no cover
        return {}


class _STHGateway(ISTHCometGateway):
    def __init__(self, total: int = 1000):
        self.total = total

    async def collect_data(
        self,
        entity_type: str,
        entity_id: str,
        attribute: str,
        h_limit: int,
        h_offset: int,
        fiware_service: str = "smart",
        fiware_servicepath: str = "/",
    ) -> List[HistoricDataPoint]:  # pragma: no cover
        return []

    async def get_total_count_from_header(
        self,
        entity_type: str,
        entity_id: str,
        attribute: str,
        fiware_service: str = "smart",
        fiware_servicepath: str = "/",
    ) -> int:
        return self.total


class _TrainingOrchestrator(ITrainingOrchestrator):
    def __init__(self) -> None:
        self.dispatched: List[dict] = []
        self.revocations: List[Sequence[str]] = []
        self.cleanup: List[tuple[UUID, int]] = []

    async def dispatch_training_job(
        self,
        *,
        training_job_id: UUID,
        model_id: UUID,
        last_n: int,
    ) -> str:
        self.dispatched.append(
            {
                "training_job_id": training_job_id,
                "model_id": model_id,
                "last_n": last_n,
            }
        )
        return "task-id"

    async def revoke_tasks(self, task_ids: Sequence[str]) -> None:
        self.revocations.append(task_ids)

    async def schedule_cleanup(
        self, training_job_id: UUID, countdown_seconds: int = 60
    ) -> None:
        self.cleanup.append((training_job_id, countdown_seconds))


class _IoTAgentGateway(IIoTAgentGateway):
    def __init__(self) -> None:
        self.devices: Dict[str, IoTDevice] = {}
        self.service_groups: Dict[Tuple[str, str], IoTServiceGroup] = {}
        self.deleted_devices: List[str] = []

    async def get_devices(
        self, service: str = "smart", service_path: str = "/"
    ) -> IoTDeviceCollection:
        devices = [
            device
            for device in self.devices.values()
            if device.service == service and device.service_path == service_path
        ]
        return IoTDeviceCollection(count=len(devices), devices=devices)

    async def get_service_groups(
        self, service: str = "smart", service_path: str = "/"
    ) -> List[IoTServiceGroup]:
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
        self.deleted_entities: List[str] = []
        self.deleted_subscriptions: List[str] = []

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
        prediction: PredictionRecord,
        *,
        service: str,
        service_path: str,
    ) -> None:
        self.entities[prediction.entity_id] = {
            "prediction": prediction,
            "service": service,
            "service_path": service_path,
        }

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
        subscription_id = f"sub-{len(self.subscriptions)+1}"
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


@pytest.fixture()
def model(sample_model):
    model = sample_model
    model.status = ModelStatus.DRAFT
    model.has_successful_training = False
    return model


@pytest.fixture()
def use_case(model):
    repo = _ModelRepository(model)
    training_repo = _TrainingJobRepository()
    artifacts = _ArtifactsRepository()
    gateway = _STHGateway(total=2000)
    orchestrator = _TrainingOrchestrator()
    iot_gateway = _IoTAgentGateway()
    orion_gateway = _OrionGateway()
    use_case = TrainingManagementUseCase(
        training_job_repository=training_repo,
        model_repository=repo,
        artifacts_repository=artifacts,
        sth_gateway=gateway,
        training_orchestrator=orchestrator,
        iot_agent_gateway=iot_gateway,
        orion_gateway=orion_gateway,
        fiware_service="smart",
        fiware_service_path="/",
    )
    return use_case, repo, training_repo, artifacts, gateway, orchestrator


@pytest.mark.asyncio
async def test_start_training_creates_job(use_case, model):
    use_case_obj, repo, training_repo, _, gateway, orchestrator = use_case

    response = await use_case_obj.start_training(
        model_id=model.id,
        request=TrainingRequestDTO(last_n=500),
    )

    assert response.status is TrainingStatus.PENDING
    assert orchestrator.dispatched
    assert response.training_job_id in training_repo.jobs
    assert training_repo.jobs[response.training_job_id].model_id == model.id
    assert repo.model.status is ModelStatus.TRAINING


@pytest.mark.asyncio
async def test_start_training_validates_availability(use_case, model):
    use_case_obj, repo, _, _, gateway, _ = use_case
    gateway.total = 10

    with pytest.raises(TrainingManagementError):
        await use_case_obj.start_training(model.id, TrainingRequestDTO(last_n=500))


@pytest.mark.asyncio
async def test_cancel_training_job_updates_status(use_case, model):
    use_case_obj, repo, training_repo, _, _, orchestrator = use_case
    now = datetime.now(timezone.utc)
    job = TrainingJob(
        id=uuid4(),
        model_id=model.id,
        status=TrainingStatus.TRAINING,
        data_collection_jobs=[
            DataCollectionJob(
                id=uuid4(),
                status=DataCollectionStatus.IN_PROGRESS,
            )
        ],
        task_refs={
            "orchestration_task_id": "task-1",
            "data_collection_task_ids": ["dc-1"],
        },
        data_collection_start=now,
        preprocessing_start=now,
        training_start=now,
    )
    training_repo.jobs[job.id] = job

    cancelled = await use_case_obj.cancel_training_job(model.id, job.id)

    assert cancelled is True
    assert len(orchestrator.revocations) == 1
    assert set(orchestrator.revocations[0]) == {"task-1", "dc-1"}
    assert repo.model.status in {ModelStatus.DRAFT, ModelStatus.TRAINED}


@pytest.mark.asyncio
async def test_cancel_training_job_returns_false_for_missing(use_case, model):
    use_case_obj, *_ = use_case

    result = await use_case_obj.cancel_training_job(model.id, uuid4())
    assert result is False


@pytest.mark.asyncio
async def test_training_metadata_history_surface_in_dto_and_summary(use_case, model):
    use_case_obj, _, training_repo, artifacts, _, _ = use_case

    metadata_id = "meta-history"
    metadata_payload = {
        "training_history": {
            "best_epoch": 8,
            "epochs_trained": 12,
        }
    }
    artifacts.artifacts[metadata_id] = ModelArtifact(
        metadata_id,
        "metadata",
        json.dumps(metadata_payload).encode("utf-8"),
    )

    job = TrainingJob(
        id=uuid4(),
        model_id=model.id,
        status=TrainingStatus.COMPLETED,
        metadata_artifact_id=metadata_id,
    )
    training_repo.jobs[job.id] = job

    dto = await use_case_obj.get_training_job(model.id, job.id)
    assert dto is not None
    assert dto.metadata_artifact_id == metadata_id
    assert dto.training_history == metadata_payload["training_history"]

    summaries = await use_case_obj.list_training_jobs_by_model(model.id)
    assert summaries
    assert summaries[0].metadata_artifact_id == metadata_id
    assert summaries[0].training_history == metadata_payload["training_history"]


@pytest.mark.asyncio
async def test_training_metadata_invalid_json_returns_none(use_case, model):
    use_case_obj, _, training_repo, artifacts, _, _ = use_case

    metadata_id = "meta-invalid"
    artifacts.artifacts[metadata_id] = ModelArtifact(
        metadata_id,
        "metadata",
        b"{not-json",
    )

    job = TrainingJob(
        id=uuid4(),
        model_id=model.id,
        status=TrainingStatus.COMPLETED,
        metadata_artifact_id=metadata_id,
    )
    training_repo.jobs[job.id] = job

    dto = await use_case_obj.get_training_job(model.id, job.id)
    assert dto is not None
    assert dto.training_history is None

    summary = (await use_case_obj.list_training_jobs_by_model(model.id))[0]
    assert summary.training_history is None


@pytest.mark.asyncio
async def test_training_metadata_fetch_failure_returns_none(use_case, model):
    use_case_obj, _, training_repo, artifacts, _, _ = use_case

    metadata_id = "meta-missing"
    artifacts.raise_on_get = True

    job = TrainingJob(
        id=uuid4(),
        model_id=model.id,
        status=TrainingStatus.COMPLETED,
        metadata_artifact_id=metadata_id,
    )
    training_repo.jobs[job.id] = job

    dto = await use_case_obj.get_training_job(model.id, job.id)
    assert dto is not None
    assert dto.training_history is None

    summary = (await use_case_obj.list_training_jobs_by_model(model.id))[0]
    assert summary.training_history is None


@pytest.mark.asyncio
async def test_cancel_training_job_rejects_completed(use_case, model):
    use_case_obj, _, training_repo, _, _, _ = use_case
    job = TrainingJob(id=uuid4(), model_id=model.id, status=TrainingStatus.COMPLETED)
    training_repo.jobs[job.id] = job

    with pytest.raises(TrainingManagementError):
        await use_case_obj.cancel_training_job(model.id, job.id)


@pytest.mark.asyncio
async def test_delete_training_job_removes_artifacts(use_case, model):
    use_case_obj, repo, training_repo, artifacts, _, _ = use_case
    job = TrainingJob(
        id=uuid4(),
        model_id=model.id,
        status=TrainingStatus.COMPLETED,
        model_artifact_id="model-art",
        x_scaler_artifact_id="x-art",
        y_scaler_artifact_id="y-art",
        metadata_artifact_id="meta-art",
    )
    training_repo.jobs[job.id] = job

    deleted = await use_case_obj.delete_training_job(model.id, job.id)

    assert deleted is True
    assert set(artifacts.deleted) == {"model-art", "x-art", "y-art", "meta-art"}
    assert repo.model.status is ModelStatus.DRAFT


@pytest.mark.asyncio
async def test_delete_training_job_skips_missing_artifacts(use_case, model):
    use_case_obj, _, training_repo, artifacts, _, _ = use_case
    job = TrainingJob(
        id=uuid4(),
        model_id=model.id,
        status=TrainingStatus.COMPLETED,
        model_artifact_id=None,
        x_scaler_artifact_id="x-art",
        y_scaler_artifact_id=None,
        metadata_artifact_id="meta-art",
    )
    training_repo.jobs[job.id] = job

    deleted = await use_case_obj.delete_training_job(model.id, job.id)

    assert deleted is True
    assert set(artifacts.deleted) == {"x-art", "meta-art"}


@pytest.mark.asyncio
async def test_delete_training_job_warns_when_artifact_missing(use_case, model):
    class _Artifacts(_ArtifactsRepository):
        async def delete_artifact(self, artifact_id: str) -> bool:
            self.deleted.append(artifact_id)
            return False

    use_case_obj, _, training_repo, _, _, _ = use_case
    artifacts = _Artifacts()
    use_case_obj.artifacts_repository = artifacts

    job = TrainingJob(
        id=uuid4(),
        model_id=model.id,
        status=TrainingStatus.COMPLETED,
        model_artifact_id="model-art",
    )
    training_repo.jobs[job.id] = job

    deleted = await use_case_obj.delete_training_job(model.id, job.id)

    assert deleted is True
    assert artifacts.deleted == ["model-art"]


@pytest.mark.asyncio
async def test_delete_training_job_reverts_when_remaining_failed(use_case, model):
    use_case_obj, repo, training_repo, artifacts, _, _ = use_case
    job_success = TrainingJob(
        id=uuid4(),
        model_id=model.id,
        status=TrainingStatus.COMPLETED,
        model_artifact_id="model-art",
    )
    job_failed = TrainingJob(
        id=uuid4(),
        model_id=model.id,
        status=TrainingStatus.FAILED,
    )
    training_repo.jobs[job_success.id] = job_success
    training_repo.jobs[job_failed.id] = job_failed

    await use_case_obj.delete_training_job(model.id, job_success.id)

    assert repo.model.status is ModelStatus.DRAFT
    assert artifacts.deleted == ["model-art"]


@pytest.mark.asyncio
async def test_delete_training_job_handles_repository_failure(
    monkeypatch, use_case, model
):
    use_case_obj, _, training_repo, _, _, _ = use_case
    job = TrainingJob(id=uuid4(), model_id=model.id, status=TrainingStatus.COMPLETED)
    training_repo.jobs[job.id] = job

    async def _fail_delete(training_job_id: UUID) -> bool:
        raise RuntimeError("db down")

    monkeypatch.setattr(training_repo, "delete", _fail_delete)

    with pytest.raises(TrainingManagementError):
        await use_case_obj.delete_training_job(model.id, job.id)


@pytest.mark.asyncio
async def test_get_training_job_returns_dto(use_case, model):
    use_case_obj, _, training_repo, _, _, _ = use_case
    job = TrainingJob(
        id=uuid4(),
        model_id=model.id,
        status=TrainingStatus.COMPLETED,
        metrics=TrainingMetrics(
            mae=0.1, mse=0.2, rmse=0.3, theil_u=0.4, mape=0.5, r2=0.6
        ),
    )
    training_repo.jobs[job.id] = job

    dto = await use_case_obj.get_training_job(model.id, job.id)
    assert dto is not None
    assert dto.status is TrainingStatus.COMPLETED


@pytest.mark.asyncio
async def test_get_training_job_returns_none_for_other_model(use_case, model):
    use_case_obj, _, training_repo, _, _, _ = use_case
    other_model = uuid4()
    job = TrainingJob(id=uuid4(), model_id=other_model, status=TrainingStatus.COMPLETED)
    training_repo.jobs[job.id] = job

    result = await use_case_obj.get_training_job(model.id, job.id)
    assert result is None


@pytest.mark.asyncio
async def test_list_training_jobs_returns_summaries(use_case, model):
    use_case_obj, _, training_repo, _, _, _ = use_case
    job = TrainingJob(id=uuid4(), model_id=model.id, status=TrainingStatus.FAILED)
    training_repo.jobs[job.id] = job

    summaries = await use_case_obj.list_training_jobs_by_model(model.id)
    assert len(summaries) == 1
    assert summaries[0].status is TrainingStatus.FAILED


@pytest.mark.asyncio
async def test_start_training_requires_existing_model(use_case):
    use_case_obj, *_ = use_case

    with pytest.raises(TrainingManagementError):
        await use_case_obj.start_training(uuid4(), TrainingRequestDTO(last_n=100))


@pytest.mark.asyncio
async def test_start_training_requires_entity_configuration(use_case, model):
    use_case_obj, repo, *_ = use_case
    model.entity_type = ""

    with pytest.raises(TrainingManagementError) as exc:
        await use_case_obj.start_training(model.id, TrainingRequestDTO(last_n=500))

    assert "entity_type" in str(exc.value)
    assert repo.model.status is ModelStatus.DRAFT


@pytest.mark.asyncio
async def test_start_training_validates_model_structure(monkeypatch, use_case, model):
    use_case_obj, repo, *_ = use_case

    def _raise_validation(config):
        raise ModelValidationError("bad")

    monkeypatch.setattr(
        "src.application.use_cases.training_management_use_case"
        ".validate_model_configuration",
        _raise_validation,
    )

    with pytest.raises(TrainingManagementError) as exc:
        await use_case_obj.start_training(model.id, TrainingRequestDTO(last_n=500))

    assert "Model configuration invalid" in str(exc.value)
    assert repo.model.status is ModelStatus.DRAFT


@pytest.mark.asyncio
async def test_start_training_requires_train_ratio(use_case, model):
    use_case_obj, *_ = use_case
    model.validation_ratio = 0.6
    model.test_ratio = 0.5

    with pytest.raises(TrainingManagementError):
        await use_case_obj.start_training(model.id, TrainingRequestDTO(last_n=500))


@pytest.mark.asyncio
async def test_start_training_enforces_minimum_points(use_case, model):
    use_case_obj, *_ = use_case

    with pytest.raises(TrainingManagementError):
        await use_case_obj.start_training(model.id, TrainingRequestDTO(last_n=5))


@pytest.mark.asyncio
async def test_start_training_handles_sth_gateway_failure(monkeypatch, use_case, model):
    use_case_obj, _, _, _, gateway, _ = use_case

    async def _fail(**kwargs):
        raise RuntimeError("sth down")

    monkeypatch.setattr(gateway, "get_total_count_from_header", _fail)

    with pytest.raises(TrainingManagementError) as exc:
        await use_case_obj.start_training(model.id, TrainingRequestDTO(last_n=500))

    assert "Failed to validate" in str(exc.value)


@pytest.mark.asyncio
async def test_start_training_requires_available_history(use_case, model):
    use_case_obj, _, _, _, gateway, _ = use_case
    gateway.total = 0

    with pytest.raises(TrainingManagementError):
        await use_case_obj.start_training(model.id, TrainingRequestDTO(last_n=500))


@pytest.mark.asyncio
async def test_start_training_insufficient_historical_data(use_case, model):
    use_case_obj, _, _, _, gateway, _ = use_case
    gateway.total = model.lookback_window + model.forecast_horizon + 1

    with pytest.raises(TrainingManagementError):
        await use_case_obj.start_training(
            model.id, TrainingRequestDTO(last_n=gateway.total)
        )


@pytest.mark.asyncio
async def test_start_training_rejects_request_beyond_available(use_case, model):
    use_case_obj, _, _, _, gateway, _ = use_case
    gateway.total = 100

    with pytest.raises(TrainingManagementError):
        await use_case_obj.start_training(model.id, TrainingRequestDTO(last_n=200))


@pytest.mark.asyncio
async def test_start_training_detects_running_jobs(use_case, model):
    use_case_obj, _, training_repo, _, _, _ = use_case
    training_repo.jobs[uuid4()] = TrainingJob(
        id=uuid4(),
        model_id=model.id,
        status=TrainingStatus.PENDING,
    )

    with pytest.raises(TrainingManagementError):
        await use_case_obj.start_training(model.id, TrainingRequestDTO(last_n=500))


@pytest.mark.asyncio
async def test_start_training_rolls_back_status_on_error(monkeypatch, use_case, model):
    use_case_obj, repo, training_repo, _, _, _ = use_case
    model.status = ModelStatus.TRAINED
    model.has_successful_training = True

    async def _fail_create(training_job):
        raise RuntimeError("db down")

    monkeypatch.setattr(training_repo, "create", _fail_create)

    with pytest.raises(TrainingManagementError) as exc:
        await use_case_obj.start_training(model.id, TrainingRequestDTO(last_n=500))

    assert "Failed to start training" in str(exc.value)
    assert repo.model.status is ModelStatus.TRAINED


@pytest.mark.asyncio
async def test_cancel_training_job_wraps_repository_errors(
    monkeypatch, use_case, model
):
    use_case_obj, _, training_repo, _, _, _ = use_case

    async def _raise(job_id):
        raise RuntimeError("boom")

    monkeypatch.setattr(training_repo, "get_by_id", _raise)

    with pytest.raises(TrainingManagementError):
        await use_case_obj.cancel_training_job(model.id, uuid4())


@pytest.mark.asyncio
async def test_delete_training_job_returns_false_for_missing(use_case, model):
    use_case_obj, *_ = use_case

    result = await use_case_obj.delete_training_job(model.id, uuid4())
    assert result is False


@pytest.mark.asyncio
async def test_delete_training_job_rejects_running_job(use_case, model):
    use_case_obj, _, training_repo, _, _, _ = use_case
    job = TrainingJob(
        id=uuid4(),
        model_id=model.id,
        status=TrainingStatus.TRAINING,
    )
    training_repo.jobs[job.id] = job

    with pytest.raises(TrainingManagementError):
        await use_case_obj.delete_training_job(model.id, job.id)


@pytest.mark.asyncio
async def test_delete_training_job_handles_artifact_errors(
    monkeypatch, use_case, model
):
    use_case_obj, _, training_repo, artifacts, _, _ = use_case
    job = TrainingJob(
        id=uuid4(),
        model_id=model.id,
        status=TrainingStatus.COMPLETED,
        model_artifact_id="artifact",
    )
    training_repo.jobs[job.id] = job

    async def _fail(artifact_id):
        raise RuntimeError("storage down")

    monkeypatch.setattr(artifacts, "delete_artifact", _fail)

    with pytest.raises(TrainingManagementError):
        await use_case_obj.delete_training_job(model.id, job.id)


@pytest.mark.asyncio
async def test_get_training_job_wraps_errors(monkeypatch, use_case, model):
    use_case_obj, _, training_repo, _, _, _ = use_case

    async def _boom(job_id):
        raise RuntimeError("db")

    monkeypatch.setattr(training_repo, "get_by_id", _boom)

    with pytest.raises(TrainingManagementError):
        await use_case_obj.get_training_job(model.id, uuid4())


@pytest.mark.asyncio
async def test_list_training_jobs_wraps_errors(monkeypatch, use_case, model):
    use_case_obj, _, training_repo, _, _, _ = use_case

    async def _boom(model_id):
        raise RuntimeError("db")

    monkeypatch.setattr(training_repo, "get_by_model_id", _boom)

    with pytest.raises(TrainingManagementError):
        await use_case_obj.list_training_jobs_by_model(model.id)
