from __future__ import annotations

from datetime import datetime, timezone
from typing import cast
from uuid import uuid4

import pytest
from fastapi import HTTPException

from src.application.dtos.training_dto import (
    StartTrainingResponseDTO,
    TrainingJobDTO,
    TrainingJobSummaryDTO,
    TrainingRequestDTO,
)
from src.application.use_cases.training_management_use_case import (
    TrainingManagementError,
    TrainingManagementUseCase,
)
from src.domain.entities.training_job import TrainingStatus
from src.presentation.controllers.training_controller import (
    cancel_training_job,
    delete_training_job,
    get_training_job,
    list_model_training_jobs,
    start_training,
)


class _StubTrainingUseCase:
    def __init__(self):
        now = datetime.now(timezone.utc)
        self.list_response = [
            TrainingJobSummaryDTO(
                id=uuid4(),
                model_id=uuid4(),
                status=TrainingStatus.COMPLETED,
                data_collection_progress=100.0,
                start_time=None,
                end_time=None,
                error=None,
                created_at=now,
            )
        ]
        self.job_dto = TrainingJobDTO(
            id=uuid4(),
            model_id=uuid4(),
            status=TrainingStatus.PENDING,
            last_n=100,
            data_collection_jobs=[],
            total_data_points_requested=0,
            total_data_points_collected=0,
            data_collection_progress=0.0,
            start_time=None,
            end_time=None,
            data_collection_start=None,
            data_collection_end=None,
            preprocessing_start=None,
            preprocessing_end=None,
            training_start=None,
            training_end=None,
            metrics=None,
            model_artifact_id=None,
            error=None,
            error_details=None,
            created_at=now,
            updated_at=now,
            total_duration_seconds=None,
            training_duration_seconds=None,
        )

    async def list_training_jobs_by_model(self, model_id, skip=0, limit=100):
        return self.list_response

    async def get_training_job(
        self, model_id, training_job_id
    ) -> TrainingJobDTO | None:
        return self.job_dto

    async def start_training(self, model_id, request: TrainingRequestDTO):
        return StartTrainingResponseDTO(training_job_id=uuid4())

    async def cancel_training_job(self, model_id, training_job_id) -> bool:
        return True

    async def delete_training_job(self, model_id, training_job_id) -> bool:
        return True


def _as_use_case(use_case: _StubTrainingUseCase) -> TrainingManagementUseCase:
    return cast(TrainingManagementUseCase, use_case)


@pytest.mark.asyncio
async def test_list_training_jobs_returns_data():
    response = await list_model_training_jobs(
        model_id=uuid4(),
        skip=0,
        limit=10,
        training_use_case=_as_use_case(_StubTrainingUseCase()),
    )
    assert len(response) == 1
    assert response[0].status is TrainingStatus.COMPLETED


@pytest.mark.asyncio
async def test_list_training_jobs_handles_unexpected_error():
    class _Fail(_StubTrainingUseCase):
        async def list_training_jobs_by_model(self, model_id, skip=0, limit=100):
            raise RuntimeError("boom")

    with pytest.raises(HTTPException) as exc:
        await list_model_training_jobs(
            model_id=uuid4(),
            skip=0,
            limit=10,
            training_use_case=_as_use_case(_Fail()),
        )
    assert exc.value.status_code == 500


@pytest.mark.asyncio
async def test_get_training_job_not_found():
    class _Missing(_StubTrainingUseCase):
        async def get_training_job(self, model_id, training_job_id) -> TrainingJobDTO:
            return cast(TrainingJobDTO, None)

    with pytest.raises(HTTPException) as exc:
        await get_training_job(
            model_id=uuid4(),
            training_job_id=uuid4(),
            training_use_case=_as_use_case(_Missing()),
        )
    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_get_training_job_handles_unexpected_error():
    class _Fail(_StubTrainingUseCase):
        async def get_training_job(self, model_id, training_job_id):
            raise RuntimeError("boom")

    with pytest.raises(HTTPException) as exc:
        await get_training_job(
            model_id=uuid4(),
            training_job_id=uuid4(),
            training_use_case=_as_use_case(_Fail()),
        )
    assert exc.value.status_code == 500


@pytest.mark.asyncio
async def test_start_training_returns_response():
    stub = _StubTrainingUseCase()
    response = await start_training(
        model_id=uuid4(),
        request=TrainingRequestDTO(),
        training_use_case=_as_use_case(stub),
    )
    assert response.message


@pytest.mark.asyncio
async def test_start_training_handles_unexpected_error():
    class _Fail(_StubTrainingUseCase):
        async def start_training(self, model_id, request: TrainingRequestDTO):
            raise RuntimeError("boom")

    with pytest.raises(HTTPException) as exc:
        await start_training(
            model_id=uuid4(),
            request=TrainingRequestDTO(),
            training_use_case=_as_use_case(_Fail()),
        )
    assert exc.value.status_code == 500


@pytest.mark.asyncio
async def test_cancel_training_job_handles_errors():
    class _Fail(_StubTrainingUseCase):
        async def cancel_training_job(self, model_id, training_job_id):
            raise TrainingManagementError("cannot")

    with pytest.raises(HTTPException) as exc:
        await cancel_training_job(
            model_id=uuid4(),
            training_job_id=uuid4(),
            training_use_case=_as_use_case(_Fail()),
        )
    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_cancel_training_job_returns_not_found():
    class _Missing(_StubTrainingUseCase):
        async def cancel_training_job(self, model_id, training_job_id):
            return False

    with pytest.raises(HTTPException) as exc:
        await cancel_training_job(
            model_id=uuid4(),
            training_job_id=uuid4(),
            training_use_case=_as_use_case(_Missing()),
        )

    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_cancel_training_job_handles_unexpected_error():
    class _Fail(_StubTrainingUseCase):
        async def cancel_training_job(self, model_id, training_job_id):
            raise RuntimeError("boom")

    with pytest.raises(HTTPException) as exc:
        await cancel_training_job(
            model_id=uuid4(),
            training_job_id=uuid4(),
            training_use_case=_as_use_case(_Fail()),
        )
    assert exc.value.status_code == 500


@pytest.mark.asyncio
async def test_delete_training_job_success():
    stub = _StubTrainingUseCase()
    response = await delete_training_job(
        model_id=uuid4(),
        training_job_id=uuid4(),
        training_use_case=_as_use_case(stub),
    )
    assert "successfully" in response["message"]


@pytest.mark.asyncio
async def test_delete_training_job_returns_not_found():
    class _Missing(_StubTrainingUseCase):
        async def delete_training_job(self, model_id, training_job_id):
            return False

    with pytest.raises(HTTPException) as exc:
        await delete_training_job(
            model_id=uuid4(),
            training_job_id=uuid4(),
            training_use_case=_as_use_case(_Missing()),
        )

    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_delete_training_job_handles_management_error():
    class _Fail(_StubTrainingUseCase):
        async def delete_training_job(self, model_id, training_job_id):
            raise TrainingManagementError("cannot")

    with pytest.raises(HTTPException) as exc:
        await delete_training_job(
            model_id=uuid4(),
            training_job_id=uuid4(),
            training_use_case=_as_use_case(_Fail()),
        )
    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_delete_training_job_handles_unexpected_error():
    class _Fail(_StubTrainingUseCase):
        async def delete_training_job(self, model_id, training_job_id):
            raise RuntimeError("boom")

    with pytest.raises(HTTPException) as exc:
        await delete_training_job(
            model_id=uuid4(),
            training_job_id=uuid4(),
            training_use_case=_as_use_case(_Fail()),
        )
    assert exc.value.status_code == 500
