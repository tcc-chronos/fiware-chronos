from __future__ import annotations

from datetime import datetime, timezone
from typing import cast
from uuid import uuid4

import pytest
from fastapi import HTTPException

from src.application.dtos.prediction_dto import (
    ForecastPointDTO,
    HistoricalPointDTO,
    PredictionHistoryPointDTO,
    PredictionHistoryResponseDTO,
    PredictionResponseDTO,
    PredictionToggleRequestDTO,
    PredictionToggleResponseDTO,
)
from src.application.use_cases.model_prediction_use_case import (
    ModelPredictionDependencyError,
    ModelPredictionError,
    ModelPredictionNotFoundError,
    ModelPredictionUseCase,
)
from src.application.use_cases.prediction_management_use_case import (
    GetPredictionHistoryUseCase,
    PredictionManagementError,
    PredictionNotReadyError,
    TogglePredictionUseCase,
)
from src.presentation.controllers.predictions_controller import (
    generate_prediction,
    get_prediction_history,
    toggle_prediction,
)


class _StubPredictionUseCase:
    def __init__(self):
        now = datetime.now(timezone.utc)
        self.response = PredictionResponseDTO(
            model_id=uuid4(),
            training_job_id=uuid4(),
            lookback_window=3,
            forecast_horizon=1,
            generated_at=now,
            context_window=[
                HistoricalPointDTO(timestamp=now, value=1.0),
                HistoricalPointDTO(timestamp=now, value=2.0),
                HistoricalPointDTO(timestamp=now, value=3.0),
            ],
            predictions=[ForecastPointDTO(step=1, value=4.0, timestamp=now)],
            metadata={},
        )

    async def execute(self, model_id, training_job_id) -> PredictionResponseDTO:
        return self.response


def _as_prediction_use_case(use_case: _StubPredictionUseCase) -> ModelPredictionUseCase:
    return cast(ModelPredictionUseCase, use_case)


class _StubToggleUseCase:
    def __init__(self):
        now = datetime.now(timezone.utc)
        self.response = PredictionToggleResponseDTO(
            model_id=uuid4(),
            training_job_id=uuid4(),
            enabled=True,
            entity_id="urn:entity",
            next_prediction_at=now,
            sampling_interval_seconds=300,
        )

    async def execute(
        self,
        model_id,
        training_job_id,
        payload: PredictionToggleRequestDTO,
    ) -> PredictionToggleResponseDTO:
        return self.response


def _as_toggle_use_case(use_case: _StubToggleUseCase) -> TogglePredictionUseCase:
    return cast(TogglePredictionUseCase, use_case)


class _StubHistoryUseCase:
    def __init__(self):
        point = PredictionHistoryPointDTO(
            timestamp=datetime.now(timezone.utc),
            value=1.23,
        )
        self.response = PredictionHistoryResponseDTO(
            entity_id="entity",
            feature="temperature",
            points=[point],
        )

    async def execute(self, model_id, training_job_id, request):
        return self.response


def _as_history_use_case(use_case: _StubHistoryUseCase) -> GetPredictionHistoryUseCase:
    return cast(GetPredictionHistoryUseCase, use_case)


@pytest.mark.asyncio
async def test_generate_prediction_returns_response():
    stub = _StubPredictionUseCase()
    response = await generate_prediction(
        model_id=uuid4(),
        training_job_id=uuid4(),
        prediction_use_case=_as_prediction_use_case(stub),
    )
    assert response is stub.response


@pytest.mark.asyncio
async def test_generate_prediction_handles_not_found():
    class _NotFound(_StubPredictionUseCase):
        async def execute(self, model_id, training_job_id):
            raise ModelPredictionNotFoundError("missing")

    with pytest.raises(HTTPException) as exc:
        await generate_prediction(
            model_id=uuid4(),
            training_job_id=uuid4(),
            prediction_use_case=_as_prediction_use_case(_NotFound()),
        )
    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_generate_prediction_handles_dependency_error():
    class _Dependency(_StubPredictionUseCase):
        async def execute(self, model_id, training_job_id):
            raise ModelPredictionDependencyError("sth")

    with pytest.raises(HTTPException) as exc:
        await generate_prediction(
            model_id=uuid4(),
            training_job_id=uuid4(),
            prediction_use_case=_as_prediction_use_case(_Dependency()),
        )
    assert exc.value.status_code == 502


@pytest.mark.asyncio
async def test_generate_prediction_handles_generic_error():
    class _Generic(_StubPredictionUseCase):
        async def execute(self, model_id, training_job_id):
            raise ModelPredictionError("invalid")

    with pytest.raises(HTTPException) as exc:
        await generate_prediction(
            model_id=uuid4(),
            training_job_id=uuid4(),
            prediction_use_case=_as_prediction_use_case(_Generic()),
        )
    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_toggle_prediction_returns_response():
    stub = _StubToggleUseCase()
    payload = PredictionToggleRequestDTO(enabled=True)

    response = await toggle_prediction(
        model_id=uuid4(),
        training_job_id=uuid4(),
        payload=payload,
        toggle_use_case=_as_toggle_use_case(stub),
    )

    assert response is stub.response


@pytest.mark.asyncio
async def test_toggle_prediction_handles_not_ready():
    class _NotReady(_StubToggleUseCase):
        async def execute(self, model_id, training_job_id, payload):
            raise PredictionNotReadyError("collecting data")

    with pytest.raises(HTTPException) as exc:
        await toggle_prediction(
            model_id=uuid4(),
            training_job_id=uuid4(),
            payload=PredictionToggleRequestDTO(enabled=False),
            toggle_use_case=_as_toggle_use_case(_NotReady()),
        )
    assert exc.value.status_code == 409


@pytest.mark.asyncio
async def test_toggle_prediction_handles_management_error():
    class _ManagementError(_StubToggleUseCase):
        async def execute(self, model_id, training_job_id, payload):
            raise PredictionManagementError("invalid state")

    with pytest.raises(HTTPException) as exc:
        await toggle_prediction(
            model_id=uuid4(),
            training_job_id=uuid4(),
            payload=PredictionToggleRequestDTO(enabled=True),
            toggle_use_case=_as_toggle_use_case(_ManagementError()),
        )
    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_get_prediction_history_returns_response():
    stub = _StubHistoryUseCase()
    start = datetime.now(timezone.utc)
    end = start

    response = await get_prediction_history(
        model_id=uuid4(),
        training_job_id=uuid4(),
        start=start,
        end=end,
        limit=50,
        history_use_case=_as_history_use_case(stub),
    )

    assert response is stub.response


@pytest.mark.asyncio
async def test_get_prediction_history_handles_management_error():
    class _HistoryError(_StubHistoryUseCase):
        async def execute(self, model_id, training_job_id, request):
            raise PredictionManagementError("bad filters")

    with pytest.raises(HTTPException) as exc:
        await get_prediction_history(
            model_id=uuid4(),
            training_job_id=uuid4(),
            start=None,
            end=None,
            limit=10,
            history_use_case=_as_history_use_case(_HistoryError()),
        )

    assert exc.value.status_code == 400
