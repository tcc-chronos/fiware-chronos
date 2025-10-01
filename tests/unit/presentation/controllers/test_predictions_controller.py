from __future__ import annotations

from datetime import datetime, timezone
from typing import cast
from uuid import uuid4

import pytest
from fastapi import HTTPException

from src.application.dtos.prediction_dto import (
    ForecastPointDTO,
    HistoricalPointDTO,
    PredictionResponseDTO,
)
from src.application.use_cases.model_prediction_use_case import (
    ModelPredictionDependencyError,
    ModelPredictionError,
    ModelPredictionNotFoundError,
    ModelPredictionUseCase,
)
from src.presentation.controllers.predictions_controller import generate_prediction


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
