"""
Presentation Layer - Predictions Controller

Exposes endpoints to retrieve forecasts generated from trained models.
"""

from datetime import datetime
from typing import Optional
from uuid import UUID

import structlog
from dependency_injector.wiring import Provide, inject
from fastapi import APIRouter, Depends, HTTPException, Query

from src.application.dtos.prediction_dto import (
    PredictionHistoryRequestDTO,
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
from src.main.container import AppContainer

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/models", tags=["Predictions"])


@router.post(
    "/{model_id}/training-jobs/{training_job_id}/predict",
    response_model=PredictionResponseDTO,
    summary="Generate predictions for a training job",
    description="""
    Use the artifacts created by a completed training job to forecast future points
    for the associated sensor. The prediction leverages the model's lookback window
    and returns both the recent context window and the forecast horizon.
    """,
)
@inject
async def generate_prediction(
    model_id: UUID,
    training_job_id: UUID,
    prediction_use_case: ModelPredictionUseCase = Depends(
        Provide[AppContainer.model_prediction_use_case]
    ),
) -> PredictionResponseDTO:
    try:
        return await prediction_use_case.execute(model_id, training_job_id)
    except ModelPredictionNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ModelPredictionDependencyError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    except ModelPredictionError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.error(
            "prediction.unexpected_error",
            model_id=str(model_id),
            training_job_id=str(training_job_id),
            error=str(exc),
            exc_info=exc,
        )
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post(
    "/{model_id}/training-jobs/{training_job_id}/prediction-toggle",
    response_model=PredictionToggleResponseDTO,
    summary="Enable or disable recurring predictions",
)
@inject
async def toggle_prediction(
    model_id: UUID,
    training_job_id: UUID,
    payload: PredictionToggleRequestDTO,
    toggle_use_case: TogglePredictionUseCase = Depends(
        Provide[AppContainer.toggle_prediction_use_case]
    ),
) -> PredictionToggleResponseDTO:
    try:
        return await toggle_use_case.execute(model_id, training_job_id, payload)
    except PredictionNotReadyError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except PredictionManagementError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get(
    "/{model_id}/training-jobs/{training_job_id}/predictions/history",
    response_model=PredictionHistoryResponseDTO,
    summary="Retrieve historical predictions stored in STH-Comet",
)
@inject
async def get_prediction_history(
    model_id: UUID,
    training_job_id: UUID,
    start: Optional[datetime] = Query(
        default=None, description="Start timestamp filter (ISO8601)"
    ),
    end: Optional[datetime] = Query(
        default=None, description="End timestamp filter (ISO8601)"
    ),
    limit: int = Query(
        default=100,
        ge=1,
        le=1000,
        description="Maximum number of prediction records to return",
    ),
    history_use_case: GetPredictionHistoryUseCase = Depends(
        Provide[AppContainer.get_prediction_history_use_case]
    ),
) -> PredictionHistoryResponseDTO:
    try:
        request = PredictionHistoryRequestDTO(start=start, end=end, limit=limit)
        return await history_use_case.execute(model_id, training_job_id, request)
    except PredictionManagementError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
