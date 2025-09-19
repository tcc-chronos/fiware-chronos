"""
Presentation Layer - Training Controller

This module contains the FastAPI controller for training operations.
"""

from typing import List, Optional
from uuid import UUID

import structlog
from dependency_injector.wiring import Provide, inject
from fastapi import APIRouter, Depends, HTTPException, Query

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
from src.main.container import AppContainer

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/models", tags=["training"])


@router.post(
    "/{model_id}/train",
    response_model=StartTrainingResponseDTO,
    summary="Start model training",
    description="""
    Start training a deep learning model with data collected from STH-Comet.

    The training process includes:
    1. Data collection from STH-Comet (parallelized for large datasets)
    2. Data preprocessing and sequence creation
    3. Model training with configured hyperparameters
    4. Model evaluation and artifact storage

    The process is asynchronous and returns a training job ID for status tracking.
    """,
)
@inject
async def start_training(
    model_id: UUID,
    request: TrainingRequestDTO,
    training_use_case: TrainingManagementUseCase = Depends(
        Provide[AppContainer.training_management_use_case]
    ),
) -> StartTrainingResponseDTO:
    """Start training a model."""
    try:
        logger.info(
            "Starting model training", model_id=str(model_id), last_n=request.last_n
        )

        result = await training_use_case.start_training(model_id, request)

        logger.info(
            "Training started successfully",
            model_id=str(model_id),
            training_job_id=str(result.training_job_id),
        )

        return result

    except TrainingManagementError as e:
        logger.error("Training start failed", model_id=str(model_id), error=str(e))
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        logger.error(
            "Unexpected error starting training", model_id=str(model_id), error=str(e)
        )
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get(
    "/training-jobs/{training_job_id}",
    response_model=TrainingJobDTO,
    summary="Get training job details",
    description="""
    Get detailed information about a specific training job including:
    - Current status and progress
    - Data collection jobs and their status
    - Training metrics (if completed)
    - Error information (if failed)
    - Timing information for each phase
    """,
)
@inject
async def get_training_job(
    training_job_id: UUID,
    training_use_case: TrainingManagementUseCase = Depends(
        Provide[AppContainer.training_management_use_case]
    ),
) -> TrainingJobDTO:
    """Get training job details."""
    try:
        result = await training_use_case.get_training_job(training_job_id)

        if not result:
            raise HTTPException(
                status_code=404, detail=f"Training job {training_job_id} not found"
            )

        return result

    except HTTPException:
        raise

    except Exception as e:
        logger.error(
            "Unexpected error getting training job",
            training_job_id=str(training_job_id),
            error=str(e),
        )
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get(
    "/training-jobs",
    response_model=List[TrainingJobSummaryDTO],
    summary="List training jobs",
    description="""
    List training jobs with optional filtering by model ID.
    Returns a summary of each training job including status and progress.
    """,
)
@inject
async def list_training_jobs(
    model_id: Optional[UUID] = Query(None, description="Filter by model ID"),
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(
        100, ge=1, le=1000, description="Maximum number of records to return"
    ),
    training_use_case: TrainingManagementUseCase = Depends(
        Provide[AppContainer.training_management_use_case]
    ),
) -> List[TrainingJobSummaryDTO]:
    """List training jobs."""
    try:
        result = await training_use_case.list_training_jobs(
            model_id=model_id, skip=skip, limit=limit
        )

        return result

    except Exception as e:
        logger.error(
            "Unexpected error listing training jobs",
            model_id=str(model_id) if model_id else None,
            error=str(e),
        )
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get(
    "/{model_id}/training-jobs",
    response_model=List[TrainingJobSummaryDTO],
    summary="List training jobs for a model",
    description="""
    List all training jobs for a specific model.
    Returns training jobs sorted by creation date (newest first).
    """,
)
@inject
async def list_model_training_jobs(
    model_id: UUID,
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(
        100, ge=1, le=1000, description="Maximum number of records to return"
    ),
    training_use_case: TrainingManagementUseCase = Depends(
        Provide[AppContainer.training_management_use_case]
    ),
) -> List[TrainingJobSummaryDTO]:
    """List training jobs for a specific model."""
    try:
        result = await training_use_case.list_training_jobs(
            model_id=model_id, skip=skip, limit=limit
        )

        return result

    except Exception as e:
        logger.error(
            "Unexpected error listing model training jobs",
            model_id=str(model_id),
            error=str(e),
        )
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post(
    "/training-jobs/{training_job_id}/cancel",
    summary="Cancel training job",
    description="""
    Cancel a running training job.
    Only jobs in pending, collecting_data, preprocessing, or training
    status can be cancelled.
    """,
)
@inject
async def cancel_training_job(
    training_job_id: UUID,
    training_use_case: TrainingManagementUseCase = Depends(
        Provide[AppContainer.training_management_use_case]
    ),
) -> dict:
    """Cancel a training job."""
    try:
        success = await training_use_case.cancel_training_job(training_job_id)

        if success:
            return {"message": f"Training job {training_job_id} cancelled successfully"}
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to cancel training job {training_job_id}",
            )

    except TrainingManagementError as e:
        logger.error(
            "Training cancellation failed",
            training_job_id=str(training_job_id),
            error=str(e),
        )
        raise HTTPException(status_code=400, detail=str(e))

    except HTTPException:
        raise

    except Exception as e:
        logger.error(
            "Unexpected error cancelling training job",
            training_job_id=str(training_job_id),
            error=str(e),
        )
        raise HTTPException(status_code=500, detail="Internal server error")


@router.delete(
    "/training-jobs/{training_job_id}",
    summary="Delete training job",
    description="""
    Delete a training job and its generated artifacts.
    Only completed, failed, or cancelled jobs can be deleted.
    """,
)
@inject
async def delete_training_job(
    training_job_id: UUID,
    training_use_case: TrainingManagementUseCase = Depends(
        Provide[AppContainer.training_management_use_case]
    ),
) -> dict:
    """Delete a training job and associated artifacts."""
    try:
        deleted = await training_use_case.delete_training_job(training_job_id)

        if not deleted:
            raise HTTPException(
                status_code=404,
                detail=f"Training job {training_job_id} not found",
            )

        logger.info(
            "Training job deleted",
            training_job_id=str(training_job_id),
        )

        return {"message": f"Training job {training_job_id} deleted successfully"}

    except TrainingManagementError as e:
        logger.error(
            "Training deletion failed",
            training_job_id=str(training_job_id),
            error=str(e),
        )
        raise HTTPException(status_code=400, detail=str(e))

    except HTTPException:
        raise

    except Exception as e:
        logger.error(
            "Unexpected error deleting training job",
            training_job_id=str(training_job_id),
            error=str(e),
        )
        raise HTTPException(status_code=500, detail="Internal server error")
