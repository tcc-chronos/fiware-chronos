"""
Application Use Cases - Training Management

This module contains use cases for managing training jobs.
"""

from datetime import datetime
from typing import List, Optional
from uuid import UUID

import structlog

from src.application.dtos.training_dto import (
    DataCollectionJobDTO,
    StartTrainingResponseDTO,
    TrainingJobDTO,
    TrainingJobSummaryDTO,
    TrainingMetricsDTO,
    TrainingRequestDTO,
)
from src.domain.entities.training_job import TrainingJob, TrainingStatus
from src.domain.repositories.model_repository import IModelRepository
from src.domain.repositories.training_job_repository import ITrainingJobRepository

logger = structlog.get_logger(__name__)


class TrainingManagementError(Exception):
    """Exception raised when training management operations fail."""

    pass


class TrainingManagementUseCase:
    """Use case for managing training jobs."""

    def __init__(
        self,
        training_job_repository: ITrainingJobRepository,
        model_repository: IModelRepository,
    ):
        """
        Initialize the training management use case.

        Args:
            training_job_repository: Repository for training jobs
            model_repository: Repository for models
        """
        self.training_job_repository = training_job_repository
        self.model_repository = model_repository

    async def start_training(
        self, model_id: UUID, request: TrainingRequestDTO
    ) -> StartTrainingResponseDTO:
        """
        Start a new training job for a model.

        Args:
            model_id: ID of the model to train
            request: Training request parameters

        Returns:
            Response with training job ID

        Raises:
            TrainingManagementError: When training cannot be started
        """
        try:
            # Validate model exists
            model = await self.model_repository.find_by_id(model_id)
            if not model:
                raise TrainingManagementError(f"Model {model_id} not found")

            logger.info(
                "Starting training job",
                model_id=str(model_id),
                lookback_window=model.lookback_window,
                last_n=request.last_n,
            )

            # Validate model configuration
            if not model.entity_type or not model.entity_id or not model.feature:
                raise TrainingManagementError(
                    "Model must have entity_type, entity_id, and feature configured"
                )

            # Check for existing running training jobs
            existing_jobs = await self.training_job_repository.get_by_model_id(model_id)
            running_jobs = [
                job
                for job in existing_jobs
                if job.status
                in [
                    TrainingStatus.PENDING,
                    TrainingStatus.COLLECTING_DATA,
                    TrainingStatus.PREPROCESSING,
                    TrainingStatus.TRAINING,
                ]
            ]

            if running_jobs:
                raise TrainingManagementError(
                    f"Model {model_id} already has a running training job: "
                    f"{running_jobs[0].id}"
                )

            # Create new training job
            training_job = TrainingJob(
                model_id=model_id,
                last_n=request.last_n,
                total_data_points_requested=request.last_n,
                start_time=datetime.utcnow(),
            )

            # Save training job
            created_job = await self.training_job_repository.create(training_job)

            # Start Celery orchestration task
            from src.infrastructure.services.celery_config import celery_app

            celery_app.tasks["orchestrate_training"].delay(
                training_job_id=str(created_job.id),
                model_id=str(model_id),
                last_n=request.last_n,
            )

            logger.info(
                "Training job started successfully",
                training_job_id=str(created_job.id),
                model_id=str(model_id),
            )

            return StartTrainingResponseDTO(
                training_job_id=created_job.id, status=created_job.status
            )

        except Exception as e:
            logger.error(
                "Failed to start training", model_id=str(model_id), error=str(e)
            )
            if isinstance(e, TrainingManagementError):
                raise e
            raise TrainingManagementError(f"Failed to start training: {str(e)}") from e

    async def get_training_job(self, training_job_id: UUID) -> Optional[TrainingJobDTO]:
        """
        Get detailed information about a training job.

        Args:
            training_job_id: ID of the training job

        Returns:
            Training job details or None if not found
        """
        try:
            training_job = await self.training_job_repository.get_by_id(training_job_id)

            if not training_job:
                return None

            return self._to_training_job_dto(training_job)

        except Exception as e:
            logger.error(
                "Failed to get training job",
                training_job_id=str(training_job_id),
                error=str(e),
            )
            raise TrainingManagementError(
                f"Failed to get training job: {str(e)}"
            ) from e

    async def list_training_jobs(
        self, model_id: Optional[UUID] = None, skip: int = 0, limit: int = 100
    ) -> List[TrainingJobSummaryDTO]:
        """
        List training jobs with optional filtering by model.

        Args:
            model_id: Optional model ID to filter by
            skip: Number of records to skip
            limit: Maximum number of records to return

        Returns:
            List of training job summaries
        """
        try:
            if model_id:
                training_jobs = await self.training_job_repository.get_by_model_id(
                    model_id
                )
                # Apply pagination manually for model-specific results
                training_jobs = training_jobs[skip : skip + limit]
            else:
                training_jobs = await self.training_job_repository.list_all(skip, limit)

            return [self._to_training_job_summary_dto(job) for job in training_jobs]

        except Exception as e:
            logger.error(
                "Failed to list training jobs",
                model_id=str(model_id) if model_id else None,
                error=str(e),
            )
            raise TrainingManagementError(
                f"Failed to list training jobs: {str(e)}"
            ) from e

    async def cancel_training_job(self, training_job_id: UUID) -> bool:
        """
        Cancel a running training job.

        Args:
            training_job_id: ID of the training job to cancel

        Returns:
            True if cancelled successfully

        Raises:
            TrainingManagementError: When job cannot be cancelled
        """
        try:
            training_job = await self.training_job_repository.get_by_id(training_job_id)

            if not training_job:
                raise TrainingManagementError(
                    f"Training job {training_job_id} not found"
                )

            # Check if job can be cancelled
            if training_job.status in [
                TrainingStatus.COMPLETED,
                TrainingStatus.FAILED,
                TrainingStatus.CANCELLED,
            ]:
                raise TrainingManagementError(
                    f"Training job {training_job_id} is already "
                    f"{training_job.status.value}"
                )

            # Update job status to cancelled
            await self.training_job_repository.update_training_job_status(
                training_job_id, TrainingStatus.CANCELLED
            )

            # TODO: Cancel Celery tasks if possible
            # This would require storing task IDs and implementing task cancellation

            logger.info("Training job cancelled", training_job_id=str(training_job_id))

            return True

        except Exception as e:
            logger.error(
                "Failed to cancel training job",
                training_job_id=str(training_job_id),
                error=str(e),
            )
            if isinstance(e, TrainingManagementError):
                raise e
            raise TrainingManagementError(
                f"Failed to cancel training job: {str(e)}"
            ) from e

    def _to_training_job_dto(self, training_job: TrainingJob) -> TrainingJobDTO:
        """Convert TrainingJob entity to DTO."""

        # Convert data collection jobs
        data_collection_jobs = [
            DataCollectionJobDTO(
                id=job.id,
                h_offset=job.h_offset,
                last_n=job.last_n,
                status=job.status,
                start_time=job.start_time,
                end_time=job.end_time,
                error=job.error,
                data_points_collected=job.data_points_collected,
            )
            for job in training_job.data_collection_jobs
        ]

        # Convert metrics
        metrics = None
        if training_job.metrics:
            metrics = TrainingMetricsDTO(
                mse=training_job.metrics.mse,
                mae=training_job.metrics.mae,
                rmse=training_job.metrics.rmse,
                mape=training_job.metrics.mape,
                r2=training_job.metrics.r2,
                mae_pct=training_job.metrics.mae_pct,
                rmse_pct=training_job.metrics.rmse_pct,
                best_train_loss=training_job.metrics.best_train_loss,
                best_val_loss=training_job.metrics.best_val_loss,
                best_epoch=training_job.metrics.best_epoch,
            )

        return TrainingJobDTO(
            id=training_job.id,
            model_id=training_job.model_id,
            status=training_job.status,
            last_n=training_job.last_n,
            data_collection_jobs=data_collection_jobs,
            total_data_points_requested=training_job.total_data_points_requested,
            total_data_points_collected=training_job.total_data_points_collected,
            data_collection_progress=training_job.get_data_collection_progress(),
            start_time=training_job.start_time,
            end_time=training_job.end_time,
            data_collection_start=training_job.data_collection_start,
            data_collection_end=training_job.data_collection_end,
            preprocessing_start=training_job.preprocessing_start,
            preprocessing_end=training_job.preprocessing_end,
            training_start=training_job.training_start,
            training_end=training_job.training_end,
            metrics=metrics,
            model_artifact_path=training_job.model_artifact_path,
            error=training_job.error,
            error_details=training_job.error_details,
            created_at=training_job.created_at,
            updated_at=training_job.updated_at,
            total_duration_seconds=training_job.get_total_duration(),
            training_duration_seconds=training_job.get_training_duration(),
        )

    def _to_training_job_summary_dto(
        self, training_job: TrainingJob
    ) -> TrainingJobSummaryDTO:
        """Convert TrainingJob entity to summary DTO."""

        return TrainingJobSummaryDTO(
            id=training_job.id,
            model_id=training_job.model_id,
            status=training_job.status,
            data_collection_progress=training_job.get_data_collection_progress(),
            start_time=training_job.start_time,
            end_time=training_job.end_time,
            error=training_job.error,
            created_at=training_job.created_at,
        )
