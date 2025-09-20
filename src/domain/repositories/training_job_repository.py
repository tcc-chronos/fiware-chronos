"""
Domain Repository Interface - Training Job

This module defines the repository interface for training job persistence.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Optional
from uuid import UUID

from src.domain.entities.training_job import (
    DataCollectionJob,
    DataCollectionStatus,
    TrainingJob,
    TrainingMetrics,
    TrainingStatus,
)


class ITrainingJobRepository(ABC):
    """Interface for training job repository."""

    @abstractmethod
    async def create(self, training_job: TrainingJob) -> TrainingJob:
        """Create a new training job."""
        pass

    @abstractmethod
    async def get_by_id(self, training_job_id: UUID) -> Optional[TrainingJob]:
        """Get training job by ID."""
        pass

    @abstractmethod
    async def get_by_model_id(self, model_id: UUID) -> List[TrainingJob]:
        """Get all training jobs for a specific model."""
        pass

    @abstractmethod
    async def update(self, training_job: TrainingJob) -> TrainingJob:
        """Update a training job."""
        pass

    @abstractmethod
    async def delete(self, training_job_id: UUID) -> bool:
        """Delete a training job."""
        pass

    @abstractmethod
    async def add_data_collection_job(
        self, training_job_id: UUID, job: DataCollectionJob
    ) -> bool:
        """Add a data collection job to a training job."""
        pass

    @abstractmethod
    async def update_data_collection_job_status(
        self,
        training_job_id: UUID,
        job_id: UUID,
        status: DataCollectionStatus,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        error: Optional[str] = None,
        data_points_collected: Optional[int] = None,
    ) -> bool:
        """Update the status of a data collection job."""
        pass

    @abstractmethod
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
    ) -> bool:
        """Update the status and timestamps of a training job."""
        pass

    @abstractmethod
    async def complete_training_job(
        self,
        training_job_id: UUID,
        metrics: TrainingMetrics,
        model_artifact_id: str,
        x_scaler_artifact_id: str,
        y_scaler_artifact_id: str,
        metadata_artifact_id: str,
    ) -> bool:
        """Complete a training job with results."""
        pass

    @abstractmethod
    async def fail_training_job(
        self, training_job_id: UUID, error: str, error_details: Optional[dict] = None
    ) -> bool:
        """Mark a training job as failed."""
        pass
