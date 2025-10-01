from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from src.application.dtos.training_dto import (
    DataCollectionJobDTO,
    StartTrainingResponseDTO,
    TrainingJobDTO,
    TrainingMetricsDTO,
    TrainingRequestDTO,
)
from src.domain.entities.training_job import DataCollectionStatus, TrainingStatus


def test_training_request_dto_defaults() -> None:
    dto = TrainingRequestDTO()
    assert dto.last_n == 1000


def test_training_job_dto_serialization_roundtrip() -> None:
    job_id = uuid4()
    now = datetime.now(timezone.utc)
    dto = TrainingJobDTO(
        id=job_id,
        model_id=uuid4(),
        status=TrainingStatus.PENDING,
        last_n=100,
        data_collection_jobs=[
            DataCollectionJobDTO(
                id=uuid4(),
                h_offset=0,
                last_n=50,
                status=DataCollectionStatus.PENDING,
                data_points_collected=10,
            )
        ],
        total_data_points_requested=100,
        total_data_points_collected=50,
        data_collection_progress=50.0,
        start_time=now,
        end_time=None,
        data_collection_start=now,
        data_collection_end=None,
        preprocessing_start=None,
        preprocessing_end=None,
        training_start=None,
        training_end=None,
        metrics=TrainingMetricsDTO(mse=0.1),
        model_artifact_id="artifact",
        error=None,
        error_details=None,
        created_at=now,
        updated_at=now,
        total_duration_seconds=None,
        training_duration_seconds=None,
    )

    payload = dto.model_dump()
    assert payload["status"] == TrainingStatus.PENDING
    assert payload["data_collection_jobs"][0]["status"] == DataCollectionStatus.PENDING


def test_start_training_response_defaults() -> None:
    job_id = uuid4()
    response = StartTrainingResponseDTO(training_job_id=job_id)
    assert response.training_job_id == job_id
    assert response.status is TrainingStatus.PENDING
    assert "successfully" in response.message
