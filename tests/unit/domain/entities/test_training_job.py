from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.domain.entities.training_job import (
    DataCollectionJob,
    DataCollectionStatus,
    TrainingJob,
    TrainingMetrics,
    TrainingStatus,
)


def test_add_data_collection_job_updates_timestamp(
    sample_training_job: TrainingJob,
) -> None:
    initial = sample_training_job.updated_at
    new_job = DataCollectionJob(
        h_offset=10,
        last_n=100,
        status=DataCollectionStatus.IN_PROGRESS,
    )
    sample_training_job.add_data_collection_job(new_job)
    assert sample_training_job.data_collection_jobs[-1] is new_job
    assert sample_training_job.updated_at >= initial


def test_get_data_collection_progress_handles_zero(
    sample_training_job: TrainingJob,
) -> None:
    sample_training_job.total_data_points_requested = 0
    assert sample_training_job.get_data_collection_progress() == 0.0


def test_get_data_collection_progress_calculates_percentage(
    sample_training_job: TrainingJob,
) -> None:
    sample_training_job.total_data_points_requested = 200
    sample_training_job.total_data_points_collected = 150
    assert sample_training_job.get_data_collection_progress() == pytest.approx(75.0)


def test_mark_training_complete_sets_status(
    sample_training_job: TrainingJob, sample_training_metrics: TrainingMetrics
) -> None:
    sample_training_job.training_start = datetime.now(timezone.utc) - timedelta(
        minutes=5
    )
    sample_training_job.start_time = sample_training_job.training_start
    sample_training_job.mark_training_complete(sample_training_metrics)
    assert sample_training_job.status is TrainingStatus.COMPLETED
    assert sample_training_job.metrics is sample_training_metrics
    assert sample_training_job.end_time is not None


def test_mark_failed_sets_error(sample_training_job: TrainingJob) -> None:
    sample_training_job.mark_failed("boom", {"detail": "trace"})
    assert sample_training_job.status is TrainingStatus.FAILED
    assert sample_training_job.error == "boom"
    assert sample_training_job.error_details == {"detail": "trace"}


def test_get_total_duration(sample_training_job: TrainingJob) -> None:
    sample_training_job.start_time = datetime.now(timezone.utc) - timedelta(minutes=5)
    sample_training_job.end_time = datetime.now(timezone.utc)
    assert sample_training_job.get_total_duration() == pytest.approx(300.0, abs=1.0)


def test_get_training_duration(sample_training_job: TrainingJob) -> None:
    sample_training_job.training_start = datetime.now(timezone.utc) - timedelta(
        minutes=2
    )
    sample_training_job.training_end = datetime.now(timezone.utc)
    assert sample_training_job.get_training_duration() == pytest.approx(120.0, abs=1.0)
