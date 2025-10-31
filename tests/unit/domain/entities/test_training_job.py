from __future__ import annotations

from datetime import datetime, timedelta, timezone

from src.domain.entities.training_job import (
    TrainingJob,
    TrainingMetrics,
    TrainingStatus,
)


def test_set_sampling_interval_initialises_next_prediction() -> None:
    job = TrainingJob()
    assert job.next_prediction_at is None

    job.set_sampling_interval(300)

    assert job.sampling_interval_seconds == 300
    assert job.next_prediction_at is not None

    previous_update = job.updated_at
    job.set_sampling_interval(-5)
    # Negative values do not change state
    assert job.sampling_interval_seconds == 300
    assert job.updated_at == previous_update


def test_schedule_and_toggle_predictions() -> None:
    job = TrainingJob()
    base_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
    job.set_sampling_interval(60)

    job.schedule_next_prediction(base_time=base_time)
    assert job.next_prediction_at == base_time + timedelta(seconds=60)

    job.enable_predictions(
        service_group="sg",
        entity_id="urn:prediction:entity",
        entity_type="Forecast",
        metadata={"device_id": "device-1"},
        subscription_id="sub-1",
    )
    assert job.prediction_config.enabled is True
    assert job.prediction_config.entity_id == "urn:prediction:entity"
    assert job.prediction_config.metadata["device_id"] == "device-1"
    assert job.prediction_config.subscription_id == "sub-1"

    job.disable_predictions(clear_subscription=True)
    assert job.prediction_config.enabled is False
    assert job.prediction_config.subscription_id is None


def test_schedule_next_prediction_without_interval_keeps_state() -> None:
    job = TrainingJob()
    reference = datetime(2024, 1, 1, tzinfo=timezone.utc)
    job.schedule_next_prediction(base_time=reference)
    assert job.next_prediction_at is None

    job.enable_predictions(
        service_group="sg",
        entity_id="urn:prediction:entity",
        metadata=None,
    )
    assert job.prediction_config.metadata == {}


def test_data_collection_progress_and_exceptions() -> None:
    job = TrainingJob()
    job.total_data_points_requested = 200
    job.total_data_points_collected = 50

    assert job.get_data_collection_progress() == 25.0

    job.total_data_points_requested = "invalid"  # type: ignore[assignment]
    assert job.get_data_collection_progress() == 0.0


def test_marking_phases_updates_state() -> None:
    job = TrainingJob()
    metrics = TrainingMetrics(mse=0.1)

    job.mark_data_collection_complete()
    job.mark_preprocessing_complete()
    job.mark_training_complete(metrics=metrics)

    assert job.metrics is metrics
    assert job.status is TrainingStatus.COMPLETED
    assert job.get_total_duration() is None or job.get_total_duration() >= 0.0


def test_get_total_and_training_duration_calculations() -> None:
    job = TrainingJob()
    job.start_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
    job.end_time = job.start_time + timedelta(hours=2)
    job.training_start = job.start_time + timedelta(minutes=30)
    job.training_end = job.training_start + timedelta(minutes=45)

    assert job.get_total_duration() == 7200.0
    assert job.get_training_duration() == 2700.0


def test_mark_failed_sets_status_and_error_details() -> None:
    job = TrainingJob()
    job.mark_failed("boom", {"cause": "unexpected"})

    assert job.status is TrainingStatus.FAILED
    assert job.error == "boom"
    assert job.error_details == {"cause": "unexpected"}
    assert job.end_time is not None


def test_disable_predictions_without_clearing_subscription() -> None:
    job = TrainingJob()
    job.enable_predictions(
        service_group="sg",
        entity_id="entity",
        subscription_id="sub-1",
    )

    job.disable_predictions(clear_subscription=False)

    assert job.prediction_config.enabled is False
    assert job.prediction_config.subscription_id == "sub-1"
