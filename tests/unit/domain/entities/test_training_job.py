from __future__ import annotations

from datetime import datetime, timedelta, timezone

from src.domain.entities.training_job import TrainingJob


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
