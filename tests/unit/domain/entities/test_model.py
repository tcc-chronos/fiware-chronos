from __future__ import annotations

from datetime import datetime, timezone

from src.domain.entities.model import Model, ModelStatus, ModelType


def test_model_defaults() -> None:
    model = Model()
    assert model.model_type is ModelType.LSTM
    assert model.status is ModelStatus.DRAFT
    assert len(model.rnn_layers) == 1
    assert isinstance(model.created_at, datetime)
    assert model.created_at.tzinfo == timezone.utc


def test_update_timestamp_changes_value(sample_model: Model) -> None:
    before = sample_model.updated_at
    sample_model.update_timestamp()
    assert sample_model.updated_at >= before
    assert sample_model.updated_at.tzinfo == timezone.utc


def test_has_trained_artifacts_flag(sample_model: Model) -> None:
    assert sample_model.has_trained_artifacts() is False
    sample_model.has_successful_training = True
    assert sample_model.has_trained_artifacts() is True


def test_clear_artifacts_resets_flag(sample_model: Model) -> None:
    sample_model.has_successful_training = True
    sample_model.clear_artifacts()
    assert sample_model.has_successful_training is False
    assert sample_model.updated_at.tzinfo == timezone.utc
