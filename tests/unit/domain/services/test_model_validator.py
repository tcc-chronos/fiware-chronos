from __future__ import annotations

import pytest

from src.domain.entities.errors import ModelValidationError
from src.domain.entities.model import DenseLayerConfig, RNNLayerConfig
from src.domain.services.model_validator import validate_model_configuration


def test_validate_model_configuration_accepts_valid(sample_model) -> None:
    validate_model_configuration(sample_model)


def test_validate_model_configuration_raises_detailed_errors(sample_model) -> None:
    sample_model.rnn_layers = [RNNLayerConfig(units=0, dropout=-1, recurrent_dropout=2)]
    sample_model.dense_layers = [DenseLayerConfig(units=0, dropout=1.5, activation="")]
    sample_model.batch_size = 0
    sample_model.epochs = 0
    sample_model.learning_rate = 0
    sample_model.validation_ratio = 0
    sample_model.test_ratio = 1
    sample_model.lookback_window = 0
    sample_model.forecast_horizon = 0
    sample_model.early_stopping_patience = 999
    sample_model.entity_type = ""
    sample_model.entity_id = ""
    sample_model.feature = ""

    with pytest.raises(ModelValidationError) as exc:
        validate_model_configuration(sample_model)

    details = exc.value.details["errors"]
    assert "RNN layer #1" in details[0]
    assert any("Batch size" in item for item in details)
    assert any("Feature attribute" in item for item in details)
