from __future__ import annotations

import pytest

from src.application.dtos.model_dto import (
    DenseLayerDTO,
    ModelCreateDTO,
    ModelUpdateDTO,
    RNNLayerDTO,
)
from src.domain.entities.model import ModelType


def test_model_create_dto_generates_defaults() -> None:
    dto = ModelCreateDTO(
        model_type=ModelType.LSTM,
        feature="temperature",
        entity_type="Sensor",
        entity_id="urn:ngsi-ld:Sensor:001",
        rnn_layers=[RNNLayerDTO(units=64)],
        dense_layers=[DenseLayerDTO(units=32)],
        epochs=80,
    )

    assert dto.name == "LSTM - temperature"
    assert dto.description == "LSTM model for temperature forecasting"
    assert 5 <= dto.early_stopping_patience <= 20


def test_model_create_dto_validates_ratios() -> None:
    with pytest.raises(ValueError):
        ModelCreateDTO(
            name="Invalid",
            description="Invalid ratios",
            model_type=ModelType.GRU,
            feature="humidity",
            entity_type="Sensor",
            entity_id="urn:ngsi-ld:Sensor:002",
            rnn_layers=[RNNLayerDTO(units=32)],
            dense_layers=[DenseLayerDTO(units=16)],
            validation_ratio=0.6,
            test_ratio=0.5,
        )


def test_model_update_dto_preserves_fields() -> None:
    dto = ModelUpdateDTO(
        name="Updated",
        description="Updated description",
        epochs=200,
    )

    assert dto.name == "Updated"
    assert dto.description == "Updated description"
    assert dto.epochs == 200
