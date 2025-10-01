from __future__ import annotations

import pytest

from src.application.dtos.model_dto import ModelCreateDTO, ModelUpdateDTO
from src.domain.entities.model import ModelType


def _base_payload() -> dict:
    return {
        "model_type": ModelType.LSTM,
        "rnn_layers": [{"units": 64, "dropout": 0.1, "recurrent_dropout": 0.0}],
        "dense_layers": [{"units": 32, "dropout": 0.1, "activation": "relu"}],
        "feature": "temperature",
        "entity_type": "Sensor",
        "entity_id": "urn:ngsi-ld:Sensor:001",
    }


def test_model_create_dto_generates_defaults() -> None:
    dto = ModelCreateDTO(**_base_payload())
    assert dto.early_stopping_patience == 10


def test_model_create_dto_validates_ratios() -> None:
    payload = _base_payload()
    payload["validation_ratio"] = 0.7
    payload["test_ratio"] = 0.5
    with pytest.raises(ValueError):
        ModelCreateDTO(**payload)


def test_model_update_dto_autofills_patience() -> None:
    dto = ModelUpdateDTO.model_validate({"epochs": 150})
    assert dto.early_stopping_patience == 15
