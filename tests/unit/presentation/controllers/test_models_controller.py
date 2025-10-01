from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, List, cast
from uuid import uuid4

import pytest
from fastapi import HTTPException

from src.application.dtos.model_dto import (
    DenseLayerDTO,
    ModelCreateDTO,
    ModelResponseDTO,
    ModelTypeOptionDTO,
    ModelUpdateDTO,
    RNNLayerDTO,
)
from src.application.use_cases.model_use_cases import (
    CreateModelUseCase,
    DeleteModelUseCase,
    GetModelByIdUseCase,
    GetModelsUseCase,
    GetModelTypesUseCase,
    UpdateModelUseCase,
)
from src.domain.entities.errors import (
    ModelNotFoundError,
    ModelOperationError,
    ModelValidationError,
)
from src.domain.entities.model import ModelStatus, ModelType
from src.presentation.controllers import models_controller


class _StubGetModels:
    def __init__(self, result: List[ModelResponseDTO]):
        self.result = result

    async def execute(
        self,
        skip: int = 0,
        limit: int = 100,
        model_type: str | None = None,
        status: str | None = None,
        entity_id: str | None = None,
        feature: str | None = None,
    ) -> List[ModelResponseDTO]:
        return self.result


def _as_get_models_use_case(use_case: _StubGetModels) -> GetModelsUseCase:
    return cast(GetModelsUseCase, use_case)


def _as_get_model_types_use_case(use_case: object) -> GetModelTypesUseCase:
    return cast(GetModelTypesUseCase, use_case)


def _as_get_model_by_id_use_case(use_case: object) -> GetModelByIdUseCase:
    return cast(GetModelByIdUseCase, use_case)


def _as_create_model_use_case(use_case: object) -> CreateModelUseCase:
    return cast(CreateModelUseCase, use_case)


def _as_update_model_use_case(use_case: object) -> UpdateModelUseCase:
    return cast(UpdateModelUseCase, use_case)


def _as_delete_model_use_case(use_case: object) -> DeleteModelUseCase:
    return cast(DeleteModelUseCase, use_case)


def _create_model_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model_type": ModelType.LSTM,
        "rnn_layers": [{"units": 32, "dropout": 0.1, "recurrent_dropout": 0.0}],
        "dense_layers": [],
        "feature": "temp",
        "entity_type": "Sensor",
        "entity_id": "urn",
    }
    payload.update(overrides)
    return payload


def _update_model_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    payload.update(overrides)
    return payload


@pytest.mark.asyncio
async def test_get_models_returns_list():
    now = datetime.now(timezone.utc)
    dto = ModelResponseDTO(
        id=uuid4(),
        name="model",
        description=None,
        model_type=ModelType.LSTM,
        status=ModelStatus.DRAFT,
        batch_size=32,
        epochs=10,
        learning_rate=0.01,
        validation_ratio=0.1,
        test_ratio=0.1,
        rnn_layers=[RNNLayerDTO(units=64, dropout=0.1, recurrent_dropout=0)],
        dense_layers=[],
        early_stopping_patience=None,
        lookback_window=24,
        forecast_horizon=1,
        feature="temp",
        entity_type="Sensor",
        entity_id="urn",
        created_at=now,
        updated_at=now,
        trainings=[],
    )
    response = await models_controller.get_models(
        get_models_use_case=_as_get_models_use_case(_StubGetModels([dto]))
    )
    assert len(response) == 1
    assert response[0].name == "model"


@pytest.mark.asyncio
async def test_get_models_handles_exception():
    class _Fail(_StubGetModels):
        def __init__(self) -> None:
            super().__init__([])

        async def execute(
            self,
            skip: int = 0,
            limit: int = 100,
            model_type: str | None = None,
            status: str | None = None,
            entity_id: str | None = None,
            feature: str | None = None,
        ) -> List[ModelResponseDTO]:
            raise RuntimeError("boom")

    with pytest.raises(HTTPException) as exc:
        await models_controller.get_models(
            get_models_use_case=_as_get_models_use_case(_Fail())
        )
    assert exc.value.status_code == 500


@pytest.mark.asyncio
async def test_get_model_types():
    class _Types:
        async def execute(self) -> List[ModelTypeOptionDTO]:
            return [ModelTypeOptionDTO(value=ModelType.LSTM, label="LSTM")]

    result = await models_controller.get_model_types(
        get_model_types_use_case=_as_get_model_types_use_case(_Types())
    )
    assert result[0].label == "LSTM"


@pytest.mark.asyncio
async def test_get_model_by_id_not_found():
    class _Getter:
        async def execute(self, model_id):
            raise ModelNotFoundError(str(model_id))

    with pytest.raises(HTTPException) as exc:
        await models_controller.get_model_by_id(
            model_id=uuid4(),
            get_model_use_case=_as_get_model_by_id_use_case(_Getter()),
        )
    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_create_model_validation_error():
    class _Creator:
        async def execute(self, model_dto):
            raise ModelValidationError("invalid", details={"errors": []})

    with pytest.raises(HTTPException) as exc:
        await models_controller.create_model(
            model_dto=ModelCreateDTO.model_validate(_create_model_payload()),
            create_model_use_case=_as_create_model_use_case(_Creator()),
        )
    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_update_model_dispatches():
    class _Updater:
        async def execute(self, model_id, model_dto):
            now = datetime.now(timezone.utc)
            return ModelResponseDTO(
                id=model_id,
                name="updated",
                description=None,
                model_type=ModelType.LSTM,
                status=ModelStatus.DRAFT,
                batch_size=32,
                epochs=20,
                learning_rate=0.01,
                validation_ratio=0.1,
                test_ratio=0.1,
                rnn_layers=[RNNLayerDTO(units=64, dropout=0.1, recurrent_dropout=0)],
                dense_layers=[],
                early_stopping_patience=None,
                lookback_window=24,
                forecast_horizon=1,
                feature="temp",
                entity_type="Sensor",
                entity_id="urn",
                created_at=now,
                updated_at=now,
                trainings=[],
            )

    result = await models_controller.update_model(
        model_id=uuid4(),
        model_dto=ModelUpdateDTO.model_validate({}),
        update_model_use_case=_as_update_model_use_case(_Updater()),
    )
    assert result.name == "updated"


@pytest.mark.asyncio
async def test_delete_model_not_found():
    class _Deleter:
        async def execute(self, model_id):
            raise ModelNotFoundError(str(model_id))

    with pytest.raises(HTTPException) as exc:
        await models_controller.delete_model(
            model_id=uuid4(),
            delete_model_use_case=_as_delete_model_use_case(_Deleter()),
        )
    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_get_model_types_handles_exception():
    class _Fail:
        async def execute(self):
            raise RuntimeError("boom")

    with pytest.raises(HTTPException) as exc:
        await models_controller.get_model_types(
            get_model_types_use_case=_as_get_model_types_use_case(_Fail())
        )
    assert exc.value.status_code == 500


@pytest.mark.asyncio
async def test_create_model_warns_for_redundant_dense_layer(monkeypatch):
    class _Creator:
        async def execute(self, model_dto):
            return ModelResponseDTO(
                id=uuid4(),
                name="test",
                description=None,
                model_type=ModelType.LSTM,
                status=ModelStatus.DRAFT,
                batch_size=32,
                epochs=10,
                learning_rate=0.01,
                validation_ratio=0.1,
                test_ratio=0.1,
                rnn_layers=[RNNLayerDTO(units=16, dropout=0.1, recurrent_dropout=0)],
                dense_layers=[DenseLayerDTO(units=1, dropout=0.1, activation="relu")],
                early_stopping_patience=5,
                lookback_window=24,
                forecast_horizon=1,
                feature="temp",
                entity_type="Sensor",
                entity_id="urn",
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
                trainings=[],
            )

    class _Logger:
        def __init__(self):
            self.warning_called = False

        def warning(self, *args, **kwargs):
            self.warning_called = True

        def error(self, *args, **kwargs):
            pass

    fake_logger = _Logger()
    monkeypatch.setattr(models_controller, "logger", fake_logger)

    await models_controller.create_model(
        model_dto=ModelCreateDTO.model_validate(
            _create_model_payload(
                dense_layers=[{"units": 1, "dropout": 0.1, "activation": "relu"}]
            )
        ),
        create_model_use_case=_as_create_model_use_case(_Creator()),
    )

    assert fake_logger.warning_called is True


@pytest.mark.asyncio
async def test_create_model_handles_operation_error():
    class _Creator:
        async def execute(self, model_dto):
            raise ModelOperationError("cannot")

    with pytest.raises(HTTPException) as exc:
        await models_controller.create_model(
            model_dto=ModelCreateDTO.model_validate(_create_model_payload()),
            create_model_use_case=_as_create_model_use_case(_Creator()),
        )
    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_create_model_handles_unexpected_exception():
    class _Creator:
        async def execute(self, model_dto):
            raise RuntimeError("boom")

    with pytest.raises(HTTPException) as exc:
        await models_controller.create_model(
            model_dto=ModelCreateDTO.model_validate(_create_model_payload()),
            create_model_use_case=_as_create_model_use_case(_Creator()),
        )
    assert exc.value.status_code == 500


@pytest.mark.asyncio
async def test_update_model_warns_about_redundant_dense_layer(monkeypatch):
    now = datetime.now(timezone.utc)

    class _Updater:
        async def execute(self, model_id, model_dto):
            return ModelResponseDTO(
                id=model_id,
                name="updated",
                description=None,
                model_type=ModelType.LSTM,
                status=ModelStatus.DRAFT,
                batch_size=32,
                epochs=20,
                learning_rate=0.01,
                validation_ratio=0.1,
                test_ratio=0.1,
                rnn_layers=[RNNLayerDTO(units=64, dropout=0.1, recurrent_dropout=0)],
                dense_layers=[],
                early_stopping_patience=1,
                lookback_window=24,
                forecast_horizon=1,
                feature="temp",
                entity_type="Sensor",
                entity_id="urn",
                created_at=now,
                updated_at=now,
                trainings=[],
            )

    class _Logger:
        def __init__(self):
            self.warning_called = False

        def warning(self, *args, **kwargs):
            self.warning_called = True

        def error(self, *args, **kwargs):
            pass

    fake_logger = _Logger()
    monkeypatch.setattr(models_controller, "logger", fake_logger)

    await models_controller.update_model(
        model_id=uuid4(),
        model_dto=ModelUpdateDTO.model_validate(
            _update_model_payload(
                dense_layers=[{"units": 1, "dropout": 0.1, "activation": "relu"}]
            )
        ),
        update_model_use_case=_as_update_model_use_case(_Updater()),
    )

    assert fake_logger.warning_called is True


@pytest.mark.asyncio
async def test_update_model_handles_operation_error():
    class _Updater:
        async def execute(self, model_id, model_dto):
            raise ModelOperationError("cannot")

    with pytest.raises(HTTPException) as exc:
        await models_controller.update_model(
            model_id=uuid4(),
            model_dto=ModelUpdateDTO.model_validate({}),
            update_model_use_case=_as_update_model_use_case(_Updater()),
        )
    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_update_model_handles_unexpected_exception():
    class _Updater:
        async def execute(self, model_id, model_dto):
            raise RuntimeError("boom")

    with pytest.raises(HTTPException) as exc:
        await models_controller.update_model(
            model_id=uuid4(),
            model_dto=ModelUpdateDTO.model_validate({}),
            update_model_use_case=_as_update_model_use_case(_Updater()),
        )
    assert exc.value.status_code == 500


@pytest.mark.asyncio
async def test_delete_model_handles_operation_error():
    class _Deleter:
        async def execute(self, model_id):
            raise ModelOperationError("cannot")

    with pytest.raises(HTTPException) as exc:
        await models_controller.delete_model(
            model_id=uuid4(),
            delete_model_use_case=_as_delete_model_use_case(_Deleter()),
        )
    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_delete_model_handles_unexpected_exception():
    class _Deleter:
        async def execute(self, model_id):
            raise RuntimeError("boom")

    with pytest.raises(HTTPException) as exc:
        await models_controller.delete_model(
            model_id=uuid4(),
            delete_model_use_case=_as_delete_model_use_case(_Deleter()),
        )
    assert exc.value.status_code == 500


@pytest.mark.asyncio
async def test_get_model_by_id_handles_unexpected_error():
    class _Getter:
        async def execute(self, model_id):
            raise RuntimeError("db down")

    with pytest.raises(HTTPException) as exc:
        await models_controller.get_model_by_id(
            model_id=uuid4(),
            get_model_use_case=_as_get_model_by_id_use_case(_Getter()),
        )

    assert exc.value.status_code == 500


@pytest.mark.asyncio
async def test_update_model_not_found():
    class _Updater:
        async def execute(self, model_id, model_dto):
            raise ModelNotFoundError(str(model_id))

    with pytest.raises(HTTPException) as exc:
        await models_controller.update_model(
            model_id=uuid4(),
            model_dto=ModelUpdateDTO.model_validate({}),
            update_model_use_case=_as_update_model_use_case(_Updater()),
        )

    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_update_model_handles_validation_error():
    class _Updater:
        async def execute(self, model_id, model_dto):
            raise ModelValidationError("invalid", details={"field": ["bad"]})

    with pytest.raises(HTTPException) as exc:
        await models_controller.update_model(
            model_id=uuid4(),
            model_dto=ModelUpdateDTO.model_validate({}),
            update_model_use_case=_as_update_model_use_case(_Updater()),
        )

    assert exc.value.status_code == 400
    detail = cast(dict[str, Any], exc.value.detail)
    assert detail["field"] == ["bad"]
    assert detail["message"] == "invalid"
