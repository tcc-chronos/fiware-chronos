from __future__ import annotations

import math
from datetime import datetime, timezone
from uuid import uuid4

import numpy as np
import pytest

from src.application.use_cases.model_training_use_case import (
    ModelTrainingError,
    ModelTrainingUseCase,
)
from src.domain.entities.model import (
    DenseLayerConfig,
    Model,
    ModelStatus,
    ModelType,
    RNNLayerConfig,
)
from src.domain.entities.time_series import HistoricDataPoint
from src.domain.repositories.model_artifacts_repository import (
    IModelArtifactsRepository,
    ModelArtifact,
)


class _ArtifactsRepo(IModelArtifactsRepository):
    async def save_artifact(
        self,
        model_id,
        artifact_type,
        content,
        metadata=None,
        filename=None,
    ):
        return "artifact"

    async def get_artifact(
        self, model_id, artifact_type
    ):  # pragma: no cover - helper stub
        return None

    async def get_artifact_by_id(
        self, artifact_id: str
    ) -> ModelArtifact | None:  # pragma: no cover
        return None

    async def delete_artifact(self, artifact_id: str) -> bool:  # pragma: no cover
        return True

    async def delete_model_artifacts(self, model_id) -> int:  # pragma: no cover
        return 0

    async def list_model_artifacts(self, model_id):  # pragma: no cover
        return {}


class _StubModel:
    def predict(self, x, verbose=0):
        return np.zeros((x.shape[0], 1))

    def save(self, path: str) -> None:  # pragma: no cover - helper stub
        with open(path, "wb") as handler:
            handler.write(b"model")


class _RecordingArtifactsRepo(IModelArtifactsRepository):
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def save_artifact(
        self,
        model_id,
        artifact_type,
        content,
        metadata=None,
        filename=None,
    ):
        self.calls.append(artifact_type)
        return f"{artifact_type}-id"

    async def get_artifact(
        self, model_id, artifact_type
    ) -> ModelArtifact | None:  # pragma: no cover
        return None

    async def get_artifact_by_id(
        self, artifact_id: str
    ) -> ModelArtifact | None:  # pragma: no cover
        return None

    async def delete_artifact(self, artifact_id: str) -> bool:  # pragma: no cover
        return True

    async def delete_model_artifacts(self, model_id) -> int:  # pragma: no cover
        return 0

    async def list_model_artifacts(self, model_id):  # pragma: no cover
        return {}


@pytest.fixture()
def model_config() -> Model:
    return Model(
        id=uuid4(),
        name="Test",
        description="",
        model_type=ModelType.LSTM,
        status=ModelStatus.DRAFT,
        batch_size=32,
        epochs=10,
        learning_rate=0.01,
        validation_ratio=0.1,
        test_ratio=0.1,
        lookback_window=5,
        forecast_horizon=1,
        feature="temp",
        entity_type="Sensor",
        entity_id="urn",
        rnn_layers=[RNNLayerConfig(units=32)],
    )


@pytest.mark.asyncio
async def test_model_training_use_case_executes(monkeypatch, model_config):
    use_case = ModelTrainingUseCase(artifacts_repository=_ArtifactsRepo())

    monkeypatch.setattr(use_case, "_validate_config", lambda cfg: None)
    monkeypatch.setattr(
        use_case.data_preprocessing,
        "execute",
        lambda **kwargs: (
            np.ones((50, 1, 1), dtype=np.float32),
            np.ones((10, 1, 1), dtype=np.float32),
            np.ones((10, 1, 1), dtype=np.float32),
            np.ones((50, 1, 1), dtype=np.float32),
            np.ones((10, 1, 1), dtype=np.float32),
            np.ones((10, 1, 1), dtype=np.float32),
            object(),
            object(),
            ["value"],
        ),
    )
    monkeypatch.setattr(use_case, "_build_model", lambda **kwargs: _StubModel())
    monkeypatch.setattr(
        use_case,
        "_train_model",
        lambda **kwargs: ({"history": []}, 0.1, 0.2),
    )
    monkeypatch.setattr(
        use_case,
        "_evaluate_model",
        lambda **kwargs: {
            "mse": 0.1,
            "mae": 0.05,
            "rmse": 0.2,
            "theil_u": 0.1,
            "mape": 1.0,
            "r2": 0.9,
            "mae_pct": 0.1,
            "rmse_pct": 0.2,
        },
    )

    async def save_artifacts(**kwargs):
        return ("m", "x", "y", "meta")

    monkeypatch.setattr(use_case, "_save_artifacts", save_artifacts)

    data = [
        HistoricDataPoint(timestamp=datetime.now(timezone.utc), value=float(i))
        for i in range(200)
    ]

    metrics, model_id, x_id, y_id, meta_id = await use_case.execute(
        model_config=model_config,
        collected_data=data,
        window_size=5,
        training_job_id="job",
    )

    assert metrics.mse == 0.1
    assert model_id == "m"
    assert x_id == "x"
    assert y_id == "y"
    assert meta_id == "meta"


def _base_config() -> Model:
    return Model(
        id=uuid4(),
        name="Sample",
        description="",
        model_type=ModelType.GRU,
        status=ModelStatus.DRAFT,
        batch_size=8,
        epochs=5,
        learning_rate=0.01,
        validation_ratio=0.1,
        test_ratio=0.1,
        lookback_window=4,
        forecast_horizon=1,
        feature="temp",
        entity_type="Sensor",
        entity_id="urn",
        rnn_layers=[RNNLayerConfig(units=8)],
    )


@pytest.mark.parametrize(
    "modifier,expected",
    [
        (lambda cfg: cfg.rnn_layers.clear(), "At least one RNN layer"),
        (
            lambda cfg: cfg.rnn_layers.__setitem__(
                slice(None), [RNNLayerConfig(units=0)]
            ),
            "positive number of units",
        ),
        (
            lambda cfg: cfg.rnn_layers.__setitem__(
                slice(None), [RNNLayerConfig(units=4, dropout=1.0)]
            ),
            "dropout must be between 0 and 1",
        ),
        (
            lambda cfg: cfg.rnn_layers.__setitem__(
                slice(None), [RNNLayerConfig(units=4, recurrent_dropout=1.0)]
            ),
            "recurrent_dropout must be between 0 and 1",
        ),
        (
            lambda cfg: cfg.dense_layers.extend([DenseLayerConfig(units=0)]),
            "dense layers must have a positive number",
        ),
        (
            lambda cfg: cfg.dense_layers.extend(
                [DenseLayerConfig(units=8, dropout=1.0)]
            ),
            "Dense layer dropout must be between 0 and 1",
        ),
        (
            lambda cfg: setattr(cfg, "validation_ratio", 1.1),
            "Validation ratio must be between",
        ),
        (lambda cfg: setattr(cfg, "test_ratio", 1.1), "Test ratio must be between"),
        (
            lambda cfg: (
                setattr(cfg, "validation_ratio", 0.6),
                setattr(cfg, "test_ratio", 0.5),
            ),
            "must be less than 1",
        ),
        (
            lambda cfg: setattr(cfg, "learning_rate", 0.0),
            "Learning rate must be positive",
        ),
        (lambda cfg: setattr(cfg, "batch_size", 0), "Batch size must be positive"),
        (lambda cfg: setattr(cfg, "epochs", 0), "Epochs must be positive"),
    ],
)
def test_validate_config_rejects_invalid_parameters(modifier, expected):
    use_case = ModelTrainingUseCase(artifacts_repository=_ArtifactsRepo())
    cfg = _base_config()

    modifier(cfg)

    with pytest.raises(ModelTrainingError) as exc:
        use_case._validate_config(cfg)

    assert expected in str(exc.value)


def test_calculate_percentage_metrics_with_zero_baseline():
    use_case = ModelTrainingUseCase(artifacts_repository=_ArtifactsRepo())

    mae_pct, rmse_pct, mape = use_case._calculate_percentage_metrics(
        np.array([0.0, 0.0]), np.array([1.0, -1.0])
    )

    assert math.isnan(mae_pct)
    assert math.isnan(rmse_pct)
    assert math.isnan(mape)


def test_calculate_percentage_metrics_returns_values():
    use_case = ModelTrainingUseCase(artifacts_repository=_ArtifactsRepo())
    y_true = np.array([10.0, 12.0, 15.0])
    y_pred = np.array([11.0, 11.0, 14.0])

    mae_pct, rmse_pct, mape = use_case._calculate_percentage_metrics(y_true, y_pred)

    assert pytest.approx(mae_pct, rel=1e-3) == 8.333
    assert pytest.approx(mape, rel=1e-3) == mae_pct
    assert pytest.approx(rmse_pct, rel=1e-3) == 8.444


def test_calculate_theil_u_handles_small_series():
    use_case = ModelTrainingUseCase(artifacts_repository=_ArtifactsRepo())
    result = use_case._calculate_theil_u(np.array([1.0]), np.array([1.0]))
    assert math.isnan(result)


def test_calculate_theil_u_returns_ratio():
    use_case = ModelTrainingUseCase(artifacts_repository=_ArtifactsRepo())
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.0, 2.2, 2.8, 4.2])

    result = use_case._calculate_theil_u(y_true, y_pred)
    assert result > 0.0


@pytest.mark.asyncio
async def test_save_artifacts_persists_all_types(model_config):
    use_case = ModelTrainingUseCase(artifacts_repository=_RecordingArtifactsRepo())
    model = _StubModel()

    artifacts = await use_case._save_artifacts(
        model=model,
        x_scaler={"x": 1},
        y_scaler={"y": 1},
        model_config=model_config,
        window_size=5,
        feature_columns=["value"],
        training_history={"best_epoch": 1},
        test_metrics={"mse": 0.1},
        training_duration=1.23,
        data_info={"points": 10},
        training_job_id="job",
    )

    assert artifacts == (
        "model-id",
        "x_scaler-id",
        "y_scaler-id",
        "metadata-id",
    )


@pytest.mark.asyncio
async def test_save_artifacts_wraps_errors(model_config):
    class _FailingRepo(IModelArtifactsRepository):
        async def save_artifact(
            self,
            model_id,
            artifact_type,
            content,
            metadata=None,
            filename=None,
        ):
            raise RuntimeError("disk full")

        async def get_artifact(
            self, model_id, artifact_type
        ) -> ModelArtifact | None:  # pragma: no cover
            return None

        async def get_artifact_by_id(
            self, artifact_id: str
        ) -> ModelArtifact | None:  # pragma: no cover
            return None

        async def delete_artifact(self, artifact_id: str) -> bool:  # pragma: no cover
            return False

        async def delete_model_artifacts(self, model_id) -> int:  # pragma: no cover
            return 0

        async def list_model_artifacts(self, model_id):  # pragma: no cover
            return {}

    use_case = ModelTrainingUseCase(artifacts_repository=_FailingRepo())

    with pytest.raises(ModelTrainingError) as exc:
        await use_case._save_artifacts(
            model=_StubModel(),
            x_scaler={"x": 1},
            y_scaler={"y": 1},
            model_config=model_config,
            window_size=5,
            feature_columns=["value"],
            training_history={},
            test_metrics={},
            training_duration=0.0,
            data_info={},
            training_job_id="job",
        )

    assert "Failed to save artifacts" in str(exc.value)


@pytest.mark.asyncio
async def test_execute_wraps_unexpected_errors(monkeypatch, model_config):
    use_case = ModelTrainingUseCase(artifacts_repository=_ArtifactsRepo())

    def explode(config):
        raise ValueError("broken config")

    monkeypatch.setattr(use_case, "_validate_config", explode)

    data = [
        HistoricDataPoint(timestamp=datetime.now(timezone.utc), value=float(i))
        for i in range(50)
    ]

    with pytest.raises(ModelTrainingError) as exc:
        await use_case.execute(
            model_config=model_config,
            collected_data=data,
            window_size=5,
            training_job_id="job",
        )

    assert "Model training failed" in str(exc.value)
