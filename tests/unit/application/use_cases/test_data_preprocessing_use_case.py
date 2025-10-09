from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from src.application.use_cases.data_preprocessing_use_case import (
    DataPreprocessingUseCase,
)
from src.domain.entities.time_series import HistoricDataPoint


def _make_series(
    count: int, start: datetime | None = None, step_minutes: int = 5
) -> list[HistoricDataPoint]:
    base = start or datetime.now(timezone.utc)
    return [
        HistoricDataPoint(
            timestamp=base + timedelta(minutes=i * step_minutes), value=float(i)
        )
        for i in range(count)
    ]


def test_preprocessing_creates_sequences() -> None:
    data = _make_series(200)
    use_case = DataPreprocessingUseCase(test_size=0.1, val_size=0.1)

    (
        x_train,
        x_val,
        x_test,
        y_train,
        y_val,
        y_test,
        x_scaler,
        y_scaler,
        features,
    ) = use_case.execute(collected_data=data, window_size=5)

    assert x_train.shape[1:] == (5, 1)
    assert x_val.shape[1:] == (5, 1)
    assert x_test.shape[1:] == (5, 1)
    assert y_train.ndim == 2
    assert features == ["value"]
    assert hasattr(x_scaler, "transform")
    assert hasattr(y_scaler, "transform")


def test_preprocessing_handles_duplicates() -> None:
    base = datetime.now(timezone.utc)
    duplicates = [
        HistoricDataPoint(timestamp=base, value=10.0),
        HistoricDataPoint(timestamp=base, value=20.0),
    ] + _make_series(150, start=base + timedelta(minutes=5))

    use_case = DataPreprocessingUseCase(test_size=0.1, val_size=0.1)
    result = use_case.execute(collected_data=duplicates, window_size=5)

    assert result[0].shape[0] > 0  # train sequences exist


@pytest.mark.parametrize(
    "count",
    [10, 25],
)
def test_preprocessing_validates_minimum_data(count: int) -> None:
    use_case = DataPreprocessingUseCase(test_size=0.2, val_size=0.2)
    data = _make_series(count)

    with pytest.raises(ValueError):
        use_case.execute(collected_data=data, window_size=20)


def test_preprocessing_validates_ratios() -> None:
    data = _make_series(150)
    use_case = DataPreprocessingUseCase()

    with pytest.raises(ValueError):
        use_case.execute(
            collected_data=data,
            window_size=10,
            validation_ratio=0.0,
        )

    with pytest.raises(ValueError):
        use_case.execute(
            collected_data=data,
            window_size=10,
            test_ratio=1.0,
        )


def test_preprocessing_extends_feature_columns() -> None:
    data = _make_series(180)
    use_case = DataPreprocessingUseCase(test_size=0.1, val_size=0.1)

    with pytest.raises(ValueError) as exc:
        use_case.execute(
            collected_data=data,
            window_size=5,
            feature_columns=["extra_feature"],
        )

    assert "Column 'extra_feature' not found" in str(exc.value)


def test_make_supervised_skips_windows_with_nan() -> None:
    use_case = DataPreprocessingUseCase()
    # Force NaNs so sequences are discarded
    x_scaled = np.array([[np.nan], [np.nan], [np.nan]], dtype=np.float32)
    y_scaled = np.array([[np.nan], [np.nan], [np.nan]], dtype=np.float32)

    x_seq, y_seq = use_case._make_supervised(x_scaled, y_scaled, window_size=2)

    assert x_seq.size == 0
    assert y_seq.size == 0


def test_convert_to_dataframe_rejects_empty_and_missing_values() -> None:
    use_case = DataPreprocessingUseCase()

    with pytest.raises(ValueError):
        use_case._convert_to_dataframe([])

    base = datetime.now(timezone.utc)
    points = [
        HistoricDataPoint(timestamp=base, value=float("nan")),
        HistoricDataPoint(timestamp=base + timedelta(minutes=1), value=float("nan")),
    ]

    with pytest.raises(ValueError):
        use_case._convert_to_dataframe(points)


def test_preprocessing_validates_combined_ratios() -> None:
    data = _make_series(200)
    use_case = DataPreprocessingUseCase(test_size=0.6, val_size=0.4)

    with pytest.raises(ValueError):
        use_case.execute(collected_data=data, window_size=5)


def test_preprocessing_requires_sufficient_trainval_data() -> None:
    data = _make_series(50)
    use_case = DataPreprocessingUseCase(test_size=0.4, val_size=0.3)

    with pytest.raises(ValueError):
        use_case.execute(collected_data=data, window_size=40)


def test_preprocessing_requires_sufficient_train_data() -> None:
    data = _make_series(80)
    use_case = DataPreprocessingUseCase(test_size=0.2, val_size=0.2)

    with pytest.raises(ValueError):
        use_case.execute(collected_data=data, window_size=55)
