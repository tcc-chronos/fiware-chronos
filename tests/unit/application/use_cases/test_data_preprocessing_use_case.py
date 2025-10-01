from __future__ import annotations

from datetime import datetime, timedelta, timezone

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
