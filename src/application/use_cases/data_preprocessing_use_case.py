"""
Application Use Cases - Data Preprocessing

This module contains the use case for preprocessing data for RNN training.
It handles data normalization, sequence creation, and train/validation/test splits.
"""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import structlog
from sklearn.preprocessing import StandardScaler

from src.application.dtos.training_dto import CollectedDataDTO

logger = structlog.get_logger(__name__)


class DataPreprocessingUseCase:
    """
    Prepares data for RNN training:
      - Converts collected data to DataFrame
      - Sorts by timestamp
      - Normalizes features and target (fit ONLY on train set)
      - Creates sequences with window_size
      - Returns train/val/test splits with scalers
    """

    def __init__(
        self,
        test_size: float = 0.15,
        val_size: float = 0.15,
        x_scaler_cls=StandardScaler,
        y_scaler_cls=StandardScaler,
    ):
        """
        Initialize the preprocessing use case.

        Args:
            test_size: Fraction of data for test set
            val_size: Fraction of data for validation set
            x_scaler_cls: Scaler class for features
            y_scaler_cls: Scaler class for target
        """
        self.test_size = float(test_size)
        self.val_size = float(val_size)
        self.x_scaler_cls = x_scaler_cls
        self.y_scaler_cls = y_scaler_cls

    def execute(
        self,
        collected_data: List[CollectedDataDTO],
        window_size: int,
        target_column: str = "value",
        feature_columns: Optional[List[str]] = None,
        validation_ratio: Optional[float] = None,
        test_ratio: Optional[float] = None,
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        object,
        object,
        List[str],
    ]:
        """
        Execute data preprocessing.

        Args:
            collected_data: List of collected data points
            window_size: Size of sequence windows
            target_column: Name of target column
            feature_columns: List of feature column names (if None, uses only target)

        Returns:
            Tuple of (x_train, x_val, x_test, y_train, y_val,
            y_test, x_scaler, y_scaler, feature_columns)

        Raises:
            ValueError: When data is insufficient or invalid
        """
        logger.info(
            "Starting data preprocessing",
            data_points=len(collected_data),
            window_size=window_size,
            target_column=target_column,
        )

        val_ratio = (
            self.val_size if validation_ratio is None else float(validation_ratio)
        )
        test_ratio_value = self.test_size if test_ratio is None else float(test_ratio)

        if not 0.0 < val_ratio < 1.0:
            raise ValueError(
                "Validation ratio must be between 0 (exclusive) and 1 (exclusive)."
            )
        if not 0.0 < test_ratio_value < 1.0:
            raise ValueError(
                "Test ratio must be between 0 (exclusive) and 1 (exclusive)."
            )

        train_ratio = 1.0 - val_ratio - test_ratio_value
        if train_ratio <= 0.0:
            raise ValueError(
                "The sum of validation and test ratios must be less than 1."
            )

        logger.info(
            "Using data split ratios",
            train_ratio=train_ratio,
            validation_ratio=val_ratio,
            test_ratio=test_ratio_value,
        )

        # Convert to DataFrame
        df = self._convert_to_dataframe(collected_data)

        # Sort by timestamp
        df = df.sort_values("timestamp")
        df.reset_index(drop=True, inplace=True)

        logger.info(
            "Data converted and sorted",
            rows=len(df),
            date_range_start=df["timestamp"].min() if len(df) > 0 else None,
            date_range_end=df["timestamp"].max() if len(df) > 0 else None,
        )

        # Determine feature columns
        if feature_columns is None:
            feature_columns = [target_column]
        else:
            # Ensure target column is in feature columns
            if target_column not in feature_columns:
                feature_columns = feature_columns + [target_column]

        # Validate columns exist
        for col in feature_columns + [target_column]:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in data")

        # Extract features and target
        X_raw = df[feature_columns].to_numpy(dtype=np.float32)
        y_raw = df[target_column].to_numpy(dtype=np.float32).reshape(-1, 1)

        # Validate data size
        n = len(df)
        min_required = window_size + 10  # Need extra points for meaningful splits
        if n <= min_required:
            raise ValueError(
                f"Insufficient data: {n} points available, but need at least "
                f"{min_required} for window_size={window_size} "
                f"with train/val/test splits. "
                f"Consider collecting more data or reducing window_size."
            )

        # Temporal split (maintaining order)
        # First separate test set from the end
        n_test = max(1, int(np.floor(test_ratio_value * n)))
        n_trainval = n - n_test

        if n_trainval <= window_size + 2:
            raise ValueError(
                f"Insufficient data for training: {n_trainval} points "
                f"available after test split, but need at least {window_size + 3}"
            )

        X_trainval, X_test = X_raw[:n_trainval], X_raw[n_trainval:]
        y_trainval, y_test = y_raw[:n_trainval], y_raw[n_trainval:]

        # Within trainval, separate validation from the end
        n_val = max(1, int(np.floor(val_ratio * n)))
        if n_val >= n_trainval:
            n_val = max(1, n_trainval - 1)

        n_train = n_trainval - n_val

        if n_train <= window_size + 1:
            raise ValueError(
                f"Insufficient data for training: {n_train} points available "
                f"after val split, but need at least {window_size + 2}"
            )

        X_train, X_val = X_trainval[:n_train], X_trainval[n_train:]
        y_train, y_val = y_trainval[:n_train], y_trainval[n_train:]

        logger.info(
            "Data split completed",
            train_size=len(X_train),
            val_size=len(X_val),
            test_size=len(X_test),
        )

        # Fit scalers on training data only
        x_scaler = self.x_scaler_cls()
        y_scaler = self.y_scaler_cls()

        X_train_scaled = x_scaler.fit_transform(X_train)
        y_train_scaled = y_scaler.fit_transform(y_train)

        # Transform validation and test data
        X_val_scaled = x_scaler.transform(X_val)
        y_val_scaled = y_scaler.transform(y_val)

        X_test_scaled = x_scaler.transform(X_test)
        y_test_scaled = y_scaler.transform(y_test)

        logger.info("Data scaling completed")

        # Create sequences
        x_train, y_train = self._make_supervised(
            X_train_scaled, y_train_scaled, window_size
        )
        x_val, y_val = self._make_supervised(X_val_scaled, y_val_scaled, window_size)
        x_test, y_test = self._make_supervised(
            X_test_scaled, y_test_scaled, window_size
        )

        # Validate sequences
        if len(x_train) == 0 or len(x_val) == 0 or len(x_test) == 0:
            raise ValueError(
                "After creating sequences, one or more splits became empty"
            )

        logger.info(
            "Sequence creation completed",
            train_sequences=len(x_train),
            val_sequences=len(x_val),
            test_sequences=len(x_test),
            sequence_shape=x_train.shape if len(x_train) > 0 else None,
        )

        return (
            x_train,
            x_val,
            x_test,
            y_train,
            y_val,
            y_test,
            x_scaler,
            y_scaler,
            feature_columns,
        )

    def _convert_to_dataframe(
        self, collected_data: List[CollectedDataDTO]
    ) -> pd.DataFrame:
        """Convert collected data to pandas DataFrame."""

        if not collected_data:
            raise ValueError("No data provided for preprocessing")

        # Convert to list of dictionaries
        data_dicts = []
        for item in collected_data:
            data_dicts.append({"timestamp": item.timestamp, "value": item.value})

        df = pd.DataFrame(data_dicts)

        # Ensure timestamp is datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Check for missing values
        if df["value"].isna().any():
            logger.warning(
                "Found missing values in data",
                missing_count=df["value"].isna().sum(),
                total_count=len(df),
            )
            # Drop missing values
            df = df.dropna(subset=["value"])

        # Check for duplicate timestamps
        duplicates = df.duplicated(subset=["timestamp"]).sum()
        if duplicates > 0:
            logger.warning("Found duplicate timestamps", duplicate_count=duplicates)
            # Aggregate duplicate timestamps by taking the mean value
            # This preserves more data than just dropping duplicates
            df = df.groupby("timestamp", as_index=False).agg({"value": "mean"})
            logger.info(
                "Aggregated duplicate timestamps",
                original_count=len(data_dicts),
                final_count=len(df),
                duplicates_resolved=duplicates,
            )

        if len(df) == 0:
            raise ValueError("No valid data points after cleaning")

        return df

    def _make_supervised(
        self, X_scaled: np.ndarray, y_scaled: np.ndarray, window_size: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create supervised learning sequences from time series data.

        Args:
            X_scaled: Scaled feature data
            y_scaled: Scaled target data
            window_size: Size of input sequences

        Returns:
            Tuple of (X_sequences, y_sequences)
        """
        X_seq, y_seq = [], []

        for i in range(window_size, len(X_scaled)):
            # Input window
            window = X_scaled[i - window_size : i]
            # Target (next value)
            target = y_scaled[i]

            # Skip if any NaN values
            if np.any(np.isnan(window)) or np.isnan(target):
                continue

            X_seq.append(window)
            y_seq.append(target)

        if len(X_seq) == 0:
            logger.warning("No valid sequences created")
            return np.array([]), np.array([])

        return np.array(X_seq, dtype=np.float32), np.array(y_seq, dtype=np.float32)
