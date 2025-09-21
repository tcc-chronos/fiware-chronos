"""
Application Use Cases - Model Training

This module contains the use case for training deep learning models (LSTM/GRU).
It handles model creation, compilation, training, and evaluation.
"""

import io
import json
import os
import tempfile
import time
from datetime import datetime
from typing import List, Tuple

import joblib
import numpy as np
import structlog
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau  # type: ignore
from tensorflow.keras.layers import GRU, LSTM, Dense, Dropout, Input  # type: ignore
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore

from src.application.dtos.training_dto import CollectedDataDTO
from src.application.use_cases.data_preprocessing_use_case import (
    DataPreprocessingUseCase,
)
from src.domain.entities.model import Model
from src.domain.entities.training_job import TrainingMetrics
from src.domain.repositories.model_artifacts_repository import IModelArtifactsRepository

logger = structlog.get_logger(__name__)


class ModelTrainingError(Exception):
    """Exception raised when model training fails."""

    pass


class ModelTrainingUseCase:
    """Use case for training deep learning models."""

    def __init__(
        self,
        artifacts_repository: IModelArtifactsRepository,
        save_directory: str = "/tmp/models",
    ):
        """
        Initialize the model training use case.

        Args:
            artifacts_repository: Repository for storing model artifacts
            save_directory: Temporary directory for intermediate files (deprecated)
        """
        self.artifacts_repository = artifacts_repository
        self.save_directory = save_directory  # Keep for backward compatibility
        self.data_preprocessing = DataPreprocessingUseCase()

        # Create save directory if it doesn't exist
        os.makedirs(save_directory, exist_ok=True)

    async def execute(
        self,
        model_config: Model,
        collected_data: List[CollectedDataDTO],
        window_size: int,
    ) -> Tuple[TrainingMetrics, str, str, str, str]:
        """
        Execute model training.

        Args:
            model_config: Model configuration with hyperparameters
            collected_data: Collected training data
            window_size: Sequence window size

        Returns:
            Tuple of (metrics, model_artifact_id, x_scaler_artifact_id,
            y_scaler_artifact_id, metadata_artifact_id)

        Raises:
            ModelTrainingError: When training fails
        """
        start_time = time.time()

        logger.info(
            "Starting model training",
            model_id=str(model_config.id),
            model_type=model_config.model_type,
            data_points=len(collected_data),
            window_size=window_size,
        )

        try:
            # Validate configuration
            self._validate_config(model_config)

            # Preprocess data
            (
                x_train,
                x_val,
                x_test,
                y_train,
                y_val,
                y_test,
                x_scaler,
                y_scaler,
                feature_columns,
            ) = self.data_preprocessing.execute(
                collected_data=collected_data,
                window_size=window_size,
                target_column="value",  # Use "value" as the consistent column name
                feature_columns=["value"],  # Use "value" for consistency
            )

            logger.info(
                "Data preprocessing completed",
                train_sequences=len(x_train),
                val_sequences=len(x_val),
                test_sequences=len(x_test),
                feature_count=x_train.shape[2] if len(x_train) > 0 else 0,
            )

            # Build and compile model
            model = self._build_model(
                window_size=window_size,
                n_features=x_train.shape[2],
                config=model_config,
            )

            # Train model
            training_history, best_train_loss, best_val_loss = self._train_model(
                model=model,
                x_train=x_train,
                x_val=x_val,
                y_train=y_train,
                y_val=y_val,
                config=model_config,
            )

            # Evaluate on test set
            test_metrics = self._evaluate_model(
                model=model, x_test=x_test, y_test=y_test, y_scaler=y_scaler
            )

            end_time = time.time()
            training_duration = end_time - start_time

            logger.info(
                "Model training completed",
                training_duration=training_duration,
                test_metrics=test_metrics,
            )

            # Save artifacts
            (
                model_artifact_id,
                x_scaler_artifact_id,
                y_scaler_artifact_id,
                metadata_artifact_id,
            ) = await self._save_artifacts(
                model=model,
                x_scaler=x_scaler,
                y_scaler=y_scaler,
                model_config=model_config,
                window_size=window_size,
                feature_columns=feature_columns,
                training_history=training_history,
                test_metrics=test_metrics,
                training_duration=training_duration,
                data_info={
                    "total_points": len(collected_data),
                    "train_sequences": len(x_train),
                    "val_sequences": len(x_val),
                    "test_sequences": len(x_test),
                },
            )

            # Create metrics object
            metrics = TrainingMetrics(
                mse=test_metrics["mse"],
                mae=test_metrics["mae"],
                rmse=test_metrics["rmse"],
                mape=test_metrics["mape"],
                r2=test_metrics["r2"],
                mae_pct=test_metrics["mae_pct"],
                rmse_pct=test_metrics["rmse_pct"],
                best_train_loss=best_train_loss,
                best_val_loss=best_val_loss,
                best_epoch=training_history.get("best_epoch"),
            )

            return (
                metrics,
                model_artifact_id,
                x_scaler_artifact_id,
                y_scaler_artifact_id,
                metadata_artifact_id,
            )

        except Exception as e:
            logger.error(
                "model_training.execution_failed",
                model_id=str(model_config.id),
                error=str(e),
                exc_info=e,
            )
            raise ModelTrainingError(f"Model training failed: {str(e)}") from e

    def _validate_config(self, config: Model) -> None:
        """Validate model configuration."""

        if not config.rnn_units or len(config.rnn_units) == 0:
            raise ModelTrainingError("RNN units must be specified")

        if any(units <= 0 for units in config.rnn_units):
            raise ModelTrainingError("All RNN units must be positive")

        if config.dense_units and any(units <= 0 for units in config.dense_units):
            raise ModelTrainingError("All dense units must be positive")

        if not (0 <= config.rnn_dropout < 1):
            raise ModelTrainingError("RNN dropout must be between 0 and 1")

        if not (0 <= config.dense_dropout < 1):
            raise ModelTrainingError("Dense dropout must be between 0 and 1")

        if config.learning_rate <= 0:
            raise ModelTrainingError("Learning rate must be positive")

        if config.batch_size <= 0:
            raise ModelTrainingError("Batch size must be positive")

        if config.epochs <= 0:
            raise ModelTrainingError("Epochs must be positive")

    def _build_model(
        self, window_size: int, n_features: int, config: Model
    ) -> Sequential:
        """Build and compile the neural network model."""

        model = Sequential()
        model.add(Input(shape=(window_size, n_features)))

        # Determine RNN layer type
        RNNLayer = LSTM if config.model_type.lower() == "lstm" else GRU

        # Add RNN layers
        for i, units in enumerate(config.rnn_units):
            return_sequences = i < len(config.rnn_units) - 1

            model.add(
                RNNLayer(
                    units,
                    return_sequences=return_sequences,
                    dropout=config.rnn_dropout,
                    recurrent_dropout=0.0,  # Avoid performance penalty
                )
            )

            # Add dropout after each RNN layer if specified
            if config.rnn_dropout > 0:
                model.add(Dropout(config.rnn_dropout))

        # Add dense layers
        for units in config.dense_units:
            model.add(Dense(units, activation="relu"))

            # Add dropout after each dense layer if specified
            if config.dense_dropout > 0:
                model.add(Dropout(config.dense_dropout))

        # Output layer
        model.add(Dense(1))

        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=config.learning_rate),
            loss="mean_squared_error",
        )

        logger.info(
            "Model built and compiled",
            total_params=model.count_params(),
            model_type=config.model_type,
            rnn_units=config.rnn_units,
            dense_units=config.dense_units,
        )

        return model

    def _train_model(
        self,
        model: Sequential,
        x_train: np.ndarray,
        x_val: np.ndarray,
        y_train: np.ndarray,
        y_val: np.ndarray,
        config: Model,
    ) -> Tuple[dict, float, float]:
        """Train the model."""

        # Prepare callbacks
        callbacks = []

        # Early stopping
        if config.early_stopping_patience:
            early_stop = EarlyStopping(
                monitor="val_loss",
                patience=config.early_stopping_patience,
                restore_best_weights=True,
                verbose=1,
            )
            callbacks.append(early_stop)

        # Learning rate reduction
        reduce_lr = ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=max(
                3,
                (
                    config.early_stopping_patience // 3
                    if config.early_stopping_patience
                    else 3
                ),
            ),
            min_lr=1e-6,
            verbose=1,
        )
        callbacks.append(reduce_lr)

        logger.info(
            "Starting model training",
            epochs=config.epochs,
            batch_size=config.batch_size,
            callbacks=len(callbacks),
        )

        # Train model
        history = model.fit(
            x_train,
            y_train,
            epochs=config.epochs,
            batch_size=config.batch_size,
            validation_data=(x_val, y_val),
            callbacks=callbacks,
            shuffle=False,  # Important for time series
            verbose=1,
        )

        # Find best epoch
        val_losses = history.history["val_loss"]
        best_epoch = int(np.argmin(val_losses))
        best_val_loss = float(val_losses[best_epoch])
        best_train_loss = float(history.history["loss"][best_epoch])

        training_history = {
            "best_epoch": best_epoch,
            "epochs_trained": len(val_losses),
            "history": {
                "loss": [float(x) for x in history.history["loss"]],
                "val_loss": [float(x) for x in history.history["val_loss"]],
            },
        }

        logger.info(
            "Model training completed",
            best_epoch=best_epoch,
            best_train_loss=best_train_loss,
            best_val_loss=best_val_loss,
            epochs_trained=len(val_losses),
        )

        return training_history, best_train_loss, best_val_loss

    def _evaluate_model(
        self, model: Sequential, x_test: np.ndarray, y_test: np.ndarray, y_scaler
    ) -> dict:
        """Evaluate model on test set."""

        # Make predictions
        y_pred_scaled = model.predict(x_test, verbose=0).reshape(-1, 1)
        y_test_scaled = y_test.reshape(-1, 1)

        # Transform back to original scale
        y_pred = y_scaler.inverse_transform(y_pred_scaled).ravel()
        y_true = y_scaler.inverse_transform(y_test_scaled).ravel()

        # Calculate metrics
        mse = float(mean_squared_error(y_true, y_pred))
        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(mse))
        r2 = float(r2_score(y_true, y_pred))

        # Calculate percentage metrics
        mae_pct, rmse_pct, mape = self._calculate_percentage_metrics(y_true, y_pred)

        metrics = {
            "mse": mse,
            "mae": mae,
            "rmse": rmse,
            "mape": mape,
            "r2": r2,
            "mae_pct": mae_pct,
            "rmse_pct": rmse_pct,
        }

        logger.info("Model evaluation completed", metrics=metrics)

        return metrics

    def _calculate_percentage_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Tuple[float, float, float]:
        """Calculate percentage-based metrics."""

        eps = 1e-8
        mask = np.abs(y_true) > eps

        if not np.any(mask):
            return float("nan"), float("nan"), float("nan")

        # Calculate percentage errors
        abs_pct_errors = np.abs((y_true[mask] - y_pred[mask]) / np.abs(y_true[mask]))

        mae_pct = float(np.mean(abs_pct_errors) * 100.0)
        rmse_pct = float(
            np.sqrt(
                np.mean(((y_true[mask] - y_pred[mask]) / np.abs(y_true[mask])) ** 2)
            )
            * 100.0
        )
        mape = mae_pct  # MAPE is equivalent to MAE% when relative to |y_true|

        return mae_pct, rmse_pct, mape

    async def _save_artifacts(
        self,
        model: Sequential,
        x_scaler,
        y_scaler,
        model_config: Model,
        window_size: int,
        feature_columns: List[str],
        training_history: dict,
        test_metrics: dict,
        training_duration: float,
        data_info: dict,
    ) -> Tuple[str, str, str, str]:
        """Save model artifacts to GridFS and return artifact IDs."""

        model_id = model_config.id
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        try:
            # Prepare metadata
            metadata = {
                "model_id": str(model_id),
                "timestamp": timestamp,
                "window_size": window_size,
                "feature_columns": feature_columns,
                "training_history": training_history,
                "test_metrics": test_metrics,
                "training_duration": training_duration,
                "data_info": data_info,
                "model_config": {
                    "model_type": model_config.model_type.value,
                    "rnn_units": model_config.rnn_units,
                    "dense_units": model_config.dense_units,
                    "batch_size": model_config.batch_size,
                    "epochs": model_config.epochs,
                    "learning_rate": model_config.learning_rate,
                    "validation_split": model_config.validation_split,
                    "lookback_window": model_config.lookback_window,
                    "forecast_horizon": model_config.forecast_horizon,
                },
            }

            # Save model to bytes (Keras 3 format)
            with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as tmp:
                model.save(tmp.name)  # ou tmp.name + ".h5"
                tmp.seek(0)
                model_bytes = tmp.read()

            os.unlink(tmp.name)

            # Save x_scaler to bytes
            x_scaler_buffer = io.BytesIO()
            joblib.dump(x_scaler, x_scaler_buffer)
            x_scaler_bytes = x_scaler_buffer.getvalue()
            x_scaler_buffer.close()

            # Save y_scaler to bytes
            y_scaler_buffer = io.BytesIO()
            joblib.dump(y_scaler, y_scaler_buffer)
            y_scaler_bytes = y_scaler_buffer.getvalue()
            y_scaler_buffer.close()

            # Save metadata to bytes
            metadata_bytes = json.dumps(metadata, indent=2, default=str).encode("utf-8")

            # Save all artifacts to GridFS
            model_artifact_id = await self.artifacts_repository.save_artifact(
                model_id=model_id,
                artifact_type="model",
                content=model_bytes,
                metadata={"format": "tensorflow", "size": len(model_bytes)},
                filename=f"{model_id}_{timestamp}_model.tf",
            )

            x_scaler_artifact_id = await self.artifacts_repository.save_artifact(
                model_id=model_id,
                artifact_type="x_scaler",
                content=x_scaler_bytes,
                metadata={"format": "pickle", "size": len(x_scaler_bytes)},
                filename=f"{model_id}_{timestamp}_x_scaler.pkl",
            )

            y_scaler_artifact_id = await self.artifacts_repository.save_artifact(
                model_id=model_id,
                artifact_type="y_scaler",
                content=y_scaler_bytes,
                metadata={"format": "pickle", "size": len(y_scaler_bytes)},
                filename=f"{model_id}_{timestamp}_y_scaler.pkl",
            )

            metadata_artifact_id = await self.artifacts_repository.save_artifact(
                model_id=model_id,
                artifact_type="metadata",
                content=metadata_bytes,
                metadata={"format": "json", "size": len(metadata_bytes)},
                filename=f"{model_id}_{timestamp}_metadata.json",
            )

            logger.info(
                "Model artifacts saved to GridFS",
                model_id=str(model_id),
                model_artifact_id=model_artifact_id,
                x_scaler_artifact_id=x_scaler_artifact_id,
                y_scaler_artifact_id=y_scaler_artifact_id,
                metadata_artifact_id=metadata_artifact_id,
            )

            return (
                model_artifact_id,
                x_scaler_artifact_id,
                y_scaler_artifact_id,
                metadata_artifact_id,
            )

        except Exception as e:
            logger.error(
                "Failed to save model artifacts to GridFS",
                model_id=str(model_id),
                error=str(e),
            )
            raise ModelTrainingError(f"Failed to save artifacts: {e}") from e
