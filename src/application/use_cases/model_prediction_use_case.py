"""
Application Use Case - Model Prediction

Provides real-time prediction capabilities leveraging previously trained
model artifacts. The use case orchestrates:
  * Model and training job validation
  * Retrieval of persisted artifacts and metadata
  * Collection of the latest sensor readings from STH-Comet
  * Generation of multi-step forecasts using the trained network
"""

from __future__ import annotations

import io
import json
import os
import tempfile
from dataclasses import asdict, is_dataclass
from datetime import datetime, timedelta, timezone
from statistics import median
from typing import Any, Callable, Dict, List, Optional, Sequence
from uuid import UUID

import numpy as np
import structlog
from joblib import load as joblib_load
from tensorflow.keras.models import load_model  # type: ignore

from src.application.dtos.prediction_dto import (
    ForecastPointDTO,
    HistoricalPointDTO,
    PredictionMetadataDTO,
    PredictionResponseDTO,
)
from src.domain.entities.model import Model
from src.domain.entities.time_series import HistoricDataPoint
from src.domain.entities.training_job import (
    TrainingJob,
    TrainingMetrics,
    TrainingStatus,
)
from src.domain.gateways.iot_agent_gateway import IIoTAgentGateway
from src.domain.gateways.sth_comet_gateway import ISTHCometGateway
from src.domain.repositories.model_artifacts_repository import (
    IModelArtifactsRepository,
    ModelArtifact,
)
from src.domain.repositories.model_repository import IModelRepository
from src.domain.repositories.training_job_repository import ITrainingJobRepository

logger = structlog.get_logger(__name__)


class ModelPredictionError(Exception):
    """Base exception for prediction failures."""

    pass


class ModelPredictionNotFoundError(ModelPredictionError):
    """Raised when a required resource cannot be located."""

    pass


class ModelPredictionDependencyError(ModelPredictionError):
    """Raised when dependent services respond with errors."""

    pass


class ModelPredictionUseCase:
    """Coordinates prediction flow using stored model artifacts."""

    def __init__(
        self,
        model_repository: IModelRepository,
        training_job_repository: ITrainingJobRepository,
        artifacts_repository: IModelArtifactsRepository,
        sth_gateway: ISTHCometGateway,
        iot_agent_gateway: IIoTAgentGateway,
        fiware_service: str = "smart",
        fiware_service_path: str = "/",
        model_loader: Optional[Callable[[bytes], Any]] = None,
        scaler_loader: Optional[Callable[[bytes], Any]] = None,
    ):
        self.model_repository = model_repository
        self.training_job_repository = training_job_repository
        self.artifacts_repository = artifacts_repository
        self.sth_gateway = sth_gateway
        self.iot_agent_gateway = iot_agent_gateway
        self.fiware_service = fiware_service
        self.fiware_service_path = fiware_service_path
        self._model_loader = model_loader or self._load_model_from_bytes
        self._scaler_loader = scaler_loader or self._load_scaler_from_bytes

    async def execute(
        self, model_id: UUID, training_job_id: UUID
    ) -> PredictionResponseDTO:
        """Generate predictions for a model using a completed training job."""

        logger.info(
            "prediction.start",
            model_id=str(model_id),
            training_job_id=str(training_job_id),
        )

        model = await self._get_model(model_id)
        training_job = await self._get_training_job(training_job_id, model_id)
        self._validate_training_job(training_job)
        await self._ensure_entity_active(model)

        (
            model_artifact,
            x_scaler_artifact,
            y_scaler_artifact,
            metadata_artifact,
        ) = await self._get_required_artifacts(training_job)

        metadata_payload = self._parse_metadata(metadata_artifact)
        lookback_window = self._resolve_lookback_window(metadata_payload, model)
        forecast_horizon = self._resolve_forecast_horizon(metadata_payload, model)
        feature_columns = metadata_payload.get("feature_columns") or ["value"]

        if not feature_columns or feature_columns != ["value"]:
            raise ModelPredictionError(
                "Prediction currently supports only single feature 'value'."
            )

        trained_model = self._load_trained_model(model_artifact)
        x_scaler = self._load_scaler(x_scaler_artifact)
        y_scaler = self._load_scaler(y_scaler_artifact)

        context_window = await self._collect_recent_points(model, lookback_window)
        predictions = self._generate_predictions(
            context_window=context_window,
            trained_model=trained_model,
            x_scaler=x_scaler,
            y_scaler=y_scaler,
            forecast_horizon=forecast_horizon,
        )

        prediction_timestamps = self._infer_prediction_timestamps(
            context_window, forecast_horizon
        )

        response = self._build_response(
            model=model,
            training_job=training_job,
            context_window=context_window,
            predictions=predictions,
            prediction_timestamps=prediction_timestamps,
            lookback_window=lookback_window,
            forecast_horizon=forecast_horizon,
            metadata_payload=metadata_payload,
        )

        logger.info(
            "prediction.completed",
            model_id=str(model_id),
            training_job_id=str(training_job_id),
            forecast_horizon=forecast_horizon,
        )

        return response

    async def _get_model(self, model_id: UUID) -> Model:
        model = await self.model_repository.find_by_id(model_id)
        if not model:
            logger.warning("prediction.model_not_found", model_id=str(model_id))
            raise ModelPredictionNotFoundError(f"Model {model_id} was not found")
        if not model.entity_id or not model.entity_type or not model.feature:
            raise ModelPredictionError(
                "Model must have entity_type, entity_id and feature configured"
            )
        return model

    async def _get_training_job(
        self, training_job_id: UUID, model_id: UUID
    ) -> TrainingJob:
        training_job = await self.training_job_repository.get_by_id(training_job_id)
        if not training_job:
            logger.warning(
                "prediction.training_not_found", training_job_id=str(training_job_id)
            )
            raise ModelPredictionNotFoundError(
                f"Training job {training_job_id} was not found"
            )
        if training_job.model_id != model_id:
            raise ModelPredictionError(
                "Training job does not belong to the specified model"
            )
        return training_job

    def _validate_training_job(self, training_job: TrainingJob) -> None:
        if training_job.status != TrainingStatus.COMPLETED:
            raise ModelPredictionError(
                "Training job must be completed before predictions can be generated"
            )
        required = [
            training_job.model_artifact_id,
            training_job.x_scaler_artifact_id,
            training_job.y_scaler_artifact_id,
            training_job.metadata_artifact_id,
        ]
        if not all(required):
            raise ModelPredictionError(
                "Training job is missing required artifacts for prediction"
            )

    async def _ensure_entity_active(self, model: Model) -> None:
        try:
            devices = await self.iot_agent_gateway.get_devices(
                service=self.fiware_service, service_path=self.fiware_service_path
            )
        except Exception as exc:  # pragma: no cover - gateway layer handles logging
            raise ModelPredictionDependencyError(
                f"Failed to verify entity availability via IoT Agent: {exc}"
            ) from exc

        normalized_model_entity = model.entity_id.strip()
        for device in devices.devices:
            device_entity_id = (device.entity_name or "").strip()
            alt_entity_id = (device.device_id or "").strip()
            if device.entity_type == model.entity_type and normalized_model_entity in {
                device_entity_id,
                alt_entity_id,
            }:
                attribute_names = {attr.name for attr in device.attributes}
                if model.feature not in attribute_names:
                    raise ModelPredictionError(
                        "Configured feature is not exposed by the IoT Agent entity"
                    )
                return

        raise ModelPredictionNotFoundError(
            "Configured entity is not available in the IoT Agent anymore"
        )

    async def _get_required_artifacts(
        self, training_job: TrainingJob
    ) -> tuple[ModelArtifact, ModelArtifact, ModelArtifact, ModelArtifact]:
        model_artifact_id = self._require_artifact_id(
            training_job.model_artifact_id, "model"
        )
        x_scaler_artifact_id = self._require_artifact_id(
            training_job.x_scaler_artifact_id, "x_scaler"
        )
        y_scaler_artifact_id = self._require_artifact_id(
            training_job.y_scaler_artifact_id, "y_scaler"
        )
        metadata_artifact_id = self._require_artifact_id(
            training_job.metadata_artifact_id, "metadata"
        )

        model_artifact = await self.artifacts_repository.get_artifact_by_id(
            model_artifact_id
        )
        x_scaler_artifact = await self.artifacts_repository.get_artifact_by_id(
            x_scaler_artifact_id
        )
        y_scaler_artifact = await self.artifacts_repository.get_artifact_by_id(
            y_scaler_artifact_id
        )
        metadata_artifact = await self.artifacts_repository.get_artifact_by_id(
            metadata_artifact_id
        )

        if (
            model_artifact is None
            or x_scaler_artifact is None
            or y_scaler_artifact is None
            or metadata_artifact is None
        ):
            raise ModelPredictionNotFoundError(
                "One or more artifacts referenced by the training job were not found"
            )

        return (
            model_artifact,
            x_scaler_artifact,
            y_scaler_artifact,
            metadata_artifact,
        )

    def _require_artifact_id(self, value: Optional[str], label: str) -> str:
        if not value:
            raise ModelPredictionNotFoundError(
                f"Training job does not reference a {label} artifact"
            )
        return value

    def _parse_metadata(self, artifact: ModelArtifact) -> Dict[str, Any]:
        try:
            return json.loads(artifact.content.decode("utf-8"))
        except Exception as exc:
            raise ModelPredictionError(
                "Failed to parse training metadata artifact"
            ) from exc

    def _resolve_lookback_window(self, metadata: Dict[str, Any], model: Model) -> int:
        window = metadata.get("window_size") or model.lookback_window
        try:
            window_int = int(window)
        except (TypeError, ValueError):
            raise ModelPredictionError("Invalid lookback window in metadata")
        if window_int <= 0:
            raise ModelPredictionError("Lookback window must be greater than zero")
        return window_int

    def _resolve_forecast_horizon(self, metadata: Dict[str, Any], model: Model) -> int:
        meta_config = metadata.get("model_config") or {}
        horizon = meta_config.get("forecast_horizon") or model.forecast_horizon
        try:
            horizon_int = int(horizon)
        except (TypeError, ValueError):
            raise ModelPredictionError("Invalid forecast horizon in metadata")
        if horizon_int <= 0:
            raise ModelPredictionError("Forecast horizon must be greater than zero")
        return horizon_int

    def _load_trained_model(self, artifact: ModelArtifact) -> Any:
        try:
            return self._model_loader(artifact.content)
        except Exception as exc:
            raise ModelPredictionError("Failed to load trained model artifact") from exc

    def _load_scaler(self, artifact: ModelArtifact) -> Any:
        try:
            return self._scaler_loader(artifact.content)
        except Exception as exc:
            raise ModelPredictionError("Failed to load scaler artifact") from exc

    def _load_model_from_bytes(self, content: bytes) -> Any:
        with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as tmp_file:
            tmp_file.write(content)
            tmp_file.flush()
            tmp_path = tmp_file.name
        try:
            return load_model(tmp_path)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:  # pragma: no cover - best effort clean-up
                logger.warning(
                    "prediction.model_tempfile_cleanup_failed", path=tmp_path
                )

    def _load_scaler_from_bytes(self, content: bytes) -> Any:
        buffer = io.BytesIO(content)
        try:
            return joblib_load(buffer)
        finally:
            buffer.close()

    async def _collect_recent_points(
        self, model: Model, lookback_window: int
    ) -> List[HistoricDataPoint]:
        try:
            total_available = await self.sth_gateway.get_total_count_from_header(
                entity_type=model.entity_type,
                entity_id=model.entity_id,
                attribute=model.feature,
                fiware_service=self.fiware_service,
                fiware_servicepath=self.fiware_service_path,
            )
        except Exception as exc:
            raise ModelPredictionDependencyError(
                f"Failed to inspect historical availability in STH-Comet: {exc}"
            ) from exc

        if total_available <= 0:
            raise ModelPredictionError(
                "No historical data available for the configured entity and feature"
            )

        if total_available < lookback_window:
            raise ModelPredictionError(
                "Insufficient historical data available for the configured "
                "lookback window"
            )

        collected: List[HistoricDataPoint] = []
        remaining = lookback_window
        current_offset = max(0, total_available - lookback_window)

        while remaining > 0:
            limit = remaining if remaining <= 100 else 100
            try:
                chunk = await self.sth_gateway.collect_data(
                    entity_type=model.entity_type,
                    entity_id=model.entity_id,
                    attribute=model.feature,
                    h_limit=limit,
                    h_offset=current_offset,
                    fiware_service=self.fiware_service,
                    fiware_servicepath=self.fiware_service_path,
                )
            except Exception as exc:
                raise ModelPredictionDependencyError(
                    f"Failed to collect data from STH-Comet: {exc}"
                ) from exc

            if not chunk:
                break

            collected.extend(chunk)
            fetched = len(chunk)
            remaining -= fetched
            current_offset += fetched

            if fetched < limit:
                break

        if len(collected) < lookback_window:
            raise ModelPredictionError(
                "Insufficient historical data available for the "
                "configured lookback window"
            )

        collected.sort(key=lambda point: point.timestamp)
        return collected[-lookback_window:]

    def _generate_predictions(
        self,
        context_window: Sequence[HistoricDataPoint],
        trained_model: Any,
        x_scaler: Any,
        y_scaler: Any,
        forecast_horizon: int,
    ) -> List[float]:
        values = np.array([[point.value] for point in context_window], dtype=np.float32)
        predictions: List[float] = []

        window_values = values.copy()
        for _ in range(forecast_horizon):
            scaled_window = x_scaler.transform(window_values)
            model_input = np.expand_dims(scaled_window, axis=0)
            scaled_prediction = trained_model.predict(model_input, verbose=0)
            unscaled_prediction = y_scaler.inverse_transform(
                scaled_prediction.reshape(-1, 1)
            )
            next_value = float(unscaled_prediction[0][0])
            predictions.append(next_value)
            window_values = np.vstack([window_values[1:], [[next_value]]])

        return predictions

    def _infer_prediction_timestamps(
        self,
        context_window: Sequence[HistoricDataPoint],
        horizon: int,
    ) -> List[Optional[datetime]]:
        if horizon <= 0:
            return []
        if len(context_window) < 2:
            return [None for _ in range(horizon)]

        deltas_seconds = [
            (
                context_window[i].timestamp - context_window[i - 1].timestamp
            ).total_seconds()
            for i in range(1, len(context_window))
        ]
        valid_deltas = [delta for delta in deltas_seconds if delta > 0]
        if not valid_deltas:
            return [None for _ in range(horizon)]

        avg_delta_seconds = median(valid_deltas)
        inferred_seconds = (
            avg_delta_seconds if avg_delta_seconds > 0 else valid_deltas[-1]
        )
        delta = timedelta(seconds=float(inferred_seconds))
        base_time = context_window[-1].timestamp

        timestamps: List[Optional[datetime]] = []
        for _ in range(horizon):
            base_time = base_time + delta
            timestamps.append(base_time)
        return timestamps

    def _build_response(
        self,
        model: Model,
        training_job: TrainingJob,
        context_window: Sequence[HistoricDataPoint],
        predictions: Sequence[float],
        prediction_timestamps: Sequence[Optional[datetime]],
        lookback_window: int,
        forecast_horizon: int,
        metadata_payload: Dict[str, Any],
    ) -> PredictionResponseDTO:
        generated_at = datetime.now(timezone.utc)
        context_dto = [
            HistoricalPointDTO(timestamp=point.timestamp, value=point.value)
            for point in context_window
        ]

        prediction_dto = []
        for idx, value in enumerate(predictions, start=1):
            timestamp = (
                prediction_timestamps[idx - 1]
                if idx - 1 < len(prediction_timestamps)
                else None
            )
            prediction_dto.append(
                ForecastPointDTO(step=idx, value=value, timestamp=timestamp)
            )

        metadata = self._build_metadata(model, training_job, metadata_payload)

        return PredictionResponseDTO(
            model_id=model.id,
            training_job_id=training_job.id,
            lookback_window=lookback_window,
            forecast_horizon=forecast_horizon,
            generated_at=generated_at,
            context_window=context_dto,
            predictions=prediction_dto,
            metadata=metadata,
        )

    def _build_metadata(
        self,
        model: Model,
        training_job: TrainingJob,
        metadata_payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        metrics_payload: Optional[Dict[str, Any]] = None
        if isinstance(training_job.metrics, TrainingMetrics) or is_dataclass(
            training_job.metrics
        ):
            metrics_payload = {
                key: value
                for key, value in asdict(training_job.metrics).items()
                if value is not None
            }

        metadata_dto = PredictionMetadataDTO(
            entity_type=model.entity_type,
            entity_id=model.entity_id,
            feature=model.feature,
            model_status=model.status.value,
            training_metrics=metrics_payload,
            model_info={
                "name": model.name,
                "description": model.description,
                "model_type": model.model_type.value,
                "epochs": model.epochs,
                "batch_size": model.batch_size,
                "learning_rate": model.learning_rate,
            },
        )

        extra_metadata = {
            "training_job_metadata_id": training_job.metadata_artifact_id,
            "model_artifact_id": training_job.model_artifact_id,
            "x_scaler_artifact_id": training_job.x_scaler_artifact_id,
            "y_scaler_artifact_id": training_job.y_scaler_artifact_id,
            "stored_metadata": metadata_payload,
        }

        result = metadata_dto.model_dump()
        result.update(extra_metadata)
        return result
