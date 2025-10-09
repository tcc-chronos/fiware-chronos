"""Use cases for managing recurring model predictions."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

from src.application.dtos.prediction_dto import (
    PredictionHistoryPointDTO,
    PredictionHistoryRequestDTO,
    PredictionHistoryResponseDTO,
    PredictionToggleRequestDTO,
    PredictionToggleResponseDTO,
)
from src.domain.entities.iot import DeviceAttribute
from src.domain.entities.model import ModelStatus
from src.domain.entities.time_series import HistoricDataPoint
from src.domain.entities.training_job import TrainingJob, TrainingStatus
from src.domain.gateways.iot_agent_gateway import IIoTAgentGateway
from src.domain.gateways.orion_gateway import IOrionGateway
from src.domain.gateways.sth_comet_gateway import ISTHCometGateway
from src.domain.repositories.model_repository import IModelRepository
from src.domain.repositories.training_job_repository import ITrainingJobRepository


class PredictionManagementError(Exception):
    """Base exception for prediction management failures."""


class PredictionNotReadyError(PredictionManagementError):
    """Raised when a training job is not eligible for recurring predictions."""


class TogglePredictionUseCase:
    """Enable or disable recurring predictions for a training job."""

    def __init__(
        self,
        model_repository: IModelRepository,
        training_job_repository: ITrainingJobRepository,
        iot_agent_gateway: IIoTAgentGateway,
        orion_gateway: IOrionGateway,
        *,
        fiware_service: str,
        fiware_service_path: str,
        forecast_service_group: str,
        forecast_service_apikey: str,
        forecast_service_resource: str,
        forecast_entity_type: str,
        orion_cb_url: str,
        sth_notification_url: str,
        forecast_device_transport: str,
        forecast_device_protocol: str,
    ) -> None:
        self._model_repository = model_repository
        self._training_job_repository = training_job_repository
        self._iot_agent_gateway = iot_agent_gateway
        self._orion_gateway = orion_gateway
        self._fiware_service = fiware_service
        self._fiware_service_path = fiware_service_path
        self._forecast_service_group = forecast_service_group
        self._forecast_service_apikey = forecast_service_apikey
        self._forecast_service_resource = forecast_service_resource
        self._forecast_entity_type = forecast_entity_type
        self._orion_cb_url = orion_cb_url
        self._sth_notification_url = sth_notification_url.rstrip("/") + "/notify"
        self._forecast_device_transport = forecast_device_transport
        self._forecast_device_protocol = forecast_device_protocol

    async def execute(
        self,
        model_id: UUID,
        training_job_id: UUID,
        request: PredictionToggleRequestDTO,
    ) -> PredictionToggleResponseDTO:
        model = await self._model_repository.find_by_id(model_id)
        if not model:
            raise PredictionManagementError(f"Model {model_id} was not found")
        if model.status != ModelStatus.TRAINED:
            raise PredictionManagementError(
                "Model must be in 'trained' status before enabling predictions"
            )

        training_job = await self._training_job_repository.get_by_id(training_job_id)
        if not training_job:
            raise PredictionManagementError(
                f"Training job {training_job_id} was not found"
            )

        if training_job.model_id != model.id:
            raise PredictionManagementError(
                "Training job does not belong to the specified model"
            )

        if request.enabled:
            await self._enable_predictions(
                model_id,
                training_job_id,
                model,
                training_job,
                metadata=dict(request.metadata or {}),
            )
        else:
            await self._disable_predictions(training_job)

        refreshed = await self._training_job_repository.get_by_id(training_job_id)
        if not refreshed:
            raise PredictionManagementError(
                f"Training job {training_job_id} was not found after update"
            )

        return PredictionToggleResponseDTO(
            model_id=model.id,
            training_job_id=training_job_id,
            enabled=refreshed.prediction_config.enabled,
            entity_id=refreshed.prediction_config.entity_id,
            next_prediction_at=refreshed.next_prediction_at,
            sampling_interval_seconds=refreshed.sampling_interval_seconds,
        )

    async def _enable_predictions(
        self,
        model_id: UUID,
        training_job_id: UUID,
        model,
        training_job,
        *,
        metadata: dict,
    ) -> None:
        if training_job.status != TrainingStatus.COMPLETED:
            raise PredictionNotReadyError(
                "Training job must be completed before enabling predictions"
            )
        if not model.entity_id:
            raise PredictionManagementError(
                "Model entity_id must be configured to emit predictions"
            )
        if not model.feature:
            raise PredictionManagementError(
                "Model feature must be set before enabling predictions"
            )

        await self._iot_agent_gateway.ensure_service_group(
            service=self._fiware_service,
            service_path=self._fiware_service_path,
            apikey=self._forecast_service_apikey,
            entity_type=self._forecast_entity_type,
            resource=self._forecast_service_resource,
            cbroker=self._orion_cb_url,
        )

        entity_id = self._build_entity_id(training_job_id)
        if training_job.prediction_config.entity_id != entity_id:
            training_job.prediction_config.entity_id = entity_id

        device_id = metadata.get("device_id")
        if not device_id:
            device_id = self._build_device_id(model_id, training_job_id)
            metadata.setdefault("device_id", device_id)
        device_transport = metadata.get(
            "device_transport", self._forecast_device_transport
        )
        metadata.setdefault("device_transport", device_transport)
        device_protocol = metadata.get(
            "device_protocol", self._forecast_device_protocol
        )
        metadata.setdefault("device_protocol", device_protocol)

        device_attributes = [
            DeviceAttribute(
                object_id="fs", name="forecastSeries", type="StructuredValue"
            )
        ]

        await self._iot_agent_gateway.ensure_device(
            device_id=device_id,
            entity_name=entity_id,
            entity_type=self._forecast_entity_type,
            attributes=device_attributes,
            transport=device_transport,
            protocol=device_protocol,
            service=self._fiware_service,
            service_path=self._fiware_service_path,
        )

        now_value = self._format_datetime(datetime.now(timezone.utc))

        base_payload = {
            "sourceEntity": {"type": "Relationship", "value": model.entity_id},
            "feature": {"type": "Text", "value": model.feature},
            "forecastSeries": {
                "type": "StructuredValue",
                "value": {
                    "generatedAt": now_value,
                    "horizon": model.forecast_horizon,
                    "points": [],
                },
                "metadata": {
                    "TimeInstant": {
                        "type": "DateTime",
                        "value": now_value,
                    }
                },
            },
        }

        await self._orion_gateway.ensure_entity(
            entity_id=entity_id,
            entity_type=self._forecast_entity_type,
            payload=base_payload,
            service=self._fiware_service,
            service_path=self._fiware_service_path,
        )

        subscription_id = training_job.prediction_config.subscription_id
        if not subscription_id:
            subscription_id = await self._orion_gateway.create_subscription(
                entity_id=entity_id,
                entity_type=self._forecast_entity_type,
                attrs=["forecastSeries"],
                notification_url=self._sth_notification_url,
                service=self._fiware_service,
                service_path=self._fiware_service_path,
            )

        training_job.enable_predictions(
            service_group=self._forecast_service_group,
            entity_id=entity_id,
            entity_type=self._forecast_entity_type,
            metadata=metadata,
            subscription_id=subscription_id,
        )
        await self._training_job_repository.update(training_job)

    async def _disable_predictions(self, training_job: TrainingJob) -> None:
        subscription_id = training_job.prediction_config.subscription_id
        if subscription_id:
            await self._orion_gateway.delete_subscription(
                subscription_id,
                service=self._fiware_service,
                service_path=self._fiware_service_path,
            )

        training_job.disable_predictions(clear_subscription=True)
        await self._training_job_repository.update(training_job)

    def _build_entity_id(self, training_job_id: UUID) -> str:
        compact_id = str(training_job_id).replace("-", "")[:12]
        return f"urn:ngsi-ld:Chronos:Prediction:{compact_id}"

    def _build_device_id(self, model_id: UUID, training_job_id: UUID) -> str:
        return (
            f"chronos-forecast-{str(model_id).replace('-', '')[:12]}-"
            f"{str(training_job_id).replace('-', '')[:12]}"
        )

    def _format_datetime(self, value: datetime) -> str:
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        value = value.astimezone(timezone.utc)
        return value.strftime("%Y-%m-%dT%H:%M:%S.%fZ")


class GetPredictionHistoryUseCase:
    """Retrieve persisted predictions for reporting purposes."""

    def __init__(
        self,
        model_repository: IModelRepository,
        training_job_repository: ITrainingJobRepository,
        sth_gateway: ISTHCometGateway,
        *,
        fiware_service: str,
        fiware_service_path: str,
    ) -> None:
        self._model_repository = model_repository
        self._training_job_repository = training_job_repository
        self._sth_gateway = sth_gateway
        self._fiware_service = fiware_service
        self._fiware_service_path = fiware_service_path

    async def execute(
        self,
        model_id: UUID,
        training_job_id: UUID,
        request: PredictionHistoryRequestDTO,
    ) -> PredictionHistoryResponseDTO:
        training_job = await self._training_job_repository.get_by_id(training_job_id)
        if not training_job or training_job.model_id != model_id:
            raise PredictionManagementError(
                "Training job not found for the specified model"
            )
        model = await self._model_repository.find_by_id(model_id)
        if not model:
            raise PredictionManagementError("Model not found for history retrieval")
        entity_id = training_job.prediction_config.entity_id
        if not entity_id:
            raise PredictionManagementError(
                "Training job does not have an associated prediction entity"
            )

        attribute = "forecastSeries"
        raw_limit = self._determine_sth_limit(request.limit, model.forecast_horizon)

        try:
            total_count = await self._sth_gateway.get_total_count_from_header(
                entity_type=training_job.prediction_config.entity_type or "Prediction",
                entity_id=entity_id,
                attribute=attribute,
                fiware_service=self._fiware_service,
                fiware_servicepath=self._fiware_service_path,
            )
        except Exception as exc:
            raise PredictionManagementError(
                f"Failed to inspect prediction history availability: {exc}"
            ) from exc

        if total_count <= 0:
            return PredictionHistoryResponseDTO(
                entity_id=entity_id,
                feature=attribute,
                points=[],
            )

        h_offset = max(0, total_count - raw_limit)

        data_points = await self._sth_gateway.collect_data(
            entity_type=training_job.prediction_config.entity_type or "Prediction",
            entity_id=entity_id,
            attribute=attribute,
            h_limit=raw_limit,
            h_offset=h_offset,
            fiware_service=self._fiware_service,
            fiware_servicepath=self._fiware_service_path,
        )

        filtered_points: list[HistoricDataPoint] = []
        for item in data_points:
            if request.start and item.timestamp < request.start:
                continue
            if request.end and item.timestamp > request.end:
                continue
            filtered_points.append(item)

        grouped_points: dict[datetime, list[HistoricDataPoint]] = {}
        for item in filtered_points:
            group_key = item.group_timestamp or item.timestamp
            grouped_points.setdefault(group_key, []).append(item)

        ordered_groups: list[tuple[datetime, list[HistoricDataPoint]]] = []
        if grouped_points:
            ordered_groups = sorted(
                grouped_points.items(), key=lambda kv: kv[0], reverse=True
            )

        unique_points: dict[datetime, PredictionHistoryPointDTO] = {}
        insertion_order: list[datetime] = []
        remaining = request.limit

        if ordered_groups:
            latest_ts, latest_entries = ordered_groups[0]
            future_candidates = sorted(latest_entries, key=lambda dp: dp.timestamp)
            for point in future_candidates:
                ts = self._normalize_timestamp(point.timestamp)
                if ts in unique_points:
                    continue
                unique_points[ts] = PredictionHistoryPointDTO(
                    timestamp=ts, value=point.value
                )
                insertion_order.append(ts)
                remaining -= 1
                if remaining <= 0:
                    break

            if remaining > 0:
                for _, entries in ordered_groups[1:]:
                    past_candidates = sorted(
                        entries, key=lambda dp: dp.timestamp, reverse=True
                    )
                    for point in past_candidates:
                        ts = self._normalize_timestamp(point.timestamp)
                        if ts in unique_points:
                            continue
                        unique_points[ts] = PredictionHistoryPointDTO(
                            timestamp=ts, value=point.value
                        )
                        insertion_order.append(ts)
                        remaining -= 1
                        if remaining <= 0:
                            break
                    if remaining <= 0:
                        break
        else:
            for point in sorted(
                filtered_points, key=lambda dp: dp.timestamp, reverse=True
            ):
                ts = self._normalize_timestamp(point.timestamp)
                if ts in unique_points:
                    continue
                unique_points[ts] = PredictionHistoryPointDTO(
                    timestamp=ts, value=point.value
                )
                insertion_order.append(ts)
                remaining -= 1
                if remaining <= 0:
                    break

        points = sorted(
            (unique_points[ts] for ts in insertion_order),
            key=lambda item: item.timestamp,
        )

        return PredictionHistoryResponseDTO(
            entity_id=entity_id,
            feature="forecastSeries",
            points=points,
        )

    def _determine_sth_limit(self, requested_limit: int, forecast_horizon: int) -> int:
        multiplier = max(1, forecast_horizon)
        scaled = requested_limit * multiplier
        if scaled <= requested_limit:
            scaled = requested_limit
        return min(100, max(1, scaled))

    def _normalize_timestamp(self, value: datetime) -> datetime:
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        else:
            value = value.astimezone(timezone.utc)
        return value.replace(microsecond=0)
