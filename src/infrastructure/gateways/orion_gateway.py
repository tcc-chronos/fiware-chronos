"""Orion Context Broker gateway implementation."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List

import httpx

from src.domain.entities.prediction import PredictionRecord
from src.domain.gateways.orion_gateway import IOrionGateway
from src.shared import get_logger

logger = get_logger(__name__)


class OrionGateway(IOrionGateway):
    """HTTP-based Orion Context Broker gateway."""

    def __init__(self, base_url: str, timeout: float = 30.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    async def ensure_entity(
        self,
        *,
        entity_id: str,
        entity_type: str,
        payload: Dict[str, Any],
        service: str,
        service_path: str,
    ) -> None:
        """Ensure an Orion entity exists, creating it when absent."""

        url = f"{self._base_url}/v2/entities/{entity_id}"
        headers = self._build_headers(service, service_path, include_content_type=False)

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.get(url, headers=headers)
                if response.status_code == httpx.codes.OK:
                    return
                if response.status_code != httpx.codes.NOT_FOUND:
                    response.raise_for_status()

            create_url = f"{self._base_url}/v2/entities"
            entity_payload = {"id": entity_id, "type": entity_type, **payload}
            logger.info(
                "orion.ensure_entity.create",
                url=create_url,
                entity_id=entity_id,
                entity_type=entity_type,
            )
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(
                    create_url,
                    headers=self._build_headers(
                        service, service_path, include_content_type=True
                    ),
                    json=entity_payload,
                )
                response.raise_for_status()

        except httpx.HTTPStatusError as exc:
            logger.error(
                "orion.ensure_entity.http_error",
                entity_id=entity_id,
                status_code=exc.response.status_code,
                response_text=exc.response.text,
                exc_info=exc,
            )
            raise Exception(
                f"Failed to ensure Orion entity {entity_id}: {exc.response.text}"
            ) from exc
        except httpx.RequestError as exc:
            logger.error(
                "orion.ensure_entity.request_error",
                entity_id=entity_id,
                error=str(exc),
                exc_info=exc,
            )
            raise Exception(
                f"Network error ensuring Orion entity {entity_id}: {exc}"
            ) from exc

    async def upsert_prediction(
        self,
        prediction: PredictionRecord,
        *,
        service: str,
        service_path: str,
    ) -> None:
        """Upsert prediction attributes for the configured entity."""

        attrs = self._build_prediction_payload(prediction)
        url = f"{self._base_url}/v2/entities/{prediction.entity_id}/attrs"
        headers = self._build_headers(service, service_path, include_content_type=True)

        logger.info(
            "orion.prediction.upsert",
            entity_id=prediction.entity_id,
            model_id=prediction.model_id,
            training_id=prediction.training_id,
            horizon=prediction.horizon,
        )

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(url, headers=headers, json=attrs)
                response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            logger.error(
                "orion.prediction.http_error",
                entity_id=prediction.entity_id,
                status_code=exc.response.status_code,
                response_text=exc.response.text,
                exc_info=exc,
            )
            raise Exception(
                f"Failed to publish prediction to Orion: {exc.response.text}"
            ) from exc
        except httpx.RequestError as exc:
            logger.error(
                "orion.prediction.request_error",
                entity_id=prediction.entity_id,
                error=str(exc),
                exc_info=exc,
            )
            raise Exception(f"Network error publishing prediction: {exc}") from exc

    def _build_headers(
        self,
        service: str,
        service_path: str,
        *,
        include_content_type: bool,
    ) -> Dict[str, str]:
        headers: Dict[str, str] = {
            "fiware-service": service,
            "fiware-servicepath": service_path,
        }
        if include_content_type:
            headers["Content-Type"] = "application/json"
        return headers

    async def create_subscription(
        self,
        *,
        entity_id: str,
        entity_type: str,
        attrs: List[str],
        notification_url: str,
        service: str,
        service_path: str,
        attrs_format: str = "legacy",
    ) -> str:
        """Create an Orion subscription and return its ID."""

        payload = {
            "description": "Chronos - Forecast updates",
            "subject": {
                "entities": [{"id": entity_id, "type": entity_type}],
                "condition": {"attrs": attrs},
            },
            "notification": {
                "http": {"url": notification_url},
                "attrs": attrs,
                "attrsFormat": attrs_format,
            },
            "throttling": 1,
        }

        url = f"{self._base_url}/v2/subscriptions"
        headers = self._build_headers(service, service_path, include_content_type=True)

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(url, headers=headers, json=payload)
                response.raise_for_status()

            location = response.headers.get("Location", "")
            subscription_id = location.rsplit("/", 1)[-1] if location else ""
            if not subscription_id:
                data = response.json() if response.content else {}
                subscription_id = data.get("id", "")
            if not subscription_id:
                raise Exception("Failed to obtain subscription ID from Orion response")
            return subscription_id

        except httpx.HTTPStatusError as exc:
            logger.error(
                "orion.subscription.http_error",
                status_code=exc.response.status_code,
                response_text=exc.response.text,
                exc_info=exc,
            )
            raise Exception(
                f"Failed to create Orion subscription: {exc.response.text}"
            ) from exc
        except httpx.RequestError as exc:
            logger.error(
                "orion.subscription.request_error",
                error=str(exc),
                exc_info=exc,
            )
            raise Exception(f"Network error creating subscription: {exc}") from exc

    async def delete_subscription(
        self,
        subscription_id: str,
        *,
        service: str,
        service_path: str,
    ) -> None:
        """Delete an Orion subscription if it exists."""

        url = f"{self._base_url}/v2/subscriptions/{subscription_id}"
        headers = self._build_headers(service, service_path, include_content_type=False)

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.delete(url, headers=headers)
                if response.status_code not in (
                    httpx.codes.NO_CONTENT,
                    httpx.codes.NOT_FOUND,
                ):
                    response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            logger.warning(
                "orion.subscription.delete_failed",
                subscription_id=subscription_id,
                status_code=exc.response.status_code,
                response_text=exc.response.text,
            )
        except httpx.RequestError as exc:
            logger.warning(
                "orion.subscription.delete_request_error",
                subscription_id=subscription_id,
                error=str(exc),
            )

    async def delete_entity(
        self,
        entity_id: str,
        *,
        service: str,
        service_path: str,
    ) -> None:
        """Delete an Orion entity if it exists."""

        url = f"{self._base_url}/v2/entities/{entity_id}"
        headers = self._build_headers(service, service_path, include_content_type=False)

        logger.info(
            "orion.entity.delete.request",
            entity_id=entity_id,
            service=service,
            service_path=service_path,
        )

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.delete(url, headers=headers)
                if response.status_code in (
                    httpx.codes.NO_CONTENT,
                    httpx.codes.NOT_FOUND,
                ):
                    return
                response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            logger.warning(
                "orion.entity.delete_failed",
                entity_id=entity_id,
                status_code=exc.response.status_code,
                response_text=exc.response.text,
            )
        except httpx.RequestError as exc:
            logger.warning(
                "orion.entity.delete_request_error",
                entity_id=entity_id,
                error=str(exc),
            )

    def _build_prediction_payload(self, prediction: PredictionRecord) -> Dict[str, Any]:
        series_payload: List[Dict[str, Any]] = []
        for point in prediction.series:
            target_ts = point.target_timestamp
            if isinstance(target_ts, datetime):
                formatted_ts = self._format_datetime(target_ts)
            else:
                formatted_ts = str(target_ts)
            series_payload.append(
                {
                    "step": point.step,
                    "value": point.value,
                    "targetTimestamp": formatted_ts,
                }
            )

        time_value = self._format_datetime(prediction.generated_at)

        payload: Dict[str, Any] = {
            "sourceEntity": {
                "type": "Relationship",
                "value": prediction.source_entity,
            },
            "feature": {"type": "Text", "value": prediction.feature},
            "forecastSeries": self._build_forecast_series_attribute(
                series_payload, time_value, prediction.horizon
            ),
        }

        for key, value in prediction.metadata.items():
            payload[key] = self._normalize_attribute(value)

        return payload

    def _normalize_attribute(self, value: Any) -> Dict[str, Any]:
        if isinstance(value, dict):
            if "type" in value and "value" in value:
                return value  # already NGSI-style
            return {"type": "StructuredValue", "value": value}

        if isinstance(value, list):
            return {"type": "StructuredValue", "value": value}

        if isinstance(value, (int, float)):
            attr_type = "Number"
        elif isinstance(value, bool):
            attr_type = "Boolean"
        else:
            attr_type = "Text"

        return {"type": attr_type, "value": value}

    def _build_forecast_series_attribute(
        self,
        series_payload: List[Dict[str, Any]],
        time_value: str,
        horizon: int,
    ) -> Dict[str, Any]:
        try:
            points = json.loads(json.dumps(series_payload, ensure_ascii=False))
        except (TypeError, ValueError):
            points = []

        return {
            "type": "StructuredValue",
            "value": {
                "generatedAt": time_value,
                "horizon": horizon,
                "points": points,
            },
            "metadata": {
                "TimeInstant": {
                    "type": "DateTime",
                    "value": time_value,
                },
            },
        }

    def _format_datetime(self, value: datetime) -> str:
        """Format datetime in ISO8601 expected by Orion (UTC with 'Z')."""

        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        value = value.astimezone(timezone.utc)
        return value.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
