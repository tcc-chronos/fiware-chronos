from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
from typing import Callable, Deque, Dict, Iterable, List, Tuple

import httpx
import pytest

from src.domain.entities.prediction import ForecastSeriesPoint, PredictionRecord
from src.infrastructure.gateways.orion_gateway import OrionGateway

ResponseSpec = Tuple[str, str, httpx.Response]


class _MockAsyncClient:
    def __init__(self, expected_calls: Deque[ResponseSpec]) -> None:
        self._expected_calls = expected_calls
        self.requests: List[Dict[str, object]] = []

    async def __aenter__(self) -> "_MockAsyncClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:  # type: ignore[override]
        if self._expected_calls:
            raise AssertionError("Not all expected HTTP calls were consumed")
        return False

    async def get(self, url: str, *, headers=None, json=None):
        return await self._handle("GET", url, headers=headers, json=json)

    async def post(self, url: str, *, headers=None, json=None):
        return await self._handle("POST", url, headers=headers, json=json)

    async def delete(self, url: str, *, headers=None, json=None):
        return await self._handle("DELETE", url, headers=headers, json=json)

    async def _handle(self, method: str, url: str, *, headers=None, json=None):
        if not self._expected_calls:
            raise AssertionError(f"Unexpected {method} request to {url}")
        expected_method, expected_url, response = self._expected_calls.popleft()
        assert expected_method == method
        assert expected_url == url
        self.requests.append(
            {"method": method, "url": url, "headers": headers, "json": json}
        )
        return response


def _make_response(method: str, url: str, status_code: int, **kwargs) -> ResponseSpec:
    request = httpx.Request(method, url)
    payload = kwargs.get("json")
    headers = kwargs.get("headers")
    response = httpx.Response(
        status_code, json=payload, headers=headers, request=request
    )
    return method, url, response


def _mock_client_factory(
    sequences: Iterable[Iterable[ResponseSpec]],
) -> Tuple[Callable[..., _MockAsyncClient], List[_MockAsyncClient]]:
    queue: Deque[Deque[ResponseSpec]] = deque(deque(sequence) for sequence in sequences)
    created: List[_MockAsyncClient] = []

    def factory(*args, **kwargs) -> _MockAsyncClient:
        if not queue:
            raise AssertionError("Unexpected AsyncClient instantiation")
        client = _MockAsyncClient(queue.popleft())
        created.append(client)
        return client

    return factory, created


@pytest.mark.asyncio
async def test_ensure_entity_creates_when_missing(monkeypatch: pytest.MonkeyPatch):
    gateway = OrionGateway("http://orion")
    entity_url = "http://orion/v2/entities/urn:ngsi-ld:Prediction:001"
    create_url = "http://orion/v2/entities"

    factory, created = _mock_client_factory(
        sequences=[
            [_make_response("GET", entity_url, 404)],
            [_make_response("POST", create_url, 201)],
        ]
    )
    monkeypatch.setattr("httpx.AsyncClient", factory)

    await gateway.ensure_entity(
        entity_id="urn:ngsi-ld:Prediction:001",
        entity_type="Prediction",
        payload={
            "sourceEntity": {"type": "Relationship", "value": "urn:ngsi-ld:Sensor:1"}
        },
        service="smart",
        service_path="/",
    )

    assert created[1].requests[0]["headers"]["fiware-service"] == "smart"
    assert created[1].requests[0]["json"]["type"] == "Prediction"


@pytest.mark.asyncio
async def test_upsert_prediction_posts_forecast_series(monkeypatch: pytest.MonkeyPatch):
    gateway = OrionGateway("http://orion")
    attr_url = "http://orion/v2/entities/urn:ngsi-ld:Prediction:001/attrs"

    factory, created = _mock_client_factory(
        sequences=[[_make_response("POST", attr_url, 204)]]
    )
    monkeypatch.setattr("httpx.AsyncClient", factory)

    prediction = PredictionRecord(
        entity_id="urn:ngsi-ld:Prediction:001",
        entity_type="Prediction",
        source_entity="urn:ngsi-ld:Sensor:001",
        model_id="model-id",
        training_id="training-id",
        horizon=2,
        feature="humidity",
        generated_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        series=[
            ForecastSeriesPoint(
                step=1,
                value=61.2,
                target_timestamp=datetime(2025, 1, 1, 0, 15, tzinfo=timezone.utc),
            ),
            ForecastSeriesPoint(
                step=2,
                value=61.9,
                target_timestamp=datetime(2025, 1, 1, 0, 30, tzinfo=timezone.utc),
            ),
        ],
    )

    await gateway.upsert_prediction(prediction, service="smart", service_path="/")

    payload = created[0].requests[0]["json"]
    assert payload["forecastSeries"]["value"]["horizon"] == 2
    assert len(payload["forecastSeries"]["value"]["points"]) == 2


@pytest.mark.asyncio
async def test_create_subscription_returns_location_header(
    monkeypatch: pytest.MonkeyPatch,
):
    gateway = OrionGateway("http://orion")
    subscriptions_url = "http://orion/v2/subscriptions"

    response = _make_response(
        "POST",
        subscriptions_url,
        201,
        headers={"Location": "/v2/subscriptions/abc123"},
    )

    factory, _ = _mock_client_factory(sequences=[[response]])
    monkeypatch.setattr("httpx.AsyncClient", factory)

    subscription_id = await gateway.create_subscription(
        entity_id="urn:ngsi-ld:Prediction:001",
        entity_type="Prediction",
        attrs=["forecastSeries"],
        notification_url="http://chronos/callback",
        service="smart",
        service_path="/",
    )

    assert subscription_id == "abc123"
