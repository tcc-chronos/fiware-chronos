from __future__ import annotations

import httpx
import pytest

from src.infrastructure.gateways.iot_agent_gateway import IoTAgentGateway


class _StubResponse:
    def __init__(self, status_code: int, json_data: dict):
        self.status_code = status_code
        self._json = json_data

    def json(self) -> dict:
        return self._json

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            request = httpx.Request("GET", "http://iot/iot/devices")
            response = httpx.Response(self.status_code, request=request, text="error")
            raise httpx.HTTPStatusError("error", request=request, response=response)


class _StubAsyncClient:
    def __init__(self, response: _StubResponse):
        self._response = response
        self.last_headers = None

    async def __aenter__(self) -> "_StubAsyncClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    async def get(self, url: str, headers: dict):
        self.last_headers = headers
        return self._response


@pytest.mark.asyncio
async def test_get_devices_success(monkeypatch) -> None:
    response = _StubResponse(
        200,
        {
            "count": 1,
            "devices": [
                {
                    "device_id": "device-1",
                    "service": "smart",
                    "service_path": "/",
                    "entity_name": "Device",
                    "entity_type": "Sensor",
                    "transport": "mqtt",
                    "protocol": "PDI",
                    "attributes": [
                        {"object_id": "t", "name": "temp", "type": "Number"}
                    ],
                }
            ],
        },
    )

    monkeypatch.setattr(
        "httpx.AsyncClient",
        lambda timeout: _StubAsyncClient(response),
    )

    gateway = IoTAgentGateway("http://iot")
    collection = await gateway.get_devices()

    assert collection.count == 1
    assert collection.devices[0].attributes[0].name == "temp"


@pytest.mark.asyncio
async def test_get_devices_raises_on_http_error(monkeypatch) -> None:
    response = _StubResponse(500, {})

    monkeypatch.setattr(
        "httpx.AsyncClient",
        lambda timeout: _StubAsyncClient(response),
    )

    gateway = IoTAgentGateway("http://iot")

    with pytest.raises(Exception) as exc:
        await gateway.get_devices()

    assert "HTTP 500" in str(exc.value)


@pytest.mark.asyncio
async def test_get_devices_request_error(monkeypatch) -> None:
    class _FailingClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def get(self, url, headers):
            raise httpx.RequestError("boom")

    monkeypatch.setattr("httpx.AsyncClient", lambda timeout: _FailingClient())

    gateway = IoTAgentGateway("http://iot")

    with pytest.raises(Exception) as exc:
        await gateway.get_devices()

    assert "Failed to communicate" in str(exc.value)
