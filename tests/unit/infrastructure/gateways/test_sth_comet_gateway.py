from __future__ import annotations

from datetime import timezone

import httpx
import pytest

from src.infrastructure.gateways.sth_comet_gateway import STHCometError, STHCometGateway


class _StubResponse:
    def __init__(
        self,
        status_code: int,
        json_data: dict | None = None,
        headers: dict | None = None,
    ):
        self.status_code = status_code
        self._json = json_data or {}
        self.headers = headers or {}
        self.text = "error"

    def json(self) -> dict:
        return self._json

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            request = httpx.Request("GET", "http://sth")
            response = httpx.Response(self.status_code, request=request, text=self.text)
            raise httpx.HTTPStatusError("error", request=request, response=response)


class _StubAsyncClient:
    def __init__(self, response: _StubResponse):
        self._response = response

    async def __aenter__(self) -> "_StubAsyncClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    async def get(self, *args, **kwargs):
        return self._response


@pytest.mark.asyncio
async def test_collect_data_parses_points(monkeypatch) -> None:
    payload = {
        "contextResponses": [
            {
                "contextElement": {
                    "attributes": [
                        {
                            "name": "temperature",
                            "values": [
                                {
                                    "attrValue": "25.5",
                                    "recvTime": "2024-09-09T12:00:00Z",
                                }
                            ],
                        }
                    ]
                }
            }
        ]
    }
    monkeypatch.setattr(
        "httpx.AsyncClient",
        lambda timeout: _StubAsyncClient(_StubResponse(200, payload)),
    )

    gateway = STHCometGateway("http://sth")
    result = await gateway.collect_data("Sensor", "urn:1", "temperature", 10, 0)

    assert len(result) == 1
    assert result[0].value == 25.5
    assert result[0].timestamp.tzinfo == timezone.utc


@pytest.mark.asyncio
async def test_collect_data_validates_limits() -> None:
    gateway = STHCometGateway("http://sth")
    with pytest.raises(STHCometError):
        await gateway.collect_data("Sensor", "urn:1", "temperature", 0, 0)


@pytest.mark.asyncio
async def test_collect_data_handles_http_error(monkeypatch) -> None:
    monkeypatch.setattr(
        "httpx.AsyncClient",
        lambda timeout: _StubAsyncClient(_StubResponse(500)),
    )
    gateway = STHCometGateway("http://sth")
    with pytest.raises(STHCometError):
        await gateway.collect_data("Sensor", "urn:1", "temperature", 10, 0)


@pytest.mark.asyncio
async def test_get_total_count_reads_header(monkeypatch) -> None:
    response = _StubResponse(200, {}, {"fiware-total-count": "123"})
    monkeypatch.setattr(
        "httpx.AsyncClient",
        lambda timeout: _StubAsyncClient(response),
    )
    gateway = STHCometGateway("http://sth")
    total = await gateway.get_total_count_from_header("Sensor", "urn:1", "temperature")
    assert total == 123


@pytest.mark.asyncio
async def test_collect_data_missing_attribute(monkeypatch) -> None:
    payload = {
        "contextResponses": [
            {"contextElement": {"attributes": [{"name": "humidity", "values": []}]}}
        ]
    }

    monkeypatch.setattr(
        "httpx.AsyncClient",
        lambda timeout: _StubAsyncClient(_StubResponse(200, payload)),
    )

    gateway = STHCometGateway("http://sth")
    result = await gateway.collect_data("Sensor", "urn:1", "temperature", 10, 0)
    assert result == []
