from __future__ import annotations

import pytest

from src.application.use_cases.device_use_cases import GetDevicesUseCase
from src.domain.entities.iot import DeviceAttribute, IoTDevice, IoTDeviceCollection
from src.domain.gateways.iot_agent_gateway import IIoTAgentGateway


class _StubGateway(IIoTAgentGateway):
    def __init__(self, collection: IoTDeviceCollection) -> None:
        self._collection = collection
        self.calls: list[tuple[str, str]] = []

    async def get_devices(
        self, service: str = "smart", service_path: str = "/"
    ) -> IoTDeviceCollection:
        self.calls.append((service, service_path))
        return self._collection


@pytest.mark.asyncio
async def test_get_devices_groups_by_entity_type() -> None:
    collection = IoTDeviceCollection(
        count=2,
        devices=[
            IoTDevice(
                device_id="1",
                service="smart",
                service_path="/",
                entity_name="DeviceOne",
                entity_type="Sensor",
                transport="mqtt",
                protocol="PDI-IoTA-UltraLight",
                attributes=[DeviceAttribute(object_id="t", name="temp", type="Number")],
            ),
            IoTDevice(
                device_id="2",
                service="smart",
                service_path="/",
                entity_name="DeviceTwo",
                entity_type="Sensor",
                transport="mqtt",
                protocol="PDI-IoTA-UltraLight",
                attributes=[DeviceAttribute(object_id="h", name="hum", type="Number")],
            ),
        ],
    )
    gateway = _StubGateway(collection)
    use_case = GetDevicesUseCase(iot_agent_gateway=gateway)

    response = await use_case.execute(service="smart", service_path="/")

    assert gateway.calls == [("smart", "/")]
    assert len(response.devices) == 1
    assert response.devices[0].entity_type == "Sensor"
    assert sorted(response.devices[0].entities[0].attributes) == ["temp"]


@pytest.mark.asyncio
async def test_get_devices_raises_when_gateway_fails() -> None:
    class FailingGateway(IIoTAgentGateway):
        async def get_devices(
            self, service: str = "smart", service_path: str = "/"
        ) -> IoTDeviceCollection:
            raise RuntimeError("failure")

    use_case = GetDevicesUseCase(iot_agent_gateway=FailingGateway())

    with pytest.raises(RuntimeError):
        await use_case.execute()
