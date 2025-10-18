from __future__ import annotations

import pytest

from src.application.use_cases.device_use_cases import GetDevicesUseCase
from src.domain.entities.iot import (
    DeviceAttribute,
    IoTDevice,
    IoTDeviceCollection,
    IoTServiceGroup,
)
from src.domain.gateways.iot_agent_gateway import IIoTAgentGateway


class _StubGateway(IIoTAgentGateway):
    def __init__(self, collection: IoTDeviceCollection) -> None:
        self._collection = collection
        self.calls: list[tuple[str, str]] = []
        self.service_groups: dict[tuple[str, str], IoTServiceGroup] = {}
        self.devices: dict[str, IoTDevice] = {
            device.device_id: device for device in collection.devices
        }

    async def get_devices(
        self, service: str = "smart", service_path: str = "/"
    ) -> IoTDeviceCollection:
        self.calls.append((service, service_path))
        return self._collection

    async def get_service_groups(
        self, service: str = "smart", service_path: str = "/"
    ) -> list[IoTServiceGroup]:
        return [
            group
            for key, group in self.service_groups.items()
            if key == (service, service_path)
        ]

    async def ensure_service_group(
        self,
        *,
        service: str,
        service_path: str,
        apikey: str,
        entity_type: str,
        resource: str,
        cbroker: str,
    ) -> IoTServiceGroup:
        group = IoTServiceGroup(
            apikey=apikey,
            cbroker=cbroker,
            entity_type=entity_type,
            resource=resource,
            service=service,
            service_path=service_path,
        )
        self.service_groups[(service, service_path)] = group
        return group

    async def ensure_device(
        self,
        *,
        device_id: str,
        entity_name: str,
        entity_type: str,
        attributes: list[DeviceAttribute],
        transport: str,
        protocol: str,
        service: str,
        service_path: str,
    ) -> IoTDevice:
        device = IoTDevice(
            device_id=device_id,
            service=service,
            service_path=service_path,
            entity_name=entity_name,
            entity_type=entity_type,
            transport=transport,
            protocol=protocol,
            attributes=attributes,
        )
        self.devices[device_id] = device
        return device

    async def delete_device(
        self,
        device_id: str,
        *,
        service: str,
        service_path: str,
    ) -> None:
        self.devices.pop(device_id, None)


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

        async def get_service_groups(
            self, service: str = "smart", service_path: str = "/"
        ):
            raise NotImplementedError

        async def ensure_service_group(self, **kwargs):
            raise NotImplementedError

        async def ensure_device(self, **kwargs):
            raise NotImplementedError

        async def delete_device(
            self, device_id: str, *, service: str, service_path: str
        ):
            raise NotImplementedError

    use_case = GetDevicesUseCase(iot_agent_gateway=FailingGateway())

    with pytest.raises(RuntimeError):
        await use_case.execute()


@pytest.mark.asyncio
async def test_get_devices_skips_forecast_service_group() -> None:
    collection = IoTDeviceCollection(
        count=1,
        devices=[
            IoTDevice(
                device_id="forecast",
                service="forecast-group",
                service_path="/",
                entity_name="ForecastDevice",
                entity_type="Prediction",
                transport="mqtt",
                protocol="MQTT",
                attributes=[DeviceAttribute(object_id="t", name="temp", type="Number")],
            )
        ],
    )
    gateway = _StubGateway(collection)
    use_case = GetDevicesUseCase(
        iot_agent_gateway=gateway,
        forecast_service_group="forecast-group",
    )

    response = await use_case.execute(service="smart", service_path="/")

    assert response.devices == []


@pytest.mark.asyncio
async def test_get_devices_skips_prediction_entity_type() -> None:
    collection = IoTDeviceCollection(
        count=2,
        devices=[
            IoTDevice(
                device_id="prediction",
                service="smart",
                service_path="/",
                entity_name="PredictionDevice",
                entity_type="Prediction",
                transport="mqtt",
                protocol="MQTT",
                attributes=[DeviceAttribute(object_id="t", name="temp", type="Number")],
            ),
            IoTDevice(
                device_id="regular",
                service="smart",
                service_path="/",
                entity_name="RegularDevice",
                entity_type="Sensor",
                transport="mqtt",
                protocol="MQTT",
                attributes=[DeviceAttribute(object_id="h", name="hum", type="Number")],
            ),
        ],
    )
    gateway = _StubGateway(collection)
    use_case = GetDevicesUseCase(iot_agent_gateway=gateway)

    response = await use_case.execute(service="smart", service_path="/")

    assert len(response.devices) == 1
    assert response.devices[0].entity_type == "Sensor"
    assert response.devices[0].entities[0].entity_name == "RegularDevice"
