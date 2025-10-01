from __future__ import annotations

import pytest
from fastapi import HTTPException

from src.application.dtos.device_dto import DevicesResponseDTO, GroupedDevicesDTO
from src.application.use_cases.device_use_cases import GetDevicesUseCase
from src.presentation.controllers.devices_controller import get_devices


class _StubDevices(GetDevicesUseCase):
    def __init__(self, response: DevicesResponseDTO):
        self.response = response

    async def execute(
        self, service: str = "", service_path: str = ""
    ) -> DevicesResponseDTO:
        return self.response


@pytest.mark.asyncio
async def test_get_devices_returns_grouped() -> None:
    dto = DevicesResponseDTO(
        devices=[GroupedDevicesDTO(entity_type="Sensor", entities=[])]
    )
    response = await get_devices(get_devices_use_case=_StubDevices(dto))
    assert response.devices[0].entity_type == "Sensor"


@pytest.mark.asyncio
async def test_get_devices_handles_errors() -> None:
    class _Fail(GetDevicesUseCase):
        async def execute(
            self, service: str = "", service_path: str = ""
        ) -> DevicesResponseDTO:
            raise RuntimeError("failure")

    with pytest.raises(HTTPException) as exc:
        await get_devices(get_devices_use_case=_Fail())
    assert exc.value.status_code == 500
