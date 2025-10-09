"""
Device Use Cases - Application Layer

This module defines use cases for device operations.
It orchestrates the flow of data to and from the IoT Agent
and implements the business rules for device management.
"""

from collections import defaultdict
from typing import Dict, List

from dependency_injector.wiring import Provide, inject

from src.application.dtos.device_dto import (
    DeviceEntityDTO,
    DevicesResponseDTO,
    GroupedDevicesDTO,
)
from src.domain.gateways.iot_agent_gateway import IIoTAgentGateway
from src.shared import get_logger

logger = get_logger(__name__)


class GetDevicesUseCase:
    """Use case for retrieving and formatting devices from IoT Agent."""

    @inject
    def __init__(
        self,
        iot_agent_gateway: IIoTAgentGateway = Provide["iot_agent_gateway"],
        forecast_service_group: str = Provide["forecast_service_group"],
    ):
        """
        Initialize the use case with its dependencies.

        Args:
            iot_agent_gateway: Gateway for communicating with IoT Agent
        """
        self.iot_agent_gateway = iot_agent_gateway
        self._forecast_service_group = forecast_service_group

    async def execute(
        self, service: str = "smart", service_path: str = "/"
    ) -> DevicesResponseDTO:
        """
        Retrieve devices from IoT Agent and format them grouped by entity type.

        Args:
            service: FIWARE service header
            service_path: FIWARE service path header

        Returns:
            DevicesResponseDTO: Devices grouped by entity type with formatted attributes

        Raises:
            Exception: If retrieval or processing fails
        """
        logger.info(
            "devices.fetch_started",
            service=service,
            service_path=service_path,
        )

        try:
            # Get devices from IoT Agent
            iot_agent_response = await self.iot_agent_gateway.get_devices(
                service, service_path
            )

            logger.info(
                "devices.gateway_response",
                count=iot_agent_response.count,
                service=service,
                service_path=service_path,
            )

            # Group devices by entity_type
            grouped_devices: Dict[str, List[DeviceEntityDTO]] = defaultdict(list)

            for device in iot_agent_response.devices:
                if (
                    self._forecast_service_group
                    and device.service == self._forecast_service_group
                ):
                    logger.debug(
                        "devices.skip_forecast_group",
                        entity_name=device.entity_name,
                        service=device.service,
                    )
                    continue
                # Extract attribute names from device attributes
                attribute_names = [attr.name for attr in device.attributes]

                # Create device entity DTO
                device_entity = DeviceEntityDTO(
                    entity_name=device.entity_name, attributes=attribute_names
                )

                # Group by entity type
                grouped_devices[device.entity_type].append(device_entity)

            # Create grouped response
            devices = []
            for entity_type, entities in grouped_devices.items():
                group = GroupedDevicesDTO(entity_type=entity_type, entities=entities)
                devices.append(group)

            logger.info(
                "devices.grouped",
                group_count=len(devices),
                service=service,
                service_path=service_path,
            )
            for group in devices:
                logger.debug(
                    "devices.group_details",
                    entity_type=group.entity_type,
                    entity_count=len(group.entities),
                )

            return DevicesResponseDTO(devices=devices)

        except Exception as e:
            logger.error(
                "devices.fetch_failed",
                service=service,
                service_path=service_path,
                error=str(e),
                exc_info=e,
            )
            raise
