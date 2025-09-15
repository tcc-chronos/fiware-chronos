"""
IoT Agent Gateway Implementation - Infrastructure Layer

This module implements the gateway for communicating with the IoT Agent.
"""

import logging

import httpx
from dependency_injector.wiring import inject

from src.application.dtos.device_dto import IoTAgentDevicesResponseDTO
from src.domain.gateways.iot_agent_gateway import IIoTAgentGateway

logger = logging.getLogger(__name__)


class IoTAgentGateway(IIoTAgentGateway):
    """Implementation of IoT Agent Gateway."""

    @inject
    def __init__(self, iot_agent_url: str):
        """
        Initialize IoT Agent Gateway.

        Args:
            iot_agent_url: Base URL for the IoT Agent service
        """
        self.iot_agent_url = iot_agent_url.rstrip("/")
        self.timeout = 30.0

    async def get_devices(
        self, service: str = "smart", service_path: str = "/"
    ) -> IoTAgentDevicesResponseDTO:
        """
        Retrieve devices from IoT Agent.

        Args:
            service: FIWARE service header
            service_path: FIWARE service path header

        Returns:
            IoTAgentDevicesResponseDTO: Response containing devices information

        Raises:
            httpx.HTTPError: If HTTP request fails
            Exception: If communication with IoT Agent fails
        """
        url = f"{self.iot_agent_url}/iot/devices"
        headers = {
            "fiware-service": service,
            "fiware-servicepath": service_path,
            "Content-Type": "application/json",
        }

        logger.info(f"Requesting devices from IoT Agent: {url}")
        logger.debug(f"Headers: {headers}")

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(url, headers=headers)
                response.raise_for_status()

                response_data = response.json()
                logger.info(
                    f"Successfully retrieved {response_data.get('count', 0)} "
                    f"devices from IoT Agent"
                )

                return IoTAgentDevicesResponseDTO(**response_data)

        except httpx.HTTPStatusError as e:
            logger.error(
                f"HTTP error when requesting devices from IoT Agent: "
                f"{e.response.status_code} - {e.response.text}"
            )
            raise Exception(
                f"IoT Agent returned HTTP {e.response.status_code}: "
                f"{e.response.text}"
            )

        except httpx.RequestError as e:
            logger.error(f"Request error when communicating with IoT Agent: {str(e)}")
            raise Exception(f"Failed to communicate with IoT Agent: {str(e)}")

        except Exception as e:
            logger.error(
                f"Unexpected error when getting devices from IoT Agent: {str(e)}"
            )
            raise Exception(f"Unexpected error communicating with IoT Agent: {str(e)}")
