"""
IoT Agent Gateway Implementation - Infrastructure Layer

This module implements the gateway for communicating with the IoT Agent.
"""

import httpx
from dependency_injector.wiring import inject

from src.application.dtos.device_dto import IoTAgentDevicesResponseDTO
from src.domain.gateways.iot_agent_gateway import IIoTAgentGateway
from src.shared import get_logger

logger = get_logger(__name__)


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

        logger.info(
            "iot_agent.devices.request",
            url=url,
            service=service,
            service_path=service_path,
        )
        logger.debug("iot_agent.devices.request_headers", headers=headers)

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(url, headers=headers)
                response.raise_for_status()

                response_data = response.json()
                logger.info(
                    "iot_agent.devices.response",
                    count=response_data.get("count", 0),
                    status_code=response.status_code,
                )

                return IoTAgentDevicesResponseDTO(**response_data)

        except httpx.HTTPStatusError as e:
            logger.error(
                "iot_agent.devices.http_error",
                status_code=e.response.status_code,
                response_text=e.response.text,
                url=url,
                exc_info=e,
            )
            raise Exception(
                f"IoT Agent returned HTTP {e.response.status_code}: "
                f"{e.response.text}"
            )

        except httpx.RequestError as e:
            logger.error(
                "iot_agent.devices.request_error",
                error=str(e),
                url=url,
                exc_info=e,
            )
            raise Exception(f"Failed to communicate with IoT Agent: {str(e)}")

        except Exception as e:
            logger.error(
                "iot_agent.devices.unexpected_error",
                error=str(e),
                url=url,
                exc_info=e,
            )
            raise Exception(f"Unexpected error communicating with IoT Agent: {str(e)}")
