"""
IoT Agent Gateway Interface - Domain Layer

This module defines the interface for communicating with the IoT Agent.
"""

from abc import ABC, abstractmethod

from src.domain.entities.iot import IoTDeviceCollection


class IIoTAgentGateway(ABC):
    """Interface for IoT Agent Gateway."""

    @abstractmethod
    async def get_devices(
        self, service: str = "smart", service_path: str = "/"
    ) -> IoTDeviceCollection:
        """
        Retrieve devices from IoT Agent.

        Args:
            service: FIWARE service header
            service_path: FIWARE service path header

        Returns:
            IoTDeviceCollection: Response containing devices information

        Raises:
            Exception: If communication with IoT Agent fails
        """
        pass
