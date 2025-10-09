"""
IoT Agent Gateway Interface - Domain Layer

This module defines the interface for communicating with the IoT Agent.
"""

from abc import ABC, abstractmethod
from typing import List

from src.domain.entities.iot import (
    DeviceAttribute,
    IoTDevice,
    IoTDeviceCollection,
    IoTServiceGroup,
)


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

    @abstractmethod
    async def get_service_groups(
        self, service: str = "smart", service_path: str = "/"
    ) -> List[IoTServiceGroup]:
        """Retrieve configured service groups from IoT Agent."""
        pass

    @abstractmethod
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
        """Ensure a service group exists, creating it if necessary."""
        pass

    @abstractmethod
    async def ensure_device(
        self,
        *,
        device_id: str,
        entity_name: str,
        entity_type: str,
        attributes: List[DeviceAttribute],
        transport: str,
        protocol: str,
        service: str,
        service_path: str,
    ) -> IoTDevice:
        """Ensure an IoT device exists for the specified entity."""
        pass

    @abstractmethod
    async def delete_device(
        self,
        device_id: str,
        *,
        service: str,
        service_path: str,
    ) -> None:
        """Delete an IoT device from the agent if present."""
        pass
