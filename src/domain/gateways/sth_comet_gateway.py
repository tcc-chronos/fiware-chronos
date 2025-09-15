"""
Domain Gateway - STH Comet

This module defines the gateway interface for interacting with STH-Comet
to collect historical data from FIWARE Context Broker.
"""

from abc import ABC, abstractmethod
from typing import List

from src.application.dtos.training_dto import CollectedDataDTO


class ISTHCometGateway(ABC):
    """Interface for STH-Comet gateway."""

    @abstractmethod
    async def collect_data(
        self,
        entity_type: str,
        entity_id: str,
        attribute: str,
        last_n: int,
        h_offset: int = 0,
        fiware_service: str = "smart",
        fiware_servicepath: str = "/",
    ) -> List[CollectedDataDTO]:
        """
        Collect historical data from STH-Comet.

        Args:
            entity_type: Type of the entity (e.g., "Sensor")
            entity_id: ID of the entity (e.g., "urn:ngsi-ld:Chronos:ESP32:001")
            attribute: Attribute to collect (e.g., "humidity")
            last_n: Number of most recent values to collect (max 100 per request)
            h_offset: Historical offset for pagination
            fiware_service: FIWARE service header
            fiware_servicepath: FIWARE service path header

        Returns:
            List of collected data points

        Raises:
            STHCometError: When data collection fails
        """
        pass

    @abstractmethod
    async def get_total_count(
        self,
        entity_type: str,
        entity_id: str,
        attribute: str,
        fiware_service: str = "smart",
        fiware_servicepath: str = "/",
    ) -> int:
        """
        Get total count of available data points.

        Args:
            entity_type: Type of the entity
            entity_id: ID of the entity
            attribute: Attribute to check
            fiware_service: FIWARE service header
            fiware_servicepath: FIWARE service path header

        Returns:
            Total count of available data points

        Raises:
            STHCometError: When count retrieval fails
        """
        pass
