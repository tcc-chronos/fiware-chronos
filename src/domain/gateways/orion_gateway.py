"""Orion Context Broker gateway interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from src.domain.entities.prediction import PredictionRecord


class IOrionGateway(ABC):
    """Defines operations required to interact with Orion Context Broker."""

    @abstractmethod
    async def ensure_entity(
        self,
        *,
        entity_id: str,
        entity_type: str,
        payload: Dict[str, Any],
        service: str,
        service_path: str,
    ) -> None:
        """Ensure an Orion entity exists, creating it when absent."""
        raise NotImplementedError

    @abstractmethod
    async def upsert_prediction(
        self,
        prediction: PredictionRecord,
        *,
        service: str,
        service_path: str,
    ) -> None:
        """Upsert prediction attributes for the configured entity."""
        raise NotImplementedError

    @abstractmethod
    async def create_subscription(
        self,
        *,
        entity_id: str,
        entity_type: str,
        attrs: List[str],
        notification_url: str,
        service: str,
        service_path: str,
        attrs_format: str = "legacy",
    ) -> str:
        """Create an Orion subscription and return its identifier."""
        raise NotImplementedError

    @abstractmethod
    async def delete_subscription(
        self,
        subscription_id: str,
        *,
        service: str,
        service_path: str,
    ) -> None:
        """Remove an Orion subscription when present."""
        raise NotImplementedError

    @abstractmethod
    async def delete_entity(
        self,
        entity_id: str,
        *,
        service: str,
        service_path: str,
    ) -> None:
        """Remove an entity from Orion when present."""
        raise NotImplementedError
