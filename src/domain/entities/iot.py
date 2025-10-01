"""Domain entities for IoT Agent devices."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass(slots=True)
class DeviceAttribute:
    """Represents a device attribute exposed by the IoT Agent."""

    object_id: str
    name: str
    type: str


@dataclass(slots=True)
class IoTDevice:
    """Represents a single device managed by the IoT Agent."""

    device_id: str
    service: str
    service_path: str
    entity_name: str
    entity_type: str
    transport: str
    protocol: str
    attributes: List[DeviceAttribute] = field(default_factory=list)
    lazy: List[Dict[str, Any]] = field(default_factory=list)
    commands: List[Dict[str, Any]] = field(default_factory=list)
    static_attributes: List[Dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class IoTDeviceCollection:
    """Container for IoT devices returned by the IoT Agent."""

    count: int
    devices: List[IoTDevice] = field(default_factory=list)
