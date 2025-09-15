"""
Device DTOs - Application Layer

This module defines Data Transfer Objects (DTOs) for device entities
from the IoT Agent. These DTOs are used to transfer data between
the application layer and the presentation layer (API).
"""

from typing import List

from pydantic import BaseModel, Field


class DeviceAttributeDTO(BaseModel):
    """DTO for device attribute from IoT Agent."""

    object_id: str = Field(description="Object ID of the attribute")
    name: str = Field(description="Name of the attribute")
    type: str = Field(description="Type of the attribute")

    model_config = {
        "json_schema_extra": {
            "example": {"object_id": "t", "name": "temperature", "type": "Number"}
        }
    }


class IoTAgentDeviceDTO(BaseModel):
    """DTO for device response from IoT Agent."""

    device_id: str = Field(description="Device ID")
    service: str = Field(description="FIWARE service")
    service_path: str = Field(description="FIWARE service path")
    entity_name: str = Field(description="NGSI-LD entity name")
    entity_type: str = Field(description="NGSI-LD entity type")
    transport: str = Field(description="Transport protocol")
    attributes: List[DeviceAttributeDTO] = Field(description="Device attributes")
    lazy: List = Field(default_factory=list, description="Lazy attributes")
    commands: List = Field(default_factory=list, description="Device commands")
    static_attributes: List = Field(
        default_factory=list, description="Static attributes"
    )
    protocol: str = Field(description="Protocol used")

    model_config = {
        "json_schema_extra": {
            "example": {
                "device_id": "esp32-chronos00",
                "service": "smart",
                "service_path": "/",
                "entity_name": "urn:ngsi-ld:Chronos:ESP32:000",
                "entity_type": "Sensor",
                "transport": "MQTT",
                "attributes": [
                    {"object_id": "t", "name": "temperature", "type": "Number"}
                ],
                "lazy": [],
                "commands": [],
                "static_attributes": [],
                "protocol": "PDI-IoTA-UltraLight",
            }
        }
    }


class IoTAgentDevicesResponseDTO(BaseModel):
    """DTO for complete IoT Agent devices response."""

    count: int = Field(description="Total number of devices")
    devices: List[IoTAgentDeviceDTO] = Field(description="List of devices")

    model_config = {
        "json_schema_extra": {
            "example": {
                "count": 1,
                "devices": [
                    {
                        "device_id": "esp32-chronos00",
                        "service": "smart",
                        "service_path": "/",
                        "entity_name": "urn:ngsi-ld:Chronos:ESP32:000",
                        "entity_type": "Sensor",
                        "transport": "MQTT",
                        "attributes": [
                            {"object_id": "t", "name": "temperature", "type": "Number"}
                        ],
                        "lazy": [],
                        "commands": [],
                        "static_attributes": [],
                        "protocol": "PDI-IoTA-UltraLight",
                    }
                ],
            }
        }
    }


class DeviceEntityDTO(BaseModel):
    """DTO for formatted device entity."""

    entity_name: str = Field(description="NGSI-LD entity name")
    attributes: List[str] = Field(description="List of attribute names")

    model_config = {
        "json_schema_extra": {
            "example": {
                "entity_name": "urn:ngsi-ld:Chronos:ESP32:000",
                "attributes": ["temperature", "humidity", "timestamp"],
            }
        }
    }


class GroupedDevicesDTO(BaseModel):
    """DTO for devices grouped by entity type."""

    entity_type: str = Field(description="NGSI-LD entity type")
    entities: List[DeviceEntityDTO] = Field(description="List of entities of this type")

    model_config = {
        "json_schema_extra": {
            "example": {
                "entity_type": "Sensor",
                "entities": [
                    {
                        "entity_name": "urn:ngsi-ld:Chronos:ESP32:000",
                        "attributes": ["temperature", "humidity", "timestamp"],
                    }
                ],
            }
        }
    }


class DevicesResponseDTO(BaseModel):
    """DTO for final formatted devices response."""

    devices: List[GroupedDevicesDTO] = Field(
        description="Devices grouped by entity type"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "devices": [
                    {
                        "entity_type": "Sensor",
                        "entities": [
                            {
                                "entity_name": "urn:ngsi-ld:Chronos:ESP32:000",
                                "attributes": ["temperature", "humidity", "timestamp"],
                            }
                        ],
                    }
                ]
            }
        }
    }
