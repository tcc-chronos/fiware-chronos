"""IoT Agent gateway implementation - Infrastructure layer."""

from __future__ import annotations

from typing import Any, Dict, List

import httpx
from dependency_injector.wiring import inject

from src.domain.entities.iot import (
    DeviceAttribute,
    IoTDevice,
    IoTDeviceCollection,
    IoTServiceGroup,
)
from src.domain.gateways.iot_agent_gateway import IIoTAgentGateway
from src.shared import get_logger

logger = get_logger(__name__)


class IoTAgentGateway(IIoTAgentGateway):
    """HTTP client for the IoT Agent API."""

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
    ) -> IoTDeviceCollection:
        """
        Retrieve devices from IoT Agent.

        Args:
            service: FIWARE service header
            service_path: FIWARE service path header

        Returns:
            IoTDeviceCollection: Response containing devices information

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

                return self._to_domain(response_data)

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

    async def get_service_groups(
        self, service: str = "smart", service_path: str = "/"
    ) -> List[IoTServiceGroup]:
        """Retrieve service groups registered in the IoT Agent."""

        url = f"{self.iot_agent_url}/iot/services"
        headers = {
            "fiware-service": service,
            "fiware-servicepath": service_path,
            "Content-Type": "application/json",
        }

        logger.info(
            "iot_agent.service_groups.request",
            url=url,
            service=service,
            service_path=service_path,
        )

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(url, headers=headers)
                response.raise_for_status()

                payload = response.json()
                services = payload.get("services") or []
                return [
                    self._parse_service_group(item, service, service_path)
                    for item in services
                    if item
                ]

        except httpx.HTTPStatusError as e:
            logger.error(
                "iot_agent.service_groups.http_error",
                status_code=e.response.status_code,
                response_text=e.response.text,
                url=url,
                exc_info=e,
            )
            raise Exception(
                f"IoT Agent returned HTTP {e.response.status_code}: {e.response.text}"
            )
        except httpx.RequestError as e:
            logger.error(
                "iot_agent.service_groups.request_error",
                error=str(e),
                url=url,
                exc_info=e,
            )
            raise Exception(f"Failed to retrieve service groups: {str(e)}")
        except Exception as e:
            logger.error(
                "iot_agent.service_groups.unexpected_error",
                error=str(e),
                url=url,
                exc_info=e,
            )
            raise Exception(f"Unexpected error retrieving service groups: {str(e)}")

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

        existing_groups = await self.get_service_groups(
            service=service, service_path=service_path
        )
        for group in existing_groups:
            if group.apikey == apikey and group.resource == resource:
                return group

        url = f"{self.iot_agent_url}/iot/services"
        headers = {
            "fiware-service": service,
            "fiware-servicepath": service_path,
            "Content-Type": "application/json",
        }
        payload = {
            "services": [
                {
                    "apikey": apikey,
                    "cbroker": cbroker,
                    "entity_type": entity_type,
                    "resource": resource,
                }
            ]
        }

        logger.info(
            "iot_agent.service_groups.create",
            url=url,
            payload=payload,
        )

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(url, headers=headers, json=payload)
                response.raise_for_status()

            return IoTServiceGroup(
                apikey=apikey,
                cbroker=cbroker,
                entity_type=entity_type,
                resource=resource,
                service=service,
                service_path=service_path,
            )

        except httpx.HTTPStatusError as e:
            logger.error(
                "iot_agent.service_groups.create_http_error",
                status_code=e.response.status_code,
                response_text=e.response.text,
                url=url,
                exc_info=e,
            )
            raise Exception(
                f"IoT Agent returned HTTP {e.response.status_code}: {e.response.text}"
            )
        except httpx.RequestError as e:
            logger.error(
                "iot_agent.service_groups.create_request_error",
                error=str(e),
                url=url,
                exc_info=e,
            )
            raise Exception(f"Failed to create service group: {str(e)}")
        except Exception as e:
            logger.error(
                "iot_agent.service_groups.create_unexpected_error",
                error=str(e),
                url=url,
                exc_info=e,
            )
            raise Exception(f"Unexpected error creating service group: {str(e)}")

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
        """Ensure an IoT device exists, creating it if necessary."""

        existing = await self.get_devices(service=service, service_path=service_path)
        for device in existing.devices:
            if device.device_id == device_id or device.entity_name == entity_name:
                return device

        url = f"{self.iot_agent_url}/iot/devices"
        headers = {
            "fiware-service": service,
            "fiware-servicepath": service_path,
            "Content-Type": "application/json",
        }
        payload = {
            "devices": [
                {
                    "device_id": device_id,
                    "entity_name": entity_name,
                    "entity_type": entity_type,
                    "transport": transport,
                    "protocol": protocol,
                    "attributes": [
                        {
                            "object_id": attr.object_id,
                            "name": attr.name,
                            "type": attr.type,
                        }
                        for attr in attributes
                    ],
                }
            ]
        }

        logger.info(
            "iot_agent.device.create",
            url=url,
            device_id=device_id,
            entity_name=entity_name,
            entity_type=entity_type,
            service=service,
            service_path=service_path,
        )

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(url, headers=headers, json=payload)
                response.raise_for_status()

            return IoTDevice(
                device_id=device_id,
                service=service,
                service_path=service_path,
                entity_name=entity_name,
                entity_type=entity_type,
                transport=transport,
                protocol=protocol,
                attributes=attributes,
            )

        except httpx.HTTPStatusError as e:
            logger.error(
                "iot_agent.device.create_http_error",
                device_id=device_id,
                status_code=e.response.status_code,
                response_text=e.response.text,
                url=url,
                service=service,
                service_path=service_path,
                exc_info=e,
            )
            raise Exception(
                f"IoT Agent returned HTTP {e.response.status_code}: {e.response.text}"
            )
        except httpx.RequestError as e:
            logger.error(
                "iot_agent.device.create_request_error",
                device_id=device_id,
                error=str(e),
                url=url,
                service=service,
                service_path=service_path,
                exc_info=e,
            )
            raise Exception(f"Failed to create IoT device: {str(e)}")
        except Exception as e:
            logger.error(
                "iot_agent.device.create_unexpected_error",
                device_id=device_id,
                error=str(e),
                url=url,
                service=service,
                service_path=service_path,
                exc_info=e,
            )
            raise Exception(f"Unexpected error creating IoT device: {str(e)}")

    async def delete_device(
        self,
        device_id: str,
        *,
        service: str,
        service_path: str,
    ) -> None:
        """Delete a device from the IoT Agent if it exists."""

        url = f"{self.iot_agent_url}/iot/devices/{device_id}"
        headers = {
            "fiware-service": service,
            "fiware-servicepath": service_path,
            "Content-Type": "application/json",
        }

        logger.info(
            "iot_agent.device.delete.request",
            url=url,
            device_id=device_id,
            service=service,
            service_path=service_path,
        )

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.delete(url, headers=headers)
                if response.status_code in (
                    httpx.codes.NO_CONTENT,
                    httpx.codes.NOT_FOUND,
                ):
                    return
                response.raise_for_status()

        except httpx.HTTPStatusError as e:
            logger.warning(
                "iot_agent.device.delete_http_error",
                device_id=device_id,
                status_code=e.response.status_code,
                response_text=e.response.text,
                url=url,
                service=service,
                service_path=service_path,
            )
        except httpx.RequestError as e:
            logger.warning(
                "iot_agent.device.delete_request_error",
                device_id=device_id,
                error=str(e),
                url=url,
                service=service,
                service_path=service_path,
            )
        except Exception as e:
            logger.warning(
                "iot_agent.device.delete_unexpected_error",
                device_id=device_id,
                error=str(e),
                url=url,
                service=service,
                service_path=service_path,
            )

    def _to_domain(self, payload: Dict[str, Any]) -> IoTDeviceCollection:
        count = int(payload.get("count", 0))
        devices_payload = payload.get("devices") or []
        devices = [self._parse_device(item) for item in devices_payload if item]
        return IoTDeviceCollection(count=count, devices=devices)

    def _parse_service_group(
        self, data: Dict[str, Any], service: str, service_path: str
    ) -> IoTServiceGroup:
        return IoTServiceGroup(
            apikey=str(data.get("apikey", "")),
            cbroker=str(data.get("cbroker", "")),
            entity_type=str(data.get("entity_type", "")),
            resource=str(data.get("resource", "")),
            service=service,
            service_path=service_path,
        )

    def _parse_device(self, data: Dict[str, Any]) -> IoTDevice:
        return IoTDevice(
            device_id=data.get("device_id", ""),
            service=data.get("service", ""),
            service_path=data.get("service_path", ""),
            entity_name=data.get("entity_name", ""),
            entity_type=data.get("entity_type", ""),
            transport=data.get("transport", ""),
            protocol=data.get("protocol", ""),
            attributes=self._parse_attributes(data.get("attributes")),
            lazy=self._ensure_dict_list(data.get("lazy")),
            commands=self._ensure_dict_list(data.get("commands")),
            static_attributes=self._ensure_dict_list(data.get("static_attributes")),
        )

    def _parse_attributes(self, payload: Any) -> List[DeviceAttribute]:
        if not payload:
            return []
        attributes: List[DeviceAttribute] = []
        for attr in payload:
            if not isinstance(attr, dict):
                continue
            attributes.append(
                DeviceAttribute(
                    object_id=str(attr.get("object_id", "")),
                    name=str(attr.get("name", "")),
                    type=str(attr.get("type", "")),
                )
            )
        return attributes

    def _ensure_dict_list(self, payload: Any) -> List[Dict[str, Any]]:
        if not payload:
            return []
        return [item for item in payload if isinstance(item, dict)]
