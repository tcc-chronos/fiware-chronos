"""
Infrastructure Gateway - STH Comet Implementation

This module implements the STH-Comet gateway for collecting historical data
from FIWARE Context Broker using the Short Term Historic API.
"""

from datetime import datetime
from typing import List
from urllib.parse import quote

import httpx
import structlog

from src.application.dtos.training_dto import CollectedDataDTO
from src.domain.entities.errors import DomainError
from src.domain.gateways.sth_comet_gateway import ISTHCometGateway

logger = structlog.get_logger(__name__)


class STHCometError(DomainError):
    """Exception raised when STH-Comet operations fail."""

    pass


class STHCometGateway(ISTHCometGateway):
    """Implementation of STH-Comet gateway using HTTP client."""

    def __init__(self, base_url: str, timeout: float = 30.0):
        """
        Initialize STH-Comet gateway.

        Args:
            base_url: Base URL of STH-Comet service (e.g., "http://sth-comet:8666")
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

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
        """Collect historical data from STH-Comet."""

        # Validate parameters
        if last_n > 100:
            raise STHCometError(
                "last_n cannot be greater than 100 per STH-Comet limitations"
            )

        if last_n <= 0:
            raise STHCometError("last_n must be positive")

        # URL encode entity_id to handle special characters
        encoded_entity_id = quote(entity_id, safe="")

        # Build URL
        url = (
            f"{self.base_url}/STH/v1/contextEntities/type/{entity_type}/"
            f"id/{encoded_entity_id}/attributes/{attribute}"
        )

        # Build query parameters
        params = {"lastN": str(last_n), "hOffset": str(h_offset), "count": "true"}

        # Build headers
        headers = {
            "fiware-service": fiware_service,
            "fiware-servicepath": fiware_servicepath,
            "Content-Type": "application/json",
        }

        logger.info(
            "Collecting data from STH-Comet",
            url=url,
            params=params,
            headers=headers,
            entity_type=entity_type,
            entity_id=entity_id,
            attribute=attribute,
            last_n=last_n,
            h_offset=h_offset,
        )

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(url, params=params, headers=headers)
                response.raise_for_status()

                data = response.json()
                return self._parse_sth_response(data, attribute)

        except httpx.HTTPStatusError as e:
            logger.error(
                "STH-Comet HTTP error",
                status_code=e.response.status_code,
                response_text=e.response.text,
                url=url,
            )
            raise STHCometError(
                f"STH-Comet HTTP error {e.response.status_code}: {e.response.text}"
            ) from e

        except httpx.RequestError as e:
            logger.error("STH-Comet request error", error=str(e), url=url)
            raise STHCometError(f"STH-Comet request failed: {str(e)}") from e

        except Exception as e:
            logger.error("STH-Comet unexpected error", error=str(e), url=url)
            raise STHCometError(f"STH-Comet unexpected error: {str(e)}") from e

    async def get_total_count(
        self,
        entity_type: str,
        entity_id: str,
        attribute: str,
        fiware_service: str = "smart",
        fiware_servicepath: str = "/",
    ) -> int:
        """Get total count of available data points."""

        # URL encode entity_id
        encoded_entity_id = quote(entity_id, safe="")

        # Build URL for count only
        url = (
            f"{self.base_url}/STH/v1/contextEntities/type/{entity_type}/"
            f"id/{encoded_entity_id}/attributes/{attribute}"
        )

        # Build query parameters (count only, no data)
        params = {"lastN": "1", "count": "true"}  # Minimum to get count

        # Build headers
        headers = {
            "fiware-service": fiware_service,
            "fiware-servicepath": fiware_servicepath,
            "Content-Type": "application/json",
        }

        logger.info(
            "Getting total count from STH-Comet",
            url=url,
            entity_type=entity_type,
            entity_id=entity_id,
            attribute=attribute,
        )

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(url, params=params, headers=headers)
                response.raise_for_status()

                data = response.json()
                return self._extract_count_from_response(data)

        except httpx.HTTPStatusError as e:
            logger.error(
                "STH-Comet count HTTP error",
                status_code=e.response.status_code,
                response_text=e.response.text,
                url=url,
            )
            raise STHCometError(
                f"STH-Comet count HTTP error {e.response.status_code}: "
                f"{e.response.text}"
            ) from e

        except Exception as e:
            logger.error("STH-Comet count error", error=str(e), url=url)
            raise STHCometError(f"STH-Comet count error: {str(e)}") from e

    def _parse_sth_response(self, data: dict, attribute: str) -> List[CollectedDataDTO]:
        """Parse STH-Comet response and extract data points."""

        try:
            context_responses = data.get("contextResponses", [])
            if not context_responses:
                logger.warning("No context responses in STH-Comet data")
                return []

            # Get the first context response
            context_response = context_responses[0]
            context_element = context_response.get("contextElement", {})
            attributes = context_element.get("attributes", [])

            # Find the requested attribute
            target_attribute = None
            for attr in attributes:
                if attr.get("name") == attribute:
                    target_attribute = attr
                    break

            if not target_attribute:
                logger.warning(f"Attribute '{attribute}' not found in response")
                return []

            # Extract values
            values = target_attribute.get("values", [])
            collected_data = []

            for value_entry in values:
                try:
                    # Parse timestamp
                    recv_time_str = value_entry.get("recvTime")
                    if not recv_time_str:
                        logger.warning("Missing recvTime in value entry")
                        continue

                    # Parse ISO timestamp
                    timestamp = datetime.fromisoformat(
                        recv_time_str.replace("Z", "+00:00")
                    )

                    # Parse value
                    attr_value = value_entry.get("attrValue")
                    if attr_value is None:
                        logger.warning("Missing attrValue in value entry")
                        continue

                    # Convert to float
                    try:
                        value = float(attr_value)
                    except (ValueError, TypeError):
                        logger.warning(f"Cannot convert value to float: {attr_value}")
                        continue

                    collected_data.append(
                        CollectedDataDTO(timestamp=timestamp, value=value)
                    )

                except Exception as e:
                    logger.warning(
                        f"Error parsing value entry: {e}", value_entry=value_entry
                    )
                    continue

            logger.info(
                f"Parsed {len(collected_data)} data points from STH-Comet response"
            )
            return collected_data

        except Exception as e:
            logger.error(f"Error parsing STH-Comet response: {e}", response_data=data)
            raise STHCometError(f"Failed to parse STH-Comet response: {e}") from e

    def _extract_count_from_response(self, data: dict) -> int:
        """Extract total count from STH-Comet response."""

        try:
            context_responses = data.get("contextResponses", [])
            if not context_responses:
                return 0

            # Get status code from first context response
            status_code = context_responses[0].get("statusCode", {})
            details = status_code.get("details", "")

            # Parse count from details (format: "Count: X")
            if "Count:" in details:
                count_str = details.split("Count:")[1].strip()
                return int(count_str)

            return 0

        except Exception as e:
            logger.error(
                f"Error extracting count from response: {e}", response_data=data
            )
            raise STHCometError(f"Failed to extract count: {e}") from e
