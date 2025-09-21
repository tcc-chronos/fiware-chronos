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
        h_limit: int,
        h_offset: int,
        fiware_service: str = "smart",
        fiware_servicepath: str = "/",
    ) -> List[CollectedDataDTO]:
        """Collect historical data from STH-Comet using hLimit and hOffset."""

        # Validate parameters
        if h_limit > 100:
            raise STHCometError(
                "h_limit cannot be greater than 100 per STH-Comet limitations"
            )

        if h_limit <= 0:
            raise STHCometError("h_limit must be positive")

        if h_offset < 0:
            raise STHCometError("h_offset must be non-negative")

        # URL encode entity_id to handle special characters
        encoded_entity_id = quote(entity_id, safe="")

        # Build URL
        url = (
            f"{self.base_url}/STH/v1/contextEntities/type/{entity_type}/"
            f"id/{encoded_entity_id}/attributes/{attribute}"
        )

        # Build query parameters using hLimit and hOffset
        params = {"hLimit": str(h_limit), "hOffset": str(h_offset), "count": "true"}

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
            h_limit=h_limit,
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

    async def get_total_count_from_header(
        self,
        entity_type: str,
        entity_id: str,
        attribute: str,
        fiware_service: str = "smart",
        fiware_servicepath: str = "/",
    ) -> int:
        """Get total count of available data points from fiware-total-count header."""

        # URL encode entity_id
        encoded_entity_id = quote(entity_id, safe="")

        # Build URL
        url = (
            f"{self.base_url}/STH/v1/contextEntities/type/{entity_type}/"
            f"id/{encoded_entity_id}/attributes/{attribute}"
        )

        # Build query parameters (minimal request to get count)
        params = {"hLimit": "1", "hOffset": "0", "count": "true"}

        # Build headers
        headers = {
            "fiware-service": fiware_service,
            "fiware-servicepath": fiware_servicepath,
            "Content-Type": "application/json",
        }

        logger.info(
            "Getting total count from STH-Comet header",
            url=url,
            entity_type=entity_type,
            entity_id=entity_id,
            attribute=attribute,
        )

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(url, params=params, headers=headers)
                response.raise_for_status()

                # Extract total count from response headers
                total_count_header = response.headers.get("fiware-total-count")
                if total_count_header:
                    total_count = int(total_count_header)
                    logger.info(
                        "Got total count from header",
                        total_count=total_count,
                        entity_type=entity_type,
                        entity_id=entity_id,
                        attribute=attribute,
                    )
                    return total_count
                else:
                    logger.warning("fiware-total-count header not found in response")
                    return 0

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
                logger.warning(
                    "sth_comet.attribute_not_found",
                    attribute=attribute,
                )
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
                        logger.warning(
                            "sth_comet.value_conversion_failed",
                            value=attr_value,
                            attribute=attribute,
                        )
                        continue

                    collected_data.append(
                        CollectedDataDTO(timestamp=timestamp, value=value)
                    )

                except Exception as e:
                    logger.warning(
                        "sth_comet.value_entry_parse_failed",
                        error=str(e),
                        value_entry=value_entry,
                    )
                    continue

            logger.info(
                "sth_comet.data_points_parsed",
                count=len(collected_data),
            )
            return collected_data

        except Exception as e:
            logger.error(
                "sth_comet.response_parse_failed",
                error=str(e),
                response_data=data,
                exc_info=e,
            )
            raise STHCometError(f"Failed to parse STH-Comet response: {e}") from e
