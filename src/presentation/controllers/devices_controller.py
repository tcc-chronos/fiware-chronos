"""
Devices Router - Presentation Layer

This module defines the FastAPI router for device endpoints.
"""

from dependency_injector.wiring import Provide, inject
from fastapi import APIRouter, Depends, HTTPException, Query, status

from src.application.dtos.device_dto import DevicesResponseDTO
from src.application.use_cases.device_use_cases import GetDevicesUseCase
from src.shared import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/devices", tags=["Devices"])


@router.get("/", response_model=DevicesResponseDTO)
@inject
async def get_devices(
    service: str = Query(
        default="smart", description="FIWARE service to query devices from"
    ),
    service_path: str = Query(
        default="/", description="FIWARE service path to query devices from"
    ),
    get_devices_use_case: GetDevicesUseCase = Depends(Provide["get_devices_use_case"]),
) -> DevicesResponseDTO:
    """
    Get devices from IoT Agent grouped by entity type.

    This endpoint retrieves all devices from the configured IoT Agent
    and returns them grouped by entity_type with their available attributes.

    Args:
        service: FIWARE service header for the request
        service_path: FIWARE service path header for the request
        get_devices_use_case: Injected use case for device operations

    Returns:
        DevicesResponseDTO: Devices grouped by entity type

    Raises:
        HTTPException: If devices cannot be retrieved from IoT Agent
    """
    logger.info(
        "devices.requested",
        service=service,
        service_path=service_path,
    )

    try:
        devices_response = await get_devices_use_case.execute(service, service_path)

        logger.info(
            "devices.retrieved",
            device_count=len(devices_response.devices),
            service=service,
            service_path=service_path,
        )
        return devices_response

    except Exception as e:
        logger.error(
            "devices.retrieval_failed",
            service=service,
            service_path=service_path,
            error=str(e),
            exc_info=e,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve devices from IoT Agent: {str(e)}",
        )
