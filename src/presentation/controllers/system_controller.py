"""System endpoints exposing health and info."""

from dependency_injector.wiring import Provide, inject
from fastapi import APIRouter, Depends, HTTPException, Request, status

from src.application.dtos.health_dto import ApplicationInfoDTO, SystemHealthDTO
from src.application.use_cases.health_use_cases import (
    GetApplicationInfoUseCase,
    GetHealthStatusUseCase,
)
from src.shared import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["System"])


@router.get("/health", response_model=SystemHealthDTO)
@inject
async def health(
    get_health_status_use_case: GetHealthStatusUseCase = Depends(
        Provide["get_health_status_use_case"]
    ),
) -> SystemHealthDTO:
    """Return the health status of the application dependencies."""
    try:
        health_status = await get_health_status_use_case.execute()
        logger.debug("health.check.success", status=health_status.status.value)
        return health_status
    except Exception as exc:  # pragma: no cover - defensive logging path
        logger.error("health.check.failure", error=str(exc), exc_info=exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Unable to retrieve system health status",
        ) from exc


@router.get("/info", response_model=ApplicationInfoDTO)
@inject
async def info(
    request: Request,
    get_application_info_use_case: GetApplicationInfoUseCase = Depends(
        Provide["get_application_info_use_case"]
    ),
) -> ApplicationInfoDTO:
    """Return strategic information about the application."""
    started_at = getattr(request.app.state, "started_at", None)
    try:
        info_response = await get_application_info_use_case.execute(started_at)
        logger.debug("info.retrieved", status=info_response.status.value)
        return info_response
    except Exception as exc:  # pragma: no cover - defensive logging path
        logger.error("info.fetch.failure", error=str(exc), exc_info=exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to retrieve application info",
        ) from exc
