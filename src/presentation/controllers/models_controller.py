"""
Models Router - Presentation Layer

This module defines the FastAPI router for model endpoints.
"""

from typing import List, Optional

from dependency_injector.wiring import Provide, inject
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import UUID4

from src.application.dtos.model_dto import (
    ModelCreateDTO,
    ModelDetailResponseDTO,
    ModelResponseDTO,
    ModelUpdateDTO,
)
from src.application.use_cases.model_use_cases import (
    CreateModelUseCase,
    DeleteModelUseCase,
    GetModelByIdUseCase,
    GetModelsUseCase,
    UpdateModelUseCase,
)
from src.domain.entities.errors import ModelNotFoundError, ModelOperationError

router = APIRouter(prefix="/models", tags=["Models"])


@router.get("/", response_model=List[ModelResponseDTO])
@inject
async def get_models(
    skip: int = Query(0, ge=0, description="Number of models to skip"),
    limit: int = Query(
        100, ge=1, le=1000, description="Maximum number of models to return"
    ),
    model_type: Optional[str] = Query(
        None, description="Filter by model type (e.g., 'lstm', 'gru')"
    ),
    model_status: Optional[str] = Query(
        None, description="Filter by model status (e.g., 'draft', 'trained')"
    ),
    entity_id: Optional[str] = Query(None, description="Filter by FIWARE entity ID"),
    feature: Optional[str] = Query(None, description="Filter by feature name"),
    get_models_use_case: GetModelsUseCase = Depends(Provide["get_models_use_case"]),
) -> List[ModelResponseDTO]:
    """
    Get a list of models with pagination and filtering options.

    Filter models by various criteria such as model type, status,
    entity ID, and feature. All filters are optional and can be
    combined for more specific queries.
    """
    try:
        return await get_models_use_case.execute(
            skip=skip,
            limit=limit,
            model_type=model_type,
            status=model_status,
            entity_id=entity_id,
            feature=feature,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve models: {str(e)}",
        )


@router.get("/{model_id}", response_model=ModelDetailResponseDTO)
@inject
async def get_model_by_id(
    model_id: UUID4,
    get_model_use_case: GetModelByIdUseCase = Depends(
        Provide["get_model_by_id_use_case"]
    ),
) -> ModelDetailResponseDTO:
    """
    Get detailed information about a specific model by its ID.
    """
    try:
        return await get_model_use_case.execute(model_id=model_id)
    except ModelNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve model: {str(e)}",
        )


@router.post(
    "/",
    response_model=ModelResponseDTO,
    status_code=status.HTTP_201_CREATED,
)
@inject
async def create_model(
    model_dto: ModelCreateDTO,
    create_model_use_case: CreateModelUseCase = Depends(
        Provide["create_model_use_case"]
    ),
) -> ModelResponseDTO:
    """
    Create a new deep learning model configuration.
    """
    try:
        return await create_model_use_case.execute(model_dto=model_dto)
    except ModelOperationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create model: {str(e)}",
        )


@router.patch("/{model_id}", response_model=ModelResponseDTO)
@inject
async def update_model(
    model_id: UUID4,
    model_dto: ModelUpdateDTO,
    update_model_use_case: UpdateModelUseCase = Depends(
        Provide["update_model_use_case"]
    ),
) -> ModelResponseDTO:
    """
    Update an existing model with new values.
    """
    try:
        return await update_model_use_case.execute(
            model_id=model_id, model_dto=model_dto
        )
    except ModelNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except ModelOperationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update model: {str(e)}",
        )


@router.delete("/{model_id}", status_code=status.HTTP_204_NO_CONTENT)
@inject
async def delete_model(
    model_id: UUID4,
    delete_model_use_case: DeleteModelUseCase = Depends(
        Provide["delete_model_use_case"]
    ),
) -> None:
    """
    Delete a model by its ID.
    """
    try:
        await delete_model_use_case.execute(model_id=model_id)
    except ModelNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except ModelOperationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete model: {str(e)}",
        )
