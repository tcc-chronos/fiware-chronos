"""
Controllers Package - Presentation Layer

This package contains FastAPI controllers (routers) that handle
HTTP requests and responses. Controllers are responsible for
input validation, error handling, and mapping between API DTOs
and application layer use cases.
"""

from .devices_controller import router as devices_router
from .models_controller import router as models_router
from .system_controller import router as system_router

__all__ = ["models_router", "devices_router", "system_router"]
