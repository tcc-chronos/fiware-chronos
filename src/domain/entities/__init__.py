"""
Domain Entities Package

This package contains the core domain entities and business logic.
"""

from .errors import (
    DomainError,
    ModelNotFoundError,
    ModelOperationError,
    ModelValidationError,
)
from .health import ApplicationInfo, DependencyStatus, ServiceStatus, SystemHealth
from .iot import DeviceAttribute, IoTDevice, IoTDeviceCollection
from .model import DenseLayerConfig, Model, ModelStatus, ModelType, RNNLayerConfig
from .time_series import HistoricDataPoint

__all__ = [
    "Model",
    "ModelType",
    "ModelStatus",
    "RNNLayerConfig",
    "DenseLayerConfig",
    "SystemHealth",
    "DependencyStatus",
    "ServiceStatus",
    "ApplicationInfo",
    "DeviceAttribute",
    "IoTDevice",
    "IoTDeviceCollection",
    "HistoricDataPoint",
    "DomainError",
    "ModelNotFoundError",
    "ModelValidationError",
    "ModelOperationError",
]
