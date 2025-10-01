"""
DTOs Package - Application Layer

This package contains Data Transfer Objects (DTOs) used for data exchange
between the application layer and the presentation layer.
"""

from .device_dto import (
    DeviceEntityDTO,
    DevicesResponseDTO,
    GroupedDevicesDTO,
    IoTAgentDeviceDTO,
    IoTAgentDevicesResponseDTO,
)
from .health_dto import ApplicationInfoDTO, DependencyStatusDTO, SystemHealthDTO
from .model_dto import (
    DenseLayerDTO,
    ModelCreateDTO,
    ModelDetailResponseDTO,
    ModelResponseDTO,
    ModelTrainingSummaryDTO,
    ModelTypeOptionDTO,
    ModelUpdateDTO,
    RNNLayerDTO,
)
from .training_dto import (
    DataCollectionJobDTO,
    StartTrainingResponseDTO,
    TrainingJobDTO,
    TrainingJobSummaryDTO,
    TrainingMetricsDTO,
    TrainingRequestDTO,
)

__all__ = [
    "DenseLayerDTO",
    "ModelCreateDTO",
    "ModelUpdateDTO",
    "ModelResponseDTO",
    "ModelDetailResponseDTO",
    "ModelTrainingSummaryDTO",
    "ModelTypeOptionDTO",
    "RNNLayerDTO",
    "IoTAgentDeviceDTO",
    "IoTAgentDevicesResponseDTO",
    "DeviceEntityDTO",
    "GroupedDevicesDTO",
    "DevicesResponseDTO",
    "SystemHealthDTO",
    "DependencyStatusDTO",
    "ApplicationInfoDTO",
    "TrainingRequestDTO",
    "TrainingMetricsDTO",
    "TrainingJobDTO",
    "TrainingJobSummaryDTO",
    "DataCollectionJobDTO",
    "StartTrainingResponseDTO",
]
