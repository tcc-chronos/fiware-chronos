"""Domain ports package."""

from .health_check import IHealthCheckService
from .training_orchestrator import ITrainingOrchestrator

__all__ = ["IHealthCheckService", "ITrainingOrchestrator"]
