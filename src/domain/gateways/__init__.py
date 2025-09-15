"""
Gateways Package - Domain Layer

This package contains interfaces defining gateway contracts
for external service communications. Specific implementations
are provided by the infrastructure layer.
"""

from .iot_agent_gateway import IIoTAgentGateway

__all__ = ["IIoTAgentGateway"]
