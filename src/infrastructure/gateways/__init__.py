"""
Gateways Package - Infrastructure Layer

This package contains concrete implementations of the gateway
interfaces defined in the domain layer. These implementations
handle the details of external service communications.
"""

from .iot_agent_gateway import IoTAgentGateway

__all__ = ["IoTAgentGateway"]
