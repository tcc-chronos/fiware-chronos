"""
Domain Layer Package

This package contains the core business logic and rules of the application.
It defines entities, repositories, and services without dependencies on
external frameworks or infrastructure concerns.
"""

# Re-export submodules
from src.domain import entities, repositories

__all__ = ["entities", "repositories"]
