"""
Database package - Infrastructure Layer

This package contains database-related implementations for the FIWARE Chronos system.
It provides concrete implementations of database connections, query handlers,
and other database-specific functionality needed by the application.
"""

from src.infrastructure.database.mongo_database import MongoDatabase

__all__ = ["MongoDatabase"]
