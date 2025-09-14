"""
Main module - Main/Composition Root Layer

This module serves as the entry point for the application, orchestrating
the initialization and configuration of all other layers.

Its primary responsibilities include:
- Setting up the application environment
- Configuring dependencies and services (Composition Root)
- Initializing the frameworks (FastAPI)
- Orchestrating the interaction between different layers
"""

from .config import AppSettings, get_settings
from .container import AppContainer, get_container, init_container

__all__ = [
    "AppSettings",
    "get_settings",
    "AppContainer",
    "init_container",
    "get_container",
]
