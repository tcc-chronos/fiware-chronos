"""
Main module entry point.

This allows running the worker as: python -m src.main
(defaulting to worker functionality when the main module is executed)
"""

from .worker import main

if __name__ == "__main__":
    main()
