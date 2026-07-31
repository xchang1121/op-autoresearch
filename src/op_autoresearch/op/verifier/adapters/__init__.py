"""Adapter modules for verification system."""

from .factory import get_framework_adapter, get_dsl_adapter, get_backend_adapter

__all__ = [
    'get_framework_adapter',
    'get_dsl_adapter',
    'get_backend_adapter',
]

