"""
Type handlers for different base systems.

Each handler implements the logic for tracking and validating schemas
according to the specific rules of its base system.
"""

from .docetl_handler import DocETLTypeHandler

__all__ = [
    'DocETLTypeHandler',
]