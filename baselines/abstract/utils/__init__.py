"""
Utility functions for abstract layer.
"""
from .data_io import load_from_reference, save_to_reference, DataReference

__all__ = [
    'DataReference',
    'load_from_reference',
    'save_to_reference',
]
