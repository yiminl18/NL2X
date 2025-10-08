"""
Convert module for mapping between system-specific operators and abstract operators.

This module provides conversion utilities for:
- DocETL operators
- Lotus operators
- Palimpzest operators

Usage:
    from baselines.abstract.convert import docetl_to_abstract, abstract_to_docetl
"""

# Import from new docetl submodule for backward compatibility
from .docetl import docetl_to_abstract, abstract_to_docetl, docetl_pipeline_to_abstract

__all__ = ['docetl_to_abstract', 'abstract_to_docetl', 'docetl_pipeline_to_abstract']
