"""
Validation Interface

Validation result types and protocol definitions.

For validation functionality, use:
- DocETLSchemaTracker directly for schema tracking and validation
- validate_pipeline_static() from abstract_step_utils.validation for pipeline validation
"""

from .interface import ValidatorProtocol
from .result import ValidationResult, ValidationError, ValidationWarning


__all__ = [
    'ValidatorProtocol',
    'ValidationResult',
    'ValidationError',
    'ValidationWarning',
]
