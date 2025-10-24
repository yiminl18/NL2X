"""
Validation Interface

Protocol definition for pipeline validation implementations.
"""

from typing import Protocol, Dict, Any, Optional
from .result import ValidationResult


class ValidatorProtocol(Protocol):
    """
    Protocol for pipeline validator implementations.

    Defines the interface that all system-specific validators must implement.
    """

    def validate_pipeline(
        self,
        pipeline_config: Dict[str, Any],
        dataset_schema: Optional[Dict[str, str]] = None
    ) -> ValidationResult:
        """
        Validate complete pipeline configuration.

        Args:
            pipeline_config: Pipeline configuration dictionary
            dataset_schema: Optional initial dataset schema (inferred if not provided)

        Returns:
            ValidationResult with errors, warnings, and schema history
        """
        ...

    def validate_operator(
        self,
        operator_config: Dict[str, Any],
        current_schema: Dict[str, str]
    ) -> ValidationResult:
        """
        Validate single operator against current schema.

        Args:
            operator_config: Operator configuration
            current_schema: Current available schema

        Returns:
            ValidationResult for this operator
        """
        ...
