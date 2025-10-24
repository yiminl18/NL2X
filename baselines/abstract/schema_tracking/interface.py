"""
Schema Tracking Interface

Protocol definition for schema tracking implementations.
"""

from typing import Protocol, Dict, Any, List, Tuple, Optional


class SchemaTrackerProtocol(Protocol):
    """
    Protocol for schema tracking implementations.

    Defines the interface that all system-specific schema trackers must implement.
    """

    def get_current_schema(self) -> Dict[str, str]:
        """
        Get current schema state.

        Returns:
            Dictionary mapping field names to type strings
        """
        ...

    def get_history(self) -> List[Dict[str, str]]:
        """
        Get schema evolution history.

        Returns:
            List of schema states (oldest to newest)
        """
        ...

    def validate_operator(self, operator_config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate operator against current schema without applying changes.

        Args:
            operator_config: Operator configuration dictionary

        Returns:
            Tuple of (is_valid, error_messages)
        """
        ...

    def update_schema(self, operator_config: Dict[str, Any]) -> None:
        """
        Update schema without validation (for pipeline validation scenarios).

        Args:
            operator_config: Operator configuration dictionary

        Raises:
            Exception: If schema transformation fails
        """
        ...

    def apply_operator(self, operator_config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate and apply operator transformation to schema.

        Only updates schema if validation passes (used for runtime schema tracking).

        Args:
            operator_config: Operator configuration dictionary

        Returns:
            Tuple of (is_valid, error_messages)
        """
        ...

    def validate_pipeline(
        self,
        operators: List[Dict[str, Any]],
        continue_on_error: bool = True
    ) -> Tuple[bool, List[str], List[str]]:
        """
        Validate entire pipeline (continues updating schema even if validation fails).

        This method is designed for static pipeline validation where we want to:
        1. Collect all validation errors across all operators
        2. Continue updating schema even if validation fails (to check subsequent operators)

        Args:
            operators: List of operator configurations
            continue_on_error: If True, continue validating even if errors occur (default: True)

        Returns:
            Tuple of (passed, errors, warnings)
        """
        ...

    def rollback(self, steps: int = 1) -> None:
        """
        Rollback schema changes.

        Args:
            steps: Number of steps to rollback
        """
        ...

    @classmethod
    def convert_schema_to_abstract(cls, schema: Dict[str, str]) -> Dict[str, str]:
        """
        Convert base system schema to abstract layer format.

        Args:
            schema: Schema dictionary in base system format (e.g., {'file': 'str', 'medications': 'list[{...}]'})

        Returns:
            Schema dictionary in abstract layer format (e.g., {'file': 'String', 'medications': 'List[Dict{...}]'})
        """
        ...
