"""
Base System Schema Tracker

Wrapper for system-specific schema trackers used by abstract_step baseline.
"""

from typing import Dict, Any, List, Tuple, Optional

from ..abstract.support import BaseSystem
from ..abstract.ops.base import Operator
from ..abstract.convert.docetl import abstract_to_docetl
from ..abstract.schema_tracking import create_tracker, create_tracker_from_dataset


class BaseSystemSchemaTracker:
    """
    Schema tracker wrapper for abstract_step.py.

    Now a thin wrapper that delegates to baselines/abstract/schema_tracking/tracker.py.

    Usage:
        tracker = BaseSystemSchemaTracker(base_system, initial_schema)
        # or
        tracker = BaseSystemSchemaTracker.from_dataset(base_system, dataset_path)

        is_valid, errors = tracker.apply_operator(abstract_operator)
        schema = tracker.get_current_schema()
    """

    def __init__(
        self,
        base_system: BaseSystem,
        initial_schema: Dict[str, str],
        verbose: bool = False
    ):
        """
        Initialize schema tracker.

        Args:
            base_system: The base system to use (DocETL, Lotus, etc.)
            initial_schema: Initial dataset schema in abstract format
            verbose: Enable verbose logging
        """
        self.base_system = base_system
        self.verbose = verbose

        # Create core tracker using factory function
        self.core_tracker = create_tracker(base_system, initial_schema, verbose)

    @classmethod
    def from_dataset(
        cls,
        base_system: BaseSystem,
        dataset_path: str,
        verbose: bool = False
    ) -> 'BaseSystemSchemaTracker':
        """
        Create tracker from dataset file by inferring initial schema.

        Args:
            base_system: The base system
            dataset_path: Path to dataset file
            verbose: Enable verbose logging

        Returns:
            BaseSystemSchemaTracker instance
        """
        # Use factory function to create tracker from dataset
        core_tracker = create_tracker_from_dataset(base_system, dataset_path, verbose)
        initial_schema = core_tracker.get_current_schema()

        # Create wrapper instance
        instance = cls(base_system, initial_schema, verbose)
        instance.core_tracker = core_tracker  # Reuse the tracker
        return instance

    def get_current_schema(self) -> Dict[str, str]:
        """
        Get current schema in abstract format (for LLM prompts).

        Returns:
            Schema dictionary in abstract format
        """
        raw_schema = self.core_tracker.get_current_schema()
        return self.core_tracker.convert_schema_to_abstract(raw_schema)

    def get_base_schema(self) -> Dict[str, str]:
        """
        Get current schema in base system format.

        Returns:
            Schema dictionary in base system format
        """
        # For now, return abstract format (can be extended if needed)
        return self.core_tracker.get_current_schema()

    def apply_operator(self, abstract_operator: Operator) -> Tuple[bool, List[str]]:
        """
        Apply operator transformation to schema.

        This method:
        1. Converts abstract operator to base system format
        2. Validates against base system rules
        3. Updates schema using core tracker
        4. Returns validation result

        Args:
            abstract_operator: Abstract operator to apply

        Returns:
            Tuple of (success, list_of_errors)
        """
        try:
            # Convert abstract operator to base system format
            base_operator = abstract_to_docetl(abstract_operator)

            if self.verbose:
                print(f"Applying operator: {abstract_operator.name} ({abstract_operator.type})")
                print(f"Current schema fields: {list(self.core_tracker.get_current_schema().keys())}")

            # Apply using core tracker
            is_valid, errors = self.core_tracker.apply_operator(base_operator)

            if not is_valid and self.verbose:
                print(f"  Validation failed: {errors}")

            return is_valid, errors

        except Exception as e:
            error_msg = f"Failed to apply operator: {str(e)}"
            if self.verbose:
                print(f"Error: {error_msg}")

            return False, [error_msg]

    def validate_operator(self, base_operator: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate operator against current schema without updating it.

        Args:
            base_operator: Operator in base system format (e.g., DocETL dict)

        Returns:
            Tuple of (is_valid, error_messages)
        """
        return self.core_tracker.validate_operator(base_operator)

    def update_schema(self, base_operator: Dict[str, Any]) -> None:
        """
        Update schema based on operator transformation without validation.

        Used when validation has already been performed separately.

        Args:
            base_operator: Operator in base system format (e.g., DocETL dict)

        Raises:
            Exception: If schema transformation fails
        """
        self.core_tracker.update_schema(base_operator)

    def rollback(self, steps: int = 1) -> None:
        """
        Rollback to previous schema state.

        Args:
            steps: Number of steps to rollback

        Raises:
            ValueError: If no history to rollback to
        """
        self.core_tracker.rollback(steps)

        if self.verbose:
            schema = self.core_tracker.get_current_schema()
            print(f"Rolled back {steps} step(s). Current fields: {list(schema.keys())}")

    def get_schema_history(self) -> List[Dict[str, str]]:
        """
        Get the full history of schema transformations.

        Returns:
            List of schema states (oldest to newest)
        """
        return self.core_tracker.get_history()


def format_fields_for_prompt(schema: Dict[str, str]) -> str:
    """
    Format field schema for LLM prompts.

    Args:
        schema: Field schema dictionary in abstract format

    Returns:
        Formatted string for prompts

    Example:
        >>> format_fields_for_prompt({"id": "Integer", "text": "String"})
        '- id: Integer\\n- text: String'
    """
    if not schema:
        return "(No fields available)"

    lines = []
    for field, field_type in sorted(schema.items()):
        lines.append(f"- {field}: {field_type}")

    result = '\n'.join(lines)

    # Check for complex types and add access hints
    has_simple_list = any(field_type.startswith("List[") and "Dict" not in field_type for field_type in schema.values())
    has_complex_list = any("List[Dict" in field_type for field_type in schema.values())
    has_dict = any("Dict" in field_type for field_type in schema.values())

    if has_simple_list:
        result += "\nNote: For simple List fields (List[String], List[Integer]), access elements using `{{ input.field[idx] }}` or use loops `{% for item in input.field %}`. Do NOT use `{{ input.field }}` directly."

    if has_complex_list:
        result += "\nNote: For List[Dict] fields, access elements using `{{ input.field[idx].subfield }}` or use Unnest to expand the list."

    if has_dict:
        result += "\nNote: For Dict fields, access nested fields using `{{ input.field.subfield }}` or use Unnest to expand the dictionary."

    return result


def infer_schema_from_samples(samples: List[Dict[str, Any]]) -> Dict[str, str]:
    """
    Infer abstract schema from sample records with full recursive type inference.

    Returns schema using Abstract layer type format (capitalized):
    - Basic: 'String', 'Integer', 'Float', 'Boolean'
    - List: 'List[String]' or 'List[Dict{name: String, age: Integer}]'
    - Dict: 'Dict{name: String, age: Integer}'

    Args:
        samples: List of sample data records

    Returns:
        Schema dictionary in abstract format with detailed types

    Examples:
        >>> samples = [{'items': ['a', 'b'], 'user': {'name': 'John', 'age': 30}}]
        >>> infer_schema_from_samples(samples)
        {'items': 'List[String]', 'user': 'Dict{age: Integer, name: String}'}

    Note: For full schema tracking, use create_tracker_from_dataset()
    """
    if not samples or not isinstance(samples[0], dict):
        return {}

    def _infer_abstract_type(value: Any) -> str:
        """
        Recursively infer Abstract layer type from value.
        No depth limit - complete recursive inference.
        """
        # None
        if value is None:
            return 'String'  # Default to String for None

        # Basic types (bool before int!)
        if isinstance(value, bool):
            return 'Boolean'
        elif isinstance(value, int):
            return 'Integer'
        elif isinstance(value, float):
            return 'Float'
        elif isinstance(value, str):
            return 'String'

        # List type - recursively infer element type
        elif isinstance(value, list):
            if len(value) == 0:
                return 'List'  # Can't infer element type

            # Sample multiple elements
            sample_size = min(10, len(value))
            element_types = set()

            for i in range(sample_size):
                elem_type = _infer_abstract_type(value[i])
                element_types.add(elem_type)

            if len(element_types) == 1:
                # All same type
                elem_type = element_types.pop()
                return f'List[{elem_type}]'
            else:
                # Mixed types
                return 'List'

        # Dict type - recursively infer all field types
        elif isinstance(value, dict):
            if len(value) == 0:
                return 'Dict'

            # Recursively infer each field
            field_types = {}
            for field_name, field_value in value.items():
                field_types[field_name] = _infer_abstract_type(field_value)

            # Format as Dict{field1: Type1, field2: Type2, ...}
            # Sort for consistency
            fields_str = ', '.join(f'{k}: {v}' for k, v in sorted(field_types.items()))
            return f'Dict{{{fields_str}}}'

        else:
            # Unknown type - raise error
            raise ValueError(f"Unsupported type for schema inference: {type(value)}")

    # Infer from first sample
    schema = {}
    for field_name, field_value in samples[0].items():
        schema[field_name] = _infer_abstract_type(field_value)

    return schema
