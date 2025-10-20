"""
Abstract Layer Type System

Provides type inference and tracking for abstract layer operators.
Unlike DocETL's type system, this uses a simpler, more generic approach.
"""

from typing import Dict, Any, List, Optional, Set
import json

# Import unified type system from abstract layer
from ..abstract.type import (
    parse_type_string,
    serialize_type_dict,
    TypeSystem as UnifiedTypeSystem
)


class AbstractTypeSystem:
    """
    Type system for abstract layer operators.

    Tracks field types through pipeline transformations and validates
    operator input/output schemas.

    Supported Types:
    - String, Integer, Float, Boolean
    - List[Type] (e.g., List[String])
    - Dict (generic dictionary)
    - Any (fallback for unknown types)
    """

    # Basic type mappings
    PYTHON_TO_ABSTRACT = {
        str: 'String',
        int: 'Integer',
        float: 'Float',
        bool: 'Boolean',
        list: 'List',
        dict: 'Dict',
    }

    ABSTRACT_TYPES = {
        'String', 'Integer', 'Float', 'Boolean',
        'List', 'Dict', 'Any'
    }

    def __init__(self, initial_schema: Optional[Dict[str, str]] = None):
        """
        Initialize type system.

        Args:
            initial_schema: Initial field types, e.g., {"id": "Integer", "text": "String"}
        """
        self.current_schema: Dict[str, str] = initial_schema or {}
        self.schema_history: List[Dict[str, str]] = []

    def get_current_schema(self) -> Dict[str, str]:
        """
        Get the current schema (available fields and their types).

        Returns:
            Dictionary mapping field names to type strings
        """
        return self.current_schema.copy()

    def set_schema(self, schema: Dict[str, str]) -> None:
        """
        Set the current schema and save to history.

        Args:
            schema: New schema to set
        """
        self.schema_history.append(self.current_schema.copy())
        self.current_schema = schema.copy()

    def update_schema(self, new_fields: Dict[str, str]) -> None:
        """
        Update current schema with new fields (preserves existing fields).

        Args:
            new_fields: New fields to add or update
        """
        self.schema_history.append(self.current_schema.copy())
        self.current_schema.update(new_fields)

    def apply_map_operator(self, output_schema: Dict[str, str]) -> None:
        """
        Apply Map operator transformation to schema.
        Map preserves all input fields and adds new fields.

        Args:
            output_schema: New fields to add (must not include existing fields)
        """
        # Validate that output_schema only contains new fields
        for field in output_schema.keys():
            if field in self.current_schema:
                raise ValueError(
                    f"Map operator output_schema can only contain new fields. "
                    f"Field '{field}' already exists in current schema."
                )

        self.update_schema(output_schema)

    def apply_filter_operator(self) -> None:
        """
        Apply Filter operator transformation to schema.
        Filter does not change the schema (same fields in/out).
        """
        # Filter doesn't change schema, but we still record it in history
        self.schema_history.append(self.current_schema.copy())

    def apply_reduce_operator(self, output_schema: Dict[str, str], reduce_key: str) -> None:
        """
        Apply Reduce operator transformation to schema.
        Reduce groups by key and produces aggregated fields.

        Args:
            output_schema: Output schema (must include reduce_key)
            reduce_key: The field used for grouping
        """
        # Validate reduce_key exists in current schema
        if reduce_key not in self.current_schema:
            raise ValueError(
                f"Reduce key '{reduce_key}' not found in current schema. "
                f"Available: {list(self.current_schema.keys())}"
            )

        # Validate reduce_key is in output
        if reduce_key not in output_schema:
            raise ValueError(
                f"Reduce operator output must include reduce_key: {reduce_key}"
            )

        self.set_schema(output_schema)

    def apply_resolve_operator(self, output_schema: Dict[str, str]) -> None:
        """
        Apply Resolve operator transformation to schema.
        Resolve merges similar records, potentially creating new fields.

        Args:
            output_schema: Output schema after resolution
        """
        self.set_schema(output_schema)

    def apply_extract_operator(self, output_schema: Dict[str, str]) -> None:
        """
        Apply Extract operator transformation to schema.
        Extract adds extracted fields to existing schema.

        Args:
            output_schema: Complete output schema
        """
        self.set_schema(output_schema)

    def apply_generic_operator(self, output_schema: Dict[str, str]) -> None:
        """
        Apply a generic operator transformation.

        Args:
            output_schema: Output schema
        """
        self.set_schema(output_schema)

    def validate_input_schema(self, required_fields: Dict[str, str]) -> bool:
        """
        Validate that current schema contains required fields with correct types.

        Args:
            required_fields: Fields required by operator

        Returns:
            True if valid

        Raises:
            ValueError: If validation fails
        """
        for field, expected_type in required_fields.items():
            if field not in self.current_schema:
                raise ValueError(
                    f"Required field '{field}' not found in current schema. "
                    f"Available: {list(self.current_schema.keys())}"
                )

            current_type = self.current_schema[field]

            # Allow Any type to match anything
            if expected_type == 'Any' or current_type == 'Any':
                continue

            # Check type compatibility
            if not self._types_compatible(current_type, expected_type):
                raise ValueError(
                    f"Type mismatch for field '{field}': "
                    f"expected {expected_type}, got {current_type}"
                )

        return True

    def _types_compatible(self, type1: str, type2: str) -> bool:
        """
        Check if two types are compatible.

        Args:
            type1: First type
            type2: Second type

        Returns:
            True if compatible
        """
        # Exact match
        if type1 == type2:
            return True

        # Any matches everything
        if type1 == 'Any' or type2 == 'Any':
            return True

        # List type compatibility (simple check)
        if type1.startswith('List[') and type2.startswith('List['):
            return True  # Simplified - could check inner types

        return False

    def get_schema_history(self) -> List[Dict[str, str]]:
        """
        Get the full history of schema transformations.

        Returns:
            List of schemas (oldest to newest)
        """
        return self.schema_history.copy()

    def rollback(self) -> None:
        """
        Rollback to previous schema state.

        Raises:
            ValueError: If no history to rollback to
        """
        if not self.schema_history:
            raise ValueError("No schema history to rollback to")

        self.current_schema = self.schema_history.pop()


def infer_type_from_sample(value: Any, max_depth: int = 3, current_depth: int = 0) -> str:
    """
    Infer abstract type from a sample value with support for complex nested types.

    Args:
        value: Sample value
        max_depth: Maximum recursion depth for nested structures
        current_depth: Current recursion depth

    Returns:
        Abstract type string (e.g., "String", "List[String]", "Dict[{name: String, age: Integer}]")
    """
    if value is None:
        return 'Any'

    # Prevent infinite recursion
    if current_depth >= max_depth:
        value_type = type(value)
        if value_type in AbstractTypeSystem.PYTHON_TO_ABSTRACT:
            return AbstractTypeSystem.PYTHON_TO_ABSTRACT[value_type]
        return 'Any'

    value_type = type(value)

    # Handle basic types
    if value_type in AbstractTypeSystem.PYTHON_TO_ABSTRACT:
        base_type = AbstractTypeSystem.PYTHON_TO_ABSTRACT[value_type]

        # Enhanced List handling - infer element type with better accuracy
        if base_type == 'List' and isinstance(value, list):
            if len(value) == 0:
                return 'List'  # Empty list, can't infer element type

            # Sample multiple elements to determine consistent type
            sample_size = min(5, len(value))
            element_types = set()

            for i in range(sample_size):
                if value[i] is not None:
                    elem_type = infer_type_from_sample(value[i], max_depth, current_depth + 1)
                    element_types.add(elem_type)

            if len(element_types) == 0:
                return 'List'  # All elements are None
            elif len(element_types) == 1:
                # All sampled elements have the same type
                return f'List[{element_types.pop()}]'
            else:
                # Mixed types - check if they're all basic types
                all_basic = all(t in AbstractTypeSystem.ABSTRACT_TYPES for t in element_types)
                if all_basic:
                    # Mixed basic types, return generic List
                    return 'List'
                else:
                    # Try to use the most common type
                    return f'List[{sorted(element_types)[0]}]'

        # Enhanced Dict handling - infer field structure
        if base_type == 'Dict' and isinstance(value, dict):
            if len(value) == 0:
                return 'Dict'  # Empty dict

            # Infer field types
            fields = {}
            for key, val in value.items():
                if isinstance(key, str):  # Only string keys are supported
                    field_type = infer_type_from_sample(val, max_depth, current_depth + 1)
                    fields[key] = field_type

            if len(fields) == 0:
                return 'Dict'

            # Format as Dict[{field1: Type1, field2: Type2}]
            type_dict = {'type': 'Dict', 'fields': {}}
            for field_name, field_type_str in fields.items():
                # Parse the field type string to dict format
                is_valid, parsed, _ = parse_type_string(field_type_str)
                if is_valid:
                    type_dict['fields'][field_name] = parsed
                else:
                    type_dict['fields'][field_name] = {'type': field_type_str}
            return serialize_type_dict(type_dict)

        return base_type

    # Fallback
    return 'Any'


def infer_schema_from_samples(samples: List[Dict[str, Any]]) -> Dict[str, str]:
    """
    Infer schema from a list of sample records.

    Args:
        samples: List of sample data records

    Returns:
        Inferred schema dictionary
    """
    if not samples:
        return {}

    schema: Dict[str, str] = {}
    all_fields: Set[str] = set()

    # Collect all fields
    for sample in samples:
        all_fields.update(sample.keys())

    # Infer type for each field
    for field in all_fields:
        # Find first non-None value
        for sample in samples:
            if field in sample and sample[field] is not None:
                schema[field] = infer_type_from_sample(sample[field])
                break
        else:
            # All values are None or missing
            schema[field] = 'Any'

    return schema


def format_fields_for_prompt(schema: Dict[str, str]) -> str:
    """
    Format field schema for LLM prompts.

    Args:
        schema: Field schema dictionary

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

    return '\n'.join(lines)


def validate_operator_schemas(
    operator_type: str,
    input_schema: Dict[str, str],
    output_schema: Dict[str, str],
    current_schema: Dict[str, str],
    **kwargs
) -> bool:
    """
    Validate operator input/output schemas.

    Args:
        operator_type: Type of operator (Map, Filter, etc.)
        input_schema: Operator's input schema requirement
        output_schema: Operator's output schema
        current_schema: Current available fields
        **kwargs: Operator-specific parameters (e.g., reduce_key)

    Returns:
        True if valid

    Raises:
        ValueError: If validation fails
    """
    # Check input schema is satisfied
    for field, expected_type in input_schema.get('fields', {}).items():
        if field not in current_schema:
            raise ValueError(
                f"{operator_type} operator requires field '{field}' which is not available. "
                f"Available: {list(current_schema.keys())}"
            )

    # Operator-specific validation
    if operator_type == 'Map':
        # Map output_schema must only contain new fields
        for field in output_schema.keys():
            if field in current_schema:
                raise ValueError(
                    f"Map operator output_schema can only contain new fields. "
                    f"Field '{field}' already exists in current schema."
                )

    elif operator_type == 'Filter':
        # Filter must have same input/output schema
        if set(output_schema.keys()) != set(current_schema.keys()):
            raise ValueError(
                "Filter operator must have identical input and output schemas"
            )

    elif operator_type == 'Reduce':
        # Reduce must have reduce_key in output
        reduce_key = kwargs.get('reduce_key')
        if not reduce_key:
            raise ValueError("Reduce operator requires 'reduce_key' parameter")

        if reduce_key not in output_schema:
            raise ValueError(
                f"Reduce operator output must include reduce_key: {reduce_key}"
            )

    return True


def merge_schemas(schema1: Dict[str, str], schema2: Dict[str, str]) -> Dict[str, str]:
    """
    Merge two schemas (for Join operations).

    Args:
        schema1: First schema
        schema2: Second schema

    Returns:
        Merged schema

    Raises:
        ValueError: If there are conflicting field types
    """
    merged = schema1.copy()

    for field, field_type in schema2.items():
        if field in merged:
            # Check for conflicts
            if merged[field] != field_type and merged[field] != 'Any' and field_type != 'Any':
                raise ValueError(
                    f"Schema conflict for field '{field}': "
                    f"{merged[field]} vs {field_type}"
                )
            # Keep the more specific type (not Any)
            if merged[field] == 'Any':
                merged[field] = field_type
        else:
            merged[field] = field_type

    return merged
