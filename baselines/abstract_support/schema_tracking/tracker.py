"""
Schema tracker for Abstract operators.

Validates operators and tracks schema transformations during pipeline generation.
"""
from typing import Dict, List, Tuple, Any
from pathlib import Path
import json
from ...abstract.ops.base import Operator
from .schema_inference import compute_schema_transformation


class AbstractSchemaTracker:
    """
    Schema tracker for abstract-only operators.

    Validates operator constraints and tracks schema evolution.
    """

    def __init__(self, initial_schema: Dict[str, str], verbose: bool = False):
        """
        Initialize schema tracker.

        Args:
            initial_schema: Initial schema (field_name -> type_string)
            verbose: Enable verbose logging
        """
        self.current_schema = initial_schema.copy()
        self.verbose = verbose

    @classmethod
    def from_dataset(cls, dataset_path: str, verbose: bool = False) -> 'AbstractSchemaTracker':
        """
        Create tracker from dataset file by inferring initial schema.

        Args:
            dataset_path: Path to dataset file (JSON or CSV)
            verbose: Enable verbose logging

        Returns:
            AbstractSchemaTracker instance

        Raises:
            FileNotFoundError: If dataset file not found
            ValueError: If schema inference fails
        """
        dataset_path = Path(dataset_path)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

        try:
            # Load data based on format
            if dataset_path.suffix == '.json':
                with open(dataset_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            elif dataset_path.suffix == '.csv':
                import pandas as pd
                df = pd.read_csv(dataset_path)
                data = df.to_dict('records')
            else:
                raise ValueError(f"Unsupported file format: {dataset_path.suffix}")

            if not isinstance(data, list) or len(data) == 0:
                raise ValueError("Dataset must be a non-empty list")

            first_item = data[0]
            if not isinstance(first_item, dict):
                raise ValueError("Dataset items must be dictionaries")

            # Infer schema from first item
            initial_schema = {}
            for field_name, field_value in first_item.items():
                initial_schema[field_name] = cls._infer_type_from_value(field_value)

            if verbose:
                print(f"Inferred schema from {dataset_path}:")
                for field, field_type in initial_schema.items():
                    print(f"  {field}: {field_type}")

            return cls(initial_schema, verbose)

        except (json.JSONDecodeError, KeyError, IndexError) as e:
            raise ValueError(f"Failed to infer schema from dataset: {e}")

    @staticmethod
    def _infer_type_from_value(value: Any) -> str:
        """
        Infer Abstract type from value.

        Args:
            value: Value to infer type from

        Returns:
            Type string in Abstract format (String, Integer, List, Dict, etc.)
        """
        if value is None:
            return 'String'  # Default for null values
        elif isinstance(value, bool):
            return 'Boolean'
        elif isinstance(value, int):
            return 'Integer'
        elif isinstance(value, float):
            return 'Float'
        elif isinstance(value, str):
            return 'String'
        elif isinstance(value, list):
            if len(value) == 0:
                return 'List'
            # Infer element type from first item
            elem_type = AbstractSchemaTracker._infer_type_from_value(value[0])
            return f'List[{elem_type}]'
        elif isinstance(value, dict):
            if len(value) == 0:
                return 'Dict'
            # For dicts, create detailed type signature
            field_types = {k: AbstractSchemaTracker._infer_type_from_value(v)
                          for k, v in value.items()}
            field_sig = ', '.join(f'{k}: {v}' for k, v in field_types.items())
            return f'Dict{{{field_sig}}}'
        else:
            return 'String'  # Default fallback

    def validate_operator(self, operator: Operator) -> Tuple[bool, List[str]]:
        """
        Validate operator against current schema.

        Args:
            operator: Operator to validate

        Returns:
            (is_valid, error_messages)
        """
        errors = []
        op_type = operator.type.lower()

        # Dispatch to type-specific validation
        if op_type == 'convert':
            errors = self._validate_convert(operator)
        elif op_type == 'project':
            errors = self._validate_project(operator)
        elif op_type == 'sample':
            errors = self._validate_sample(operator)
        elif op_type == 'unnest':
            errors = self._validate_unnest(operator)
        elif op_type == 'split':
            errors = self._validate_split(operator)
        elif op_type == 'gather':
            errors = self._validate_gather(operator)
        elif op_type == 'pythoncode':
            errors = self._validate_pythoncode(operator)
        else:
            # Unknown operator type - allow it (may be custom)
            if self.verbose:
                print(f"  Warning: Unknown operator type '{operator.type}', skipping validation")

        is_valid = len(errors) == 0
        return is_valid, errors

    def update_schema(self, operator: Operator) -> None:
        """
        Update current schema based on operator transformation.

        Args:
            operator: Operator that transforms the schema
        """
        new_schema = compute_schema_transformation(
            operator_type=operator.type,
            operator_properties=operator.properties,
            operator_output=operator.output if operator.output else {},
            current_schema=self.current_schema
        )

        # Auto-fill operator.output if empty (for operators like Convert)
        if not operator.output or operator.output == {}:
            operator.output = new_schema.copy()
            if self.verbose:
                print(f"  Auto-inferred output schema for {operator.type}: {list(new_schema.keys())}")

        self.current_schema = new_schema

        if self.verbose:
            print(f"  Schema after {operator.name}: {list(self.current_schema.keys())}")

    def get_current_schema(self) -> Dict[str, str]:
        """Get current schema."""
        return self.current_schema.copy()

    @classmethod
    def convert_schema_to_abstract(cls, schema: Dict[str, str]) -> Dict[str, str]:
        """
        Convert schema to abstract format.

        For AbstractSchemaTracker, schema is already in abstract format.

        Args:
            schema: Schema dictionary (already in abstract format)

        Returns:
            Schema dictionary in abstract format (unchanged)
        """
        return schema.copy()

    @classmethod
    def convert_schema_from_abstract(cls, schema: Dict[str, str]) -> Dict[str, str]:
        """
        Convert schema from abstract format.

        For AbstractSchemaTracker, schema is already in abstract format.

        Args:
            schema: Schema dictionary in abstract format

        Returns:
            Schema dictionary (unchanged)
        """
        return schema.copy()

    # ===== Type-specific validation methods =====

    def _validate_convert(self, operator: Operator) -> List[str]:
        """Validate Convert operator."""
        errors = []

        # Check required properties
        if 'source_format' not in operator.properties:
            errors.append("Convert operator missing 'source_format' property")
        if 'target_format' not in operator.properties:
            errors.append("Convert operator missing 'target_format' property")

        # No schema validation needed - format conversion is agnostic
        return errors

    def _validate_project(self, operator: Operator) -> List[str]:
        """Validate Project operator."""
        errors = []

        fields = operator.properties.get('fields', [])
        if not fields:
            errors.append("Project operator requires 'fields' property")
            return errors

        # Check that all requested fields exist
        missing_fields = [f for f in fields if f not in self.current_schema]
        if missing_fields:
            errors.append(
                f"Project operator references non-existent fields: {missing_fields}. "
                f"Available fields: {list(self.current_schema.keys())}"
            )

        return errors

    def _validate_sample(self, operator: Operator) -> List[str]:
        """Validate Sample operator."""
        errors = []

        # Check required properties
        if 'method' not in operator.properties:
            errors.append("Sample operator missing 'method' property")
        if 'samples' not in operator.properties:
            errors.append("Sample operator missing 'samples' property")

        # No schema validation needed
        return errors

    def _validate_unnest(self, operator: Operator) -> List[str]:
        """Validate Unnest operator."""
        errors = []

        unnest_key = operator.properties.get('unnest_key')
        if not unnest_key:
            errors.append("Unnest operator missing 'unnest_key' property")
            return errors

        # Check that unnest_key exists in current schema
        if unnest_key not in self.current_schema:
            errors.append(
                f"Unnest operator references non-existent field: '{unnest_key}'. "
                f"Available fields: {list(self.current_schema.keys())}"
            )

        # TODO: Could validate that unnest_key is a List or Dict type
        return errors

    def _validate_split(self, operator: Operator) -> List[str]:
        """Validate Split operator."""
        errors = []

        split_key = operator.properties.get('split_key')
        if not split_key:
            errors.append("Split operator missing 'split_key' property")
            return errors

        # Check that split_key exists in current schema
        if split_key not in self.current_schema:
            errors.append(
                f"Split operator references non-existent field: '{split_key}'. "
                f"Available fields: {list(self.current_schema.keys())}"
            )

        # Check method
        method = operator.properties.get('method')
        if not method:
            errors.append("Split operator missing 'method' property")
        elif method not in ['token_count', 'delimiter']:
            errors.append(f"Split operator has invalid method: '{method}'. Must be 'token_count' or 'delimiter'")

        # TODO: Could validate that split_key is a String type
        return errors

    def _validate_gather(self, operator: Operator) -> List[str]:
        """Validate Gather operator."""
        errors = []

        # Check that required Split metadata fields exist
        required_fields = ['_chunk_text', '_split_id', '_chunk_index']
        missing_fields = [f for f in required_fields if f not in self.current_schema]

        if missing_fields:
            errors.append(
                f"Gather operator requires Split metadata fields: {missing_fields}. "
                f"Available fields: {list(self.current_schema.keys())}. "
                f"Did you forget to add a Split operator first?"
            )

        return errors

    def _validate_pythoncode(self, operator: Operator) -> List[str]:
        """Validate PythonCode operator."""
        errors = []

        # Check that function property exists
        if 'function' not in operator.properties:
            errors.append("PythonCode operator missing 'function' property")

        # TODO: Could extract field references from Python code and validate they exist
        # This would require implementing field_extraction.py

        return errors
