"""
Base System Schema Tracker

This module provides schema tracking using the base system's native type system
instead of maintaining a separate abstract layer type system.

The key insight is that different base systems (DocETL, Lotus, etc.) have different
type systems and schema rules. Instead of trying to unify them at the abstract layer,
we delegate schema tracking to each base system.
"""

from typing import Dict, Any, List, Optional, Tuple, Protocol
from abc import ABC, abstractmethod
import json

from ..abstract.support import BaseSystem
from ..abstract.ops.base import Operator
from ..abstract.convert.docetl import abstract_to_docetl, operator_to_dict


class BaseTypeHandler(Protocol):
    """Protocol for base system type handlers."""

    def apply_operator(self, operator: Dict[str, Any], current_schema: Dict[str, str]) -> Dict[str, str]:
        """
        Apply operator transformation and return new schema.

        Args:
            operator: Base system operator configuration
            current_schema: Current schema in base system format

        Returns:
            Updated schema in base system format
        """
        ...

    def validate_operator(self, operator: Dict[str, Any], current_schema: Dict[str, str]) -> Tuple[bool, List[str]]:
        """
        Validate operator against current schema using base system rules.

        Args:
            operator: Base system operator configuration
            current_schema: Current schema in base system format

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        ...

    def to_abstract_schema(self, base_schema: Dict[str, str]) -> Dict[str, str]:
        """
        Convert base system schema to abstract format for LLM prompts.

        Args:
            base_schema: Schema in base system format

        Returns:
            Schema in abstract format
        """
        ...

    def from_abstract_schema(self, abstract_schema: Dict[str, str]) -> Dict[str, str]:
        """
        Convert abstract schema to base system format.

        Args:
            abstract_schema: Schema in abstract format

        Returns:
            Schema in base system format
        """
        ...


class BaseSystemSchemaTracker:
    """
    Schema tracker that delegates to base system's native type system.

    This replaces the AbstractTypeSystem by using each base system's
    native type tracking and validation mechanisms.
    """

    def __init__(self,
                 base_system: BaseSystem,
                 initial_schema: Dict[str, str],
                 verbose: bool = False):
        """
        Initialize schema tracker.

        Args:
            base_system: The base system to use (DocETL, Lotus, etc.)
            initial_schema: Initial dataset schema in abstract format
            verbose: Enable verbose logging
        """
        self.base_system = base_system
        self.verbose = verbose

        # Get appropriate type handler
        self.type_handler = self._get_type_handler(base_system)

        # Convert initial schema to base system format
        self.base_schema = self.type_handler.from_abstract_schema(initial_schema)

        # Keep abstract version for LLM prompts
        self.abstract_schema = initial_schema.copy()

        # Track schema history for debugging
        self.schema_history = []

    def _get_type_handler(self, base_system: BaseSystem) -> BaseTypeHandler:
        """
        Get the appropriate type handler for the base system.

        Args:
            base_system: The base system enum

        Returns:
            Type handler instance

        Raises:
            ValueError: If base system is not supported
        """
        if base_system == BaseSystem.DOCETL:
            from .type_handlers.docetl_handler import DocETLTypeHandler
            return DocETLTypeHandler(verbose=self.verbose)
        # Future: Add handlers for other systems
        # elif base_system == BaseSystem.LOTUS:
        #     from .type_handlers.lotus_handler import LotusTypeHandler
        #     return LotusTypeHandler(verbose=self.verbose)
        else:
            raise ValueError(f"Unsupported base system: {base_system.value}")

    def get_current_schema(self) -> Dict[str, str]:
        """
        Get current schema in abstract format (for LLM prompts).

        Returns:
            Schema dictionary in abstract format
        """
        return self.abstract_schema.copy()

    def get_base_schema(self) -> Dict[str, str]:
        """
        Get current schema in base system format.

        Returns:
            Schema dictionary in base system format
        """
        return self.base_schema.copy()

    def apply_operator(self, abstract_operator: Operator) -> Tuple[bool, List[str]]:
        """
        Apply operator transformation to schema.

        This method:
        1. Converts abstract operator to base system format
        2. Validates against base system rules
        3. Updates schema using base system tracking
        4. Converts updated schema back to abstract format

        Args:
            abstract_operator: Abstract operator to apply

        Returns:
            Tuple of (success, list_of_errors)
        """
        # Save current state for potential rollback
        self.schema_history.append({
            'base': self.base_schema.copy(),
            'abstract': self.abstract_schema.copy()
        })

        try:
            # Convert abstract operator to base system format
            base_operator = abstract_to_docetl(abstract_operator)

            if self.verbose:
                print(f"Applying operator: {abstract_operator.name} ({abstract_operator.type})")

            # Validate operator against current schema
            is_valid, errors = self.type_handler.validate_operator(
                base_operator,
                self.base_schema
            )

            if not is_valid:
                if self.verbose:
                    print(f"Validation errors: {errors}")
                return False, errors

            # Apply operator to get new schema
            new_base_schema = self.type_handler.apply_operator(
                base_operator,
                self.base_schema
            )

            # Update schemas
            self.base_schema = new_base_schema
            self.abstract_schema = self.type_handler.to_abstract_schema(new_base_schema)

            if self.verbose:
                print(f"Schema updated. New fields: {list(self.abstract_schema.keys())}")

            return True, []

        except Exception as e:
            # Rollback on error
            if self.schema_history:
                prev_state = self.schema_history.pop()
                self.base_schema = prev_state['base']
                self.abstract_schema = prev_state['abstract']

            error_msg = f"Failed to apply operator: {str(e)}"
            if self.verbose:
                print(f"Error: {error_msg}")

            return False, [error_msg]

    def validate_operator_input(self, abstract_operator: Operator) -> Tuple[bool, List[str]]:
        """
        Validate that operator has required input fields.

        Args:
            abstract_operator: Operator to validate

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        base_operator = abstract_to_docetl(abstract_operator)
        return self.type_handler.validate_operator(base_operator, self.base_schema)

    def rollback(self) -> None:
        """
        Rollback to previous schema state.

        Raises:
            ValueError: If no history to rollback to
        """
        if not self.schema_history:
            raise ValueError("No schema history to rollback to")

        prev_state = self.schema_history.pop()
        self.base_schema = prev_state['base']
        self.abstract_schema = prev_state['abstract']

        if self.verbose:
            print(f"Rolled back schema. Current fields: {list(self.abstract_schema.keys())}")

    def get_schema_history(self) -> List[Dict[str, Dict[str, str]]]:
        """
        Get the full history of schema transformations.

        Returns:
            List of schema states (oldest to newest)
        """
        return self.schema_history.copy()


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

    return '\n'.join(lines)


def infer_schema_from_samples(samples: List[Dict[str, Any]]) -> Dict[str, str]:
    """
    Infer abstract schema from a list of sample records.

    This is still needed for initial schema inference from datasets.

    Args:
        samples: List of sample data records

    Returns:
        Inferred schema dictionary in abstract format
    """
    if not samples:
        return {}

    # Import the existing inference function
    from .type_utils_abstract import infer_schema_from_samples as original_infer
    return original_infer(samples)