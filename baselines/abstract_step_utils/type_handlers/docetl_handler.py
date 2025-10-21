"""
DocETL Type Handler

Implements schema tracking and validation using DocETL's native type system.
"""

from typing import Dict, Any, List, Tuple, Optional, Set
import re

from ...abstract.convert.docetl.schema import DocETLTypeMapper


class DocETLTypeHandler:
    """
    Type handler for DocETL system.

    This handler understands DocETL's specific type system and operator behaviors,
    providing accurate schema tracking and validation.
    """

    def __init__(self, verbose: bool = False):
        """
        Initialize DocETL type handler.

        Args:
            verbose: Enable verbose logging
        """
        self.verbose = verbose

    def apply_operator(self, operator: Dict[str, Any], current_schema: Dict[str, str]) -> Dict[str, str]:
        """
        Apply DocETL operator transformation and return new schema.

        Args:
            operator: DocETL operator configuration
            current_schema: Current schema in DocETL format (field -> type string)

        Returns:
            Updated schema in DocETL format
        """
        op_type = operator.get('type', '')
        new_schema = current_schema.copy()

        # Handle different operator types according to DocETL semantics
        if op_type in ['map', 'code_map']:
            # Map adds new fields from output schema
            output_schema = operator.get('output', {}).get('schema', {})
            new_schema.update(output_schema)

        elif op_type in ['filter', 'code_filter']:
            # Filter doesn't change schema
            pass

        elif op_type in ['reduce', 'code_reduce']:
            # Reduce changes schema completely to output schema
            # It typically includes the reduce_key and aggregated fields
            output_schema = operator.get('output', {}).get('schema', {})
            if output_schema:
                new_schema = output_schema.copy()
                # Ensure reduce_key is in output
                reduce_key = operator.get('reduce_key')
                if reduce_key and reduce_key not in new_schema:
                    # Preserve reduce_key type from input
                    if reduce_key in current_schema:
                        new_schema[reduce_key] = current_schema[reduce_key]
                    else:
                        new_schema[reduce_key] = 'str'  # Default

        elif op_type == 'resolve':
            # Resolve typically outputs merged records with same or modified schema
            output_schema = operator.get('output', {}).get('schema', {})
            if output_schema:
                new_schema.update(output_schema)

        elif op_type == 'extract':
            # Extract adds extracted fields to existing schema
            output_schema = operator.get('output', {}).get('schema', {})
            new_schema.update(output_schema)

        elif op_type == 'unnest':
            # Unnest flattens a field - removes the nested field and adds flattened ones
            unnest_key = operator.get('unnest_key')
            if unnest_key and unnest_key in new_schema:
                del new_schema[unnest_key]
            # Add any output schema fields
            output_schema = operator.get('output', {}).get('schema', {})
            new_schema.update(output_schema)

        elif op_type == 'split':
            # Split adds _split_id and _split_index fields
            new_schema['_split_id'] = 'str'
            new_schema['_split_index'] = 'int'

        elif op_type == 'gather':
            # Gather might add an output_key field
            output_key = operator.get('output_key')
            if output_key:
                new_schema[output_key] = 'list[str]'

        elif op_type == 'cluster':
            # Cluster adds cluster identifier
            output_key = operator.get('output_key')
            if output_key:
                new_schema[output_key] = 'str'

        elif op_type == 'rank':
            # Rank adds rank field
            new_schema['rank'] = 'int'

        elif op_type == 'join':
            # Join merges schemas from two datasets
            # This is simplified - actual join would need second dataset schema
            output_schema = operator.get('output', {}).get('schema', {})
            new_schema.update(output_schema)

        elif op_type == 'sample':
            # Sample doesn't change schema
            pass

        else:
            # Unknown operator - try to use output schema if available
            output_schema = operator.get('output', {}).get('schema', {})
            if output_schema:
                new_schema.update(output_schema)

        return new_schema

    def validate_operator(self, operator: Dict[str, Any], current_schema: Dict[str, str]) -> Tuple[bool, List[str]]:
        """
        Validate DocETL operator against current schema.

        Args:
            operator: DocETL operator configuration
            current_schema: Current schema in DocETL format

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        op_type = operator.get('type', '')
        op_name = operator.get('name', 'unnamed')

        # Extract required fields based on operator type and configuration
        required_fields = self._get_required_fields(operator)

        # Check if all required fields exist in current schema
        for field in required_fields:
            if field not in current_schema:
                errors.append(
                    f"Operator '{op_name}' ({op_type}) requires field '{field}' which is not available. "
                    f"Available fields: {list(current_schema.keys())}"
                )

        # Operator-specific validation
        if op_type in ['reduce', 'code_reduce']:
            reduce_key = operator.get('reduce_key')
            if not reduce_key:
                errors.append(f"Reduce operator '{op_name}' missing required 'reduce_key'")
            elif isinstance(reduce_key, list):
                for key in reduce_key:
                    if key not in current_schema:
                        errors.append(f"Reduce key '{key}' not found in current schema")
            elif reduce_key not in current_schema:
                errors.append(f"Reduce key '{reduce_key}' not found in current schema")

        elif op_type == 'unnest':
            unnest_key = operator.get('unnest_key')
            if not unnest_key:
                errors.append(f"Unnest operator '{op_name}' missing required 'unnest_key'")
            elif unnest_key not in current_schema:
                errors.append(f"Unnest key '{unnest_key}' not found in current schema")

        elif op_type == 'split':
            split_key = operator.get('split_key')
            if not split_key:
                errors.append(f"Split operator '{op_name}' missing required 'split_key'")
            elif split_key not in current_schema:
                errors.append(f"Split key '{split_key}' not found in current schema")

        elif op_type == 'gather':
            gather_key = operator.get('gather_key')
            if not gather_key:
                errors.append(f"Gather operator '{op_name}' missing required 'gather_key'")

        # Check output schema validity
        output_schema = operator.get('output', {}).get('schema', {})
        if output_schema:
            for field, type_str in output_schema.items():
                if not self._is_valid_docetl_type(type_str):
                    errors.append(f"Invalid type '{type_str}' for output field '{field}' in operator '{op_name}'")

        return len(errors) == 0, errors

    def to_abstract_schema(self, base_schema: Dict[str, str]) -> Dict[str, str]:
        """
        Convert DocETL schema to abstract format for LLM prompts.

        Args:
            base_schema: Schema in DocETL format (field -> type string)

        Returns:
            Schema in abstract format
        """
        abstract_schema = {}
        for field, docetl_type in base_schema.items():
            abstract_schema[field] = DocETLTypeMapper.to_abstract_type_string(docetl_type)
        return abstract_schema

    def from_abstract_schema(self, abstract_schema: Dict[str, str]) -> Dict[str, str]:
        """
        Convert abstract schema to DocETL format.

        Args:
            abstract_schema: Schema in abstract format

        Returns:
            Schema in DocETL format
        """
        docetl_schema = {}
        for field, abstract_type in abstract_schema.items():
            docetl_schema[field] = DocETLTypeMapper.from_abstract_type_string(abstract_type)
        return docetl_schema

    def _get_required_fields(self, operator: Dict[str, Any]) -> Set[str]:
        """
        Extract required input fields from operator configuration.

        Args:
            operator: DocETL operator configuration

        Returns:
            Set of required field names
        """
        fields = set()
        op_type = operator.get('type', '')

        # Extract fields from prompts (Jinja2 templates)
        if 'prompt' in operator:
            fields.update(self._extract_fields_from_jinja2(operator['prompt']))

        if 'comparison_prompt' in operator:
            fields.update(self._extract_fields_from_jinja2(operator['comparison_prompt']))

        if 'resolution_prompt' in operator:
            fields.update(self._extract_fields_from_jinja2(operator['resolution_prompt']))

        # Extract fields from Python code
        if 'code' in operator:
            fields.update(self._extract_fields_from_python_code(operator['code']))

        # Add operator-specific required fields
        if op_type in ['reduce', 'code_reduce']:
            reduce_key = operator.get('reduce_key')
            if reduce_key:
                if isinstance(reduce_key, list):
                    fields.update(reduce_key)
                else:
                    fields.add(reduce_key)

        elif op_type == 'unnest':
            unnest_key = operator.get('unnest_key')
            if unnest_key:
                fields.add(unnest_key)

        elif op_type == 'split':
            split_key = operator.get('split_key')
            if split_key:
                fields.add(split_key)

        return fields

    def _extract_fields_from_jinja2(self, template: str) -> Set[str]:
        """Extract field names from Jinja2 template."""
        if not template:
            return set()

        fields = set()

        # Patterns for field extraction
        patterns = [
            r'\{\{\s*input\.(\w+)',  # {{ input.field }}
            r'\{\{\s*input\["([^"]+)"\]',  # {{ input["field"] }}
            r"\{\{\s*input\['([^']+)'\]",  # {{ input['field'] }}
            r'\{\{\s*inputs\[\d+\]\.(\w+)',  # {{ inputs[0].field }}
        ]

        for pattern in patterns:
            fields.update(re.findall(pattern, template))

        return fields

    def _extract_fields_from_python_code(self, code: str) -> Set[str]:
        """Extract field names from Python code."""
        if not code:
            return set()

        fields = set()

        # Patterns for field extraction from Python code
        patterns = [
            r'doc\[[\'"]([\w_]+)[\'"]\]',  # doc["field"] or doc['field']
            r'doc\.get\([\'"]([^\'",]+)[\'"]',  # doc.get("field")
            r'doc\.(\w+)',  # doc.field
            r'input\[[\'"]([\w_]+)[\'"]\]',  # input["field"] or input['field']
            r'input\.(\w+)',  # input.field
        ]

        for pattern in patterns:
            fields.update(re.findall(pattern, code))

        # Remove common method names that aren't fields
        fields.discard('get')
        fields.discard('keys')
        fields.discard('values')
        fields.discard('items')

        return fields

    def _is_valid_docetl_type(self, type_str: str) -> bool:
        """
        Check if a type string is valid in DocETL.

        Args:
            type_str: Type string to validate

        Returns:
            True if valid DocETL type
        """
        if not type_str:
            return False

        # Basic DocETL types
        basic_types = {'str', 'int', 'float', 'bool', 'dict', 'list'}
        if type_str in basic_types:
            return True

        # List types: list[type]
        if type_str.startswith('list[') and type_str.endswith(']'):
            inner_type = type_str[5:-1]
            return inner_type in basic_types or inner_type == 'dict'

        # Complex dict types are represented as 'dict' in DocETL
        return False