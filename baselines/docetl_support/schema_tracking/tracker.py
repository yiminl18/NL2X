"""
DocETL Schema Tracker

DocETL-specific schema tracking implementation for pipeline validation and inference.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from .schema_inference import compute_schema_transformation, _infer_docetl_output_schema
from .complex_types import find_field_in_complex_types
from .field_extraction import extract_fields_from_jinja2, extract_fields_from_python_code, is_direct_list_access

class DocETLSchemaTracker:
    """
    DocETL-specific schema tracker for pipeline validation and inference.

    Tracks schema changes as operators are applied, validates operator input requirements,
    and maintains history for rollback.

    Usage:
        tracker = DocETLSchemaTracker.from_dataset('data.json')
        is_valid, errors = tracker.validate_operator(op_config)
        if is_valid:
            tracker.apply_operator(op_config)
        current_schema = tracker.get_current_schema()
    """

    # Valid operator types
    VALID_OPERATORS = {
        'map', 'filter', 'reduce', 'resolve', 'order', 'extract',
        'cluster', 'split', 'gather', 'unnest', 'sample', 'topk',
        'code_map', 'code_filter', 'code_reduce', 'equijoin', 'scan',
        'parallel_map', 'link_resolve', 'add_uuid'
    }

    # Universal required fields for all operators
    UNIVERSAL_REQUIRED_FIELDS = {'name', 'type'}

    # Operator-specific required configuration fields
    OPERATOR_REQUIREMENTS = {
        'map': set(),
        'filter': {'prompt', 'output'},
        'reduce': {'reduce_key', 'prompt', 'output'},
        'resolve': {'comparison_prompt', 'resolution_prompt', 'output'},
        'order': {'prompt', 'input_keys', 'direction'},
        'extract': {'prompt', 'document_keys'},
        'cluster': {'embedding_keys', 'summary_prompt', 'summary_schema'},
        'parallel_map': {'prompts', 'output'},
        'equijoin': {'comparison_prompt'},
        'split': {'split_key', 'method', 'method_kwargs'},
        'gather': {'content_key', 'doc_id_key', 'order_key'},
        'unnest': {'unnest_key'},
        'sample': {'method'},
        'topk': {'method', 'k', 'keys', 'query'},
        'code_map': {'code'},
        'code_filter': {'code'},
        'code_reduce': {'code'},
        'scan': {'dataset_name'}
    }

    def __init__(self, initial_schema: Dict[str, str], verbose: bool = False):
        """
        Initialize DocETL schema tracker.

        Args:
            initial_schema: Initial dataset schema in abstract format
            verbose: Enable verbose logging
        """
        self.verbose = verbose
        self.current_schema = initial_schema.copy()
        self.history: List[Dict[str, str]] = []

    @classmethod
    def from_dataset(cls, dataset_path: str, verbose: bool = False) -> 'DocETLSchemaTracker':
        """
        Create tracker from dataset file by inferring initial schema.

        Args:
            dataset_path: Path to dataset JSON file
            verbose: Enable verbose logging

        Returns:
            DocETLSchemaTracker instance

        Raises:
            FileNotFoundError: If dataset file not found
            ValueError: If schema inference fails
        """
        dataset_path = Path(dataset_path)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

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
        Recursively infer DocETL type from value.

        Returns DocETL type string (lowercase format):
        - Basic: 'str', 'int', 'float', 'bool'
        - List: 'list[str]' or 'list[{name: str, age: int}]'
        - Dict: '{name: str, age: int}'

        Args:
            value: Value to infer type from

        Returns:
            DocETL type string
        """
        # None
        if value is None:
            return 'str'

        # Basic types (bool must check before int since bool is subclass of int)
        if isinstance(value, bool):
            return 'bool'
        elif isinstance(value, int):
            return 'int'
        elif isinstance(value, float):
            return 'float'
        elif isinstance(value, str):
            return 'str'

        # List type - recursively infer element type
        elif isinstance(value, list):
            if len(value) == 0:
                return 'list'  # Can't infer element type from empty list

            # Sample multiple elements to determine if all have same type
            sample_size = min(10, len(value))  # Sample up to 10 elements
            element_types = set()

            for i in range(sample_size):
                elem_type = DocETLSchemaTracker._infer_type_from_value(value[i])
                element_types.add(elem_type)

            if len(element_types) == 1:
                # All elements have same type
                elem_type = element_types.pop()
                return f'list[{elem_type}]'
            else:
                # Mixed types - return generic list
                return 'list'

        # Dict type - recursively infer all field types
        elif isinstance(value, dict):
            if len(value) == 0:
                return 'dict'  # Empty dict

            # Recursively infer type for each field
            field_types = {}
            for field_name, field_value in value.items():
                field_types[field_name] = DocETLSchemaTracker._infer_type_from_value(field_value)

            # Format as {field1: type1, field2: type2, ...}
            # Sort fields for consistency
            fields_str = ', '.join(f'{k}: {v}' for k, v in sorted(field_types.items()))
            return f'{{{fields_str}}}'

        else:
            # Unknown type - fallback to string
            return 'str'

    def get_current_schema(self) -> Dict[str, str]:
        """
        Get current available schema.

        Returns:
            Schema dictionary in DocETL format (field -> type)
        """
        return self.current_schema.copy()

    @classmethod
    def convert_schema_to_abstract(cls, schema: Dict[str, str]) -> Dict[str, str]:
        """
        Convert DocETL schema to abstract layer format.

        Args:
            schema: Schema dictionary in DocETL format (e.g., {'file': 'str', 'medications': 'list[{...}]'})

        Returns:
            Schema dictionary in abstract layer format (e.g., {'file': 'String', 'medications': 'List[Dict{...}]'})
        """
        from .type_mapper import DocETLTypeMapper

        abstract_schema = {}
        for field_name, docetl_type in schema.items():
            abstract_schema[field_name] = DocETLTypeMapper.to_abstract_type_string(docetl_type)

        return abstract_schema

    @classmethod
    def convert_schema_from_abstract(cls, schema: Dict[str, str]) -> Dict[str, str]:
        """
        Convert abstract layer schema to DocETL format.

        Args:
            schema: Schema dictionary in abstract layer format (e.g., {'file': 'String', 'medications': 'List[Dict{...}]'})

        Returns:
            Schema dictionary in DocETL format (e.g., {'file': 'str', 'medications': 'list[{...}]'})
        """
        from .type_mapper import DocETLTypeMapper

        docetl_schema = {}
        for field_name, abstract_type in schema.items():
            docetl_schema[field_name] = DocETLTypeMapper.from_abstract_type_string(abstract_type)

        return docetl_schema

    def validate_operator(self, operator_config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate operator against current schema without applying it.

        1. YAML validity checking
        2. Operator validity checking
        3. Complex type extraction and checking
        4. Schema tracking and checking
        5. System-level schema tracking (config vs prompt fields)
        6. Field extraction from prompts
        7. Complex type field access validation

        Args:
            operator_config: Operator configuration dictionary

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        op_name = operator_config.get('name', 'unnamed')
        op_type = operator_config.get('type', 'unknown')

        try:
            # Feature 1: YAML validity - basic structure checking
            if 'name' not in operator_config:
                errors.append("Operator missing required field 'name'")
            if 'type' not in operator_config:
                errors.append("Operator missing required field 'type'")

            # Feature 2: Operator validity checking
            if op_type not in self.VALID_OPERATORS:
                errors.append(
                    f"Invalid operator type '{op_type}'. Valid types: {sorted(self.VALID_OPERATORS)}"
                )
                return False, errors

            # Feature 2.5: Check operator-specific required configuration fields
            required_config_fields = self.OPERATOR_REQUIREMENTS.get(op_type, set())
            missing_config_fields = required_config_fields - set(operator_config.keys())
            for field in missing_config_fields:
                errors.append(
                    f"Operator '{op_name}' ({op_type}) missing required configuration field: '{field}'"
                )

            # Feature 5: System-level schema tracking - extract config and prompt fields separately
            config_fields, prompt_fields = self._extract_operator_fields(operator_config)

            # Validate configuration fields (must exist directly in schema root level)
            for field in config_fields:
                if field not in self.current_schema:
                    field_source = self._get_field_source(operator_config, field)
                    errors.append(
                        f"Operator '{op_name}' ({op_type}) requires field '{field}' "
                        f"(specified in {field_source}) but it's not in current schema. "
                        f"Available fields: {sorted(self.current_schema.keys())}"
                    )

            # Features 6 & 7: Prompt field validation + complex type access validation
            for field in prompt_fields:
                if field not in self.current_schema:
                    # Check if field exists within complex types
                    complex_location = find_field_in_complex_types(field, self.current_schema)

                    if complex_location:
                        # Validate access method is appropriate
                        parent_type = complex_location['parent_type']

                        # Dict type fields cannot be accessed directly with {{ input.field }}
                        if parent_type.startswith('{') and parent_type.endswith('}'):
                            errors.append(
                                f"Field '{field}' is nested within dict '{complex_location['parent_field']}'. "
                                f"Cannot access with '{{{{ input.{field} }}}}'. Use one of:\n  "
                                + "\n  ".join(complex_location['access_suggestions'])
                            )
                        # List[Dict] type needs special access
                        elif parent_type.startswith('list[{'):
                            errors.append(
                                f"Field '{field}' is nested within list '{complex_location['parent_field']}'. "
                                f"Cannot access directly. Use one of:\n  "
                                + "\n  ".join(complex_location['access_suggestions'])
                            )
                    else:
                        # Field doesn't exist at all
                        errors.append(
                            f"Operator '{op_name}' references field '{field}' in prompt, "
                            f"but it's not available in current schema. "
                            f"Available fields: {sorted(self.current_schema.keys())}"
                        )
                else:
                    # Field exists - check if it's a simple list type
                    field_type = self.current_schema[field]
                    # Simple list types like list[str], list[int] cannot be accessed directly
                    if field_type.startswith('list[') and not field_type.startswith('list[{'):
                        # Check all prompts for direct access
                        prompts_to_check = []
                        if 'prompt' in operator_config:
                            prompts_to_check.append(operator_config['prompt'])
                        if 'comparison_prompt' in operator_config:
                            prompts_to_check.append(operator_config['comparison_prompt'])
                        if 'resolution_prompt' in operator_config:
                            prompts_to_check.append(operator_config['resolution_prompt'])
                        if 'summary_prompt' in operator_config:
                            prompts_to_check.append(operator_config['summary_prompt'])

                        # Check if any prompt has direct access
                        has_direct_access = any(is_direct_list_access(prompt, field) for prompt in prompts_to_check)

                        if has_direct_access:
                            errors.append(
                                f"Field '{field}' is of type '{field_type}'. "
                                f"Cannot access directly with '{{{{ input.{field} }}}}' or '{{{{ input1.{field} }}}}'. Use one of:\n"
                                f"  - Access via index: {{{{ input.{field}[0] }}}}\n"
                                f"  - Use a for loop: {{% for item in input.{field} %}}{{{{ item }}}}{{% endfor %}}\n"
                                f"  - Use an Unnest operator to expand the list first"
                            )

            # Additional operator-specific constraint validation
            self._validate_operator_specific_constraints(operator_config, errors)

        except Exception as e:
            errors.append(f"Validation error for operator '{op_name}': {str(e)}")

        is_valid = len(errors) == 0

        if self.verbose:
            if is_valid:
                print(f"✓ Operator '{op_name}' ({op_type}) validated successfully")
            else:
                print(f"✗ Operator '{op_name}' ({op_type}) validation failed:")
                for error in errors:
                    print(f"    {error}")

        return is_valid, errors

    def apply_operator(self, operator_config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate and apply operator transformation to schema.

        Only updates schema if validation passes (used for runtime schema tracking).

        Args:
            operator_config: Operator configuration dictionary

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        # Step 1: Validate
        is_valid, errors = self.validate_operator(operator_config)

        if not is_valid:
            return False, errors

        # Step 2: Update (only if validation passed)
        try:
            self.update_schema(operator_config)
            return True, []
        except Exception as e:
            op_name = operator_config.get('name', 'unnamed')
            error_msg = f"Failed to apply operator '{op_name}': {str(e)}"
            if self.verbose:
                print(f"✗ {error_msg}")
            return False, [error_msg]

    def update_schema(self, operator_config: Dict[str, Any]) -> None:
        """
        Update schema without validation (for pipeline validation scenarios).

        Used when you want to update schema even if validation fails,
        typically in pipeline validation to collect all errors.

        Args:
            operator_config: Operator configuration dictionary

        Raises:
            Exception: If schema transformation fails
        """
        op_name = operator_config.get('name', 'unnamed')

        # Save old schema before updating
        old_schema = self.current_schema.copy()

        # Save current state to history
        self.history.append(old_schema)

        try:
            # Compute new schema
            new_schema = compute_schema_transformation(operator_config, self.current_schema)

            # Update current schema
            self.current_schema = new_schema

            if self.verbose:
                # Format schema as readable string
                old_schema_str = ', '.join(f"{k}: {v}" for k, v in sorted(old_schema.items()))
                new_schema_str = ', '.join(f"{k}: {v}" for k, v in sorted(new_schema.items()))
                print(f"  Before: {{{old_schema_str}}}")
                print(f"  After:  {{{new_schema_str}}}")

        except Exception as e:
            # Rollback on error
            if self.history:
                self.current_schema = self.history.pop()
            raise Exception(f"Failed to update schema for operator '{op_name}': {str(e)}")

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
        errors = []
        warnings = []

        for op_config in operators:
            op_name = op_config.get('name', 'unnamed')

            # Step 1: Validate operator
            is_valid, op_errors = self.validate_operator(op_config)
            if not is_valid:
                errors.extend(op_errors)

            # Step 2: Update schema regardless of validation result (if continue_on_error)
            if continue_on_error:
                try:
                    self.update_schema(op_config)
                except Exception as e:
                    errors.append(f"Failed to update schema for operator '{op_name}': {str(e)}")
            elif is_valid:
                # Only update if validation passed
                try:
                    self.update_schema(op_config)
                except Exception as e:
                    errors.append(f"Failed to update schema for operator '{op_name}': {str(e)}")

        passed = len(errors) == 0
        return passed, errors, warnings

    def get_history(self) -> List[Dict[str, str]]:
        """
        Get full history of schema transformations.

        Returns:
            List of schema states (oldest to newest)
        """
        return self.history.copy()

    def _extract_operator_fields(self, operator_config: Dict[str, Any]) -> Tuple[set, set]:
        """
        Extract configuration fields and prompt fields from operator.

        Args:
            operator_config: Operator configuration

        Returns:
            Tuple of (config_fields, prompt_fields) where:
            - config_fields: Fields referenced in operator configuration (e.g., document_keys)
            - prompt_fields: Fields referenced in prompts or code
        """
        op_type = operator_config.get('type', '')
        config_fields = set()
        prompt_fields = set()

        # Extract based on operator type
        if op_type == 'extract':
            # document_keys are configuration fields
            document_keys = operator_config.get('document_keys', [])
            if isinstance(document_keys, list):
                config_fields.update(document_keys)
            # Fields from prompt
            prompt = operator_config.get('prompt', '')
            if prompt:
                prompt_fields.update(extract_fields_from_jinja2(prompt))

        elif op_type == 'cluster':
            # embedding_keys are configuration fields
            embedding_keys = operator_config.get('embedding_keys', [])
            if isinstance(embedding_keys, list):
                config_fields.update(embedding_keys)
            # Fields from summary_prompt
            summary_prompt = operator_config.get('summary_prompt', '')
            if summary_prompt:
                prompt_fields.update(extract_fields_from_jinja2(summary_prompt))

        elif op_type in ['reduce', 'code_reduce']:
            # reduce_key is configuration field
            reduce_key = operator_config.get('reduce_key')
            if isinstance(reduce_key, list):
                config_fields.update(reduce_key)
            elif reduce_key:
                config_fields.add(reduce_key)
            # Fields from prompt or code
            if op_type == 'reduce':
                prompt = operator_config.get('prompt', '')
                if prompt:
                    prompt_fields.update(extract_fields_from_jinja2(prompt))
            else:
                code = operator_config.get('code', '')
                if code:
                    prompt_fields.update(extract_fields_from_python_code(code))

        elif op_type == 'resolve':
            blocking_keys = operator_config.get('blocking_keys', [])
            if isinstance(blocking_keys, list):
                config_fields.update(blocking_keys)
            comparison_prompt = operator_config.get('comparison_prompt', '')
            resolution_prompt = operator_config.get('resolution_prompt', '')
            if comparison_prompt:
                prompt_fields.update(extract_fields_from_jinja2(comparison_prompt))
            if resolution_prompt:
                prompt_fields.update(extract_fields_from_jinja2(resolution_prompt))

        elif op_type == 'gather':
            # Gather keys are configuration fields
            for key in ['content_key', 'doc_id_key', 'order_key']:
                value = operator_config.get(key)
                if value:
                    config_fields.add(value)
            # Optional doc_header_key
            doc_header_key = operator_config.get('doc_header_key')
            if doc_header_key:
                config_fields.add(doc_header_key)

        elif op_type == 'unnest':
            # unnest_key is configuration field
            unnest_key = operator_config.get('unnest_key')
            if unnest_key:
                config_fields.add(unnest_key)

        elif op_type == 'split':
            # split_key is configuration field
            split_key = operator_config.get('split_key')
            if split_key:
                config_fields.add(split_key)

        elif op_type == 'rank':
            # input_keys are configuration fields
            input_keys = operator_config.get('input_keys', [])
            if isinstance(input_keys, list):
                config_fields.update(input_keys)
            # Fields from prompt if present
            prompt = operator_config.get('prompt', '')
            if prompt:
                prompt_fields.update(extract_fields_from_jinja2(prompt))

        elif op_type == 'topk':
            # keys are configuration fields
            keys = operator_config.get('keys', [])
            if isinstance(keys, list):
                config_fields.update(keys)

        elif op_type == 'sample':
            # stratify_key is configuration field
            stratify_key = operator_config.get('stratify_key')
            if stratify_key:
                config_fields.add(stratify_key)

        elif op_type == 'equijoin':
            # left_key and right_key are configuration fields
            left_key = operator_config.get('left_key')
            right_key = operator_config.get('right_key')
            if left_key:
                config_fields.add(left_key)
            if right_key:
                config_fields.add(right_key)
            # comparison_prompt has prompt fields
            comparison_prompt = operator_config.get('comparison_prompt', '')
            if comparison_prompt:
                prompt_fields.update(extract_fields_from_jinja2(comparison_prompt))

        elif op_type == 'order':
            # input_keys are configuration fields
            input_keys = operator_config.get('input_keys', [])
            if isinstance(input_keys, list):
                config_fields.update(input_keys)
            # prompt has prompt fields
            prompt = operator_config.get('prompt', '')
            if prompt:
                prompt_fields.update(extract_fields_from_jinja2(prompt))

        elif op_type in ['map', 'filter']:
            # These operators only have prompt fields
            prompt = operator_config.get('prompt', '')
            if prompt:
                prompt_fields.update(extract_fields_from_jinja2(prompt))

        elif op_type in ['code_map', 'code_filter']:
            # Code operators have fields from Python code
            code = operator_config.get('code', '')
            if code:
                prompt_fields.update(extract_fields_from_python_code(code))

        elif op_type == 'scan':
            # Scan doesn't have field requirements, just dataset_name
            pass

        # Handle parallel_map specially
        elif op_type == 'parallel_map':
            prompts = operator_config.get('prompts', [])
            for prompt_item in prompts:
                if isinstance(prompt_item, dict):
                    prompt_text = prompt_item.get('prompt', '')
                elif isinstance(prompt_item, str):
                    prompt_text = prompt_item
                else:
                    continue
                if prompt_text:
                    prompt_fields.update(extract_fields_from_jinja2(prompt_text))

        return config_fields, prompt_fields

    def _get_field_source(self, operator_config: Dict[str, Any], field: str) -> str:
        """
        Get the source location of a field in operator configuration.

        Args:
            operator_config: Operator configuration
            field: Field name to locate

        Returns:
            String describing where the field is referenced (e.g., 'document_keys')
        """
        op_type = operator_config.get('type', '')

        if op_type == 'extract':
            document_keys = operator_config.get('document_keys', [])
            if field in document_keys:
                return 'document_keys'

        elif op_type == 'cluster':
            embedding_keys = operator_config.get('embedding_keys', [])
            if field in embedding_keys:
                return 'embedding_keys'

        elif op_type in ['reduce', 'code_reduce']:
            reduce_key = operator_config.get('reduce_key')
            if isinstance(reduce_key, list) and field in reduce_key:
                return 'reduce_key'
            elif field == reduce_key:
                return 'reduce_key'

        elif op_type == 'gather':
            if field == operator_config.get('content_key'):
                return 'content_key'
            elif field == operator_config.get('doc_id_key'):
                return 'doc_id_key'
            elif field == operator_config.get('order_key'):
                return 'order_key'
            elif field == operator_config.get('doc_header_key'):
                return 'doc_header_key'

        elif op_type == 'unnest':
            if field == operator_config.get('unnest_key'):
                return 'unnest_key'

        elif op_type == 'split':
            if field == operator_config.get('split_key'):
                return 'split_key'

        elif op_type == 'rank':
            input_keys = operator_config.get('input_keys', [])
            if field in input_keys:
                return 'input_keys'

        elif op_type == 'order':
            input_keys = operator_config.get('input_keys', [])
            if field in input_keys:
                return 'input_keys'

        elif op_type == 'topk':
            keys = operator_config.get('keys', [])
            if field in keys:
                return 'keys'

        elif op_type == 'sample':
            if field == operator_config.get('stratify_key'):
                return 'stratify_key'

        elif op_type == 'equijoin':
            if field == operator_config.get('left_key'):
                return 'left_key'
            elif field == operator_config.get('right_key'):
                return 'right_key'

        return 'configuration'

    def _validate_operator_specific_constraints(
        self, operator_config: Dict[str, Any], errors: List[str]
    ) -> None:
        """
        Validate operator-specific constraints beyond field existence.

        Args:
            operator_config: Operator configuration
            errors: List to append errors to
        """
        op_type = operator_config.get('type', '')
        op_name = operator_config.get('name', 'unnamed')

        if op_type in ['reduce', 'code_reduce']:
            # reduce_key cannot be a list type field
            # Handle reduce_key as both string and list
            reduce_key = operator_config.get('reduce_key')
            reduce_keys = [reduce_key] if isinstance(reduce_key, str) else (reduce_key if reduce_key else [])

            for key in reduce_keys:
                if key in self.current_schema:
                    field_type = self.current_schema[key]
                    if 'list' in field_type.lower():
                        errors.append(
                            f"Reduce operator '{op_name}' cannot use list field '{key}' "
                            f"as reduce_key (type: {field_type})"
                        )

        elif op_type == 'unnest':
            # unnest_key must be list or dict type
            unnest_key = operator_config.get('unnest_key')
            if unnest_key and unnest_key in self.current_schema:
                field_type = self.current_schema[unnest_key]
                # Check if it's a list or dict type
                is_list = 'list' in field_type.lower()
                is_dict = 'dict' in field_type.lower() or '{' in field_type
                if not (is_list or is_dict):
                    errors.append(
                        f"Unnest operator '{op_name}' requires field '{unnest_key}' "
                        f"to be list or dict type, but got '{field_type}'"
                    )

        elif op_type == 'split':
            # split_key should be a string or text field
            split_key = operator_config.get('split_key')
            if split_key and split_key in self.current_schema:
                field_type = self.current_schema[split_key]
                # Warn if it's not a string type (but don't error)
                if 'str' not in field_type.lower() and 'string' not in field_type.lower():
                    if self.verbose:
                        print(
                            f"  Warning: Split operator '{op_name}' typically expects "
                            f"string field for split_key, but '{split_key}' has type '{field_type}'"
                        )

        elif op_type == 'extract':
            # document_keys fields should ideally be string type
            document_keys = operator_config.get('document_keys', [])
            for doc_key in document_keys:
                if doc_key in self.current_schema:
                    field_type = self.current_schema[doc_key]
                    # Warn if complex type
                    if '{' in field_type or 'list' in field_type.lower():
                        if self.verbose:
                            print(
                                f"  Warning: Extract operator '{op_name}' document_key '{doc_key}' "
                                f"has complex type '{field_type}' - consider unnesting first"
                            )

    def rollback(self, steps: int = 1) -> None:
        """
        Rollback schema to previous state.

        Args:
            steps: Number of steps to rollback

        Raises:
            ValueError: If not enough history to rollback
        """
        if steps > len(self.history):
            raise ValueError(
                f"Cannot rollback {steps} steps, only {len(self.history)} in history"
            )

        for _ in range(steps):
            if self.history:
                self.current_schema = self.history.pop()

        if self.verbose:
            print(f"Rolled back {steps} step(s). Current schema has {len(self.current_schema)} fields")

    def get_input_schema(
        self,
        operator_config: Dict[str, Any],
        field_types: Optional[Dict[str, str]] = None,
        dataset_schema: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Get input schema for a DocETL operator.

        Simple public method to extract input schema from an operator configuration.
        Consolidates logic from _extract_input_schema and _extract_operator_fields.

        Args:
            operator_config: Operator configuration dictionary
            field_types: Optional cumulative field types from pipeline
            dataset_schema: Optional initial dataset schema

        Returns:
            Input schema dictionary with type and fields
        """
        input_schema = {}
        op_type = operator_config.get("type", "")
        required_fields = set()
        extracted_fields = set()

        if field_types is None:
            field_types = {}

        # Process based on operator type
        if op_type in ["map", "filter"]:
            # Extract fields from prompt
            prompt = operator_config.get("prompt", "")
            if prompt:
                extracted_fields.update(extract_fields_from_jinja2(prompt))
            input_schema["type"] = "object"

        elif op_type in ["code_map", "code_filter"]:
            # Extract fields from Python code
            code = operator_config.get("code", "")
            if code:
                extracted_fields.update(extract_fields_from_python_code(code))
            input_schema["type"] = "object"

        elif op_type == "reduce":
            # Reduce requires a reduce_key and fields from prompt
            reduce_key = operator_config.get("reduce_key")
            if reduce_key:
                # reduce_key can be a string or a list
                if isinstance(reduce_key, list):
                    required_fields.update(reduce_key)
                else:
                    required_fields.add(reduce_key)

            prompt = operator_config.get("prompt", "")
            if prompt:
                extracted_fields.update(extract_fields_from_jinja2(prompt))
            input_schema["type"] = "array"

        elif op_type == "code_reduce":
            # Code reduce: reduce_key + fields from code
            reduce_key = operator_config.get("reduce_key")
            if reduce_key:
                if isinstance(reduce_key, list):
                    required_fields.update(reduce_key)
                else:
                    required_fields.add(reduce_key)

            code = operator_config.get("code", "")
            if code:
                extracted_fields.update(extract_fields_from_python_code(code))
            input_schema["type"] = "array"

        elif op_type == "resolve":
            # Resolve uses comparison_prompt and resolution_prompt
            comparison_prompt = operator_config.get("comparison_prompt", "")
            resolution_prompt = operator_config.get("resolution_prompt", "")

            # Extract from both prompts
            if comparison_prompt:
                extracted_fields.update(extract_fields_from_jinja2(comparison_prompt))
            if resolution_prompt:
                extracted_fields.update(extract_fields_from_jinja2(resolution_prompt))
            input_schema["type"] = "array"

        elif op_type == "split":
            # Split requires a split_key
            split_key = operator_config.get("split_key")
            if split_key:
                required_fields.add(split_key)
            input_schema["type"] = "object"

        elif op_type == "gather":
            # Gather requires specific keys
            content_key = operator_config.get("content_key")
            doc_id_key = operator_config.get("doc_id_key")
            order_key = operator_config.get("order_key")

            if content_key:
                required_fields.add(content_key)
            if doc_id_key:
                required_fields.add(doc_id_key)
            if order_key:
                required_fields.add(order_key)

            # Also check for doc_header_key and peripheral_chunks
            doc_header_key = operator_config.get("doc_header_key")
            if doc_header_key:
                extracted_fields.add(doc_header_key)

            # Extract from peripheral_chunks if present
            peripheral_chunks = operator_config.get("peripheral_chunks", {})
            for position, config in peripheral_chunks.items():
                if isinstance(config, dict) and "content_key" in config:
                    extracted_fields.add(config["content_key"])

            input_schema["type"] = "object"

        elif op_type == "unnest":
            # Unnest requires an unnest_key
            unnest_key = operator_config.get("unnest_key")
            if unnest_key:
                required_fields.add(unnest_key)
            input_schema["type"] = "object"

        elif op_type == "rank":
            # Rank uses input_keys and may have a prompt
            input_keys = operator_config.get("input_keys", [])
            if isinstance(input_keys, list):
                required_fields.update(input_keys)

            prompt = operator_config.get("prompt", "")
            if prompt:
                extracted_fields.update(extract_fields_from_jinja2(prompt))

            input_schema["type"] = "array"

        elif op_type == "order":
            # Order uses input_keys and prompt
            input_keys = operator_config.get("input_keys", [])
            if isinstance(input_keys, list):
                required_fields.update(input_keys)

            prompt = operator_config.get("prompt", "")
            if prompt:
                extracted_fields.update(extract_fields_from_jinja2(prompt))

            input_schema["type"] = "array"

        elif op_type == "cluster":
            # Cluster uses embedding_keys
            embedding_keys = operator_config.get("embedding_keys", [])
            if isinstance(embedding_keys, list):
                required_fields.update(embedding_keys)

            # Check for summary_prompt
            summary_prompt = operator_config.get("summary_prompt", "")
            if summary_prompt:
                extracted_fields.update(extract_fields_from_jinja2(summary_prompt))

            input_schema["type"] = "object"

        elif op_type == "extract":
            # Extract uses document_keys and prompt
            document_keys = operator_config.get("document_keys", [])
            if isinstance(document_keys, list):
                required_fields.update(document_keys)

            prompt = operator_config.get("prompt", "")
            if prompt:
                extracted_fields.update(extract_fields_from_jinja2(prompt))

            input_schema["type"] = "object"

        elif op_type == "topk":
            # TopK uses keys field
            keys = operator_config.get("keys", [])
            if isinstance(keys, list):
                required_fields.update(keys)
            input_schema["type"] = "array"

        elif op_type == "sample":
            # Sample may have stratify_key
            stratify_key = operator_config.get("stratify_key")
            if stratify_key:
                extracted_fields.add(stratify_key)
            input_schema["type"] = "array"

        elif op_type == "join":
            # Join uses comparison_prompt for input schema
            comparison_prompt = operator_config.get("comparison_prompt", "")
            if comparison_prompt:
                extracted_fields.update(extract_fields_from_jinja2(comparison_prompt))
            input_schema["type"] = "object"

        elif op_type == "equijoin":
            # Equijoin has left_key and right_key
            left_key = operator_config.get("left_key")
            right_key = operator_config.get("right_key")
            if left_key:
                required_fields.add(left_key)
            if right_key:
                required_fields.add(right_key)

            # Also check comparison_prompt
            comparison_prompt = operator_config.get("comparison_prompt", "")
            if comparison_prompt:
                extracted_fields.update(extract_fields_from_jinja2(comparison_prompt))

            input_schema["type"] = "object"

        elif op_type == "parallel_map":
            # Parallel map uses prompts list
            prompts = operator_config.get("prompts", [])
            for prompt_item in prompts:
                if isinstance(prompt_item, dict):
                    prompt_text = prompt_item.get("prompt", "")
                elif isinstance(prompt_item, str):
                    prompt_text = prompt_item
                else:
                    continue
                if prompt_text:
                    extracted_fields.update(extract_fields_from_jinja2(prompt_text))
            input_schema["type"] = "object"

        elif op_type == "scan":
            # Scan doesn't have field requirements, just dataset_name
            input_schema["type"] = "object"

        # Combine required fields and extracted fields
        all_fields = required_fields.union(extracted_fields)

        if all_fields:
            # Add fields dictionary with type information
            input_schema["fields"] = {}
            for field in sorted(all_fields):
                # First check field_types (cumulative from pipeline)
                if field in field_types:
                    input_schema["fields"][field] = field_types[field]
                # Then check dataset schema if available
                elif dataset_schema and "fields" in dataset_schema and field in dataset_schema["fields"]:
                    input_schema["fields"][field] = dataset_schema["fields"][field]
                else:
                    # Mark as Unknown for fields without type information
                    input_schema["fields"][field] = "Unknown"

        return input_schema

    def get_output_schema(
        self,
        operator_config: Dict[str, Any],
        cumulative_field_types: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Get output schema for a DocETL operator.

        This method now delegates to the centralized `_infer_docetl_output_schema`
        function to ensure consistency and avoid code duplication.

        Args:
            operator_config: Operator configuration dictionary
            cumulative_field_types: Optional cumulative field types for type inference

        Returns:
            Output schema dictionary mapping field names to types
        """
        # Delegate to the centralized schema inference function
        return _infer_docetl_output_schema(
            operator_config,
            cumulative_field_types or self.current_schema
        )
