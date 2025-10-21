#!/usr/bin/env python3
"""
Static checker for DocETL pipelines.
Validates YAML syntax, operator usage, and schema consistency.
"""

import sys
import re
import json
from typing import Dict, Set, Any, Optional, Union
from pathlib import Path

def infer_docetl_type_from_value(value: Any) -> str:
    """Infer DocETL type string from a Python value."""
    if value is None:
        return 'str'
    if isinstance(value, bool):
        return 'bool'
    elif isinstance(value, int):
        return 'int'
    elif isinstance(value, float):
        return 'float'
    elif isinstance(value, str):
        return 'str'
    elif isinstance(value, list):
        if len(value) == 0:
            return 'list'
        first_elem_type = infer_docetl_type_from_value(value[0])
        return f'list[{first_elem_type}]'
    elif isinstance(value, dict):
        return 'dict'
    else:
        return 'str'

def infer_schema_from_dataset(file_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """Infer dataset schema from a DocETL data file."""
    try:
        file_path = Path(file_path)
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if not isinstance(data, list) or len(data) == 0:
            return None

        first_item = data[0]
        if not isinstance(first_item, dict):
            return None

        fields = {}
        for field_name, field_value in first_item.items():
            fields[field_name] = infer_docetl_type_from_value(field_value)

        return {'fields': fields}

    except (FileNotFoundError, json.JSONDecodeError, KeyError, IndexError):
        return None

def parse_yaml_simple(content: str) -> dict:
    """Simple YAML parser for basic DocETL pipeline files."""
    try:
        # Remove comments
        lines = []
        for line in content.split('\n'):
            comment_idx = line.find('#')
            if comment_idx >= 0:
                line = line[:comment_idx]
            lines.append(line)
        content = '\n'.join(lines)
        
        import yaml
        return yaml.safe_load(content)
    except Exception as e:
        raise ValueError(f"Failed to parse YAML: {str(e)}")


class DocETLStaticChecker:
    VALID_OPERATORS = {
        'map', 'filter', 'reduce', 'resolve', 'order', 'extract', 
        'cluster', 'split', 'gather', 'unnest', 'sample', 'topk',
        'code_map', 'code_filter', 'code_reduce', 'equijoin', 'scan',
        'parallel_map', 'link_resolve', 'add_uuid'
    }
    
    # Universal required fields that must appear in all operators
    UNIVERSAL_REQUIRED_FIELDS = {'name', 'type'}
    
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
        'scan': {'dataset_name'},
    }

    UNIVERSAL_OPTIONAL_FIELDS = {
        'gleaning', 
        'skip_on_error'
    }

    OPERATOR_OPTIONAL_FIELDS = {
        'map': {
            'output', 'prompt', 'model', 'optimize', 'recursively_optimize', 'sample_size',
            'tools', 'validation_rules', 'num_retries_on_validate_failure',
            'drop_keys', 'timeout', 'enable_observability', 'batch_size', 'clustering_method',
            'batch_prompt', 'litellm_completion_kwargs', 'pdf_url_key', 'flush_partial_result',
            'calibrate', 'num_calibration_docs'
        },
        'filter': {
            'model', 'optimize', 'recursively_optimize', 'sample_size',
            'tools', 'validation_rules', 'num_retries_on_validate_failure',
            'drop_keys', 'timeout', 'enable_observability', 'batch_size', 'clustering_method',
            'batch_prompt', 'litellm_completion_kwargs', 'pdf_url_key', 'flush_partial_result',
            'calibrate', 'num_calibration_docs'
        },
        'reduce': {
            'optimize', 'synthesize_resolve', 'model', 'input', 'pass_through',
            'associative', 'fold_prompt', 'fold_batch_size', 'merge_prompt',
            'merge_batch_size', 'value_sampling', 'verbose', 'timeout',
            'litellm_completion_kwargs', 'enable_observability'
        },
        'resolve': {
            'embedding_model', 'resolution_model', 'comparison_model',
            'blocking_keys', 'blocking_threshold', 'blocking_conditions',
            'input', 'embedding_batch_size', 'compare_batch_size',
            'limit_comparisons', 'optimize', 'timeout', 'litellm_completion_kwargs',
            'enable_observability'
        },
        'extract': {
            'model', 'format_extraction', 'extraction_key_suffix', 'extraction_method',
            'timeout', 'litellm_completion_kwargs'
        },
        'cluster': {
            'output_key', 'max_batch_size', 'embedding_model', 'model',
            'timeout', 'litellm_completion_kwargs'
        },
        'gather': {
            'peripheral_chunks', 'doc_header_key', 'main_chunk_start', 'main_chunk_end'
        },
        'split': {
            'model'
        },
        'unnest': {
            'keep_empty', 'expand_fields', 'recursive', 'depth'
        },
        'sample': {
            'samples', 'stratify_key', 'samples_per_group', 'method_kwargs', 'random_state'
        },
        'topk': {
            'stratify_key', 'embedding_model', 'model', 'batch_size'
        },
        'order': {
            'model', 'embedding_model', 'batch_size', 'initial_ordering_method', 'k',
            'rerank_call_budget', 'num_top_items_per_window', 'overlap_fraction',
            'timeout', 'num_calibration_docs', 'verbose', 'litellm_completion_kwargs'
        },
        'equijoin': {
            'output', 'blocking_threshold', 'blocking_conditions', 'limits',
            'comparison_model', 'optimize', 'embedding_model', 'embedding_batch_size',
            'compare_batch_size', 'limit_comparisons', 'blocking_keys', 'timeout',
            'litellm_completion_kwargs'
        },
        'parallel_map': {
            'model', 'optimize', 'recursively_optimize', 'timeout', 'litellm_completion_kwargs'
        },
        'code_map': {
            'concurrent_thread_count', 'drop_keys'
        },
        'code_filter': {
            'concurrent_thread_count', 'drop_keys'
        },
        'code_reduce': {
            'concurrent_thread_count', 'reduce_key', 'pass_through'
        },
        'scan': set(),
    }

    def __init__(self):
        self.errors = []
        self.warnings = []
        self.pipeline = None
        self.pipeline_path = None
        self.dataset_schema = None
        self.field_tracker = {}  # Track fields created/used by operations
        
    def check(self, pipeline_path: str) -> dict:
        """Check a DocETL pipeline file and return validation results."""
        self.pipeline_path = pipeline_path

        if not self._check_yaml_validity_from_path(pipeline_path):
            return self._format_result()

        if not self._infer_dataset_schema():
            return self._format_result()

        self._check_operator_syntax()
        self._check_schema_consistency()
        return self._format_result()

    def check_string(self, yaml_content: str) -> dict:
        """Check YAML content string and return validation results."""
        if not self._check_yaml_validity_from_string(yaml_content):
            return self._format_result()

        if not self._infer_dataset_schema():
            return self._format_result()

        self._check_operator_syntax()
        self._check_schema_consistency()
        return self._format_result()

    def _infer_dataset_schema(self) -> bool:
        """Infer dataset schema from the datasets section."""
        if not self.pipeline or 'datasets' not in self.pipeline:
            self.errors.append({
                "type": "missing_datasets",
                "message": "Pipeline YAML must contain 'datasets' section for schema inference"
            })
            return False

        datasets = self.pipeline['datasets']
        if not datasets:
            self.errors.append({
                "type": "empty_datasets",
                "message": "No datasets defined in pipeline"
            })
            return False

        first_dataset_name = next(iter(datasets.keys()))
        first_dataset = datasets[first_dataset_name]

        if 'path' not in first_dataset:
            self.errors.append({
                "type": "missing_dataset_path",
                "dataset": first_dataset_name,
                "message": f"Dataset '{first_dataset_name}' missing 'path' field"
            })
            return False

        dataset_path = first_dataset['path']

        if self.pipeline_path:
            yaml_dir = Path(self.pipeline_path).parent
            full_path = yaml_dir / dataset_path
        else:
            full_path = Path(dataset_path)

        if not full_path.exists():
            self.errors.append({
                "type": "dataset_file_not_found",
                "dataset": first_dataset_name,
                "path": str(full_path),
                "message": f"Dataset file not found: {full_path}"
            })
            return False

        self.dataset_schema = infer_schema_from_dataset(full_path)

        if not self.dataset_schema or 'fields' not in self.dataset_schema:
            self.errors.append({
                "type": "schema_inference_failed",
                "dataset": first_dataset_name,
                "path": str(full_path),
                "message": f"Failed to infer schema from dataset: {full_path}"
            })
            return False

        return True

    def _check_yaml_validity_from_path(self, pipeline_path: str) -> bool:
        """Check if the YAML file is valid by reading from file path."""
        try:
            with open(pipeline_path, 'r') as f:
                content = f.read()
            return self._check_yaml_validity_from_string(content)
            
        except FileNotFoundError:
            self.errors.append({
                "type": "file_not_found",
                "file_path": pipeline_path,
                "message": f"File not found - {pipeline_path}"
            })
            return False
        except Exception as e:
            self.errors.append({
                "type": "file_read_error",
                "file_path": pipeline_path,
                "message": f"Error reading file - {str(e)}"
            })
            return False
    
    def _check_yaml_validity_from_string(self, yaml_content: str) -> bool:
        """Check if the YAML content string is valid."""
        try:
            self.pipeline = parse_yaml_simple(yaml_content)
            
            if not isinstance(self.pipeline, dict):
                self.errors.append({
                    "type": "yaml_structure_error",
                    "message": "Root element must be a dictionary"
                })
                return False

            required_keys = {'default_model', 'datasets', 'operations', 'pipeline'}
            missing_keys = required_keys - set(self.pipeline.keys())
            if missing_keys:
                for missing_key in missing_keys:
                    self.errors.append({
                        "type": "missing_top_level_key",
                        "key": missing_key,
                        "message": f"Missing required top-level key: {missing_key}"
                    })
                return False

            return True
            
        except ValueError as e:
            self.errors.append({
                "type": "yaml_syntax_error", 
                "message": f"Invalid YAML syntax - {str(e)}"
            })
            return False
        except Exception as e:
            self.errors.append({
                "type": "yaml_error",
                "message": f"Unexpected error - {str(e)}"
            })
            return False
    
    def _check_operator_syntax(self) -> bool:
        """Check if all operators use correct syntax."""
        if not self.pipeline or 'operations' not in self.pipeline:
            return True
            
        operations = self.pipeline['operations']
        if not isinstance(operations, list):
            self.errors.append({
                "type": "invalid_operations_structure",
                "message": "'operations' must be a list"
            })
            return False
        
        has_errors = False
        for i, op in enumerate(operations):
            if not isinstance(op, dict):
                self.errors.append({
                    "type": "invalid_operation_structure",
                    "operation_index": i,
                    "message": f"Operation {i} must be a dictionary"
                })
                has_errors = True
                continue

            if 'name' not in op:
                self.errors.append({
                    "type": "missing_operation_name",
                    "operation_index": i,
                    "message": f"Operation {i} missing 'name' field"
                })
                has_errors = True
                continue

            op_name = op['name']

            if 'type' not in op:
                self.errors.append({
                    "type": "missing_operation_type",
                    "operation": op_name,
                    "message": f"Operation '{op_name}' missing 'type' field"
                })
                has_errors = True
                continue

            op_type = op['type']
            if op_type not in self.VALID_OPERATORS:
                self.errors.append({
                    "type": "invalid_operation_type",
                    "operation": op_name,
                    "invalid_type": op_type,
                    "valid_types": list(self.VALID_OPERATORS),
                    "message": f"Operation '{op_name}' has invalid type '{op_type}'"
                })
                has_errors = True
                continue

            universal_required_fields = self.UNIVERSAL_REQUIRED_FIELDS
            operator_required_fields = self.OPERATOR_REQUIREMENTS.get(op_type, set())
            required_fields = universal_required_fields | operator_required_fields
            missing_fields = required_fields - set(op.keys())
            if missing_fields:
                for missing_field in missing_fields:
                    self.errors.append({
                        "type": "missing_required_field",
                        "operation": op_name,
                        "operation_type": op_type,
                        "field": missing_field,
                        "message": f"Operation '{op_name}' of type '{op_type}' missing required field: {missing_field}"
                    })
                has_errors = True

            operator_optional_fields = self.OPERATOR_OPTIONAL_FIELDS.get(op_type, set())
            all_valid_fields = required_fields | operator_optional_fields | self.UNIVERSAL_OPTIONAL_FIELDS
            unknown_fields = set(op.keys()) - all_valid_fields
            if unknown_fields:
                for unknown_field in unknown_fields:
                    self.warnings.append({
                        "type": "unknown_field",
                        "operation": op_name,
                        "field": unknown_field,
                        "message": f"Operation '{op_name}' has unknown field: {unknown_field}"
                    })

            self._track_operation_fields(op_name, op_type, op)

            if not self._validate_operator_fields(op_name, op_type, op):
                has_errors = True
        
        return not has_errors
    
    def _validate_operator_fields(self, op_name: str, op_type: str, op: Dict) -> bool:
        """Validate specific field constraints for an operator."""
        has_errors = False
        
        # Check Jinja2 template validity in prompts
        prompt_fields = ['prompt', 'comparison_prompt', 'resolution_prompt',
                        'batch_prompt', 'summary_prompt', 'combine_prompt',
                        'update_prompt']
        for field in prompt_fields:
            if field in op:
                if not self._validate_jinja_template(op[field]):
                    self.errors.append({
                        "type": "invalid_jinja_template",
                        "operation": op_name,
                        "field": field,
                        "message": f"Operation '{op_name}' has invalid Jinja2 template in '{field}'"
                    })
                    has_errors = True
                else:
                    jinja_fields = self._extract_jinja_fields(op[field])
                    if self.dataset_schema and 'fields' in self.dataset_schema:
                        for jinja_field in jinja_fields:
                            if jinja_field not in self.dataset_schema['fields']:
                                # Check if this field exists within a complex type
                                complex_location = self._find_field_in_complex_types(
                                    jinja_field,
                                    self.dataset_schema['fields']
                                )

                                if complex_location:
                                    # Field exists within a complex type - provide detailed error
                                    suggestions_text = "\n                    ".join(
                                        complex_location['access_suggestions']
                                    )
                                    self.errors.append({
                                        "type": "incorrect_complex_type_access",
                                        "operation": op_name,
                                        "field": field,
                                        "referenced_field": jinja_field,
                                        "parent_field": complex_location['parent_field'],
                                        "parent_type": complex_location['parent_type'],
                                        "message": (
                                            f"Operation '{op_name}' cannot directly access field '{jinja_field}' in '{field}'. "
                                            f"This field is nested within '{complex_location['parent_field']}: {complex_location['parent_type']}'. "
                                            f"To access it, use one of the following methods:\n"
                                            f"                    {suggestions_text}"
                                        )
                                    })
                                else:
                                    # Field truly doesn't exist
                                    self.errors.append({
                                        "type": "undefined_field_in_template",
                                        "operation": op_name,
                                        "field": field,
                                        "referenced_field": jinja_field,
                                        "message": f"Operation '{op_name}' references undefined field '{jinja_field}' in '{field}'"
                                    })
                                has_errors = True

        if op_type == 'reduce' and 'reduce_key' in op:
            if op['reduce_key'] is None:
                self.errors.append({
                    "type": "null_reduce_key",
                    "operation": op_name,
                    "message": f"Operation '{op_name}' of type 'reduce' has null reduce_key"
                })
                has_errors = True
            else:
                # Track the reduce_key as an input field requirement
                self.field_tracker[op_name]['inputs'].add(op['reduce_key'])
        
        # Check code_reduce operations have proper reduce_key  
        if op_type == 'code_reduce' and 'reduce_key' in op:
            if op['reduce_key'] is None:
                self.errors.append({
                    "type": "null_reduce_key", 
                    "operation": op_name,
                    "message": f"Operation '{op_name}' of type 'code_reduce' has null reduce_key"
                })
                has_errors = True
            else:
                # Track the reduce_key as an input field requirement
                self.field_tracker[op_name]['inputs'].add(op['reduce_key'])
        
        # Check code operations have valid Python code
        if op_type in ['code_map', 'code_filter', 'code_reduce'] and 'code' in op:
            if not self._validate_python_code(op['code']):
                self.errors.append({
                    "type": "invalid_python_code",
                    "operation": op_name,
                    "message": f"Operation '{op_name}' has invalid Python code"
                })
                has_errors = True
            else:
                # Try to extract output fields from return statement in code
                code_outputs = self._extract_fields_from_code(op['code'])
                self.field_tracker[op_name]['outputs'].update(code_outputs)
        
        # Check output schema format
        if 'output' in op and isinstance(op['output'], dict):
            if 'schema' in op['output']:
                if not self._validate_output_schema(op['output']['schema']):
                    self.errors.append({
                        "type": "invalid_output_schema",
                        "operation": op_name,
                        "message": f"Operation '{op_name}' has invalid output schema"
                    })
                    has_errors = True
        
        return not has_errors
    
    def _validate_python_code(self, code: str) -> bool:
        """Basic validation of Python code structure."""
        try:
            # Check for basic Python syntax
            compile(code, '<string>', 'exec')
            # Check for required transform function
            if 'def transform(' not in code:
                return False
            return True
        except SyntaxError:
            return False
    
    def _extract_fields_from_code(self, code: str) -> Set[str]:
        """Extract field names from return statements in Python code."""
        fields = set()
        
        # Look for return statements with dictionary literals (handle multi-line)
        import re
        
        # Find return dictionary blocks - handle multi-line dictionaries
        return_pattern = r'return\s*\{([^}]+)\}'
        matches = re.finditer(return_pattern, code, re.DOTALL)
        
        for match in matches:
            dict_content = match.group(1)
            # Extract field names from dictionary content
            field_matches = re.findall(r'["\'](\w+)["\']:', dict_content)
            fields.update(field_matches)
        
        return fields
    
    def _validate_jinja_template(self, template: str) -> bool:
        """Validate Jinja2 template syntax."""
        try:
            from jinja2 import Template
            Template(template)
            return True
        except Exception:
            return False
    
    def _validate_output_schema(self, schema: Any) -> bool:
        """Validate output schema format."""
        if not isinstance(schema, dict):
            return False
        
        valid_types = {'integer', 'number', 'boolean', 'str'}
        
        for field, field_type in schema.items():
            if isinstance(field_type, str):
                # Check basic types
                if field_type not in valid_types:
                    # Check for list types
                    if not (field_type.startswith('list[') and field_type.endswith(']')):
                        # Check for enum types
                        if not field_type.startswith('enum['):
                            return False
            elif isinstance(field_type, dict):
                # Nested schema
                if not self._validate_output_schema(field_type):
                    return False
            else:
                return False
        
        return True
    
    def _track_operation_fields(self, op_name: str, op_type: str, op: Dict):
        """Track input and output fields for schema consistency checking."""
        self.field_tracker[op_name] = {
            'inputs': set(),
            'outputs': set(),
            'output_types': {},  # Track field name -> type string mapping
            'type': op_type
        }

        # Extract input fields from prompts
        prompt_fields = ['prompt', 'comparison_prompt', 'resolution_prompt',
                        'batch_prompt', 'summary_prompt']
        for field in prompt_fields:
            if field in op:
                input_fields = self._extract_jinja_fields(op[field])
                self.field_tracker[op_name]['inputs'].update(input_fields)
                # Check for bare field references that should use Jinja2 syntax
                self._check_for_bare_field_references(op[field], op_name, input_fields)

        # Extract output fields from schema with their types
        if 'output' in op and isinstance(op['output'], dict):
            if 'schema' in op['output'] and isinstance(op['output']['schema'], dict):
                output_schema = op['output']['schema']
                self.field_tracker[op_name]['outputs'].update(output_schema.keys())
                # Store the complete type information
                for field_name, field_type in output_schema.items():
                    self.field_tracker[op_name]['output_types'][field_name] = field_type
        
        # Handle special operators
        if op_type == 'unnest' and 'unnest_key' in op:
            self.field_tracker[op_name]['inputs'].add(op['unnest_key'])
            # For unnest operations, we need to look at the schema of the field being unnested
            # to determine what fields become available
            unnest_key = op['unnest_key']

            # Track unnest-specific parameters
            self.field_tracker[op_name]['unnest_key'] = unnest_key
            self.field_tracker[op_name]['recursive'] = op.get('recursive', False)
            self.field_tracker[op_name]['depth'] = op.get('depth')
            self.field_tracker[op_name]['expand_fields'] = op.get('expand_fields', [])

            # For non-recursive unnest, the field remains but changes type
            # For recursive unnest, the field is removed and nested fields are added
            if not op.get('recursive', False):
                self.field_tracker[op_name]['outputs'].add(unnest_key)

            # Check if we can determine the nested fields from previous operations
            self.field_tracker[op_name]['nested_fields_from'] = unnest_key
        
        if op_type == 'split' and 'split_key' in op:
            self.field_tracker[op_name]['inputs'].add(op['split_key'])
            self.field_tracker[op_name]['outputs'].add(f"{op['split_key']}_chunk")
        
        if op_type == 'gather' and 'content_key' in op:
            self.field_tracker[op_name]['inputs'].add(op['content_key'])
            self.field_tracker[op_name]['outputs'].add(f"{op['content_key']}_rendered")
    
    def _extract_jinja_fields(self, template: str) -> Set[str]:
        """
        Extract field references from Jinja2 template.

        This method extracts the top-level field names being accessed,
        regardless of nested access patterns.

        Examples:
            {{ input.name }} -> extracts 'name'
            {{ input.medications[0].name }} -> extracts 'medications'
            {{ input.patient.name }} -> extracts 'patient'
        """
        fields = set()

        # Match patterns like {{ input.field }}, {{ inputs[0].field }}, etc.
        # Extract the first field name after 'input.'
        patterns = [
            r'{{\s*input\.(\w+)',  # {{ input.field... (captures first field)
            r'{{\s*inputs\[\d+\]\.(\w+)',  # {{ inputs[0].field...
            r'{{\s*item\.(\w+)',  # {{ item.field...
            r'{{\s*e\.(\w+)',  # {{ e.field...
            r'{{\s*input1\.(\w+)',  # {{ input1.field...
            r'{{\s*input2\.(\w+)'  # {{ input2.field...
        ]

        for pattern in patterns:
            matches = re.findall(pattern, template)
            fields.update(matches)

        return fields

    def _check_for_bare_field_references(self, template: str, op_name: str, jinja_fields: Set[str]):
        """
        Check for suspicious bare field references in prompt that should use Jinja2 syntax.

        Args:
            template: The prompt template string
            op_name: Operator name (for error messages)
            jinja_fields: Fields already extracted with Jinja2 syntax
        """
        if not template:
            return

        # Patterns for detecting bare field references
        suspicious_patterns = [
            (r'`(\w+)`', 'backtick-quoted field'),  # `fieldname`
            (r'field\s+["\'](\w+)["\']', 'field name in quotes'),  # field "name" or field 'name'
            (r'the\s+(\w+)\s+field', 'descriptive field reference'),  # "the name field"
            (r'in\s+the\s+field\s+["\'](\w+)["\']', 'field in quotes'),  # "in the field 'name'"
            (r'from\s+["\'](\w+)["\']', 'field from quotes'),  # "from 'fieldname'"
        ]

        # Common field names that are likely to be data fields
        likely_field_names = {'text', 'src', 'content', 'document', 'data', 'input',
                             'title', 'description', 'body', 'message', 'name',
                             'value', 'field', 'item', 'record', 'doc'}

        for pattern, pattern_desc in suspicious_patterns:
            matches = re.findall(pattern, template, re.IGNORECASE)
            for match in matches:
                # Check if this looks like a field reference and is not already in Jinja2 format
                if match.lower() in likely_field_names and match not in jinja_fields:
                    self.warnings.append({
                        "type": "possible_bare_field_reference",
                        "operation": op_name,
                        "field": match,
                        "pattern": pattern_desc,
                        "message": f"Operation '{op_name}': Possible bare field reference '{match}' found ({pattern_desc}). Should use {{{{ input.{match} }}}} instead?"
                    })
    
    def _check_schema_consistency(self) -> bool:
        """Check schema consistency across operations and operator usage."""
        if not self.pipeline or 'pipeline' not in self.pipeline:
            return True
        
        pipeline_config = self.pipeline['pipeline']
        if not pipeline_config or 'steps' not in pipeline_config:
            return True
        
        has_errors = False
        
        # Track which operations are actually used
        used_operations = set()
        
        for step in pipeline_config['steps']:
            if not isinstance(step, dict):
                continue
                
            if 'operations' not in step:
                continue

            # Track available fields with their types (field_name -> type_string)
            available_fields_with_types = {}

            if 'input' in step:
                dataset_name = step['input']

                if not self.dataset_schema or 'fields' not in self.dataset_schema:
                    self.errors.append({
                        "type": "missing_schema_for_validation",
                        "step": step.get('name', 'unnamed'),
                        "dataset": dataset_name,
                        "message": f"Cannot validate pipeline: dataset schema not available"
                    })
                    has_errors = True
                    continue

                # Store fields with their types from dataset schema
                available_fields_with_types.update(self.dataset_schema['fields'])

            # Check each operation in sequence
            operations = step['operations']
            for op_ref in operations:
                if isinstance(op_ref, str):
                    op_name = op_ref
                elif isinstance(op_ref, dict):
                    # Handle inline operations
                    op_name = list(op_ref.keys())[0]
                else:
                    continue
                
                # Track that this operation is used
                used_operations.add(op_name)
                
                # Find the operation definition
                op_def = None
                for op in self.pipeline.get('operations', []):
                    if op.get('name') == op_name:
                        op_def = op
                        break
                
                if not op_def:
                    self.errors.append({
                        "type": "undefined_operation",
                        "operation": op_name,
                        "message": f"Operation '{op_name}' is used in pipeline but not defined in operations section"
                    })
                    has_errors = True
                    # For continued checking, assume this is a basic map operation
                    continue
                
                # Check if required input fields are available
                if op_name in self.field_tracker:
                    required_inputs = self.field_tracker[op_name]['inputs']
                    available_field_names = set(available_fields_with_types.keys())
                    missing_fields = required_inputs - available_field_names

                    if missing_fields:
                        for missing_field in missing_fields:
                            # Check if this field exists within a complex type
                            complex_location = self._find_field_in_complex_types(
                                missing_field,
                                available_fields_with_types
                            )

                            if complex_location:
                                # Field exists within a complex type - provide detailed error
                                suggestions_text = "\n                ".join(
                                    complex_location['access_suggestions']
                                )
                                self.errors.append({
                                    "type": "incorrect_complex_type_access",
                                    "operation": op_name,
                                    "field": missing_field,
                                    "parent_field": complex_location['parent_field'],
                                    "parent_type": complex_location['parent_type'],
                                    "message": (
                                        f"Operation '{op_name}' cannot directly access field '{missing_field}'. "
                                        f"This field is nested within '{complex_location['parent_field']}: {complex_location['parent_type']}'. "
                                        f"To access it, use one of the following methods:\n"
                                        f"                {suggestions_text}"
                                    )
                                })
                            else:
                                # Field truly doesn't exist
                                self.errors.append({
                                    "type": "missing_schema_field",
                                    "operation": op_name,
                                    "field": missing_field,
                                    "message": f"Operation '{op_name}' expects field '{missing_field}' which is not available from previous operations"
                                })
                        has_errors = True

                    # Add output fields with types to available fields
                    output_types = self.field_tracker[op_name].get('output_types', {})
                    for field_name, field_type in output_types.items():
                        available_fields_with_types[field_name] = field_type

                    # Also add output fields that don't have explicit types (use 'str' as default)
                    for field_name in self.field_tracker[op_name]['outputs']:
                        if field_name not in available_fields_with_types:
                            available_fields_with_types[field_name] = 'str'

                    # Handle special operations that preserve fields
                    op_type = op_def.get('type')
                    if op_type in ['map', 'filter', 'resolve']:
                        # These operations typically preserve input fields - no change needed
                        # Fields already in available_fields_with_types stay
                        pass
                    elif op_type in ['reduce', 'code_reduce']:
                        # Reduce operations change the structure significantly - only keep outputs
                        new_fields = {}
                        for field_name in self.field_tracker[op_name]['outputs']:
                            if field_name in output_types:
                                new_fields[field_name] = output_types[field_name]
                            else:
                                new_fields[field_name] = 'str'
                        available_fields_with_types = new_fields
                    elif op_type in ['code_map']:
                        # Code map operations replace the structure with their outputs
                        new_fields = {}
                        for field_name in self.field_tracker[op_name]['outputs']:
                            if field_name in output_types:
                                new_fields[field_name] = output_types[field_name]
                            else:
                                new_fields[field_name] = 'str'
                        available_fields_with_types = new_fields
                    elif op_type == 'unnest':
                        # Unnest operations transform the schema based on type and recursive parameter
                        unnest_key = self.field_tracker[op_name].get('unnest_key')
                        recursive = self.field_tracker[op_name].get('recursive', False)
                        depth = self.field_tracker[op_name].get('depth')

                        if unnest_key and unnest_key in available_fields_with_types:
                            parent_type = available_fields_with_types[unnest_key]
                            type_info = self._parse_complex_type(parent_type)

                            # Compute the new schema after unnesting
                            unnest_changes = self._compute_unnest_schema(
                                unnest_key, parent_type, recursive, depth
                            )

                            # Determine if parent field should be removed
                            # Remove parent field if:
                            # 1. Dict type (always removes parent in DocETL)
                            # 2. List type with recursive=True
                            should_remove_parent = (
                                type_info['base_type'] == 'dict' or
                                (type_info['base_type'] == 'list' and recursive)
                            )

                            if should_remove_parent:
                                del available_fields_with_types[unnest_key]

                            # Add/update fields from unnest
                            for field_name, field_type in unnest_changes.items():
                                available_fields_with_types[field_name] = field_type
        
        # Check for defined but unused operations (warning only, not error)
        self._check_unused_operations(used_operations)
        
        return not has_errors
    
    def _get_nested_fields_from_schema(self, unnest_op_name: str, available_fields: Set[str]) -> Set[str]:
        """Extract nested fields that become available after unnesting."""
        nested_fields = set()
        
        if unnest_op_name not in self.field_tracker:
            return nested_fields
            
        unnest_key = self.field_tracker[unnest_op_name].get('nested_fields_from')
        if not unnest_key:
            return nested_fields
        
        # Look for the operation that produced the field being unnested
        for op_name, tracker in self.field_tracker.items():
            if unnest_key in tracker['outputs']:
                # Find the operation definition to get the schema
                for op in self.pipeline.get('operations', []):
                    if op.get('name') == op_name and 'output' in op:
                        schema = op.get('output', {}).get('schema', {})
                        if unnest_key in schema:
                            field_def = schema[unnest_key]
                            # Parse list[{...}] format to extract nested fields
                            nested_fields.update(self._parse_list_schema_fields(field_def))
                            break
                break
        
        return nested_fields
    
    def _parse_list_schema_fields(self, field_def: str) -> Set[str]:
        """Parse a list schema definition to extract nested field names."""
        fields = set()

        if isinstance(field_def, str) and field_def.startswith('list[{') and field_def.endswith('}]'):
            # Extract the content between list[{ and }]
            content = field_def[6:-2]  # Remove 'list[{' and '}]'

            # Split by commas and extract field names
            parts = content.split(',')
            for part in parts:
                if ':' in part:
                    field_name = part.split(':')[0].strip()
                    fields.add(field_name)

        return fields

    def _parse_complex_type(self, type_str: str) -> Dict[str, Any]:
        """
        Parse complex type strings to extract nested field information.

        Args:
            type_str: Type string (e.g., 'list[{name: str, age: int}]', 'dict', 'str')

        Returns:
            Dict with structure:
            {
                'is_complex': bool,
                'base_type': 'list' | 'dict' | 'simple',
                'nested_fields': Set[str],  # Field names within complex type
                'nested_types': Dict[str, str]  # Field name -> type mapping
            }
        """
        result = {
            'is_complex': False,
            'base_type': 'simple',
            'nested_fields': set(),
            'nested_types': {}
        }

        if not isinstance(type_str, str):
            return result

        type_str = type_str.strip()

        # Check for list[{...}] pattern (list of dicts with fields)
        if type_str.startswith('list[{') and type_str.endswith('}]'):
            result['is_complex'] = True
            result['base_type'] = 'list'

            # Extract content between list[{ and }]
            content = type_str[6:-2]

            # Parse field definitions
            parts = content.split(',')
            for part in parts:
                part = part.strip()
                if ':' in part:
                    field_name, field_type = part.split(':', 1)
                    field_name = field_name.strip()
                    field_type = field_type.strip()
                    result['nested_fields'].add(field_name)
                    result['nested_types'][field_name] = field_type

        # Check for simple list[type] pattern
        elif type_str.startswith('list[') and type_str.endswith(']'):
            result['is_complex'] = True
            result['base_type'] = 'list'
            # Simple list, no nested fields to extract

        # Check for dict type (generic dict without specific fields)
        elif type_str == 'dict':
            result['is_complex'] = True
            result['base_type'] = 'dict'

        # Check for {field: type, ...} pattern (dict with fields)
        elif type_str.startswith('{') and type_str.endswith('}'):
            result['is_complex'] = True
            result['base_type'] = 'dict'

            # Extract content between { and }
            content = type_str[1:-1]

            # Parse field definitions
            parts = content.split(',')
            for part in parts:
                part = part.strip()
                if ':' in part:
                    field_name, field_type = part.split(':', 1)
                    field_name = field_name.strip()
                    field_type = field_type.strip()
                    result['nested_fields'].add(field_name)
                    result['nested_types'][field_name] = field_type

        return result

    def _find_field_in_complex_types(self, field_name: str, schema: Dict[str, str]) -> Optional[Dict[str, Any]]:
        """
        Find if a field exists within complex types in the schema.

        Args:
            field_name: The field name to search for
            schema: Current schema (field -> type mapping)

        Returns:
            Dict with location info if found, None otherwise:
            {
                'parent_field': str,  # Name of the parent field containing this field
                'parent_type': str,   # Type of the parent field
                'field_type': str,    # Type of the nested field
                'base_type': 'list' | 'dict',  # Type of complex container
                'access_suggestions': List[str]  # Suggested ways to access the field
            }
        """
        for parent_field, parent_type in schema.items():
            # Parse the parent type to check for nested fields
            type_info = self._parse_complex_type(parent_type)

            if type_info['is_complex'] and field_name in type_info['nested_fields']:
                # Found the field within this complex type
                access_suggestions = []

                if type_info['base_type'] == 'list':
                    # List of dicts - suggest unnest or indexed access
                    access_suggestions.append(
                        f"1. Use an Unnest operator to expand the '{parent_field}' field first"
                    )
                    access_suggestions.append(
                        f"2. Access via index in Jinja2: {{{{ input.{parent_field}[0].{field_name} }}}}"
                    )
                    access_suggestions.append(
                        f"3. Use a for loop in Jinja2: {{% for item in input.{parent_field} %}}{{{{ item.{field_name} }}}}{{% endfor %}}"
                    )
                elif type_info['base_type'] == 'dict':
                    # Dict with fields - suggest dotted access
                    access_suggestions.append(
                        f"1. Access via dotted notation in Jinja2: {{{{ input.{parent_field}.{field_name} }}}}"
                    )
                    access_suggestions.append(
                        f"2. Use an Unnest operator to expand the '{parent_field}' field if needed"
                    )

                return {
                    'parent_field': parent_field,
                    'parent_type': parent_type,
                    'field_type': type_info['nested_types'].get(field_name, 'unknown'),
                    'base_type': type_info['base_type'],
                    'access_suggestions': access_suggestions
                }

        return None

    def _compute_unnest_schema(
        self,
        unnest_key: str,
        parent_type: str,
        recursive: bool = False,
        depth: Optional[int] = None
    ) -> Dict[str, str]:
        """
        Compute the schema changes after an unnest operation.

        Args:
            unnest_key: The field being unnested
            parent_type: The type of the field (e.g., 'list[{name: str}]')
            recursive: Whether recursive unnest is enabled
            depth: Depth of unnesting (if specified)

        Returns:
            Dict mapping field names to their new types after unnest
            - For non-recursive: {unnest_key: inner_type}
            - For recursive: {field1: type1, field2: type2, ...} (unnest_key removed)
        """
        result = {}

        # Parse the parent type to understand its structure
        type_info = self._parse_complex_type(parent_type)

        if not type_info['is_complex']:
            # Simple type, unnesting doesn't change much
            result[unnest_key] = parent_type
            return result

        if type_info['base_type'] == 'list':
            # Unnesting a list
            if parent_type.startswith('list[{') and parent_type.endswith('}]'):
                # list[{field: type, ...}] format
                if recursive:
                    # Recursive unnest: extract nested fields directly
                    for field_name, field_type in type_info['nested_types'].items():
                        result[field_name] = field_type
                else:
                    # Non-recursive unnest: list[{...}] → {...}
                    # Extract the dict part
                    inner_content = parent_type[5:-1]  # Remove 'list[' and ']'
                    result[unnest_key] = inner_content

            elif parent_type.startswith('list[') and parent_type.endswith(']'):
                # Simple list[type] format
                inner_type = parent_type[5:-1]  # Remove 'list[' and ']'
                result[unnest_key] = inner_type

        elif type_info['base_type'] == 'dict':
            # Unnesting a dict ALWAYS extracts nested fields and removes parent field
            # This is the default behavior of dict unnest in DocETL (regardless of recursive parameter)
            if type_info['nested_fields']:
                for field_name, field_type in type_info['nested_types'].items():
                    result[field_name] = field_type
            # Note: parent field (unnest_key) is NOT included in result
            # It will be removed when applying the schema changes

        return result

    def _check_unused_operations(self, used_operations: Set[str]):
        """Check for operations that are defined but not used in the pipeline."""
        if 'operations' not in self.pipeline:
            return
        
        # Get all defined operation names
        defined_operations = set()
        for op in self.pipeline['operations']:
            if isinstance(op, dict) and 'name' in op:
                defined_operations.add(op['name'])
        
        # Find unused operations
        unused_operations = defined_operations - used_operations
        
        # Print warnings for unused operations
        for unused_op in unused_operations:
            self.warnings.append({
                "type": "unused_operation",
                "operation": unused_op,
                "message": f"Operation '{unused_op}' is defined but never used in the pipeline"
            })
    
    def _format_result(self) -> dict:
        """Format the result as JSON with score and errors."""
        # Errors are already in structured format, just pass them through
        formatted_errors = []
        for error in self.errors:
            if isinstance(error, dict):
                formatted_errors.append(error)
            else:
                # Fallback for any remaining string errors
                formatted_errors.append({
                    "type": "unknown_error", 
                    "message": str(error)
                })
        
        # Warnings are already in structured format, just pass them through
        formatted_warnings = []
        for warning in self.warnings:
            if isinstance(warning, dict):
                formatted_warnings.append(warning)
            else:
                # Fallback for any remaining string warnings
                formatted_warnings.append({
                    "type": "unknown_warning",
                    "message": str(warning)
                })
        
        result = {
            "score": 1 if not self.errors else 0,
            "errors": formatted_errors
        }
        
        if formatted_warnings:
            result["warnings"] = formatted_warnings
            
        return result
    
    def _print_errors(self):
        """Print all errors and warnings."""
        for error in self.errors:
            if isinstance(error, dict):
                print(f"ERROR: {error.get('message', str(error))}", file=sys.stderr)
            else:
                print(f"ERROR: {error}", file=sys.stderr)
        for warning in self.warnings:
            if isinstance(warning, dict):
                print(f"WARNING: {warning.get('message', str(warning))}", file=sys.stderr)
            else:
                print(f"WARNING: {warning}", file=sys.stderr)


def check_pipeline_file(pipeline_path: str) -> dict:
    """
    Check a DocETL pipeline file.
    """
    checker = DocETLStaticChecker()
    return checker.check(pipeline_path)

def check_pipeline_string(yaml_content: str) -> dict:
    """
    Check a DocETL pipeline from YAML string.

    Args:
        yaml_content: Pipeline YAML content as string

    Returns:
        dict: Result with score and errors
    """
    checker = DocETLStaticChecker()
    return checker.check_string(yaml_content)

def main():
    """Main function to run the static checker."""
    if len(sys.argv) != 2:
        print(json.dumps({"score": 0, "errors": [{"type": "usage_error", "message": "Usage: python static_checker.py <pipeline.yaml>"}]}))
        sys.exit(1)
    
    pipeline_path = sys.argv[1]
    result = check_pipeline_file(pipeline_path)
    
    # Print the result as JSON
    print(json.dumps(result, indent=2))
    sys.exit(0 if result["score"] == 1 else 1)


if __name__ == "__main__":
    main()