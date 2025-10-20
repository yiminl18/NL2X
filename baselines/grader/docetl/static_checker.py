#!/usr/bin/env python3
"""
Static checker for DocETL pipelines.
Validates YAML syntax, operator usage, and schema consistency.

Usage:
    Command line:
        python3 static_checker.py <pipeline.yaml>
    
    Python API:
        from static_checker import check_pipeline_file, check_pipeline_string
        
        # Check from file path
        result = check_pipeline_file("pipeline.yaml")
        
        # Check from YAML string
        yaml_content = "default_model: gpt-4o-mini\n..."
        result = check_pipeline_string(yaml_content)
        
    Returns:
        dict: {"score": 1|0, "errors": [...], "warnings": [...]}
"""

import sys
import re
import json
from typing import Dict, Set, Any

try:
    from .pipeline_obfuscator import PipelineObfuscator
except ImportError:
    # Handle case when running as standalone script
    try:
        from pipeline_obfuscator import PipelineObfuscator
    except ImportError:
        PipelineObfuscator = None


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
        
        # Try to use yaml if available, otherwise fall back to custom parser
        try:
            import yaml
            return yaml.safe_load(content)
        except ImportError:
            # Fallback: Use a very basic custom parser
            # This is limited but sufficient for basic validation
            print("WARNING: PyYAML not available, using limited parser", file=sys.stderr)
            
            # For simplicity, we'll just check structure
            result = {}
            current_key = None
            current_list = None
            indent_stack = [0]
            
            for line in content.split('\n'):
                if not line.strip():
                    continue
                    
                # Check for key-value pairs
                if ':' in line:
                    parts = line.split(':', 1)
                    key = parts[0].strip().strip('-').strip()
                    value = parts[1].strip() if len(parts) > 1 else ''
                    
                    indent = len(line) - len(line.lstrip())
                    
                    if indent == 0:
                        # Top level key
                        if value:
                            result[key] = value.strip('"').strip("'")
                        else:
                            result[key] = {}
                        current_key = key
                    
            # Basic structure validation
            if 'default_model' not in result:
                result['default_model'] = None
            if 'datasets' not in result:
                result['datasets'] = {}
            if 'operations' not in result:
                result['operations'] = []
            if 'pipeline' not in result:
                result['pipeline'] = {}
                
            return result
    except Exception as e:
        raise ValueError(f"Failed to parse YAML: {str(e)}")


class DocETLStaticChecker:
    """Static checker for DocETL pipeline YAML files."""
    
    # Define valid operator types based on the prompt.py documentation
    VALID_OPERATORS = {
        'map', 'filter', 'reduce', 'resolve', 'order', 'extract', 
        'cluster', 'split', 'gather', 'unnest', 'sample', 'topk',
        'code_map', 'code_filter', 'code_reduce', 'equijoin', 'scan',
        'parallel_map', 'link_resolve', 'add_uuid'
    }
    
    # Universal required fields that must appear in all operators
    UNIVERSAL_REQUIRED_FIELDS = {'name', 'type'}
    
    # Required fields for each operator type
    OPERATOR_REQUIREMENTS = {
        # LLM-powered operations
        'map': set(),  # Only needs name/type (universal); prompt/output conditional based on drop_keys
        'filter': {'prompt', 'output'},  # Always required
        'reduce': {'reduce_key', 'prompt', 'output'},
        'resolve': {'comparison_prompt'},  # resolution_prompt is optional, output is optional
        'order': {'prompt', 'input_keys', 'direction'},  # Based on rank.py schema
        'extract': {'prompt', 'document_keys'},  # Based on extract.py schema
        'cluster': {'embedding_keys', 'summary_prompt', 'summary_schema'},  # Based on cluster.py syntax_check
        'parallel_map': {'prompts', 'output'},  # Based on usage patterns
        'equijoin': {'comparison_prompt'},  # Based on equijoin.py schema
        # Auxiliary operations
        'split': {'split_key', 'method', 'method_kwargs'},  # Based on split.py schema
        'gather': {'content_key', 'doc_id_key', 'order_key'},  # Based on gather.py schema
        'unnest': {'unnest_key'},  # Based on unnest.py schema
        'sample': {'method'},  # samples can be optional with defaults in some cases
        'topk': {'method', 'k', 'keys', 'query'},  # Based on topk.py schema
        # Code operations
        'code_map': {'code'},  # Based on code_operations.py schema
        'code_filter': {'code'},  # Based on code_operations.py - inherits from code_map but needs code
        'code_reduce': {'code'},  # Based on code_operations.py schema
        # Special operations
        'scan': {'dataset_name'},  # Based on scan.py schema - not prompt/output
        'link_resolve': {'comparison_prompt'},  # Based on link_resolve.py - resolution_prompt/output optional
        'add_uuid': set()  # Based on add_uuid.py schema - no required fields beyond universal
    }

    # Universal optional fields that can appear in any operator 
    UNIVERSAL_OPTIONAL_FIELDS = {
        'gleaning',         # per-operator validation hooks where supported
        'skip_on_error'     # commonly supported fallback behavior
    }

    # Operator-specific optional fields
    OPERATOR_OPTIONAL_FIELDS = {
        'map': {
            # Based on map.py schema
            'output', 'prompt', 'model', 'optimize', 'recursively_optimize', 'sample_size',
            'tools', 'validation_rules', 'num_retries_on_validate_failure',
            'drop_keys', 'timeout', 'enable_observability', 'batch_size', 'clustering_method',
            'batch_prompt', 'litellm_completion_kwargs', 'pdf_url_key', 'flush_partial_result',
            'calibrate', 'num_calibration_docs'
        },
        'filter': {
            # Inherits from map.py schema but prompt/output are required
            'model', 'optimize', 'recursively_optimize', 'sample_size',
            'tools', 'validation_rules', 'num_retries_on_validate_failure',
            'drop_keys', 'timeout', 'enable_observability', 'batch_size', 'clustering_method',
            'batch_prompt', 'litellm_completion_kwargs', 'pdf_url_key', 'flush_partial_result',
            'calibrate', 'num_calibration_docs'
        },
        'reduce': {
            # Based on reduce.py schema
            'optimize', 'synthesize_resolve', 'model', 'input', 'pass_through',
            'associative', 'fold_prompt', 'fold_batch_size', 'merge_prompt',
            'merge_batch_size', 'value_sampling', 'verbose', 'timeout',
            'litellm_completion_kwargs', 'enable_observability'
        },
        'resolve': {
            # Based on resolve.py schema
            'resolution_prompt', 'output', 'embedding_model', 'resolution_model', 'comparison_model',
            'blocking_keys', 'blocking_threshold', 'blocking_conditions',
            'input', 'embedding_batch_size', 'compare_batch_size',
            'limit_comparisons', 'optimize', 'timeout', 'litellm_completion_kwargs',
            'enable_observability'
        },
        'extract': {
            # Based on extract.py schema
            'model', 'format_extraction', 'extraction_key_suffix', 'extraction_method',
            'timeout', 'litellm_completion_kwargs'
        },
        'cluster': {
            # Based on cluster.py syntax check
            'output_key', 'max_batch_size', 'embedding_model', 'model',
            'timeout', 'litellm_completion_kwargs'
        },
        'gather': {
            # Based on gather.py schema
            'peripheral_chunks', 'doc_header_key', 'main_chunk_start', 'main_chunk_end'
        },
        'split': {
            # Based on split.py schema
            'model'  # method_kwargs is required, not optional
        },
        'unnest': {
            # Based on unnest.py schema
            'keep_empty', 'expand_fields', 'recursive', 'depth'
        },
        'sample': {
            # Based on sample.py schema
            'samples', 'stratify_key', 'samples_per_group', 'method_kwargs', 'random_state'
        },
        'topk': {
            # Based on topk.py schema
            'stratify_key', 'embedding_model', 'model', 'batch_size'
        },
        'order': {
            # Based on rank.py schema
            'model', 'embedding_model', 'batch_size', 'initial_ordering_method', 'k',
            'rerank_call_budget', 'num_top_items_per_window', 'overlap_fraction',
            'timeout', 'num_calibration_docs', 'verbose', 'litellm_completion_kwargs'
        },
        'equijoin': {
            # Based on equijoin.py schema
            'output', 'blocking_threshold', 'blocking_conditions', 'limits',
            'comparison_model', 'optimize', 'embedding_model', 'embedding_batch_size',
            'compare_batch_size', 'limit_comparisons', 'blocking_keys', 'timeout',
            'litellm_completion_kwargs'
        },
        'parallel_map': {
            # Based on parallel_map usage patterns
            'model', 'optimize', 'recursively_optimize', 'timeout', 'litellm_completion_kwargs'
        },
        'code_map': {
            # Based on code_operations.py schema
            'concurrent_thread_count', 'drop_keys'
        },
        'code_filter': {
            # Based on code_operations.py - inherits from code_map
            'concurrent_thread_count', 'drop_keys'
        },
        'code_reduce': {
            # Based on code_operations.py schema
            'concurrent_thread_count', 'reduce_key', 'pass_through'
        },
        'scan': set(),  # Based on scan.py schema - dataset_name is required, not optional
        'link_resolve': {
            # Based on link_resolve.py implementation
            'id_key', 'link_key', 'blocking_threshold', 'blocking_conditions',
            'embedding_model', 'comparison_model', 'compare_batch_size', 'timeout',
            'validation_rules', 'verbose', 'litellm_completion_kwargs', 'resolution_prompt', 'output'
        },
        'add_uuid': {
            # Based on add_uuid.py schema
            'id_key'
        }
    }

    def __init__(self, enable_obfuscation: bool = False):
        self.errors = []
        self.warnings = []
        self.pipeline = None
        self.field_tracker = {}  # Track fields created/used by operations
        self.enable_obfuscation = enable_obfuscation
        
    def check(self, pipeline_path: str) -> dict:
        """
        Main checking function that accepts a file path.
        Returns dict with score and errors for machine parsing.
        """
        # Step 1: Check YAML validity - if this fails, we can't continue
        if not self._check_yaml_validity_from_path(pipeline_path):
            return self._format_result()
            
        # Step 2: Check operator syntax - continue even if errors found
        self._check_operator_syntax()
            
        # Step 3: Check schema consistency - continue even if errors found
        self._check_schema_consistency()
        
        # Return formatted result
        return self._format_result()
    
    def check_string(self, yaml_content: str) -> dict:
        """
        Main checking function that accepts YAML content as string.
        Returns dict with score and errors for machine parsing.
        """
        # Apply obfuscation if enabled
        if self.enable_obfuscation:
            yaml_content = self._obfuscate_pipeline_content(yaml_content)
        
        # Step 1: Check YAML validity - if this fails, we can't continue
        if not self._check_yaml_validity_from_string(yaml_content):
            return self._format_result()
            
        # Step 2: Check operator syntax - continue even if errors found
        self._check_operator_syntax()
            
        # Step 3: Check schema consistency - continue even if errors found
        self._check_schema_consistency()
        
        # Return formatted result
        return self._format_result()
    
    def _obfuscate_pipeline_content(self, yaml_content: str) -> str:
        """
        Obfuscate pipeline content using PipelineObfuscator.
        
        Args:
            yaml_content: Original YAML content
            
        Returns:
            str: Obfuscated YAML content, or original content if obfuscation fails
        """
        if PipelineObfuscator is None:
            self.warnings.append({
                "type": "obfuscation_unavailable",
                "message": "Pipeline obfuscation requested but PipelineObfuscator not available"
            })
            return yaml_content
            
        try:
            obfuscator = PipelineObfuscator()
            return obfuscator.obfuscate(yaml_content)
        except Exception as e:
            self.warnings.append({
                "type": "obfuscation_failed",
                "message": f"Pipeline obfuscation failed: {str(e)}"
            })
            return yaml_content
    
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
                
            # Check required top-level keys
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
                
            # Check operator name
            if 'name' not in op:
                self.errors.append({
                    "type": "missing_operation_name",
                    "operation_index": i,
                    "message": f"Operation {i} missing 'name' field"
                })
                has_errors = True
                continue
                
            op_name = op['name']
            
            # Check operator type
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
            
            # Check required fields for operator type (universal + operator-specific)
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
            
            # Check for unknown fields using operator-specific optional fields
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
            
            # Track input/output fields for schema consistency (must be done first)
            self._track_operation_fields(op_name, op_type, op)
            
            # Validate specific field constraints
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
        
        # Check reduce operations have proper reduce_key
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
            # Try to import Jinja2 if available
            try:
                from jinja2 import Template, TemplateSyntaxError
                Template(template)
                return True
            except ImportError:
                # Basic validation without Jinja2
                # Check for balanced braces
                open_count = template.count('{{')
                close_count = template.count('}}')
                if open_count != close_count:
                    return False
                # Check for basic syntax patterns
                if '{{' in template and '}}' in template:
                    # Basic check passed
                    return True
                # If no template markers, it's valid
                return True
        except Exception:
            return False
    
    def _validate_output_schema(self, schema: Any) -> bool:
        """Validate output schema format."""
        if not isinstance(schema, dict):
            return False
        
        valid_types = {'string', 'integer', 'number', 'boolean', 'str', 'int', 'float', 'bool'}
        
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
        
        # Extract output fields from schema
        if 'output' in op and isinstance(op['output'], dict):
            if 'schema' in op['output'] and isinstance(op['output']['schema'], dict):
                self.field_tracker[op_name]['outputs'].update(op['output']['schema'].keys())
        
        # Handle special operators
        if op_type == 'unnest' and 'unnest_key' in op:
            self.field_tracker[op_name]['inputs'].add(op['unnest_key'])
            # For unnest operations, we need to look at the schema of the field being unnested
            # to determine what fields become available
            unnest_key = op['unnest_key']
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
        """Extract field references from Jinja2 template."""
        fields = set()

        # Match patterns like {{ input.field }}, {{ inputs[0].field }}, etc.
        patterns = [
            r'{{\s*input\.(\w+)\s*}}',
            r'{{\s*inputs\[\d+\]\.(\w+)\s*}}',
            r'{{\s*item\.(\w+)\s*}}',
            r'{{\s*e\.(\w+)\s*}}',
            r'{{\s*input1\.(\w+)\s*}}',
            r'{{\s*input2\.(\w+)\s*}}'
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
                
            # Track available fields through the pipeline
            available_fields = set()
            
            # Get initial fields from dataset
            if 'input' in step:
                dataset_name = step['input']
                # Assume dataset provides basic fields (we can't check without loading data)
                # Add common fields that are typically in datasets
                available_fields.update(['text', 'content', 'id', 'title', 'date'])
            
            # Check each operation in sequence
            operations = step['operations']
            for i, op_ref in enumerate(operations):
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
                    missing_fields = required_inputs - available_fields
                    
                    # Allow some flexibility for common variations
                    flexible_missing = set()
                    for field in missing_fields:
                        # Check for common variations
                        if field in ['document', 'doc', 'item', 'record']:
                            # These might refer to the whole document
                            flexible_missing.add(field)
                        elif any(f in available_fields for f in [field + 's', field[:-1] if field.endswith('s') else field + 's']):
                            # Check singular/plural variations
                            flexible_missing.add(field)
                    
                    missing_fields -= flexible_missing
                    
                    if missing_fields and i > 0:  # Don't check first operation too strictly
                        # Create separate error for each missing field
                        for missing_field in missing_fields:
                            self.errors.append({
                                "type": "missing_schema_field",
                                "operation": op_name,
                                "field": missing_field,
                                "message": f"Operation '{op_name}' expects field '{missing_field}' which is not available from previous operations"
                            })
                        has_errors = True
                        # For continued checking, assume these missing fields exist
                        available_fields.update(missing_fields)
                    
                    # Add output fields to available fields
                    available_fields.update(self.field_tracker[op_name]['outputs'])
                    
                    # Handle special operations that preserve fields
                    op_type = op_def.get('type')
                    if op_type in ['map', 'filter', 'resolve']:
                        # These operations typically preserve input fields
                        available_fields.update(required_inputs)
                    elif op_type in ['reduce', 'code_reduce']:
                        # Reduce operations change the structure significantly
                        available_fields = self.field_tracker[op_name]['outputs'].copy()
                    elif op_type in ['code_map']:
                        # Code map operations replace the structure with their outputs
                        available_fields = self.field_tracker[op_name]['outputs'].copy()
                    elif op_type == 'unnest':
                        # Unnest operations make nested fields available
                        if 'nested_fields_from' in self.field_tracker[op_name]:
                            nested_fields = self._get_nested_fields_from_schema(op_name, available_fields)
                            available_fields.update(nested_fields)
        
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


def check_pipeline_file(pipeline_path: str, enable_obfuscation: bool = False) -> dict:
    """
    Check a DocETL pipeline file.
    
    Args:
        pipeline_path: Path to the pipeline YAML file
        enable_obfuscation: Whether to enable operator name obfuscation (default: False)
    
    Returns:
        dict: Result with score and errors
    """
    checker = DocETLStaticChecker(enable_obfuscation=enable_obfuscation)
    return checker.check(pipeline_path)

def check_pipeline_string(yaml_content: str, enable_obfuscation: bool = False) -> dict:
    """
    Check a DocETL pipeline from YAML string.
    
    Args:
        yaml_content: Pipeline YAML content as string
        enable_obfuscation: Whether to enable operator name obfuscation (default: False)
    
    Returns:
        dict: Result with score and errors
    """
    checker = DocETLStaticChecker(enable_obfuscation=enable_obfuscation)
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