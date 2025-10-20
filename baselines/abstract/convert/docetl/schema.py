"""
Schema and Type Tracking Module for DocETL Conversion

This module provides utilities for extracting, parsing, and tracking type information
from DocETL operators. It handles:
- Field extraction from Jinja2 templates and Python code
- DocETL-specific type mapping to abstract layer types
- Input/output schema extraction from operators
"""

from typing import Any, Dict, Set, Optional
import re
import sys
sys.path.append('../../type')
from type import TypeSystem, serialize_type_dict


def _extract_fields_from_jinja2(template: str) -> Set[str]:
    """Extract field names from Jinja2 template patterns."""
    if not template:
        return set()

    fields = set()

    input_pattern = r'\{\{\s*input\.(\w+)'
    fields.update(re.findall(input_pattern, template))
    input_bracket_pattern = r'\{\{\s*input\["([^"]+)"\]'
    fields.update(re.findall(input_bracket_pattern, template))
    input_bracket_single_pattern = r"\{\{\s*input\['([^']+)'\]"
    fields.update(re.findall(input_bracket_single_pattern, template))

    input_n_pattern = r'\{\{\s*input[12]\.(\w+)'
    fields.update(re.findall(input_n_pattern, template))
    input_n_bracket_pattern = r'\{\{\s*input[12]\["([^"]+)"\]'
    fields.update(re.findall(input_n_bracket_pattern, template))

    inputs_idx_pattern = r'\{\{\s*inputs\[\d+\]\.(\w+)'
    fields.update(re.findall(inputs_idx_pattern, template))

    for_loop_pattern = r'\{%\s*for\s+(\w+)\s+in\s+inputs\s*%\}'
    loop_vars = re.findall(for_loop_pattern, template)

    for var in loop_vars:
        escaped_var = re.escape(var)
        var_pattern = r'\{\{?\s*' + escaped_var + r'\.(\w+)'
        fields.update(re.findall(var_pattern, template))
        var_bracket_pattern = r'\{\{?\s*' + escaped_var + r'\["([^"]+)"\]'
        fields.update(re.findall(var_bracket_pattern, template))

    for_entry_pattern = r'\{%\s*for\s+(\w+)\s+in\s+\w+\s*%\}'
    entry_vars = re.findall(for_entry_pattern, template)
    for var in entry_vars:
        if var not in loop_vars:
            escaped_var = re.escape(var)
            var_pattern = r'\{\{?\s*' + escaped_var + r'\.(\w+)'
            fields.update(re.findall(var_pattern, template))

    return fields


def _extract_fields_from_python_code(code: str) -> Set[str]:
    """Extract field names from Python code patterns."""
    if not code:
        return set()

    fields = set()

    doc_bracket_pattern = r'doc\[[\'"]([\w_]+)[\'"]\]'
    fields.update(re.findall(doc_bracket_pattern, code))

    doc_get_pattern = r'doc\.get\([\'"]([^\'",]+)[\'"]'
    fields.update(re.findall(doc_get_pattern, code))

    doc_dot_pattern = r'doc\.(?!get\b)(\w+)'
    fields.update(re.findall(doc_dot_pattern, code))

    input_bracket_pattern = r'input\[[\'"]([\w_]+)[\'"]\]'
    fields.update(re.findall(input_bracket_pattern, code))

    input_dot_pattern = r'input\.(\w+)'
    fields.update(re.findall(input_dot_pattern, code))

    return fields


class DocETLTypeMapper:
    """Maps DocETL type strings to abstract layer types."""

    DOCETL_TO_ABSTRACT = {
        "str": "String",
        "string": "String",
        "int": "Integer",
        "integer": "Integer",
        "number": "Integer",  # Default number to Integer
        "float": "Float",
        "bool": "Boolean",
        "boolean": "Boolean",
        "dict": "Dict",
        "object": "Dict",
        "array": "List",
        "list": "List"
    }

    @classmethod
    def parse_docetl_type(cls, type_str: str) -> str:
        """Parse simple DocETL type to abstract layer type name."""
        if not type_str:
            return "Unknown"

        type_str = str(type_str).strip()

        # Handle list types (simple check)
        if type_str.startswith("list[") or type_str.startswith("List["):
            return "List"

        # Map basic types
        lower_type = type_str.lower()
        if lower_type in cls.DOCETL_TO_ABSTRACT:
            return cls.DOCETL_TO_ABSTRACT[lower_type]

        # If it contains dictionary-like structure
        if "{" in type_str and "}" in type_str:
            return "Dict"

        return "Unknown"

    @classmethod
    def parse_docetl_schema(cls, schema_str: str) -> Dict[str, Any]:
        """Parse DocETL schema to abstract layer type dict."""
        if not schema_str:
            return {'type': 'Unknown'}

        schema_str = str(schema_str).strip()

        # Handle list types with nested structures
        if schema_str.startswith("list[") or schema_str.startswith("List["):
            inner_match = re.search(r'[Ll]ist\[(.+)\]$', schema_str)
            if inner_match:
                element_str = inner_match.group(1).strip()

                # Check if element is a dict/object
                if element_str.startswith("{") and element_str.endswith("}"):
                    fields_dict = {}
                    field_pairs = re.findall(r'(\w+)\s*:\s*(\w+)', element_str)
                    for field_name, field_type in field_pairs:
                        fields_dict[field_name] = cls.parse_docetl_schema(field_type)

                    return {
                        'type': 'List',
                        'element_type': {
                            'type': 'Dict',
                            'fields': fields_dict
                        }
                    }
                else:
                    return {
                        'type': 'List',
                        'element_type': cls.parse_docetl_schema(element_str)
                    }

        # Handle dict/object types
        if schema_str.startswith("{") and schema_str.endswith("}"):
            fields_dict = {}
            field_pairs = re.findall(r'(\w+)\s*:\s*(\w+)', schema_str)
            for field_name, field_type in field_pairs:
                fields_dict[field_name] = cls.parse_docetl_schema(field_type)
            return {'type': 'Dict', 'fields': fields_dict}

        # Handle basic types using the mapping
        lower_type = schema_str.lower()
        if lower_type in cls.DOCETL_TO_ABSTRACT:
            return {'type': cls.DOCETL_TO_ABSTRACT[lower_type]}

        return {'type': 'Unknown'}

    @classmethod
    def to_abstract_type_string(cls, docetl_schema: str) -> str:
        """Convert DocETL schema to abstract layer type string."""
        type_dict = cls.parse_docetl_schema(docetl_schema)
        return serialize_type_dict(type_dict)

    @classmethod
    def from_abstract_type_string(cls, abstract_type: str) -> str:
        """Convert abstract layer type string to DocETL type string.

        Args:
            abstract_type: Abstract type string (e.g., 'String', 'List[String]', 'Dict[{name: String}]')

        Returns:
            DocETL type string (e.g., 'str', 'list[str]', 'dict')

        Examples:
            'String' -> 'str'
            'Integer' -> 'int'
            'List[String]' -> 'list[str]'
            'Dict' -> 'dict'
            'Dict[{name: String, age: Integer}]' -> 'dict'  # DocETL doesn't support nested dict schemas
        """
        if not abstract_type:
            return 'str'  # Default to string

        abstract_type = abstract_type.strip()

        # Basic type mappings (reverse of DOCETL_TO_ABSTRACT)
        ABSTRACT_TO_DOCETL = {
            'String': 'str',
            'Integer': 'int',
            'Float': 'float',
            'Boolean': 'bool',
            'Dict': 'dict',
            'Unknown': 'str'  # Default unknown to string
        }

        # Handle basic types
        if abstract_type in ABSTRACT_TO_DOCETL:
            return ABSTRACT_TO_DOCETL[abstract_type]

        # Handle List types: List[ElementType]
        if abstract_type.startswith('List[') and abstract_type.endswith(']'):
            # Extract element type
            element_type_start = abstract_type.index('[') + 1
            element_type_end = abstract_type.rindex(']')
            element_type = abstract_type[element_type_start:element_type_end].strip()

            # Recursively convert element type
            docetl_element_type = cls.from_abstract_type_string(element_type)
            return f'list[{docetl_element_type}]'

        # Handle Dict types with fields: Dict[{...}]
        # DocETL doesn't support structured dict schemas in output.schema,
        # so we just return 'dict'
        if abstract_type.startswith('Dict[{') and abstract_type.endswith('}]'):
            return 'dict'

        # Unknown format, default to string
        return 'str'


def _extract_input_schema(docetl_operator: Dict[str, Any], field_types: Dict[str, str] = None, dataset_schema: Dict[str, Any] = None) -> Dict[str, Any]:
    """Extract input schema from DocETL operator by analyzing configuration and templates."""
    input_schema = {}
    op_type = docetl_operator.get("type", "")
    required_fields = set()
    extracted_fields = set()

    if field_types is None:
        field_types = {}

    # Process based on operator type
    if op_type in ["map", "filter"]:
        # Extract fields from prompt
        prompt = docetl_operator.get("prompt", "")
        extracted_fields.update(_extract_fields_from_jinja2(prompt))
        input_schema["type"] = "object"

    elif op_type in ["code_map", "code_filter"]:
        # Extract fields from Python code
        code = docetl_operator.get("code", "")
        extracted_fields.update(_extract_fields_from_python_code(code))
        input_schema["type"] = "object"

    elif op_type == "reduce":
        # Reduce requires a reduce_key and fields from prompt
        reduce_key = docetl_operator.get("reduce_key")
        if reduce_key:
            # reduce_key can be a string or a list
            if isinstance(reduce_key, list):
                required_fields.update(reduce_key)
            else:
                required_fields.add(reduce_key)

        prompt = docetl_operator.get("prompt", "")
        extracted_fields.update(_extract_fields_from_jinja2(prompt))
        input_schema["type"] = "array"

    elif op_type in ["code_reduce"]:
        # Code reduce: reduce_key + fields from code
        reduce_key = docetl_operator.get("reduce_key")
        if reduce_key:
            required_fields.add(reduce_key)

        code = docetl_operator.get("code", "")
        extracted_fields.update(_extract_fields_from_python_code(code))
        input_schema["type"] = "array"

    elif op_type == "resolve":
        # Resolve uses comparison_prompt and resolution_prompt
        comparison_prompt = docetl_operator.get("comparison_prompt", "")
        resolution_prompt = docetl_operator.get("resolution_prompt", "")

        # Extract from both prompts
        extracted_fields.update(_extract_fields_from_jinja2(comparison_prompt))
        extracted_fields.update(_extract_fields_from_jinja2(resolution_prompt))
        input_schema["type"] = "array"

    elif op_type == "split":
        # Split requires a split_key
        split_key = docetl_operator.get("split_key")
        if split_key:
            required_fields.add(split_key)
        input_schema["type"] = "object"

    elif op_type == "gather":
        # Gather requires specific keys
        content_key = docetl_operator.get("content_key")
        doc_id_key = docetl_operator.get("doc_id_key")
        order_key = docetl_operator.get("order_key")

        if content_key:
            required_fields.add(content_key)
        if doc_id_key:
            required_fields.add(doc_id_key)
        if order_key:
            required_fields.add(order_key)

        # Also check for doc_header_key and other peripheral keys
        doc_header_key = docetl_operator.get("doc_header_key")
        if doc_header_key:
            extracted_fields.add(doc_header_key)

        input_schema["type"] = "object"

    elif op_type == "unnest":
        # Unnest requires an unnest_key
        unnest_key = docetl_operator.get("unnest_key")
        if unnest_key:
            required_fields.add(unnest_key)
        input_schema["type"] = "object"

    elif op_type == "rank":
        # Rank uses input_keys and may have a prompt
        input_keys = docetl_operator.get("input_keys", [])
        required_fields.update(input_keys)

        prompt = docetl_operator.get("prompt", "")
        if prompt:
            extracted_fields.update(_extract_fields_from_jinja2(prompt))

        input_schema["type"] = "array"

    elif op_type == "cluster":
        # Cluster uses embedding_keys
        embedding_keys = docetl_operator.get("embedding_keys", [])
        required_fields.update(embedding_keys)

        # Check for summary_prompt
        summary_prompt = docetl_operator.get("summary_prompt", "")
        if summary_prompt:
            extracted_fields.update(_extract_fields_from_jinja2(summary_prompt))

        input_schema["type"] = "object"

    elif op_type == "extract":
        # Extract uses document_keys and prompt
        document_keys = docetl_operator.get("document_keys", [])
        required_fields.update(document_keys)

        prompt = docetl_operator.get("prompt", "")
        if prompt:
            extracted_fields.update(_extract_fields_from_jinja2(prompt))

        input_schema["type"] = "object"

    elif op_type == "topk":
        # TopK uses keys field
        keys = docetl_operator.get("keys", [])
        required_fields.update(keys)
        input_schema["type"] = "array"

    elif op_type == "sample":
        # Sample may have stratify_key
        stratify_key = docetl_operator.get("stratify_key")
        if stratify_key:
            extracted_fields.add(stratify_key)
        input_schema["type"] = "array"

    elif op_type == "join":
        # Join uses comparison_prompt for input schema
        comparison_prompt = docetl_operator.get("comparison_prompt", "")
        extracted_fields.update(_extract_fields_from_jinja2(comparison_prompt))
        input_schema["type"] = "object"

    elif op_type == "equijoin":
        # Equijoin has left_key and right_key
        left_key = docetl_operator.get("left_key")
        right_key = docetl_operator.get("right_key")
        if left_key:
            required_fields.add(left_key)
        if right_key:
            required_fields.add(right_key)
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


def _extract_output_schema(docetl_operator: Dict[str, Any], cumulative_field_types: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Extract output schema from DocETL operator, preserving full type information."""
    output_schema = {}

    # Check for nested output.schema format
    if "output" in docetl_operator and isinstance(docetl_operator["output"], dict):
        if "schema" in docetl_operator["output"]:
            schema = docetl_operator["output"]["schema"]
            if isinstance(schema, dict):
                for field, field_type_str in schema.items():
                    output_schema[field] = DocETLTypeMapper.to_abstract_type_string(field_type_str)
            else:
                output_schema = schema

    # Check for direct output_schema field (might be present before transformation)
    elif "output_schema" in docetl_operator:
        schema = docetl_operator["output_schema"]
        if isinstance(schema, dict):
            for field, field_type_str in schema.items():
                output_schema[field] = DocETLTypeMapper.to_abstract_type_string(field_type_str)
        else:
            output_schema = schema

    # For certain operators, we can infer output structure
    op_type = docetl_operator.get("type", "")

    if op_type == "extract":
        # Extract produces fields based on document_keys with '_extracted_findings' suffix
        if "document_keys" in docetl_operator:
            document_keys = docetl_operator.get("document_keys", [])
            if isinstance(document_keys, list) and document_keys:
                for doc_key in document_keys:
                    extracted_field = f"{doc_key}_extracted_findings"
                    output_schema[extracted_field] = "String"

    elif op_type == "gather":
        # Gather produces {content_key}_rendered field
        content_key = docetl_operator.get("content_key", "")
        if content_key:
            output_schema[f"{content_key}_rendered"] = "String"

    elif op_type == "cluster" and "output_key" in docetl_operator:
        # Cluster adds a new field with the cluster ID
        output_key = docetl_operator["output_key"]
        if not output_schema:
            output_schema = {}
        output_schema[output_key] = "String"  # Cluster ID is typically a string

    elif op_type == "rank":
        # Rank adds a _rank field
        if not output_schema:
            output_schema = {"_rank": "Integer"}

    elif op_type == "split":
        # Split adds {split_key}_chunk, {op_name}_id, and {op_name}_chunk_num
        split_key = docetl_operator.get("split_key", "")
        op_name = docetl_operator.get("name", "split")

        if split_key:
            output_schema[f"{split_key}_chunk"] = "String"
        output_schema[f"{op_name}_id"] = "String"
        output_schema[f"{op_name}_chunk_num"] = "Integer"

    elif op_type == "unnest":
        # Unnest transforms the unnest_key field based on its type
        unnest_key = docetl_operator.get("unnest_key")

        if unnest_key and cumulative_field_types and unnest_key in cumulative_field_types:
            field_type_str = cumulative_field_types[unnest_key]
            type_dict = DocETLTypeMapper.parse_docetl_schema(field_type_str)

            # Handle List types
            if type_dict.get('type') == 'List':
                element_type = type_dict.get('element_type', {})

                # If List[Dict], unwrap and add dict fields
                if element_type.get('type') == 'Dict':
                    output_schema[unnest_key] = serialize_type_dict(element_type)
                    element_fields = element_type.get('fields', {})
                    for field_name, field_type_dict in element_fields.items():
                        output_schema[field_name] = serialize_type_dict(field_type_dict)
                else:
                    output_schema[unnest_key] = serialize_type_dict(element_type)

            # Handle Dict types
            elif type_dict.get('type') == 'Dict':
                dict_fields = type_dict.get('fields', {})
                for field_name, field_type_dict in dict_fields.items():
                    output_schema[field_name] = serialize_type_dict(field_type_dict)
                output_schema['_removed_fields'] = [unnest_key]

    return output_schema
