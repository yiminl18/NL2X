"""
DocETL to Abstract Operator Conversion Module

This module provides utilities to convert DocETL operators to abstract operators.
"""

from typing import Any, Dict, List, Optional, Union, Set
import copy
import sys
import os
import yaml
import json
import argparse
import re
from pathlib import Path

# Add parent directory to path to import Operator class
_current_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_current_dir)
sys.path.insert(0, _parent_dir)
sys.path.insert(0, os.path.join(_parent_dir, 'tools'))

# Try relative imports first, fall back to absolute
try:
    from ..ops.base import Operator
    from ..tools.docetl_pipeline_parser import parse_pipeline_operators
    from ..tools.static_checker import validate_pipeline
    from ..pipeline import Pipeline, PipelineNode
except ImportError:
    # Fallback for direct script execution
    # Import Operator directly
    from ops.base import Operator

    # Import parser from tools
    from docetl_pipeline_parser import parse_pipeline_operators

    # Try to import static checker
    try:
        from static_checker import validate_pipeline
    except ImportError:
        # Define a stub if not available
        def validate_pipeline(operators, verbose=False):
            return {'is_valid': True, 'errors': [], 'warnings': [], 'summary': {}}

    # For Pipeline, we need to manually create a minimal version
    # since it has relative imports that can't be resolved in standalone mode
    # We'll only use operators in standalone mode, not the full DAG
    try:
        from pipeline import Pipeline, PipelineNode
    except:
        # Define minimal stubs if pipeline can't be imported
        Pipeline = None
        PipelineNode = None


# Mapping from DocETL operator types to Abstract operator types
DOCETL_TO_ABSTRACT_TYPE_MAP = {
    "map": "Map",
    "code_map": "Map",

    "filter": "Filter",
    "code_filter": "Filter",

    "reduce": "Reduce",
    "code_reduce": "Reduce",

    "equijoin": "Join",

    "rank": "Rank",

    "topk": "TopK",

    "resolve": "Resolve",

    "cluster": "Cluster",

    "extract": "Extract",

    "split": "Split",

    "gather": "Gather",

    "unnest": "Unnest",

    "sample": "Sample"
}


def docetl_to_abstract(docetl_operator: Dict[str, Any], field_types: Optional[Dict[str, str]] = None, dataset_schema: Optional[Dict[str, Any]] = None) -> Operator:
    """
    Convert a DocETL operator to an abstract operator.

    Args:
        docetl_operator: A dictionary representing a DocETL operator
        field_types: Optional dictionary mapping field names to their types
        dataset_schema: Optional dataset schema with field definitions

    Returns:
        An abstract Operator instance

    Example:
        >>> docetl_op = {
        ...     "name": "extract_info",
        ...     "type": "map",
        ...     "prompt": "Extract information from the text",
        ...     "output": {"schema": {"title": "string", "summary": "string"}}
        ... }
        >>> abstract_op = docetl_to_abstract(docetl_op)
    """
    # Create a new abstract operator instance
    abstract_op = Operator()

    # Copy the operator to avoid modifying the original
    docetl_op = copy.deepcopy(docetl_operator)

    # Set the name
    abstract_op.name = docetl_op.get("name", "")

    # Map the type
    docetl_type = docetl_op.get("type", "")
    abstract_op.type = DOCETL_TO_ABSTRACT_TYPE_MAP.get(docetl_type, docetl_type.capitalize())

    # Set the source information
    abstract_op.source = {
        "system": "docetl",
        "name": docetl_op.get("name", "")
    }

    # Extract and process properties
    # All fields except 'name' and 'type' go into properties
    properties = {}
    for key, value in docetl_op.items():
        if key not in ["name", "type"]:
            properties[key] = value

    # Special handling for output schema structure
    # DocETL uses nested format: {"output": {"schema": {...}}}
    # We'll keep it in properties as is for now

    abstract_op.properties = properties

    # Handle input/output schemas
    abstract_op.input = _extract_input_schema(docetl_op, field_types, dataset_schema)
    abstract_op.output = _extract_output_schema(docetl_op)

    return abstract_op


def _extract_fields_from_jinja2(template: str) -> Set[str]:
    """
    Extract field names from Jinja2 template patterns.

    This function identifies field references in various Jinja2 patterns:
    - {{ input.field }} - single input field access
    - {{ input1.field }}, {{ input2.field }} - comparison inputs
    - {{ inputs[0].field }} - array index access
    - {% for item in inputs %} {{ item.field }} - loop variable access

    Args:
        template: Jinja2 template string

    Returns:
        Set of unique field names found in the template
    """
    if not template:
        return set()

    fields = set()

    # Pattern for {{ input.field }} or {{ input["field"] }}
    input_pattern = r'\{\{\s*input\.(\w+)'
    fields.update(re.findall(input_pattern, template))
    input_bracket_pattern = r'\{\{\s*input\["([^"]+)"\]'
    fields.update(re.findall(input_bracket_pattern, template))
    input_bracket_single_pattern = r"\{\{\s*input\['([^']+)'\]"
    fields.update(re.findall(input_bracket_single_pattern, template))

    # Pattern for {{ input1.field }}, {{ input2.field }} (resolve comparison)
    input_n_pattern = r'\{\{\s*input[12]\.(\w+)'
    fields.update(re.findall(input_n_pattern, template))
    input_n_bracket_pattern = r'\{\{\s*input[12]\["([^"]+)"\]'
    fields.update(re.findall(input_n_bracket_pattern, template))

    # Pattern for {{ inputs[0].field }} (reduce/resolve)
    inputs_idx_pattern = r'\{\{\s*inputs\[\d+\]\.(\w+)'
    fields.update(re.findall(inputs_idx_pattern, template))

    # Pattern for {% for item in inputs %} ... {{ item.field }}
    # First find loop variables
    for_loop_pattern = r'\{%\s*for\s+(\w+)\s+in\s+inputs\s*%\}'
    loop_vars = re.findall(for_loop_pattern, template)

    # Then find fields accessed through loop variables
    for var in loop_vars:
        escaped_var = re.escape(var)
        var_pattern = r'\{\{?\s*' + escaped_var + r'\.(\w+)'
        fields.update(re.findall(var_pattern, template))
        var_bracket_pattern = r'\{\{?\s*' + escaped_var + r'\["([^"]+)"\]'
        fields.update(re.findall(var_bracket_pattern, template))

    # Pattern for {% for entry in inputs %} or similar variations
    for_entry_pattern = r'\{%\s*for\s+(\w+)\s+in\s+\w+\s*%\}'
    entry_vars = re.findall(for_entry_pattern, template)
    for var in entry_vars:
        if var not in loop_vars:  # Avoid duplicates
            escaped_var = re.escape(var)
            var_pattern = r'\{\{?\s*' + escaped_var + r'\.(\w+)'
            fields.update(re.findall(var_pattern, template))

    return fields


def _extract_fields_from_python_code(code: str) -> Set[str]:
    """
    Extract field names from Python code patterns.

    This function identifies field references in Python code:
    - doc['field'] or doc["field"]
    - doc.get('field') or doc.get("field")
    - input['field'] or input.field

    Args:
        code: Python code string

    Returns:
        Set of unique field names found in the code
    """
    if not code:
        return set()

    fields = set()

    # Pattern for doc['field'] or doc["field"]
    doc_bracket_pattern = r'doc\[[\'"]([\w_]+)[\'"]\]'
    fields.update(re.findall(doc_bracket_pattern, code))

    # Pattern for doc.get('field') or doc.get("field") - must be before doc.field pattern
    doc_get_pattern = r'doc\.get\([\'"]([^\'",]+)[\'"]'
    fields.update(re.findall(doc_get_pattern, code))

    # Pattern for doc.field (but exclude .get method)
    # Use negative lookahead to exclude 'get'
    doc_dot_pattern = r'doc\.(?!get\b)(\w+)'
    fields.update(re.findall(doc_dot_pattern, code))

    # Pattern for input['field'] or input["field"]
    input_bracket_pattern = r'input\[[\'"]([\w_]+)[\'"]\]'
    fields.update(re.findall(input_bracket_pattern, code))

    # Pattern for input.field
    input_dot_pattern = r'input\.(\w+)'
    fields.update(re.findall(input_dot_pattern, code))

    return fields


def _parse_docetl_type(type_str: str) -> str:
    """
    Parse DocETL type strings to a simplified string format.

    Handles types like:
    - "string" or "str" -> "String"
    - "number" or "int" or "integer" -> "Integer"
    - "float" -> "Float"
    - "list[str]" -> "Array"
    - "list[{theme: str, viewpoints: str}]" -> "Array"
    - Complex nested types -> simplified representation

    Args:
        type_str: DocETL type string

    Returns:
        Normalized type string compatible with TypeSystem
    """
    if not type_str:
        return "Unknown"

    # Normalize the type string
    type_str = str(type_str).strip()

    # Handle list types
    if type_str.startswith("list[") or type_str.startswith("List["):
        return "Array"

    # Handle basic type mappings (matching TypeSystem format)
    type_mappings = {
        "str": "String",
        "string": "String",
        "int": "Integer",
        "integer": "Integer",
        "float": "Float",
        "number": "Integer",  # Default number to Integer
        "bool": "Boolean",
        "boolean": "Boolean",
        "dict": "Dict",
        "object": "Dict",
        "array": "Array",
        "list": "Array"
    }

    # Check for exact match
    lower_type = type_str.lower()
    if lower_type in type_mappings:
        return type_mappings[lower_type]

    # If it contains dictionary-like structure
    if "{" in type_str and "}" in type_str:
        return "Dict"

    # Default to Unknown for unrecognized types
    return "Unknown"


def _parse_docetl_schema_to_type_dict(schema_str: str) -> Dict[str, Any]:
    """
    Parse DocETL schema strings to type dict format compatible with TypeSystem.

    Examples:
    - "str" -> {'type': 'String'}
    - "list[str]" -> {'type': 'List', 'element_type': {'type': 'String'}}
    - "list[{theme: str, viewpoints: str}]" ->
      {'type': 'List', 'element_type': {'type': 'Dict', 'fields': {...}}}

    Args:
        schema_str: DocETL schema string

    Returns:
        Type dict in TypeSystem format
    """
    if not schema_str:
        return {'type': 'Unknown'}

    schema_str = str(schema_str).strip()

    # Handle list types with nested structures
    if schema_str.startswith("list[") or schema_str.startswith("List["):
        # Extract the element type
        inner_match = re.search(r'[Ll]ist\[(.+)\]$', schema_str)
        if inner_match:
            element_str = inner_match.group(1).strip()

            # Check if element is a dict/object
            if element_str.startswith("{") and element_str.endswith("}"):
                # Parse dict fields
                fields_dict = {}
                # Extract field:type pairs from {field1: type1, field2: type2}
                field_pairs = re.findall(r'(\w+)\s*:\s*(\w+)', element_str)
                for field_name, field_type in field_pairs:
                    fields_dict[field_name] = _parse_docetl_schema_to_type_dict(field_type)

                return {
                    'type': 'List',
                    'element_type': {
                        'type': 'Dict',
                        'fields': fields_dict
                    }
                }
            else:
                # Simple element type
                return {
                    'type': 'List',
                    'element_type': _parse_docetl_schema_to_type_dict(element_str)
                }

    # Handle dict/object types
    if schema_str.startswith("{") and schema_str.endswith("}"):
        fields_dict = {}
        field_pairs = re.findall(r'(\w+)\s*:\s*(\w+)', schema_str)
        for field_name, field_type in field_pairs:
            fields_dict[field_name] = _parse_docetl_schema_to_type_dict(field_type)
        return {'type': 'Dict', 'fields': fields_dict}

    # Handle basic types
    type_mappings = {
        "str": "String",
        "string": "String",
        "int": "Integer",
        "integer": "Integer",
        "float": "Float",
        "number": "Integer",
        "bool": "Boolean",
        "boolean": "Boolean",
        "dict": "Dict",
        "object": "Dict",
        "array": "List",
        "list": "List"
    }

    lower_type = schema_str.lower()
    if lower_type in type_mappings:
        return {'type': type_mappings[lower_type]}

    return {'type': 'Unknown'}


def _type_dict_to_docetl_string(type_dict: Dict[str, Any]) -> str:
    """
    Convert type dict to DocETL-style type string representation.

    Preserves full type information including nested structures.

    Args:
        type_dict: Type dict in TypeSystem format

    Returns:
        DocETL-style type string (e.g., "List[String]", "Dict[theme: String, viewpoints: String]")
    """
    if not type_dict:
        return "Unknown"

    type_name = type_dict.get('type', 'Unknown')

    # For simple types, just return the type name
    if type_name in ['String', 'Integer', 'Float', 'Boolean', 'Unknown', 'Null']:
        return type_name

    # For lists, include element type
    if type_name == 'List':
        element_type = type_dict.get('element_type', {'type': 'Unknown'})
        element_str = _type_dict_to_docetl_string(element_type)
        return f"List[{element_str}]"

    # For dicts, include field definitions
    if type_name == 'Dict':
        fields = type_dict.get('fields', {})
        if fields:
            field_strs = []
            for field_name, field_type in fields.items():
                field_type_str = _type_dict_to_docetl_string(field_type)
                field_strs.append(f"{field_name}: {field_type_str}")
            return f"Dict[{{{', '.join(field_strs)}}}]"
        return "Dict"

    return type_name




def _extract_input_schema(docetl_operator: Dict[str, Any], field_types: Optional[Dict[str, str]] = None, dataset_schema: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Extract input schema from a DocETL operator.

    This function analyzes operator configuration and prompts to identify
    all input fields required by the operator. It uses template parsing
    to extract field references from Jinja2 templates.

    Args:
        docetl_operator: DocETL operator dictionary
        field_types: Optional dictionary mapping field names to their types
        dataset_schema: Optional dataset schema with field definitions

    Returns:
        Input schema dictionary with fields metadata
    """
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

    elif op_type == "equijoin":
        # Equijoin would have left_key and right_key
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


def _extract_output_schema(docetl_operator: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract output schema from a DocETL operator.

    DocETL operators may define output schemas in different ways:
    - Nested format: {"output": {"schema": {...}}}
    - Direct output_schema field (after processing in docetl_step.py)

    Args:
        docetl_operator: DocETL operator dictionary

    Returns:
        Output schema dictionary with full type information preserved
    """
    output_schema = {}

    # Check for nested output.schema format
    if "output" in docetl_operator and isinstance(docetl_operator["output"], dict):
        if "schema" in docetl_operator["output"]:
            schema = docetl_operator["output"]["schema"]
            if isinstance(schema, dict):
                # Convert each field type to preserve nested structure
                for field, field_type_str in schema.items():
                    # Parse to type dict and back to DocETL string to normalize
                    type_dict = _parse_docetl_schema_to_type_dict(field_type_str)
                    output_schema[field] = _type_dict_to_docetl_string(type_dict)
            else:
                output_schema = schema

    # Check for direct output_schema field (might be present before transformation)
    elif "output_schema" in docetl_operator:
        schema = docetl_operator["output_schema"]
        if isinstance(schema, dict):
            for field, field_type_str in schema.items():
                type_dict = _parse_docetl_schema_to_type_dict(field_type_str)
                output_schema[field] = _type_dict_to_docetl_string(type_dict)
        else:
            output_schema = schema

    # For certain operators, we can infer output structure
    op_type = docetl_operator.get("type", "")

    if op_type == "cluster" and "output_key" in docetl_operator:
        # Cluster adds a new field with the cluster ID
        output_key = docetl_operator["output_key"]
        if not output_schema:
            output_schema = {}
        output_schema[output_key] = "String"  # Cluster ID is typically a string

    elif op_type == "rank":
        # Rank adds a rank field
        if not output_schema:
            output_schema = {"rank": "Integer"}

    elif op_type == "split":
        # Split creates multiple documents from one
        output_schema["type"] = "array"

    elif op_type == "unnest":
        # Unnest expands arrays into individual items
        output_schema["type"] = "array"

    return output_schema


def docetl_pipeline_to_abstract(docetl_pipeline: List[Dict[str, Any]], dataset_schema: Optional[Dict[str, Any]] = None) -> List[Operator]:
    """
    Convert a list of DocETL operators (a pipeline) to abstract operators.

    This function tracks field types as they flow through the pipeline,
    ensuring that types defined in outputs are propagated to downstream inputs.

    Args:
        docetl_pipeline: List of DocETL operator dictionaries
        dataset_schema: Optional dataset schema with field definitions

    Returns:
        List of abstract Operator instances with proper type tracking
    """
    abstract_operators = []
    cumulative_field_types = {}  # Track all field types available at each step

    # Initialize cumulative field types from dataset schema if provided
    if dataset_schema and "fields" in dataset_schema:
        cumulative_field_types.update(dataset_schema["fields"])

    for i, docetl_op in enumerate(docetl_pipeline):
        # Convert operator with current field types and dataset schema
        abstract_op = docetl_to_abstract(docetl_op, cumulative_field_types, dataset_schema)
        abstract_operators.append(abstract_op)

        # Update cumulative field types based on operator type and output
        op_type = docetl_op.get("type", "")

        # Handle operators with output schemas
        if "output" in docetl_op and isinstance(docetl_op["output"], dict):
            if "schema" in docetl_op["output"]:
                schema = docetl_op["output"]["schema"]
                if isinstance(schema, dict):
                    for field, field_type_str in schema.items():
                        # Parse the schema to type dict, then convert to DocETL string
                        type_dict = _parse_docetl_schema_to_type_dict(field_type_str)
                        cumulative_field_types[field] = _type_dict_to_docetl_string(type_dict)

        # Special handling for unnest operations
        if op_type == "unnest":
            unnest_key = docetl_op.get("unnest_key")
            if unnest_key:
                # Look for the unnest key's type definition from previous operators
                for prev_i, prev_op in enumerate(docetl_pipeline[:i]):
                    if "output" in prev_op and isinstance(prev_op["output"], dict):
                        if "schema" in prev_op["output"] and unnest_key in prev_op["output"]["schema"]:
                            schema_str = prev_op["output"]["schema"][unnest_key]
                            type_dict = _parse_docetl_schema_to_type_dict(schema_str)

                            # If it's a list with dict elements, extract the nested fields
                            if (type_dict.get('type') == 'List' and
                                type_dict.get('element_type', {}).get('type') == 'Dict'):
                                element_fields = type_dict['element_type'].get('fields', {})
                                for field_name, field_type_dict in element_fields.items():
                                    # Add the unnested fields to cumulative types
                                    cumulative_field_types[field_name] = _type_dict_to_docetl_string(field_type_dict)

                # After unnesting, the original list field might be removed
                # But we preserve it for now as it might still be referenced

        # Handle other special operators
        elif op_type == "split":
            # Split adds standard fields
            cumulative_field_types["_split_id"] = "String"
            cumulative_field_types["_split_index"] = "Integer"

        elif op_type == "gather" and "output_key" in docetl_op:
            # Gather produces a list field
            cumulative_field_types[docetl_op["output_key"]] = "Array"

        elif op_type == "cluster" and "output_key" in docetl_op:
            # Cluster adds a cluster ID field
            cumulative_field_types[docetl_op["output_key"]] = "String"

        elif op_type == "rank":
            # Rank adds a rank field
            cumulative_field_types["rank"] = "Integer"

        # Update abstract operator's output if we detected special fields
        if abstract_op.output and isinstance(abstract_op.output, dict):
            for field, field_type in abstract_op.output.items():
                if field != "type":  # Skip special keys
                    # Ensure output types are consistent
                    if field in cumulative_field_types:
                        abstract_op.output[field] = cumulative_field_types[field]

    return abstract_operators


def abstract_to_docetl(abstract_operator: Operator) -> Dict[str, Any]:
    """
    Convert an abstract operator back to a DocETL operator.

    Args:
        abstract_operator: An abstract Operator instance

    Returns:
        A dictionary representing a DocETL operator

    Note:
        This is a reverse conversion that may not perfectly recreate
        the original DocETL operator if information was lost during
        the initial conversion.
    """
    docetl_op = {}

    # Set the name
    docetl_op["name"] = abstract_operator.name

    # Reverse map the type
    # Create reverse mapping
    abstract_to_docetl_type = {v: k for k, v in DOCETL_TO_ABSTRACT_TYPE_MAP.items()}

    # Use the first matching DocETL type for each abstract type
    # For types that map to multiple DocETL types (e.g., Map -> map/code_map),
    # we'll need to check properties to determine which one to use
    abstract_type = abstract_operator.type

    if abstract_type == "Map":
        # Check if there's code in properties to determine map vs code_map
        if "code" in abstract_operator.properties:
            docetl_op["type"] = "code_map"
        else:
            docetl_op["type"] = "map"
    elif abstract_type == "Filter":
        # Check if there's code in properties to determine filter vs code_filter
        if "code" in abstract_operator.properties:
            docetl_op["type"] = "code_filter"
        else:
            docetl_op["type"] = "filter"
    elif abstract_type == "Reduce":
        # Check if there's code in properties to determine reduce vs code_reduce
        if "code" in abstract_operator.properties:
            docetl_op["type"] = "code_reduce"
        else:
            docetl_op["type"] = "reduce"
    else:
        # For other types, use the direct mapping
        docetl_op["type"] = abstract_to_docetl_type.get(abstract_type, abstract_type.lower())

    # Copy all properties back
    for key, value in abstract_operator.properties.items():
        docetl_op[key] = value

    return docetl_op


def yaml_to_abstract_pipeline(yaml_file_path: Union[str, Path], dataset_schema: Optional[Dict[str, Any]] = None) -> List[Operator]:
    """
    Convert a DocETL YAML pipeline file to a list of abstract operators.

    This function uses the docetl_pipeline_parser to extract operators
    from a YAML file and converts each to an abstract Operator instance.

    Args:
        yaml_file_path: Path to the DocETL YAML pipeline file
        dataset_schema: Optional dataset schema with field definitions

    Returns:
        List of abstract Operator instances in pipeline execution order

    Raises:
        FileNotFoundError: If the YAML file doesn't exist
        ValueError: If the YAML structure is invalid

    Example:
        >>> operators = yaml_to_abstract_pipeline("pipeline.yaml")
        >>> for op in operators:
        ...     print(f"{op.name}: {op.type}")
    """
    # Parse operators from YAML file using the parser tool
    docetl_operators = parse_pipeline_operators(str(yaml_file_path))

    # Convert each DocETL operator to abstract operator
    abstract_operators = docetl_pipeline_to_abstract(docetl_operators, dataset_schema)

    return abstract_operators


def yaml_to_abstract_pipeline_dag(yaml_file_path: Union[str, Path], dataset_schema: Optional[Dict[str, Any]] = None) -> tuple[List[Operator], Pipeline]:
    """
    Convert a DocETL YAML pipeline file to abstract operators and a Pipeline DAG.

    This function reads the complete YAML structure and creates:
    1. A list of abstract operators
    2. A Pipeline instance representing the DAG structure

    Args:
        yaml_file_path: Path to the DocETL YAML pipeline file
        dataset_schema: Optional dataset schema with field definitions

    Returns:
        Tuple of (operators list, Pipeline instance)

    Raises:
        FileNotFoundError: If the YAML file doesn't exist
        ValueError: If the YAML structure is invalid
    """
    yaml_path = Path(yaml_file_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"YAML file not found: {yaml_file_path}")

    # Read the full YAML structure
    with open(yaml_path, 'r') as f:
        pipeline_config = yaml.safe_load(f)

    # Extract operators and convert to abstract format
    docetl_operators = parse_pipeline_operators(str(yaml_file_path))
    abstract_operators = docetl_pipeline_to_abstract(docetl_operators, dataset_schema)

    # Create Pipeline instance
    pipeline = Pipeline(name=pipeline_config.get('name', 'docetl_pipeline'))

    # Extract pipeline steps to understand the DAG structure
    pipeline_steps = pipeline_config.get('pipeline', {}).get('steps', [])

    # For linear pipelines (most common in DocETL)
    if pipeline_steps:
        first_step = pipeline_steps[0]
        operation_names = first_step.get('operations', [])

        # Create a mapping of operator names to abstract operators
        op_name_to_abstract = {op.name: op for op in abstract_operators}

        # Add operators to pipeline in order
        previous_node_id = None
        for i, op_name in enumerate(operation_names):
            if op_name in op_name_to_abstract:
                abstract_op = op_name_to_abstract[op_name]
                node_id = f"{op_name}_{i}"

                # Add operator to pipeline
                pipeline.add_operator(abstract_op, node_id)

                # Connect to previous operator if exists (linear chain)
                if previous_node_id:
                    pipeline.add_edge(previous_node_id, node_id)

                previous_node_id = node_id

    # Add metadata about datasets and output
    if 'datasets' in pipeline_config:
        for dataset_name, dataset_config in pipeline_config['datasets'].items():
            # Store dataset info in pipeline metadata if needed
            pass

    return abstract_operators, pipeline


def convert_yaml_or_operators(input_data: Union[str, Path, List[Dict[str, Any]]], dataset_schema: Optional[Dict[str, Any]] = None) -> List[Operator]:
    """
    Flexible conversion function that handles both YAML files and operator lists.

    Args:
        input_data: Either a path to a YAML file or a list of DocETL operators
        dataset_schema: Optional dataset schema with field definitions

    Returns:
        List of abstract Operator instances

    Example:
        >>> # From YAML file
        >>> operators = convert_yaml_or_operators("pipeline.yaml")
        >>>
        >>> # From operator list
        >>> docetl_ops = [{"name": "map_op", "type": "map", ...}]
        >>> operators = convert_yaml_or_operators(docetl_ops)
    """
    if isinstance(input_data, (str, Path)):
        # It's a file path, convert from YAML
        return yaml_to_abstract_pipeline(input_data, dataset_schema)
    elif isinstance(input_data, list):
        # It's already a list of operators
        return docetl_pipeline_to_abstract(input_data, dataset_schema)
    else:
        raise TypeError(f"Expected str, Path, or list, got {type(input_data)}")


# ============================================================================
# JSON Serialization Functions
# ============================================================================

def operator_to_dict(operator: Operator) -> Dict[str, Any]:
    """
    Convert an Operator instance to a dictionary for JSON serialization.

    Args:
        operator: An Operator instance

    Returns:
        Dictionary representation of the operator

    Example:
        >>> op = Operator()
        >>> op.name = "my_op"
        >>> op.type = "Map"
        >>> op_dict = operator_to_dict(op)
    """
    return {
        "name": operator.name,
        "type": operator.type,
        "source": operator.source,
        "properties": operator.properties,
        "input": operator.input,
        "output": operator.output
    }


def dict_to_operator(op_dict: Dict[str, Any]) -> Operator:
    """
    Convert a dictionary to an Operator instance.

    Args:
        op_dict: Dictionary representation of an operator

    Returns:
        Operator instance

    Example:
        >>> op_dict = {"name": "my_op", "type": "Map", ...}
        >>> op = dict_to_operator(op_dict)
    """
    operator = Operator()
    operator.name = op_dict.get("name", "")
    operator.type = op_dict.get("type", "")
    operator.source = op_dict.get("source", {"system": "", "name": ""})
    operator.properties = op_dict.get("properties", {})
    operator.input = op_dict.get("input", {})
    operator.output = op_dict.get("output", {})
    return operator


# ============================================================================
# Standalone Conversion Functions
# ============================================================================

def yaml_to_abstract_json(yaml_path: Union[str, Path], output_path: Union[str, Path], validate: bool = False, dataset_schema: Optional[Union[Dict[str, Any], str, Path]] = None) -> None:
    """
    Convert a DocETL YAML pipeline file to abstract JSON representation.

    This function reads a YAML pipeline file, converts it to abstract operators,
    optionally validates the pipeline, and saves the result as a JSON file.

    Args:
        yaml_path: Path to the input YAML pipeline file
        output_path: Path where the output JSON file should be saved
        validate: Whether to validate the pipeline after conversion
        dataset_schema: Optional dataset schema (Dict or path to JSON file)

    Example:
        >>> yaml_to_abstract_json("pipeline.yaml", "abstract_pipeline.json", validate=True)
    """
    yaml_path = Path(yaml_path)
    output_path = Path(output_path)

    # Load dataset schema if provided as a file path
    schema_dict = None
    if dataset_schema:
        if isinstance(dataset_schema, (str, Path)):
            schema_path = Path(dataset_schema)
            if schema_path.exists():
                with open(schema_path, 'r') as f:
                    schema_dict = json.load(f)
                print(f"Loaded dataset schema from: {schema_path}")
            else:
                raise FileNotFoundError(f"Dataset schema file not found: {schema_path}")
        elif isinstance(dataset_schema, dict):
            schema_dict = dataset_schema
        else:
            raise ValueError("dataset_schema must be a dict or path to JSON file")

    # Convert YAML to abstract operators
    print(f"Reading YAML pipeline from: {yaml_path}")
    abstract_operators = yaml_to_abstract_pipeline(yaml_path, schema_dict)
    print(f"Converted {len(abstract_operators)} operators to abstract representation")

    # Serialize operators to dictionaries
    operators_data = {
        "operators": [operator_to_dict(op) for op in abstract_operators]
    }

    # Include dataset schema in output if provided
    if schema_dict:
        operators_data["dataset_schema"] = schema_dict

    # Validate if requested
    if validate:
        print("\n" + "=" * 60)
        print("VALIDATING ABSTRACT PIPELINE")
        if schema_dict:
            print("Mode: STRICT (dataset schema provided)")
        else:
            print("Mode: NORMAL")
        print("=" * 60)

        validation_result = validate_pipeline(operators_data["operators"], dataset_schema=schema_dict)

        # Print summary
        print(f"\nValidation Result: {'VALID ✓' if validation_result['is_valid'] else 'INVALID ✗'}")
        print(f"Total Errors: {len(validation_result['errors'])}")
        print(f"Total Warnings: {len(validation_result['warnings'])}")

        # Print errors if any
        if validation_result['errors']:
            print("\nERRORS:")
            for i, error in enumerate(validation_result['errors'], 1):
                print(f"  {i}. {error}")

        # Print warnings if any
        if validation_result['warnings']:
            print("\nWARNINGS:")
            for i, warning in enumerate(validation_result['warnings'], 1):
                print(f"  {i}. {warning}")

        # If invalid, ask for confirmation to continue
        if not validation_result['is_valid']:
            print("\n⚠️  Pipeline validation failed!")
            try:
                response = input("Do you want to save the pipeline anyway? (y/N): ")
                if response.lower() != 'y':
                    print("Aborting save operation.")
                    return
            except (EOFError, KeyboardInterrupt):
                print("\nAborting save operation due to validation errors.")
                return

        print("=" * 60 + "\n")

    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save to JSON file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(operators_data, f, indent=2, ensure_ascii=False)

    print(f"Saved abstract pipeline to: {output_path}")


# ============================================================================
# Custom YAML Presenter for Multi-line Strings
# ============================================================================

def literal_presenter(dumper, data):
    """Custom presenter to use literal style for multi-line strings."""
    if '\n' in data:
        return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='|')
    return dumper.represent_scalar('tag:yaml.org,2002:str', data)


class LiteralDumper(yaml.SafeDumper):
    """Custom YAML dumper that uses literal style for multi-line strings."""
    pass


# Add the custom representer to the dumper
LiteralDumper.add_representer(str, literal_presenter)


def abstract_json_to_yaml(json_path: Union[str, Path], output_path: Union[str, Path]) -> None:
    """
    Convert an abstract JSON representation to DocETL YAML pipeline (operations only).

    This function reads an abstract JSON file, converts it back to DocETL operators,
    and saves the result as a YAML file containing only the operations section.
    If dataset schema is present in the JSON, it will be included as a comment.

    Args:
        json_path: Path to the input JSON file
        output_path: Path where the output YAML file should be saved
    Example:
        >>> abstract_json_to_yaml("abstract_pipeline.json", "pipeline_ops.yaml")
    """
    json_path = Path(json_path)
    output_path = Path(output_path)

    # Load JSON file
    print(f"Reading abstract pipeline from: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Extract dataset schema if present
    dataset_schema = data.get("dataset_schema")
    if dataset_schema:
        print(f"Found dataset schema with {len(dataset_schema.get('fields', {}))} fields")

    # Deserialize operators
    abstract_operators = [dict_to_operator(op_dict) for op_dict in data.get("operators", [])]
    print(f"Loaded {len(abstract_operators)} abstract operators")

    # Convert to DocETL format
    docetl_operators = [abstract_to_docetl(op) for op in abstract_operators]

    # Create YAML structure with only operations section
    pipeline_yaml = {
        "operations": docetl_operators
    }

    # Add dataset schema as a comment section if present
    yaml_content = ""
    if dataset_schema:
        yaml_content += "# Dataset Schema Information\n"
        yaml_content += "# This pipeline expects the following fields in the input dataset:\n"
        for field_name, field_type in dataset_schema.get("fields", {}).items():
            yaml_content += f"#   - {field_name}: {field_type}\n"
        yaml_content += "\n"

    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save to YAML file with schema comments if present
    with open(output_path, 'w', encoding='utf-8') as f:
        if yaml_content:
            f.write(yaml_content)
        yaml.dump(pipeline_yaml, f, Dumper=LiteralDumper,
                  default_flow_style=False, sort_keys=False, allow_unicode=True)

    print(f"Saved DocETL operations to: {output_path}")


# ============================================================================
# CLI Interface
# ============================================================================

def main():
    """
    Command-line interface for standalone conversion operations.

    Usage:
        # Convert YAML to abstract JSON
        python docetl.py --to-abstract input.yaml output.json

        # Convert abstract JSON to YAML
        python docetl.py --to-docetl input.json output.yaml
    """
    parser = argparse.ArgumentParser(
        description="Convert between DocETL YAML pipelines and abstract JSON representations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert YAML pipeline to abstract JSON
  python docetl.py --to-abstract examples/example1.yaml output/abstract.json

  # Convert abstract JSON back to YAML (operations only)
  python docetl.py --to-docetl output/abstract.json output/pipeline_ops.yaml
        """
    )

    # Add mutually exclusive group for conversion direction
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--to-abstract',
        action='store_true',
        help='Convert DocETL YAML to abstract JSON representation'
    )
    group.add_argument(
        '--to-docetl',
        action='store_true',
        help='Convert abstract JSON to DocETL YAML (operations only)'
    )

    # Add validation flag
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Validate the abstract pipeline after conversion (only with --to-abstract)'
    )

    # Add dataset schema option
    parser.add_argument(
        '--dataset-schema',
        type=str,
        help='Path to JSON file containing dataset schema (enables strict validation when used with --validate)'
    )

    # Add positional arguments for input and output files
    parser.add_argument(
        'input_file',
        type=str,
        help='Input file path (YAML or JSON depending on conversion direction)'
    )
    parser.add_argument(
        'output_file',
        type=str,
        help='Output file path (JSON or YAML depending on conversion direction)'
    )

    args = parser.parse_args()

    try:
        if args.to_abstract:
            # Convert YAML to abstract JSON
            yaml_to_abstract_json(args.input_file, args.output_file,
                                validate=args.validate,
                                dataset_schema=args.dataset_schema)
        elif args.to_docetl:
            # Convert abstract JSON to YAML
            if args.validate:
                print("Warning: --validate flag is only supported with --to-abstract, ignoring.")
            if args.dataset_schema:
                print("Warning: --dataset-schema flag is only supported with --to-abstract, ignoring.")
            abstract_json_to_yaml(args.input_file, args.output_file)

        print("\n✓ Conversion completed successfully!")

    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error during conversion: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()