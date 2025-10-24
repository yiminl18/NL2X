"""
DocETL Schema Inference

Infer input requirements and output schema for DocETL operators.
"""

from typing import Dict, Any, Set, Optional
from .complex_types import parse_complex_type

def _infer_docetl_output_schema(
    operator_config: Dict[str, Any],
    current_schema: Dict[str, str]
) -> Dict[str, str]:
    """
    Infer output schema for DocETL operator.

    Args:
        operator_config: DocETL operator configuration
        current_schema: Current schema (cumulative field types)

    Returns:
        Output schema dictionary
    """
    output_schema = {}

    # For most operators, output schema is explicitly defined
    if "output" in operator_config and isinstance(operator_config["output"], dict):
        if "schema" in operator_config["output"]:
            schema = operator_config["output"]["schema"]
            if isinstance(schema, dict):
                for field, field_type_str in schema.items():
                    output_schema[field] = field_type_str  # Keep DocETL format internally
            else:
                output_schema = schema
    elif "output_schema" in operator_config:
        schema = operator_config["output_schema"]
        if isinstance(schema, dict):
            for field, field_type_str in schema.items():
                output_schema[field] = field_type_str  # Keep DocETL format internally
        else:
            output_schema = schema

    # For certain operators, we can infer output structure based on schema.md
    op_type = operator_config.get("type", "")

    if op_type == "extract":
        if "document_keys" in operator_config:
            document_keys = operator_config.get("document_keys", [])
            if isinstance(document_keys, list) and document_keys:
                for doc_key in document_keys:
                    extracted_field = f"{doc_key}_extracted_findings"
                    output_schema[extracted_field] = "str"

    elif op_type == "gather":
        content_key = operator_config.get("content_key", "")
        if content_key:
            output_schema[f"{content_key}_rendered"] = "str"

    elif op_type == "cluster":
        output_key = operator_config.get("output_key", "clusters")
        summary_schema = operator_config.get("summary_schema", {})

        if isinstance(summary_schema, dict):
            # Format summary_schema into a type string (keep DocETL format)
            summary_fields_str = ", ".join(
                f"{k}: {v}"
                for k, v in summary_schema.items()
            )
            # Add distance field
            cluster_type_str = f"{{{summary_fields_str}, distance: number}}"
        else:
            # Fallback if summary_schema is not a dict
            cluster_type_str = "{distance: number}"

        output_schema[output_key] = cluster_type_str

    elif op_type == "rank":
        output_schema["_rank"] = "int"

    elif op_type == "split":
        split_key = operator_config.get("split_key", "")
        op_name = operator_config.get("name", "split")

        if split_key:
            output_schema[f"{split_key}_chunk"] = "str"
        output_schema[f"{op_name}_id"] = "str"
        output_schema[f"{op_name}_chunk_num"] = "int"

    elif op_type == "unnest":
        # Per user request, keep recursive and depth handling
        unnest_key = operator_config.get("unnest_key")

        if unnest_key and unnest_key in current_schema:
            field_type_str = current_schema[unnest_key]
            recursive = operator_config.get('recursive', False)
            depth = operator_config.get('depth')

            unnest_output = compute_unnest_transformation(
                unnest_key, field_type_str, recursive, depth
            )
            output_schema.update(unnest_output)

    return output_schema


def compute_unnest_transformation(
    unnest_key: str,
    parent_type: str,
    recursive: bool = False,
    depth: Optional[int] = None
) -> Dict[str, str]:
    """
    Compute schema changes after unnest operation.

    Args:
        unnest_key: The field being unnested
        parent_type: The type of the field (e.g., 'list[{name: str}]')
        recursive: Whether recursive unnest is enabled
        depth: Depth of unnesting (if specified)

    Returns:
        Dict mapping field names to their new types after unnest
        - For non-recursive: {unnest_key: inner_type}
        - For recursive: {field1: type1, field2: type2, ...} (unnest_key removed)

    Examples:
        >>> compute_unnest_transformation('meds', 'list[{name: str, dose: str}]', False)
        {'meds': '{name: str, dose: str}'}

        >>> compute_unnest_transformation('meds', 'list[{name: str, dose: str}]', True)
        {'name': 'str', 'dose': 'str'}
    """
    result = {}

    # Parse the parent type to understand its structure
    type_info = parse_complex_type(parent_type)

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

def compute_schema_transformation(
    operator_config: Dict[str, Any],
    current_schema: Dict[str, str]
) -> Dict[str, str]:
    """
    Compute full schema transformation for DocETL operator.

    Args:
        operator_config: DocETL operator configuration
        current_schema: Current schema

    Returns:
        Complete schema after operator
    """
    op_type = operator_config.get('type', '')
    new_schema = current_schema.copy()

    # Get output schema
    output_schema = _infer_docetl_output_schema(operator_config, current_schema)

    # Apply transformation based on operator type
    if op_type in ['map', 'code_map', 'filter', 'code_filter']:
        # Map/Filter: preserve all input fields + add output fields
        new_schema.update(output_schema)

    elif op_type in ['reduce', 'code_reduce']:
        # Reduce: preserve all input fields + add output fields
        new_schema.update(output_schema)

    elif op_type == 'resolve':
        # Resolve: typically outputs merged records with same or modified schema
        new_schema.update(output_schema)

    elif op_type == 'extract':
        # Extract: add extracted fields to existing schema
        new_schema.update(output_schema)

    elif op_type == 'unnest':
        # Unnest: complex transformation
        unnest_key = operator_config.get('unnest_key')
        if unnest_key and unnest_key in new_schema:
            parent_type = new_schema[unnest_key]
            recursive = operator_config.get('recursive', False)
            depth = operator_config.get('depth')

            # Compute unnest transformation
            unnest_changes = compute_unnest_transformation(
                unnest_key, parent_type, recursive, depth
            )

            # Determine if parent field should be removed
            type_info = parse_complex_type(parent_type)
            should_remove_parent = (
                type_info['base_type'] == 'dict' or
                (type_info['base_type'] == 'list' and recursive)
            )

            if should_remove_parent:
                del new_schema[unnest_key]

            # Add/update fields from unnest
            new_schema.update(unnest_changes)

    elif op_type == 'split':
        # Split: add metadata fields
        new_schema.update(output_schema)

    elif op_type == 'gather':
        # Gather: add rendered field
        new_schema.update(output_schema)

    elif op_type == 'cluster':
        # Cluster: add cluster identifier
        new_schema.update(output_schema)

    elif op_type == 'rank':
        # Rank: add rank field
        new_schema.update(output_schema)

    elif op_type == 'join':
        # Join: merge schemas (simplified - actual join would need second dataset schema)
        new_schema.update(output_schema)

    elif op_type == 'sample':
        # Sample: doesn't change schema
        pass

    else:
        # Unknown operator - try to use output schema if available
        if output_schema:
            new_schema.update(output_schema)

    return new_schema
