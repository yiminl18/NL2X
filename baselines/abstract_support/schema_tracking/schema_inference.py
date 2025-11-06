"""
Schema inference rules for Abstract operators.

Defines how each abstract operator transforms input schema to output schema.
"""
from typing import Dict, Any


def compute_schema_transformation(
    operator_type: str,
    operator_properties: Dict[str, Any],
    operator_output: Dict[str, Any],
    current_schema: Dict[str, str]
) -> Dict[str, str]:
    """
    Compute output schema based on operator type and current schema.

    Args:
        operator_type: Type of operator (Convert, Project, Sample, etc.)
        operator_properties: Operator properties
        operator_output: Operator's output definition (may be empty)
        current_schema: Current schema (available fields)

    Returns:
        New schema after this operator
    """
    op_type = operator_type.lower()

    # Route to specific inference function based on operator type
    if op_type == 'convert':
        return infer_convert_output_schema(operator_properties, current_schema)
    elif op_type == 'project':
        return infer_project_output_schema(operator_properties, current_schema)
    elif op_type == 'sample':
        return infer_sample_output_schema(current_schema)
    elif op_type == 'unnest':
        return infer_unnest_output_schema(operator_properties, current_schema)
    elif op_type == 'split':
        return infer_split_output_schema(operator_properties, current_schema)
    elif op_type == 'gather':
        return infer_gather_output_schema(current_schema)
    elif op_type == 'pythoncode':
        return infer_pythoncode_output_schema(operator_output, current_schema)
    else:
        # Unknown operator: assume schema unchanged
        return current_schema.copy()


def infer_convert_output_schema(
    properties: Dict[str, Any],
    current_schema: Dict[str, str]
) -> Dict[str, str]:
    """
    Convert operator: Format conversion, schema unchanged.

    CSV → JSON or JSON → CSV: Field names and types remain the same.
    """
    return current_schema.copy()


def infer_project_output_schema(
    properties: Dict[str, Any],
    current_schema: Dict[str, str]
) -> Dict[str, str]:
    """
    Project operator: Keep only specified fields.

    Removes all fields except those listed in properties['fields'].
    """
    fields = properties.get('fields', [])
    return {f: current_schema[f] for f in fields if f in current_schema}


def infer_sample_output_schema(current_schema: Dict[str, str]) -> Dict[str, str]:
    """
    Sample operator: Schema unchanged.

    Sampling records doesn't change field structure.
    """
    return current_schema.copy()


def infer_unnest_output_schema(
    properties: Dict[str, Any],
    current_schema: Dict[str, str]
) -> Dict[str, str]:
    """
    Unnest operator: Flatten list/dict field.

    - List[Dict{...}]: Expands dict fields to top level
    - Dict{...}: Expands specified fields to top level
    - List[primitive]: Replaces list with primitive type
    """
    output = current_schema.copy()
    unnest_key = properties.get('unnest_key')

    if not unnest_key or unnest_key not in current_schema:
        return output  # Can't infer without valid key

    unnest_type = current_schema[unnest_key]

    # Simple heuristic: If type is List[Dict] or Dict, we'll expand fields
    # For now, just keep the schema as-is (complex type handling)
    # TODO: Implement full type parsing if needed

    # Check for expand_fields (Dict unnesting)
    expand_fields = properties.get('expand_fields', [])
    if expand_fields:
        for field in expand_fields:
            # Add expanded fields (assume String type for simplicity)
            output[field] = 'String'

    return output


def infer_split_output_schema(
    properties: Dict[str, Any],
    current_schema: Dict[str, str]
) -> Dict[str, str]:
    """
    Split operator: Add metadata fields for chunking.

    Adds:
    - _chunk_text: The chunk content
    - _split_id: Document unique ID
    - _chunk_index: Chunk position (0-based)
    - _total_chunks: Total number of chunks
    """
    output = current_schema.copy()
    output.update({
        '_chunk_text': 'String',
        '_split_id': 'String',
        '_chunk_index': 'Integer',
        '_total_chunks': 'Integer'
    })
    return output


def infer_gather_output_schema(current_schema: Dict[str, str]) -> Dict[str, str]:
    """
    Gather operator: Add context field.

    Adds:
    - _chunk_with_context: Chunk text with surrounding context
    """
    output = current_schema.copy()
    output['_chunk_with_context'] = 'String'
    return output


def infer_pythoncode_output_schema(
    operator_output: Dict[str, Any],
    current_schema: Dict[str, str]
) -> Dict[str, str]:
    """
    PythonCode operator: Use explicit output schema.

    Output schema is specified in operator.output.
    If not specified, assume schema unchanged.
    """
    if operator_output and isinstance(operator_output, dict):
        # Filter out metadata fields (those starting with _)
        output_fields = {k: v for k, v in operator_output.items()
                        if not k.startswith('_')}
        if output_fields:
            return output_fields

    # Fallback: schema unchanged
    return current_schema.copy()
