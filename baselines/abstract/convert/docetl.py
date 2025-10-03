"""
DocETL to Abstract Operator Conversion Module

This module provides utilities to convert DocETL operators to abstract operators.
"""

from typing import Any, Dict, List, Optional, Union
import copy
import sys
import os
import yaml
import json
import argparse
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
    from ..pipeline import Pipeline, PipelineNode
except ImportError:
    # Fallback for direct script execution
    # Import Operator directly
    from ops.base import Operator

    # Import parser from tools
    from docetl_pipeline_parser import parse_pipeline_operators

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


def docetl_to_abstract(docetl_operator: Dict[str, Any]) -> Operator:
    """
    Convert a DocETL operator to an abstract operator.

    Args:
        docetl_operator: A dictionary representing a DocETL operator

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
    abstract_op.input = _extract_input_schema(docetl_op)
    abstract_op.output = _extract_output_schema(docetl_op)

    return abstract_op


def _extract_input_schema(docetl_operator: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract input schema from a DocETL operator.

    DocETL doesn't explicitly define input schemas in most cases,
    but we can infer them from the operator configuration.

    Args:
        docetl_operator: DocETL operator dictionary

    Returns:
        Input schema dictionary
    """
    input_schema = {}
    op_type = docetl_operator.get("type", "")

    # Infer input requirements based on operator type and configuration
    if op_type in ["map", "filter", "code_map", "code_filter"]:
        # These operate on individual documents
        # Input keys can be inferred from prompts or code if needed
        input_schema["type"] = "object"

    elif op_type == "reduce":
        # Reduce requires a reduce_key
        reduce_key = docetl_operator.get("reduce_key")
        if reduce_key:
            input_schema["required_fields"] = [reduce_key]
            input_schema["type"] = "array"

    elif op_type == "split":
        # Split requires a split_key
        split_key = docetl_operator.get("split_key")
        if split_key:
            input_schema["required_fields"] = [split_key]
            input_schema["type"] = "object"

    elif op_type == "gather":
        # Gather requires specific keys
        content_key = docetl_operator.get("content_key")
        doc_id_key = docetl_operator.get("doc_id_key")
        order_key = docetl_operator.get("order_key")
        required_fields = [k for k in [content_key, doc_id_key, order_key] if k]
        if required_fields:
            input_schema["required_fields"] = required_fields
            input_schema["type"] = "object"

    elif op_type == "unnest":
        # Unnest requires an unnest_key
        unnest_key = docetl_operator.get("unnest_key")
        if unnest_key:
            input_schema["required_fields"] = [unnest_key]
            input_schema["type"] = "object"

    elif op_type == "rank":
        # Rank uses input_keys
        input_keys = docetl_operator.get("input_keys", [])
        if input_keys:
            input_schema["required_fields"] = input_keys
            input_schema["type"] = "array"

    elif op_type == "cluster":
        # Cluster uses embedding_keys
        embedding_keys = docetl_operator.get("embedding_keys", [])
        if embedding_keys:
            input_schema["required_fields"] = embedding_keys
            input_schema["type"] = "object"

    elif op_type == "extract":
        # Extract uses document_keys
        document_keys = docetl_operator.get("document_keys", [])
        if document_keys:
            input_schema["required_fields"] = document_keys
            input_schema["type"] = "object"

    elif op_type == "topk":
        # TopK uses keys field
        keys = docetl_operator.get("keys", [])
        if keys:
            input_schema["required_fields"] = keys
            input_schema["type"] = "array"

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
        Output schema dictionary
    """
    output_schema = {}

    # Check for nested output.schema format
    if "output" in docetl_operator and isinstance(docetl_operator["output"], dict):
        if "schema" in docetl_operator["output"]:
            output_schema = docetl_operator["output"]["schema"]

    # Check for direct output_schema field (might be present before transformation)
    elif "output_schema" in docetl_operator:
        output_schema = docetl_operator["output_schema"]

    # For certain operators, we can infer output structure
    op_type = docetl_operator.get("type", "")

    if op_type == "cluster" and "output_key" in docetl_operator:
        # Cluster adds a new field with the cluster ID
        output_key = docetl_operator["output_key"]
        if not output_schema:
            output_schema = {}
        output_schema[output_key] = "string"  # Cluster ID is typically a string

    elif op_type == "rank":
        # Rank adds a rank field
        if not output_schema:
            output_schema = {"rank": "number"}

    elif op_type == "split":
        # Split creates multiple documents from one
        output_schema["type"] = "array"

    elif op_type == "unnest":
        # Unnest expands arrays into individual items
        output_schema["type"] = "array"

    return output_schema


def docetl_pipeline_to_abstract(docetl_pipeline: List[Dict[str, Any]]) -> List[Operator]:
    """
    Convert a list of DocETL operators (a pipeline) to abstract operators.

    Args:
        docetl_pipeline: List of DocETL operator dictionaries

    Returns:
        List of abstract Operator instances
    """
    abstract_operators = []
    for docetl_op in docetl_pipeline:
        abstract_op = docetl_to_abstract(docetl_op)
        abstract_operators.append(abstract_op)
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


def yaml_to_abstract_pipeline(yaml_file_path: Union[str, Path]) -> List[Operator]:
    """
    Convert a DocETL YAML pipeline file to a list of abstract operators.

    This function uses the docetl_pipeline_parser to extract operators
    from a YAML file and converts each to an abstract Operator instance.

    Args:
        yaml_file_path: Path to the DocETL YAML pipeline file

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
    abstract_operators = docetl_pipeline_to_abstract(docetl_operators)

    return abstract_operators


def yaml_to_abstract_pipeline_dag(yaml_file_path: Union[str, Path]) -> tuple[List[Operator], Pipeline]:
    """
    Convert a DocETL YAML pipeline file to abstract operators and a Pipeline DAG.

    This function reads the complete YAML structure and creates:
    1. A list of abstract operators
    2. A Pipeline instance representing the DAG structure

    Args:
        yaml_file_path: Path to the DocETL YAML pipeline file

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
    abstract_operators = docetl_pipeline_to_abstract(docetl_operators)

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


def convert_yaml_or_operators(input_data: Union[str, Path, List[Dict[str, Any]]]) -> List[Operator]:
    """
    Flexible conversion function that handles both YAML files and operator lists.

    Args:
        input_data: Either a path to a YAML file or a list of DocETL operators

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
        return yaml_to_abstract_pipeline(input_data)
    elif isinstance(input_data, list):
        # It's already a list of operators
        return docetl_pipeline_to_abstract(input_data)
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

def yaml_to_abstract_json(yaml_path: Union[str, Path], output_path: Union[str, Path]) -> None:
    """
    Convert a DocETL YAML pipeline file to abstract JSON representation.

    This function reads a YAML pipeline file, converts it to abstract operators,
    and saves the result as a JSON file containing only the operator definitions.

    Args:
        yaml_path: Path to the input YAML pipeline file
        output_path: Path where the output JSON file should be saved

    Example:
        >>> yaml_to_abstract_json("pipeline.yaml", "abstract_pipeline.json")
    """
    yaml_path = Path(yaml_path)
    output_path = Path(output_path)

    # Convert YAML to abstract operators
    print(f"Reading YAML pipeline from: {yaml_path}")
    abstract_operators = yaml_to_abstract_pipeline(yaml_path)
    print(f"Converted {len(abstract_operators)} operators to abstract representation")

    # Serialize operators to dictionaries
    operators_data = {
        "operators": [operator_to_dict(op) for op in abstract_operators]
    }

    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save to JSON file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(operators_data, f, indent=2, ensure_ascii=False)

    print(f"Saved abstract pipeline to: {output_path}")


def abstract_json_to_yaml(json_path: Union[str, Path], output_path: Union[str, Path]) -> None:
    """
    Convert an abstract JSON representation to DocETL YAML pipeline (operations only).

    This function reads an abstract JSON file, converts it back to DocETL operators,
    and saves the result as a YAML file containing only the operations section.

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

    # Deserialize operators
    abstract_operators = [dict_to_operator(op_dict) for op_dict in data.get("operators", [])]
    print(f"Loaded {len(abstract_operators)} abstract operators")

    # Convert to DocETL format
    docetl_operators = [abstract_to_docetl(op) for op in abstract_operators]

    # Create YAML structure with only operations section
    pipeline_yaml = {
        "operations": docetl_operators
    }

    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save to YAML file
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(pipeline_yaml, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

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
            yaml_to_abstract_json(args.input_file, args.output_file)
        elif args.to_docetl:
            # Convert abstract JSON to YAML
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