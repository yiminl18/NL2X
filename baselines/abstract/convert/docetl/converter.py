"""
DocETL to Abstract Operator Conversion Module

This module provides utilities to convert DocETL operators to abstract operators
and vice versa. It also handles YAML/JSON pipeline conversion and serialization.
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
_parent_dir = os.path.dirname(os.path.dirname(_current_dir))
sys.path.insert(0, _parent_dir)
sys.path.insert(0, os.path.join(_parent_dir, 'tools'))

# Try relative imports first, fall back to absolute
try:
    from ...ops.base import Operator
    from ...tools.docetl_pipeline_parser import parse_pipeline_operators
    from ...tools.static_checker import validate_pipeline
    from ...pipeline import Pipeline, PipelineNode
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

# Import schema functions from schema.py
try:
    from .schema import (
        _extract_input_schema,
        _extract_output_schema,
        _parse_docetl_schema_to_type_dict,
        _type_dict_to_docetl_string
    )
except ImportError:
    # For standalone execution, import from same directory
    from schema import (
        _extract_input_schema,
        _extract_output_schema,
        _parse_docetl_schema_to_type_dict,
        _type_dict_to_docetl_string
    )


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
    abstract_op.output = _extract_output_schema(docetl_op, field_types)

    return abstract_op


def docetl_pipeline_to_abstract(docetl_pipeline: List[Dict[str, Any]], dataset_schema: Optional[Dict[str, Any]] = None, verbose: bool = False) -> List[Operator]:
    """
    Convert a list of DocETL operators (a pipeline) to abstract operators.

    This function tracks field types as they flow through the pipeline,
    ensuring that types defined in outputs are propagated to downstream inputs.

    Args:
        docetl_pipeline: List of DocETL operator dictionaries
        dataset_schema: Optional dataset schema with field definitions
        verbose: If True, print detailed information about schema transformations

    Returns:
        List of abstract Operator instances with proper type tracking
    """
    abstract_operators = []
    cumulative_field_types = {}  # Track all field types available at each step

    # Initialize cumulative field types from dataset schema if provided
    if dataset_schema and "fields" in dataset_schema:
        cumulative_field_types.update(dataset_schema["fields"])

    # Print initial dataset schema in verbose mode
    if verbose:
        print("\n" + "=" * 60)
        print("INITIAL DATASET SCHEMA")
        print("=" * 60)
        if cumulative_field_types:
            print("Fields:")
            for field, field_type in sorted(cumulative_field_types.items()):
                print(f"  - {field}: {field_type}")
        else:
            print("No dataset schema provided")
        print("=" * 60 + "\n")

    for i, docetl_op in enumerate(docetl_pipeline):
        # Print operator info before processing in verbose mode
        if verbose:
            print(f"=== Processing Operator {i+1}/{len(docetl_pipeline)}: {docetl_op.get('name', 'unknown')} ({docetl_op.get('type', 'unknown')}) ===")

        # Convert operator with current field types and dataset schema
        abstract_op = docetl_to_abstract(docetl_op, cumulative_field_types, dataset_schema)
        abstract_operators.append(abstract_op)

        # Print input schema in verbose mode
        if verbose:
            print("\nInput Schema:")
            if abstract_op.input and isinstance(abstract_op.input, dict):
                if "fields" in abstract_op.input:
                    for field, field_type in sorted(abstract_op.input["fields"].items()):
                        print(f"  - {field}: {field_type}")
                else:
                    print(f"  Type: {abstract_op.input.get('type', 'unknown')}")
            else:
                print("  (no input schema)")

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
            # The output schema from _extract_output_schema already contains the transformed types
            if abstract_op.output and isinstance(abstract_op.output, dict):
                # Update cumulative_field_types with the output schema
                for field, field_type in abstract_op.output.items():
                    if field == '_removed_fields':
                        # Remove fields marked for removal
                        for removed_field in field_type:
                            if removed_field in cumulative_field_types:
                                del cumulative_field_types[removed_field]
                    elif field != "type":
                        cumulative_field_types[field] = field_type

                # Clean up the special marker
                if '_removed_fields' in abstract_op.output:
                    del abstract_op.output['_removed_fields']

        # Handle other special operators
        elif op_type == "split":
            # Split adds standard fields
            cumulative_field_types["_split_id"] = "String"
            cumulative_field_types["_split_index"] = "Integer"

        elif op_type == "gather" and "output_key" in docetl_op:
            # Gather produces a list field
            cumulative_field_types[docetl_op["output_key"]] = "List[String]"

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

        # Print output schema and available fields in verbose mode
        if verbose:
            print("\nOutput Schema:")
            if abstract_op.output and isinstance(abstract_op.output, dict):
                output_fields = {k: v for k, v in abstract_op.output.items() if k != "type"}
                if output_fields:
                    for field, field_type in sorted(output_fields.items()):
                        print(f"  - {field}: {field_type}")
                else:
                    print("  (no output fields defined)")
            else:
                print("  (no output schema)")

            print("\nAvailable Fields After This Operator:")
            if cumulative_field_types:
                for field, field_type in sorted(cumulative_field_types.items()):
                    print(f"  - {field}: {field_type}")
            else:
                print("  (no fields available)")
            print()

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


def yaml_to_abstract_pipeline(yaml_file_path: Union[str, Path], dataset_schema: Optional[Dict[str, Any]] = None, verbose: bool = False) -> List[Operator]:
    """
    Convert a DocETL YAML pipeline file to a list of abstract operators.

    This function uses the docetl_pipeline_parser to extract operators
    from a YAML file and converts each to an abstract Operator instance.

    Args:
        yaml_file_path: Path to the DocETL YAML pipeline file
        dataset_schema: Optional dataset schema with field definitions
        verbose: If True, print detailed information about schema transformations

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
    abstract_operators = docetl_pipeline_to_abstract(docetl_operators, dataset_schema, verbose=verbose)

    return abstract_operators


def yaml_to_abstract_pipeline_dag(yaml_file_path: Union[str, Path], dataset_schema: Optional[Dict[str, Any]] = None, verbose: bool = False) -> tuple[List[Operator], Pipeline]:
    """
    Convert a DocETL YAML pipeline file to abstract operators and a Pipeline DAG.

    This function reads the complete YAML structure and creates:
    1. A list of abstract operators
    2. A Pipeline instance representing the DAG structure

    Args:
        yaml_file_path: Path to the DocETL YAML pipeline file
        dataset_schema: Optional dataset schema with field definitions
        verbose: If True, print detailed information about schema transformations

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
    abstract_operators = docetl_pipeline_to_abstract(docetl_operators, dataset_schema, verbose=verbose)

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


def convert_yaml_or_operators(input_data: Union[str, Path, List[Dict[str, Any]]], dataset_schema: Optional[Dict[str, Any]] = None, verbose: bool = False) -> List[Operator]:
    """
    Flexible conversion function that handles both YAML files and operator lists.

    Args:
        input_data: Either a path to a YAML file or a list of DocETL operators
        dataset_schema: Optional dataset schema with field definitions
        verbose: If True, print detailed information about schema transformations

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
        return yaml_to_abstract_pipeline(input_data, dataset_schema, verbose=verbose)
    elif isinstance(input_data, list):
        # It's already a list of operators
        return docetl_pipeline_to_abstract(input_data, dataset_schema, verbose=verbose)
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

def yaml_to_abstract_json(yaml_path: Union[str, Path], output_path: Union[str, Path], validate: bool = False, dataset_schema: Optional[Union[Dict[str, Any], str, Path]] = None, verbose: bool = False) -> None:
    """
    Convert a DocETL YAML pipeline file to abstract JSON representation.

    This function reads a YAML pipeline file, converts it to abstract operators,
    optionally validates the pipeline, and saves the result as a JSON file.
    All non-operator fields from the YAML are preserved in the JSON output.

    Args:
        yaml_path: Path to the input YAML pipeline file
        output_path: Path where the output JSON file should be saved
        validate: Whether to validate the pipeline after conversion
        dataset_schema: Optional dataset schema (Dict or path to JSON file)
        verbose: If True, print detailed information about schema transformations

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

    # Read the complete YAML structure to preserve all fields
    print(f"Reading YAML pipeline from: {yaml_path}")
    with open(yaml_path, 'r') as f:
        complete_yaml = yaml.safe_load(f)

    # Convert YAML operators to abstract operators
    abstract_operators = yaml_to_abstract_pipeline(yaml_path, schema_dict, verbose=verbose)
    print(f"Converted {len(abstract_operators)} operators to abstract representation")

    # Serialize operators to dictionaries
    operators_data = {
        "operators": [operator_to_dict(op) for op in abstract_operators]
    }

    # Include dataset schema in output if provided
    if schema_dict:
        operators_data["dataset_schema"] = schema_dict

    # Preserve all other fields from the original YAML (except operations)
    for key, value in complete_yaml.items():
        if key != "operations":  # Skip operations as they're already converted
            operators_data[key] = value

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
    Convert an abstract JSON representation back to a complete DocETL YAML pipeline.

    This function reads an abstract JSON file, converts it back to DocETL operators,
    and restores all preserved fields to create a complete, executable YAML pipeline.

    Args:
        json_path: Path to the input JSON file
        output_path: Path where the output YAML file should be saved
    Example:
        >>> abstract_json_to_yaml("abstract_pipeline.json", "recovered_pipeline.yaml")
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

    # Create complete YAML structure with all preserved fields
    pipeline_yaml = {}

    # Add all preserved fields from JSON (except operators and dataset_schema)
    # These are the fields that were preserved from the original YAML
    for key, value in data.items():
        if key not in ["operators", "dataset_schema"]:
            pipeline_yaml[key] = value

    # Add the converted operations
    pipeline_yaml["operations"] = docetl_operators

    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save to YAML file
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(pipeline_yaml, f, Dumper=LiteralDumper,
                  default_flow_style=False, sort_keys=False, allow_unicode=True)

    print(f"Saved complete DocETL pipeline to: {output_path}")


# ============================================================================
# CLI Interface
# ============================================================================

def main():
    """
    Command-line interface for standalone conversion operations.

    Usage:
        # Convert YAML to abstract JSON
        python converter.py --to-abstract input.yaml output.json

        # Convert abstract JSON to YAML
        python converter.py --to-docetl input.json output.yaml
    """
    parser = argparse.ArgumentParser(
        description="Convert between DocETL YAML pipelines and abstract JSON representations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert YAML pipeline to abstract JSON
  python converter.py --to-abstract examples/example1.yaml output/abstract.json

  # Convert abstract JSON back to YAML (operations only)
  python converter.py --to-docetl output/abstract.json output/pipeline_ops.yaml
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

    # Add verbose flag
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed information about schema transformations during conversion'
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
                                dataset_schema=args.dataset_schema,
                                verbose=args.verbose)
        elif args.to_docetl:
            # Convert abstract JSON to YAML
            if args.validate:
                print("Warning: --validate flag is only supported with --to-abstract, ignoring.")
            if args.dataset_schema:
                print("Warning: --dataset-schema flag is only supported with --to-abstract, ignoring.")
            if args.verbose:
                print("Warning: --verbose flag is only supported with --to-abstract, ignoring.")
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
