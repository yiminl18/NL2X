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

# Setup path for standalone execution
_current_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(os.path.dirname(_current_dir))
sys.path.insert(0, _parent_dir)
sys.path.insert(0, os.path.join(_parent_dir, 'tools'))

# Import core dependencies
try:
    from ...ops.base import Operator
    from ...pipeline import Pipeline
    from ...tools.docetl_pipeline_parser import parse_pipeline_operators
    from ...tools.static_checker import validate_pipeline
except ImportError:
    from ops.base import Operator
    from pipeline import Pipeline
    from docetl_pipeline_parser import parse_pipeline_operators
    try:
        from static_checker import validate_pipeline
    except ImportError:
        def validate_pipeline(operators, verbose=False):
            return {'is_valid': True, 'errors': [], 'warnings': [], 'summary': {}}

# Import schema and type utilities
from .schema import _extract_input_schema, _extract_output_schema, DocETLTypeMapper


# ============================================================================
# Schema Inference Functions
# ============================================================================

def infer_docetl_type_from_value(value: Any) -> str:
    """Infer DocETL type string from a Python value.

    Args:
        value: A Python value to infer the type from

    Returns:
        A DocETL type string (e.g., 'str', 'int', 'list', 'dict')
    """
    if value is None:
        return 'str'  # Default to string for None values

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
        # Try to infer element type from first element
        first_elem_type = infer_docetl_type_from_value(value[0])
        return f'list[{first_elem_type}]'
    elif isinstance(value, dict):
        return 'dict'
    else:
        return 'str'  # Default fallback


def infer_schema_from_dataset(file_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """Infer dataset schema from a DocETL data file.

    DocETL datasets are List[Dict] format. This function reads the first dict
    and infers the schema from its fields.

    Args:
        file_path: Path to the dataset file (JSON format)

    Returns:
        A schema dict in the format {'fields': {field_name: docetl_type_str}}
        or None if the file cannot be read or is empty
    """
    try:
        file_path = Path(file_path)

        # Read JSON file
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Check if data is a list and has at least one element
        if not isinstance(data, list) or len(data) == 0:
            return None

        # Get the first dict element
        first_item = data[0]
        if not isinstance(first_item, dict):
            return None

        # Infer types for each field
        fields = {}
        for field_name, field_value in first_item.items():
            fields[field_name] = infer_docetl_type_from_value(field_value)

        return {'fields': fields}

    except (FileNotFoundError, json.JSONDecodeError, KeyError, IndexError):
        return None


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
    """Convert DocETL operator dict to abstract Operator instance."""
    abstract_op = Operator()
    docetl_op = copy.deepcopy(docetl_operator)

    abstract_op.name = docetl_op.get("name", "")

    docetl_type = docetl_op.get("type", "")
    abstract_op.type = DOCETL_TO_ABSTRACT_TYPE_MAP.get(docetl_type, docetl_type.capitalize())

    abstract_op.source = {
        "system": "docetl",
        "name": docetl_op.get("name", "")
    }

    properties = {}
    for key, value in docetl_op.items():
        if key not in ["name", "type"]:
            properties[key] = value

    abstract_op.properties = properties

    abstract_op.input = _extract_input_schema(docetl_op, field_types, dataset_schema)
    abstract_op.output = _extract_output_schema(docetl_op, field_types)

    return abstract_op


def docetl_pipeline_to_abstract(yaml_path: Union[str, Path], pipeline_config: Optional[Dict[str, Any]] = None, dataset_schema: Optional[Dict[str, Any]] = None, verbose: bool = False) -> Pipeline:
    """Convert DocETL YAML to abstract Pipeline, tracking field types through pipeline."""
    # Parse DocETL operators from YAML
    yaml_path = Path(yaml_path)
    docetl_pipeline = parse_pipeline_operators(yaml_path)

    abstract_operators = []
    cumulative_field_types = {}

    if dataset_schema and "fields" in dataset_schema:
        cumulative_field_types.update(dataset_schema["fields"])

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
        if verbose:
            print(f"=== Processing Operator {i+1}/{len(docetl_pipeline)}: {docetl_op.get('name', 'unknown')} ({docetl_op.get('type', 'unknown')}) ===")

        abstract_op = docetl_to_abstract(docetl_op, cumulative_field_types, dataset_schema)
        abstract_operators.append(abstract_op)

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

        op_type = docetl_op.get("type", "")

        if "output" in docetl_op and isinstance(docetl_op["output"], dict):
            if "schema" in docetl_op["output"]:
                schema = docetl_op["output"]["schema"]
                if isinstance(schema, dict):
                    for field, field_type_str in schema.items():
                        cumulative_field_types[field] = DocETLTypeMapper.to_abstract_type_string(field_type_str)

        if op_type == "unnest":
            if abstract_op.output and isinstance(abstract_op.output, dict):
                for field, field_type in abstract_op.output.items():
                    if field == '_removed_fields':
                        for removed_field in field_type:
                            if removed_field in cumulative_field_types:
                                del cumulative_field_types[removed_field]
                    elif field != "type":
                        cumulative_field_types[field] = field_type

                if '_removed_fields' in abstract_op.output:
                    del abstract_op.output['_removed_fields']


        elif op_type == "split":
            cumulative_field_types["_split_id"] = "String"
            cumulative_field_types["_split_index"] = "Integer"

        elif op_type == "gather" and "output_key" in docetl_op:
            cumulative_field_types[docetl_op["output_key"]] = "List[String]"

        elif op_type == "cluster" and "output_key" in docetl_op:
            cumulative_field_types[docetl_op["output_key"]] = "String"

        elif op_type == "rank":
            cumulative_field_types["rank"] = "Integer"

        if abstract_op.output and isinstance(abstract_op.output, dict):
            for field, field_type in abstract_op.output.items():
                if field != "type":
                    if field in cumulative_field_types:
                        abstract_op.output[field] = cumulative_field_types[field]


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

    input_path = None
    output_path = None
    properties = {}
    pipeline_name = "docetl_pipeline"

    if pipeline_config:
        pipeline_name = pipeline_config.get('name', 'docetl_pipeline')

        datasets = pipeline_config.get('datasets', {})
        if datasets:
            first_dataset = next(iter(datasets.values()), {})
            input_path = first_dataset.get('path')

        pipeline_output = pipeline_config.get('pipeline', {}).get('output', {})
        output_path = pipeline_output.get('path')

        for key, value in pipeline_config.items():
            if key not in ['operations', 'datasets', 'pipeline', 'name']:
                properties[key] = value

        if 'pipeline' in pipeline_config:
            pipeline_props = pipeline_config['pipeline'].copy()
            pipeline_props.pop('output', None)
            pipeline_props.pop('steps', None)
            if pipeline_props:
                properties['pipeline'] = pipeline_props


    return Pipeline.from_operators(
        abstract_operators,
        name=pipeline_name,
        input_path=input_path,
        output_path=output_path,
        properties=properties,
        dataset_schema=dataset_schema
    )


def abstract_to_docetl(abstract_operator: Operator) -> Dict[str, Any]:
    """Convert abstract Operator to DocETL operator dict."""
    from .schema import DocETLTypeMapper

    docetl_op = {}

    docetl_op["name"] = abstract_operator.name

    abstract_to_docetl_type = {v: k for k, v in DOCETL_TO_ABSTRACT_TYPE_MAP.items()}
    abstract_type = abstract_operator.type

    if abstract_type == "Map":
        if "code" in abstract_operator.properties:
            docetl_op["type"] = "code_map"
        else:
            docetl_op["type"] = "map"
    elif abstract_type == "Filter":
        if "code" in abstract_operator.properties:
            docetl_op["type"] = "code_filter"
        else:
            docetl_op["type"] = "filter"
    elif abstract_type == "Reduce":
        if "code" in abstract_operator.properties:
            docetl_op["type"] = "code_reduce"
        else:
            docetl_op["type"] = "reduce"
    else:
        docetl_op["type"] = abstract_to_docetl_type.get(abstract_type, abstract_type.lower())

    # Copy properties (prompt, reduce_key, etc.)
    # Skip empty or None values to avoid cluttering the output
    for key, value in abstract_operator.properties.items():
        # Skip empty dicts, empty lists, and None values
        if value is not None and value != {} and value != []:
            docetl_op[key] = value

    # Convert output schema from abstract format to DocETL format
    if abstract_operator.output and isinstance(abstract_operator.output, dict):
        # Filter out internal fields and 'type' metadata
        output_fields = {k: v for k, v in abstract_operator.output.items()
                        if k not in ['type', '_removed_fields']}

        if output_fields:
            # Convert each field type from abstract to DocETL format
            docetl_schema = {}
            for field_name, abstract_type_str in output_fields.items():
                if isinstance(abstract_type_str, str):
                    docetl_type_str = DocETLTypeMapper.from_abstract_type_string(abstract_type_str)
                    docetl_schema[field_name] = docetl_type_str
                else:
                    # If it's not a string (shouldn't happen), keep as-is
                    docetl_schema[field_name] = abstract_type_str

            # Create proper DocETL output.schema structure
            docetl_op["output"] = {
                "schema": docetl_schema
            }

    return docetl_op

def _load_and_convert_yaml(yaml_path: Union[str, Path], dataset_schema: Optional[Union[Dict[str, Any], str, Path]] = None, verbose: bool = False) -> Dict[str, Any]:
    """Core function: load YAML and convert to unified abstract format."""
    yaml_path = Path(yaml_path)

    # Load YAML pipeline
    with open(yaml_path, 'r') as f:
        complete_yaml = yaml.safe_load(f)

    # Load and convert dataset schema to abstract layer format
    schema_dict = None
    converted_schema = None
    if dataset_schema:
        if isinstance(dataset_schema, (str, Path)):
            with open(Path(dataset_schema), 'r') as f:
                schema_dict = json.load(f)
        elif isinstance(dataset_schema, dict):
            schema_dict = dataset_schema

        # Convert to abstract layer types
        if schema_dict and 'fields' in schema_dict:
            converted_schema = {'fields': {}}
            for field, field_type in schema_dict['fields'].items():
                if isinstance(field_type, str):
                    converted_schema['fields'][field] = DocETLTypeMapper.to_abstract_type_string(field_type)
                else:
                    converted_schema['fields'][field] = field_type

    # If schema_dict is still None, try to infer from input path
    if not schema_dict:
        datasets = complete_yaml.get('datasets', {})
        if datasets:
            first_dataset = next(iter(datasets.values()), {})
            input_path = first_dataset.get('path')
            if input_path:
                # Resolve input path relative to YAML file location
                input_file = yaml_path.parent / input_path
                if input_file.exists():
                    schema_dict = infer_schema_from_dataset(input_file)
                    if schema_dict and 'fields' in schema_dict:
                        # Convert inferred DocETL types to abstract layer types
                        converted_schema = {'fields': {}}
                        for field, field_type in schema_dict['fields'].items():
                            converted_schema['fields'][field] = DocETLTypeMapper.to_abstract_type_string(field_type)

                        if verbose:
                            print("\n" + "=" * 60)
                            print("INFERRED DATASET SCHEMA FROM INPUT FILE")
                            print(f"File: {input_file}")
                            print("=" * 60)
                            print("Fields:")
                            for field, field_type in sorted(converted_schema['fields'].items()):
                                print(f"  - {field}: {field_type}")
                            print("=" * 60 + "\n")

    # Parse and convert operators (parse_pipeline_operators called inside)
    abstract_pipeline = docetl_pipeline_to_abstract(yaml_path, complete_yaml, converted_schema, verbose=verbose)
    abstract_operators = abstract_pipeline.to_operators()

    # Perform static validation (always)
    operators_list = [operator_to_dict(op) for op in abstract_operators]
    validation_result = validate_pipeline(operators_list, dataset_schema=converted_schema)

    return {
        "input_path": abstract_pipeline.input_path,
        "output_path": abstract_pipeline.output_path,
        "dataset_schema": converted_schema,
        "properties": abstract_pipeline.properties,
        "operators": operators_list,
        "pipeline": abstract_pipeline,
        "validation": validation_result
    }


def yaml_to_abstract_pipeline(yaml_file_path: Union[str, Path], dataset_schema: Optional[Dict[str, Any]] = None, verbose: bool = False) -> Pipeline:
    """Convert DocETL YAML file to abstract Pipeline."""
    result = _load_and_convert_yaml(yaml_file_path, dataset_schema, verbose)
    if result["validation"]["errors"]:
        print("Validation errors found:")
        for error in result["validation"]["errors"]:
            print(f"- {error}")
        raise ValueError("Pipeline validation failed")
    
    return result["pipeline"]


# ============================================================================
# JSON Serialization Functions
# ============================================================================

def operator_to_dict(operator: Operator) -> Dict[str, Any]:
    """Convert Operator instance to dict for JSON serialization."""
    return {
        "name": operator.name,
        "type": operator.type,
        "source": operator.source,
        "properties": operator.properties,
        "input": operator.input,
        "output": operator.output
    }


def dict_to_operator(op_dict: Dict[str, Any]) -> Operator:
    """Convert dict to Operator instance."""
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

def yaml_to_abstract_json(yaml_path: Union[str, Path], output_path: Union[str, Path], dataset_schema: Optional[Union[Dict[str, Any], str, Path]] = None, verbose: bool = False) -> None:
    """Convert DocETL YAML pipeline to abstract JSON representation."""
    yaml_path = Path(yaml_path)
    output_path = Path(output_path)

    if dataset_schema and isinstance(dataset_schema, (str, Path)):
        if not Path(dataset_schema).exists():
            raise FileNotFoundError(f"Dataset schema file not found: {dataset_schema}")
        print(f"Loaded dataset schema from: {dataset_schema}")

    print(f"Reading YAML pipeline from: {yaml_path}")

    # Use core conversion function (includes validation)
    result = _load_and_convert_yaml(yaml_path, dataset_schema, verbose)
    validation_result = result.pop("validation")
    result.pop("pipeline")  # Remove pipeline object from output

    print(f"Converted {len(result['operators'])} operators to abstract representation")

    # Display validation results
    print("=" * 60)
    print("VALIDATING ABSTRACT PIPELINE")
    if result["dataset_schema"]:
        print("Mode: STRICT (dataset schema provided)")
    else:
        print("Mode: NORMAL")
    print("=" * 60)

    print(f"Validation Result: {'VALID ✓' if validation_result['is_valid'] else 'INVALID ✗'}")

    if validation_result['errors']:
        print(f"Total Errors: {len(validation_result['errors'])}")
        print("\nERRORS:")
        for i, error in enumerate(validation_result['errors'], 1):
            print(f"  {i}. {error}")

    if validation_result['warnings']:
        print(f"Total Warnings: {len(validation_result['warnings'])}")
        print("\nWARNINGS:")
        for i, warning in enumerate(validation_result['warnings'], 1):
            print(f"  {i}. {warning}")

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

    print("=" * 60)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"Saved abstract pipeline to: {output_path}")


# ============================================================================
# Custom YAML Presenter for Multi-line Strings
# ============================================================================

def literal_presenter(dumper, data):
    """Custom presenter for multi-line strings in YAML."""
    if '\n' in data:
        return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='|')
    return dumper.represent_scalar('tag:yaml.org,2002:str', data)


class LiteralDumper(yaml.SafeDumper):
    """YAML dumper with literal style for multi-line strings."""
    pass


LiteralDumper.add_representer(str, literal_presenter)


def abstract_json_to_yaml(json_path: Union[str, Path], output_path: Union[str, Path]) -> None:
    """Convert abstract JSON representation back to DocETL YAML pipeline."""
    json_path = Path(json_path)
    output_yaml_path = Path(output_path)

    print(f"Reading abstract pipeline from: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    pipeline_input_path = data.get("input_path")
    pipeline_output_path = data.get("output_path")
    dataset_schema = data.get("dataset_schema")
    properties = data.get("properties", {})

    if dataset_schema:
        print(f"Found dataset schema with {len(dataset_schema.get('fields', {}))} fields")

    abstract_operators = [dict_to_operator(op_dict) for op_dict in data.get("operators", [])]
    print(f"Loaded {len(abstract_operators)} abstract operators")

    docetl_operators = [abstract_to_docetl(op) for op in abstract_operators]

    pipeline_yaml = {}

    # Build datasets from input_path
    if pipeline_input_path:
        pipeline_yaml["datasets"] = {
            "input_data": {
                "type": "file",
                "path": pipeline_input_path
            }
        }

    # Add properties
    pipeline_yaml.update(properties)

    # Add operations
    pipeline_yaml["operations"] = docetl_operators

    # Build pipeline.output from output_path
    if pipeline_output_path:
        pipeline_yaml["pipeline"] = {
            "steps": [{
                "name": "main",
                "input": "input_data",
                "operations": [op["name"] for op in docetl_operators]
            }],
            "output": {
                "type": "file",
                "path": pipeline_output_path
            }
        }

    output_yaml_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(pipeline_yaml, f, Dumper=LiteralDumper,
                  default_flow_style=False, sort_keys=False, allow_unicode=True)

    print(f"Saved complete DocETL pipeline to: {output_yaml_path}")


# ============================================================================
# CLI Interface
# ============================================================================

def main():
    """Command-line interface for standalone conversion operations."""
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
        help='Path to JSON file containing dataset schema (enables strict validation)'
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
                                dataset_schema=args.dataset_schema,
                                verbose=args.verbose)
        elif args.to_docetl:
            # Convert abstract JSON to YAML
            if args.dataset_schema:
                print("Warning: --dataset-schema flag is only supported with --to-abstract, ignoring.")
            if args.verbose:
                print("Warning: --verbose flag is only supported with --to-abstract, ignoring.")
            abstract_json_to_yaml(args.input_file, args.output_file)

        print("✓ Conversion completed successfully!")

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
