#!/usr/bin/env python3
"""
DocETL Procedure Parser

This tool parses DocETL YAML procedure files and extracts operators in execution order.
The output can be directly used as the `operators` parameter for the `_connect_pipeline` function.

Usage:
    python docetl_pipeline_parser.py <yaml_file_path>

Or import and use:
    from tools.docetl_pipeline_parser import parse_procedure_operators
    operators = parse_procedure_operators("path/to/pipeline.yaml")
"""

import yaml
import sys
import json
from typing import List, Dict, Any, Optional
from pathlib import Path


def parse_procedure_operators(yaml_path: Path) -> List[Dict[str, Any]]:
    """
    Parse a DocETL YAML procedure file and extract operators in execution order.

    Args:
        yaml_file_path: Path to the YAML procedure file

    Returns:
        List of operator dictionaries in pipeline execution order, ready to be passed
        to _connect_pipeline() as the operators parameter

    """
    # Read YAML file
    if not yaml_path.exists():
        raise FileNotFoundError(f"YAML file not found: {yaml_path}")

    with open(yaml_path, 'r') as f:
        pipeline_config = yaml.safe_load(f)

    # Validate basic structure
    if not isinstance(pipeline_config, dict):
        raise ValueError("Invalid YAML: root must be a dictionary")

    # Extract operator definitions from 'operations' section
    operations_section = pipeline_config.get('operations', [])
    if not operations_section:
        raise ValueError("No 'operations' section found in YAML")

    # Create a mapping of operator name to operator definition
    operator_definitions = {}
    for op in operations_section:
        if not isinstance(op, dict):
            raise ValueError(f"Invalid operator definition: {op}")

        op_name = op.get('name')
        if not op_name:
            raise ValueError(f"Operator missing 'name' field: {op}")

        operator_definitions[op_name] = op

    # Extract pipeline execution order
    pipeline_section = pipeline_config.get('pipeline', {})
    if not pipeline_section:
        raise ValueError("No 'pipeline' section found in YAML")

    steps = pipeline_section.get('steps', [])
    if not steps:
        raise ValueError("No 'steps' found in pipeline section")

    # Get the first step's operations (assuming single-step pipeline)
    first_step = steps[0]
    if not isinstance(first_step, dict):
        raise ValueError(f"Invalid step definition: {first_step}")

    operation_names = first_step.get('operations', [])
    if not operation_names:
        raise ValueError("No operations found in pipeline step")

    # Build ordered list of operators
    ordered_operators = []
    for op_name in operation_names:
        if op_name not in operator_definitions:
            raise ValueError(f"Operator '{op_name}' referenced in pipeline but not defined in operations section")

        ordered_operators.append(operator_definitions[op_name])

    return ordered_operators


def main():
    """CLI interface for the parser."""
    if len(sys.argv) != 2:
        print("Usage: python docetl_pipeline_parser.py <yaml_file_path>")
        sys.exit(1)

    yaml_file_path = sys.argv[1]

    try:
        operators = parse_procedure_operators(yaml_file_path)

        print(f"Successfully parsed {len(operators)} operators in execution order:\n")

        for i, op in enumerate(operators, 1):
            print(f"{i}. {op.get('name', 'unnamed')} ({op.get('type', 'unknown')})")

        print("\n" + "="*60)
        print("Full operator definitions (JSON format):")
        print("="*60)
        print(json.dumps(operators, indent=2))

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
