#!/usr/bin/env python3
"""
Static comparator for abstract pipeline operators.

Compares two pipelines to verify:
1. Both pipelines are individually valid
2. They have compatible input and output schemas
"""

import json
import sys
import argparse
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

# Import static checker
from .static_checker import validate_pipeline

from ..type import check_type_compatibility


# ============================================================================
# Schema Extraction Functions
# ============================================================================

def extract_input_schema(operators: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Extract input schema from the first operator in the pipeline.

    Args:
        operators: List of operator dictionaries

    Returns:
        Input schema dictionary or None if pipeline is empty
    """
    if not operators:
        return None

    first_op = operators[0]
    return first_op.get('input', {})


def extract_output_schema(operators: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Extract output schema from the last operator in the pipeline.

    Args:
        operators: List of operator dictionaries

    Returns:
        Output schema dictionary or None if pipeline is empty
    """
    if not operators:
        return None

    last_op = operators[-1]
    return last_op.get('output', {})


def compare_schemas(
    schema1: Dict[str, Any],
    schema2: Dict[str, Any],
    schema_name: str = "schema"
) -> Tuple[bool, List[str]]:
    """
    Compare two schemas for compatibility.

    Args:
        schema1: First schema
        schema2: Second schema
        schema_name: Name of schema for error messages

    Returns:
        Tuple of (is_compatible, differences_list)
    """
    differences = []

    if not schema1 and not schema2:
        return True, []

    if not schema1:
        differences.append(f"{schema_name}: Pipeline 1 has no schema")
        return False, differences

    if not schema2:
        differences.append(f"{schema_name}: Pipeline 2 has no schema")
        return False, differences

    if 'fields' in schema1 and 'fields' in schema2:
        fields1 = schema1['fields']
        fields2 = schema2['fields']

        all_fields = set(fields1.keys()) | set(fields2.keys())

        for field in all_fields:
            if field not in fields1:
                differences.append(f"{schema_name}: Field '{field}' missing in Pipeline 1")
            elif field not in fields2:
                differences.append(f"{schema_name}: Field '{field}' missing in Pipeline 2")
            else:
                type1 = fields1[field]
                type2 = fields2[field]

                try:
                    compatible = check_type_compatibility(type1, type2)
                    if not compatible:
                        differences.append(
                            f"{schema_name}: Field '{field}' type mismatch - "
                            f"Pipeline 1: {type1}, Pipeline 2: {type2}"
                        )
                except Exception:
                    if str(type1) != str(type2):
                        differences.append(
                            f"{schema_name}: Field '{field}' type mismatch - "
                            f"Pipeline 1: {type1}, Pipeline 2: {type2}"
                        )

    else:
        schema1_str = json.dumps(schema1, sort_keys=True)
        schema2_str = json.dumps(schema2, sort_keys=True)

        if schema1_str != schema2_str:
            differences.append(
                f"{schema_name}: Schema structure mismatch - "
                f"Pipeline 1: {schema1}, Pipeline 2: {schema2}"
            )

    is_compatible = len(differences) == 0
    return is_compatible, differences


# ============================================================================
# Main Comparison Function
# ============================================================================

def compare_pipelines(
    operators1: List[Dict[str, Any]],
    operators2: List[Dict[str, Any]],
    dataset_schema: Optional[Dict[str, Any]] = None,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Compare two pipelines for validity and schema compatibility.

    This function:
    1. Validates both pipelines individually using static_checker
    2. Compares input schemas (first operator of each pipeline)
    3. Compares output schemas (last operator of each pipeline)

    Note: Pipelines can have different numbers of operators - only the
    first and last operators' schemas are compared.

    Args:
        operators1: First pipeline operators list
        operators2: Second pipeline operators list
        dataset_schema: Optional dataset schema for strict validation
        verbose: Show detailed comparison information

    Returns:
        Dictionary with comparison results:
        {
            "is_compatible": bool,
            "pipelines_valid": {
                "pipeline1": bool,
                "pipeline2": bool
            },
            "schema_comparison": {
                "input_compatible": bool,
                "output_compatible": bool,
                "input_differences": [...],
                "output_differences": [...]
            },
            "pipeline_info": {
                "pipeline1_operators": int,
                "pipeline2_operators": int
            },
            "errors": [...],
            "warnings": [...]
        }
    """
    errors = []
    warnings = []

    validation1 = validate_pipeline(operators1, verbose=False, dataset_schema=dataset_schema)
    validation2 = validate_pipeline(operators2, verbose=False, dataset_schema=dataset_schema)

    pipeline1_valid = validation1['is_valid']
    pipeline2_valid = validation2['is_valid']

    if not pipeline1_valid:
        errors.append("Pipeline 1 validation failed:")
        errors.extend([f"  - {e}" for e in validation1['errors']])

    if not pipeline2_valid:
        errors.append("Pipeline 2 validation failed:")
        errors.extend([f"  - {e}" for e in validation2['errors']])

    if not pipeline1_valid or not pipeline2_valid:
        return {
            "is_compatible": False,
            "pipelines_valid": {
                "pipeline1": pipeline1_valid,
                "pipeline2": pipeline2_valid
            },
            "schema_comparison": {
                "input_compatible": False,
                "output_compatible": False,
                "input_differences": [],
                "output_differences": []
            },
            "pipeline_info": {
                "pipeline1_operators": len(operators1),
                "pipeline2_operators": len(operators2)
            },
            "errors": errors,
            "warnings": warnings,
            "validation_details": {
                "pipeline1": validation1,
                "pipeline2": validation2
            } if verbose else {}
        }

    input1 = extract_input_schema(operators1)
    input2 = extract_input_schema(operators2)

    input_compatible, input_diffs = compare_schemas(input1, input2, "Input schema")

    if not input_compatible:
        errors.extend(input_diffs)

    output1 = extract_output_schema(operators1)
    output2 = extract_output_schema(operators2)

    output_compatible, output_diffs = compare_schemas(output1, output2, "Output schema")

    if not output_compatible:
        errors.extend(output_diffs)

    if len(operators1) != len(operators2):
        warnings.append(
            f"Pipelines have different numbers of operators: "
            f"Pipeline 1 has {len(operators1)}, Pipeline 2 has {len(operators2)}"
        )

    is_compatible = pipeline1_valid and pipeline2_valid and input_compatible and output_compatible

    result = {
        "is_compatible": is_compatible,
        "pipelines_valid": {
            "pipeline1": pipeline1_valid,
            "pipeline2": pipeline2_valid
        },
        "schema_comparison": {
            "input_compatible": input_compatible,
            "output_compatible": output_compatible,
            "input_differences": input_diffs,
            "output_differences": output_diffs
        },
        "pipeline_info": {
            "pipeline1_operators": len(operators1),
            "pipeline2_operators": len(operators2)
        },
        "errors": errors,
        "warnings": warnings
    }

    if verbose:
        result["validation_details"] = {
            "pipeline1": validation1,
            "pipeline2": validation2
        }
        result["schemas"] = {
            "pipeline1_input": input1,
            "pipeline1_output": output1,
            "pipeline2_input": input2,
            "pipeline2_output": output2
        }

    return result


# ============================================================================
# CLI Interface
# ============================================================================

def format_comparison_report(result: Dict[str, Any], format: str = 'text') -> str:
    """Format comparison report for display."""
    if format == 'json':
        return json.dumps(result, indent=2)

    # Text format
    lines = []

    # Header
    lines.append("=" * 70)
    lines.append("PIPELINE COMPARISON REPORT")
    lines.append("=" * 70)
    lines.append("")

    # Compatibility status
    is_compatible = result['is_compatible']
    status = "✓ COMPATIBLE" if is_compatible else "✗ INCOMPATIBLE"
    lines.append(f"Status: {status}")
    lines.append("")

    # Pipeline validation status
    lines.append("Pipeline Validation:")
    p1_valid = result['pipelines_valid']['pipeline1']
    p2_valid = result['pipelines_valid']['pipeline2']
    lines.append(f"  Pipeline 1: {'✓ Valid' if p1_valid else '✗ Invalid'}")
    lines.append(f"  Pipeline 2: {'✓ Valid' if p2_valid else '✗ Invalid'}")
    lines.append("")

    # Pipeline info
    info = result['pipeline_info']
    lines.append("Pipeline Information:")
    lines.append(f"  Pipeline 1: {info['pipeline1_operators']} operators")
    lines.append(f"  Pipeline 2: {info['pipeline2_operators']} operators")
    lines.append("")

    # Schema comparison
    schema_comp = result['schema_comparison']
    lines.append("Schema Comparison:")
    input_status = "✓ Compatible" if schema_comp['input_compatible'] else "✗ Incompatible"
    output_status = "✓ Compatible" if schema_comp['output_compatible'] else "✗ Incompatible"
    lines.append(f"  Input schemas:  {input_status}")
    lines.append(f"  Output schemas: {output_status}")
    lines.append("")

    # Errors
    if result['errors']:
        lines.append(f"Errors ({len(result['errors'])}):")
        for error in result['errors']:
            lines.append(f"  • {error}")
        lines.append("")

    # Warnings
    if result['warnings']:
        lines.append(f"Warnings ({len(result['warnings'])}):")
        for warning in result['warnings']:
            lines.append(f"  • {warning}")
        lines.append("")

    lines.append("=" * 70)

    return "\n".join(lines)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Static Comparator for Abstract Pipelines",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare two pipelines
  python static_comparator.py pipeline1.json pipeline2.json

  # Compare with verbose output
  python static_comparator.py pipeline1.json pipeline2.json --verbose

  # Compare with dataset schema for strict validation
  python static_comparator.py pipeline1.json pipeline2.json --dataset-schema schema.json

  # Output as JSON
  python static_comparator.py pipeline1.json pipeline2.json --format json
        """
    )

    parser.add_argument(
        'pipeline1_file',
        type=str,
        help='Path to the first pipeline JSON file'
    )

    parser.add_argument(
        'pipeline2_file',
        type=str,
        help='Path to the second pipeline JSON file'
    )

    parser.add_argument(
        '--dataset-schema',
        type=str,
        help='Path to dataset schema JSON file for strict validation'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show detailed comparison information'
    )

    parser.add_argument(
        '--format', '-f',
        choices=['text', 'json'],
        default='text',
        help='Output format (default: text)'
    )

    args = parser.parse_args()

    # Load pipeline 1
    pipeline1_path = Path(args.pipeline1_file)
    if not pipeline1_path.exists():
        print(f"Error: Pipeline 1 file not found: {args.pipeline1_file}", file=sys.stderr)
        sys.exit(1)

    try:
        with open(pipeline1_path, 'r') as f:
            pipeline1_data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in Pipeline 1 file: {e}", file=sys.stderr)
        sys.exit(1)

    # Load pipeline 2
    pipeline2_path = Path(args.pipeline2_file)
    if not pipeline2_path.exists():
        print(f"Error: Pipeline 2 file not found: {args.pipeline2_file}", file=sys.stderr)
        sys.exit(1)

    try:
        with open(pipeline2_path, 'r') as f:
            pipeline2_data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in Pipeline 2 file: {e}", file=sys.stderr)
        sys.exit(1)

    # Extract operators
    if 'operators' not in pipeline1_data:
        print("Error: Pipeline 1 file must contain 'operators' field", file=sys.stderr)
        sys.exit(1)

    if 'operators' not in pipeline2_data:
        print("Error: Pipeline 2 file must contain 'operators' field", file=sys.stderr)
        sys.exit(1)

    operators1 = pipeline1_data['operators']
    operators2 = pipeline2_data['operators']

    # Load dataset schema if provided
    dataset_schema = None
    if args.dataset_schema:
        schema_path = Path(args.dataset_schema)
        if not schema_path.exists():
            print(f"Error: Dataset schema file not found: {args.dataset_schema}", file=sys.stderr)
            sys.exit(1)

        try:
            with open(schema_path, 'r') as f:
                dataset_schema = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in dataset schema file: {e}", file=sys.stderr)
            sys.exit(1)

    # Compare pipelines
    result = compare_pipelines(
        operators1,
        operators2,
        dataset_schema=dataset_schema,
        verbose=args.verbose
    )

    # Format and print report
    output = format_comparison_report(result, args.format)
    print(output)

    # Exit with appropriate code
    sys.exit(0 if result['is_compatible'] else 1)


if __name__ == "__main__":
    main()
