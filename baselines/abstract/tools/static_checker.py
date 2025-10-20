#!/usr/bin/env python3
"""
Static validation tool for abstract pipeline operators.
"""

import json
import sys
import argparse
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from pathlib import Path
from collections import defaultdict

from ..type import TypeSystem, parse_type_string, check_type_compatibility, validate_type_syntax


# ============================================================================
# Core Validation Functions
# ============================================================================

def validate_operator_fields(operators: List[Dict[str, Any]]) -> Tuple[bool, List[str], List[str]]:
    """Validate that each operator has all required fields with correct structure."""
    errors = []
    warnings = []

    required_fields = ['name', 'type', 'source', 'properties', 'input', 'output']

    for i, op in enumerate(operators):
        op_name = op.get('name', f'<unnamed operator {i}>')

        for field in required_fields:
            if field not in op:
                errors.append(f"Operator '{op_name}' missing required field: {field}")

        if 'source' in op:
            source = op['source']
            if not isinstance(source, dict):
                errors.append(f"Operator '{op_name}': 'source' must be a dictionary")
            else:
                if 'system' not in source:
                    errors.append(f"Operator '{op_name}': 'source' missing 'system' field")
                if 'name' not in source:
                    errors.append(f"Operator '{op_name}': 'source' missing 'name' field")

        if 'input' in op:
            input_schema = op['input']
            if not isinstance(input_schema, dict):
                errors.append(f"Operator '{op_name}': 'input' must be a dictionary")
            else:
                if 'fields' in input_schema and 'type' not in input_schema:
                    warnings.append(f"Operator '{op_name}': 'input' has 'fields' but no 'type' specification")

                if 'fields' in input_schema:
                    if not isinstance(input_schema['fields'], dict):
                        errors.append(f"Operator '{op_name}': 'input.fields' must be a dictionary")

        if 'output' in op:
            output_schema = op['output']
            if not isinstance(output_schema, dict):
                errors.append(f"Operator '{op_name}': 'output' must be a dictionary")

        if 'properties' in op:
            if not isinstance(op['properties'], dict):
                errors.append(f"Operator '{op_name}': 'properties' must be a dictionary")

    is_valid = len(errors) == 0
    return is_valid, errors, warnings


def validate_unique_names(operators: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
    """Check that all operator names are unique within the pipeline."""
    errors = []
    name_counts = defaultdict(int)

    for op in operators:
        name = op.get('name')
        if name:
            name_counts[name] += 1

    duplicates = [name for name, count in name_counts.items() if count > 1]

    for dup_name in duplicates:
        count = name_counts[dup_name]
        errors.append(f"Duplicate operator name '{dup_name}' appears {count} times")

    is_valid = len(duplicates) == 0
    return is_valid, errors

def validate_type_representations(operators: List[Dict[str, Any]]) -> Tuple[bool, List[str], List[str]]:
    """Validate that all type representations in the pipeline are correctly formatted."""
    errors = []
    warnings = []

    for op in operators:
        op_name = op.get('name', '<unnamed>')

        if 'input' in op and isinstance(op['input'], dict):
            if 'fields' in op['input'] and isinstance(op['input']['fields'], dict):
                for field_name, field_type in op['input']['fields'].items():
                    if isinstance(field_type, str):
                        is_valid, parsed, error = parse_type_string(field_type)
                        if not is_valid:
                            errors.append(f"Operator '{op_name}' input field '{field_name}': {error}")

        if 'output' in op and isinstance(op['output'], dict):
            for field_name, field_type in op['output'].items():
                if field_name == 'type':
                    continue
                if isinstance(field_type, str):
                    is_valid, parsed, error = parse_type_string(field_type)
                    if not is_valid:
                        errors.append(f"Operator '{op_name}' output field '{field_name}': {error}")

    is_valid = len(errors) == 0
    return is_valid, errors, warnings


def validate_schema_consistency(operators: List[Dict[str, Any]], dataset_schema: Optional[Dict[str, Any]] = None) -> Tuple[bool, List[str], List[str]]:
    """Validate schema consistency through the pipeline."""
    errors = []
    warnings = []
    strict_mode = dataset_schema is not None

    available_fields = {}  # field_name -> type_string

    if dataset_schema and "fields" in dataset_schema:
        available_fields.update(dataset_schema["fields"])

    for i, op in enumerate(operators):
        op_name = op.get('name', f'<operator {i}>')

        if 'input' in op and isinstance(op['input'], dict):
            input_fields = op['input'].get('fields', {})
            if isinstance(input_fields, dict):
                for field_name, expected_type in input_fields.items():
                    if field_name in available_fields:
                        available_type = available_fields[field_name]
                        is_compatible, reason = check_type_compatibility(available_type, expected_type)
                        if not is_compatible:
                            errors.append(
                                f"Operator '{op_name}' expects field '{field_name}' "
                                f"with type '{expected_type}' but pipeline provides '{available_type}': {reason}"
                            )
                    else:
                        if strict_mode:
                            errors.append(
                                f"Operator '{op_name}' references field '{field_name}' "
                                f"which does not exist in dataset or upstream operators"
                            )
                        elif expected_type != 'Unknown':
                            warnings.append(
                                f"Operator '{op_name}' references field '{field_name}' "
                                f"which may not exist in pipeline"
                            )

        if 'output' in op and isinstance(op['output'], dict):
            for field_name, field_type in op['output'].items():
                if field_name == 'type':
                    continue
                available_fields[field_name] = field_type

        op_type = op.get('type', '')

        if op_type == 'Unnest':
            unnest_key = op.get('properties', {}).get('unnest_key')
            if unnest_key and unnest_key in available_fields:
                type_str = available_fields.get(unnest_key, '')
                dict_match = re.search(r'Dict\[\{(.+?)\}\]', type_str)
                if dict_match:
                    fields_str = dict_match.group(1)
                    field_pairs = re.findall(r'(\w+)\s*:\s*(\w+)', fields_str)
                    for field_name, field_type in field_pairs:
                        available_fields[field_name] = field_type

    is_valid = len(errors) == 0
    return is_valid, errors, warnings


def validate_dataset_compliance(operators: List[Dict[str, Any]], dataset_schema: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate strict compliance with dataset schema (no Unknown types allowed)."""
    errors = []
    dataset_fields = dataset_schema.get('fields', {})

    for op in operators:
        op_name = op.get('name', '<unnamed>')

        # Check input fields for Unknown types
        if 'input' in op and isinstance(op['input'], dict):
            input_fields = op['input'].get('fields', {})
            if isinstance(input_fields, dict):
                for field_name, field_type in input_fields.items():
                    if field_type == "Unknown":
                        errors.append(
                            f"Operator '{op_name}': Field '{field_name}' has Unknown type - "
                            f"not allowed in strict mode (dataset schema provided)"
                        )

        # Check output fields for Unknown types
        if 'output' in op and isinstance(op['output'], dict):
            for field_name, field_type in op['output'].items():
                if field_name != 'type' and field_type == "Unknown":
                    errors.append(
                        f"Operator '{op_name}': Output field '{field_name}' has Unknown type - "
                        f"not allowed in strict mode (dataset schema provided)"
                    )

    is_valid = len(errors) == 0
    return is_valid, errors


def validate_prompt_fields(operators: List[Dict[str, Any]]) -> Tuple[bool, List[str], List[str]]:
    """
    Validate prompt field references in operators.

    Checks that:
    1. Fields referenced in prompts (via Jinja2) exist in input schema
    2. Prompts don't have suspicious bare field references
    """
    errors = []
    warnings = []

    # Import the Jinja2 field extraction utility from schema module
    try:
        import sys
        import os
        schema_module_path = os.path.join(os.path.dirname(__file__), '..', 'convert', 'docetl')
        sys.path.insert(0, schema_module_path)
        from schema import _extract_fields_from_jinja2
    except ImportError:
        # If we can't import the schema module, skip this validation
        warnings.append("Could not import Jinja2 field extraction utility - skipping prompt validation")
        return True, [], warnings

    for op in operators:
        op_name = op.get('name', '<unnamed>')

        # Check if operator has a prompt in properties
        if 'properties' not in op or not isinstance(op['properties'], dict):
            continue

        prompt = op['properties'].get('prompt')
        if not prompt or not isinstance(prompt, str):
            continue

        # Extract input fields
        input_fields = set()
        if 'input' in op and isinstance(op['input'], dict):
            if 'fields' in op['input'] and isinstance(op['input']['fields'], dict):
                input_fields = set(op['input']['fields'].keys())

        # Extract Jinja2 field references from prompt
        jinja_fields = _extract_fields_from_jinja2(prompt)

        # Check that referenced fields exist in input
        for field in jinja_fields:
            if field not in input_fields:
                errors.append(
                    f"Operator '{op_name}': prompt references field '{field}' "
                    f"which is not defined in input schema"
                )

        # Check for suspicious bare field references
        # (similar to DocETL checker but adapted for Abstract layer)
        suspicious_patterns = [
            (r'`(\w+)`', 'backtick-quoted field'),
            (r'field\s+["\'](\w+)["\']', 'field in quotes'),
            (r'the\s+(\w+)\s+field', 'descriptive field reference'),
        ]

        likely_field_names = {'text', 'src', 'content', 'document', 'data', 'input',
                             'title', 'description', 'body', 'message', 'name',
                             'value', 'field', 'item', 'record', 'doc'}

        for pattern, pattern_desc in suspicious_patterns:
            matches = re.findall(pattern, prompt, re.IGNORECASE)
            for match in matches:
                # Check if this looks like a field reference and is not already in Jinja2 format
                if match.lower() in likely_field_names and match not in jinja_fields:
                    warnings.append(
                        f"Operator '{op_name}': Possible bare field reference '{match}' "
                        f"found ({pattern_desc}). Should use {{{{ input.{match} }}}} instead?"
                    )

    is_valid = len(errors) == 0
    return is_valid, errors, warnings


def extract_fields_from_operator(operator: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
    """Extract all input and output fields from an operator."""
    result = {'input': {}, 'output': {}}

    # Extract input fields
    if 'input' in operator and isinstance(operator['input'], dict):
        input_fields = operator['input'].get('fields', {})
        if isinstance(input_fields, dict):
            result['input'] = input_fields.copy()

    # Extract output fields
    if 'output' in operator and isinstance(operator['output'], dict):
        for field_name, field_type in operator['output'].items():
            if field_name != 'type':  # Skip special keys
                result['output'][field_name] = field_type

    return result


# ============================================================================
# Main Validation Function
# ============================================================================

def validate_pipeline(operators: List[Dict[str, Any]], verbose: bool = False, dataset_schema: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Perform comprehensive validation of an abstract pipeline."""
    all_errors = []
    all_warnings = []
    validation_results = {}
    mode = "STRICT" if dataset_schema else "NORMAL"

    # 1. Validate operator fields
    fields_valid, fields_errors, fields_warnings = validate_operator_fields(operators)
    all_errors.extend(fields_errors)
    all_warnings.extend(fields_warnings)
    validation_results['fields'] = {
        'valid': fields_valid,
        'errors': len(fields_errors),
        'warnings': len(fields_warnings)
    }

    # 2. Validate unique names
    names_valid, names_errors = validate_unique_names(operators)
    all_errors.extend(names_errors)
    validation_results['unique_names'] = {
        'valid': names_valid,
        'errors': len(names_errors)
    }

    # 3. Validate type representations
    types_valid, types_errors, types_warnings = validate_type_representations(operators)
    all_errors.extend(types_errors)
    all_warnings.extend(types_warnings)
    validation_results['type_syntax'] = {
        'valid': types_valid,
        'errors': len(types_errors),
        'warnings': len(types_warnings)
    }

    # 4. Validate schema consistency
    schema_valid, schema_errors, schema_warnings = validate_schema_consistency(operators, dataset_schema)
    all_errors.extend(schema_errors)
    all_warnings.extend(schema_warnings)
    validation_results['schema_consistency'] = {
        'valid': schema_valid,
        'errors': len(schema_errors),
        'warnings': len(schema_warnings)
    }

    # 5. Validate prompt field references
    prompt_valid, prompt_errors, prompt_warnings = validate_prompt_fields(operators)
    all_errors.extend(prompt_errors)
    all_warnings.extend(prompt_warnings)
    validation_results['prompt_fields'] = {
        'valid': prompt_valid,
        'errors': len(prompt_errors),
        'warnings': len(prompt_warnings)
    }

    # 6. In strict mode, validate dataset compliance
    if dataset_schema:
        compliance_valid, compliance_errors = validate_dataset_compliance(operators, dataset_schema)
        all_errors.extend(compliance_errors)
        validation_results['dataset_compliance'] = {
            'valid': compliance_valid,
            'errors': len(compliance_errors)
        }

    # Create summary
    is_valid = len(all_errors) == 0
    summary = {
        'mode': mode,
        'total_operators': len(operators),
        'total_errors': len(all_errors),
        'total_warnings': len(all_warnings),
        'validation_checks': validation_results
    }

    result = {
        'is_valid': is_valid,
        'mode': mode,
        'errors': all_errors,
        'warnings': all_warnings,
        'summary': summary
    }

    # Add dataset schema fields to result if in strict mode
    if dataset_schema and verbose:
        result['dataset_fields'] = list(dataset_schema.get('fields', {}).keys())

    if verbose:
        # Add additional details
        result['operators_analyzed'] = [op.get('name', f'<unnamed {i}>')
                                       for i, op in enumerate(operators)]
        result['field_inventory'] = {}
        for i, op in enumerate(operators):
            op_name = op.get('name', f'<operator {i}>')
            result['field_inventory'][op_name] = extract_fields_from_operator(op)

    return result


# ============================================================================
# CLI Interface
# ============================================================================

def format_validation_report(report: Dict[str, Any], format: str = 'text') -> str:
    """Format validation report for display."""
    if format == 'json':
        return json.dumps(report, indent=2)

    # Text format
    lines = []
    lines.append("=" * 60)
    lines.append("ABSTRACT PIPELINE VALIDATION REPORT")
    lines.append("=" * 60)

    # Summary
    summary = report['summary']
    lines.append(f"\nValidation Mode: {summary.get('mode', 'NORMAL')}")
    lines.append(f"\nPipeline Summary:")
    lines.append(f"  Total Operators: {summary['total_operators']}")
    lines.append(f"  Total Errors: {summary['total_errors']}")
    lines.append(f"  Total Warnings: {summary['total_warnings']}")

    # Validation Results
    lines.append(f"\nValidation Results:")
    status_symbol = lambda valid: "✓" if valid else "✗"
    for check_name, check_result in summary['validation_checks'].items():
        symbol = status_symbol(check_result['valid'])
        lines.append(f"  {symbol} {check_name.replace('_', ' ').title()}: "
                    f"{check_result['errors']} errors, {check_result.get('warnings', 0)} warnings")

    # Overall Status
    lines.append(f"\nOverall Status: {'VALID ✓' if report['is_valid'] else 'INVALID ✗'}")

    # Errors
    if report['errors']:
        lines.append("\n" + "=" * 60)
        lines.append("ERRORS:")
        lines.append("=" * 60)
        for i, error in enumerate(report['errors'], 1):
            lines.append(f"{i}. {error}")

    # Warnings
    if report['warnings']:
        lines.append("\n" + "=" * 60)
        lines.append("WARNINGS:")
        lines.append("=" * 60)
        for i, warning in enumerate(report['warnings'], 1):
            lines.append(f"{i}. {warning}")

    # Verbose information
    if 'operators_analyzed' in report:
        lines.append("\n" + "=" * 60)
        lines.append("OPERATORS ANALYZED:")
        lines.append("=" * 60)
        for op_name in report['operators_analyzed']:
            lines.append(f"  - {op_name}")

    if 'field_inventory' in report:
        lines.append("\n" + "=" * 60)
        lines.append("FIELD INVENTORY:")
        lines.append("=" * 60)
        for op_name, fields in report['field_inventory'].items():
            lines.append(f"\n  {op_name}:")
            if fields['input']:
                lines.append("    Input fields:")
                for field_name, field_type in fields['input'].items():
                    lines.append(f"      - {field_name}: {field_type}")
            if fields['output']:
                lines.append("    Output fields:")
                for field_name, field_type in fields['output'].items():
                    lines.append(f"      - {field_name}: {field_type}")

    lines.append("\n" + "=" * 60)
    return "\n".join(lines)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Static Syntax Checker for Abstract Pipelines",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate a pipeline with default settings
  python static_checker.py pipeline.json

  # Verbose output with field inventory
  python static_checker.py pipeline.json --verbose

  # Only show errors (quiet mode)
  python static_checker.py pipeline.json --quiet

  # Output as JSON
  python static_checker.py pipeline.json --format json
        """
    )

    parser.add_argument(
        'pipeline_file',
        type=str,
        help='Path to the abstract pipeline JSON file'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show detailed validation information'
    )

    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Only show errors (no warnings or summary)'
    )

    parser.add_argument(
        '--format', '-f',
        choices=['text', 'json'],
        default='text',
        help='Output format (default: text)'
    )

    args = parser.parse_args()

    # Load pipeline file
    pipeline_path = Path(args.pipeline_file)
    if not pipeline_path.exists():
        print(f"Error: Pipeline file not found: {args.pipeline_file}", file=sys.stderr)
        sys.exit(1)

    try:
        with open(pipeline_path, 'r') as f:
            pipeline_data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in pipeline file: {e}", file=sys.stderr)
        sys.exit(1)

    # Extract operators
    if 'operators' not in pipeline_data:
        print("Error: Pipeline file must contain 'operators' field", file=sys.stderr)
        sys.exit(1)

    operators = pipeline_data['operators']

    # Validate pipeline
    report = validate_pipeline(operators, verbose=args.verbose)

    # Filter report if quiet mode
    if args.quiet:
        filtered_report = {
            'is_valid': report['is_valid'],
            'errors': report['errors'],
            'warnings': [],  # Suppress warnings in quiet mode
            'summary': {
                'total_operators': report['summary']['total_operators'],
                'total_errors': len(report['errors']),
                'total_warnings': 0,
                'validation_checks': {}
            }
        }
        output = format_validation_report(filtered_report, args.format)
    else:
        output = format_validation_report(report, args.format)

    # Print report
    print(output)

    # Exit with appropriate code
    sys.exit(0 if report['is_valid'] else 1)


if __name__ == '__main__':
    main()