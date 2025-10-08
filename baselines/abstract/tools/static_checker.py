#!/usr/bin/env python3
"""
Static Syntax Checker for Abstract Pipeline

This tool validates the correctness of abstract pipeline operators after conversion.
It performs comprehensive checks including field validation, type checking, and schema consistency.

Usage:
    As a library:
        from static_checker import validate_pipeline
        result = validate_pipeline(operators)

    As CLI tool:
        python static_checker.py pipeline.json [--verbose] [--quiet] [--format text|json]
"""

import json
import sys
import argparse
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from pathlib import Path
from collections import defaultdict


# ============================================================================
# Core Validation Functions
# ============================================================================

def validate_operator_fields(operators: List[Dict[str, Any]]) -> Tuple[bool, List[str], List[str]]:
    """
    Validate that each operator has all required fields with correct structure.

    Args:
        operators: List of operator dictionaries

    Returns:
        Tuple of (is_valid, errors, warnings)
    """
    errors = []
    warnings = []

    required_fields = ['name', 'type', 'source', 'properties', 'input', 'output']

    for i, op in enumerate(operators):
        op_name = op.get('name', f'<unnamed operator {i}>')

        for field in required_fields:
            if field not in op:
                errors.append(f"Operator '{op_name}' missing required field: {field}")

        # Validate source structure
        if 'source' in op:
            source = op['source']
            if not isinstance(source, dict):
                errors.append(f"Operator '{op_name}': 'source' must be a dictionary")
            else:
                if 'system' not in source:
                    errors.append(f"Operator '{op_name}': 'source' missing 'system' field")
                if 'name' not in source:
                    errors.append(f"Operator '{op_name}': 'source' missing 'name' field")

        # Validate input structure
        if 'input' in op:
            input_schema = op['input']
            if not isinstance(input_schema, dict):
                errors.append(f"Operator '{op_name}': 'input' must be a dictionary")
            else:
                # Check for type field when input has fields
                if 'fields' in input_schema and 'type' not in input_schema:
                    warnings.append(f"Operator '{op_name}': 'input' has 'fields' but no 'type' specification")

                # Validate fields structure
                if 'fields' in input_schema:
                    if not isinstance(input_schema['fields'], dict):
                        errors.append(f"Operator '{op_name}': 'input.fields' must be a dictionary")

        # Validate output structure
        if 'output' in op:
            output_schema = op['output']
            if not isinstance(output_schema, dict):
                errors.append(f"Operator '{op_name}': 'output' must be a dictionary")

        # Validate properties structure
        if 'properties' in op:
            if not isinstance(op['properties'], dict):
                errors.append(f"Operator '{op_name}': 'properties' must be a dictionary")

    is_valid = len(errors) == 0
    return is_valid, errors, warnings


def validate_unique_names(operators: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
    """
    Check that all operator names are unique within the pipeline.

    Args:
        operators: List of operator dictionaries

    Returns:
        Tuple of (is_valid, errors)
    """
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


def parse_type_string(type_str: str) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
    """
    Parse a type string and validate its syntax.

    Args:
        type_str: Type string to parse

    Returns:
        Tuple of (is_valid, parsed_type, error_message)
    """
    if not type_str:
        return False, None, "Empty type string"

    # Basic types
    basic_types = ['String', 'Integer', 'Float', 'Boolean', 'Unknown', 'Array', 'Object', 'Null']
    if type_str in basic_types:
        return True, {'type': type_str}, None

    # List type: List[ElementType]
    list_match = re.match(r'^List\[(.+)\]$', type_str)
    if list_match:
        element_type_str = list_match.group(1)
        # Recursively parse element type
        is_valid, element_type, error = parse_type_string(element_type_str)
        if not is_valid:
            return False, None, f"Invalid List element type: {error}"
        return True, {'type': 'List', 'element_type': element_type}, None

    # Dict type with fields: Dict[{field1: Type1, field2: Type2}]
    dict_match = re.match(r'^Dict\[\{(.+)\}\]$', type_str)
    if dict_match:
        fields_str = dict_match.group(1)
        fields = {}

        # Parse field definitions
        # Simple parser for field: Type pairs
        field_pairs = re.findall(r'(\w+)\s*:\s*([^,}]+)', fields_str)
        for field_name, field_type_str in field_pairs:
            field_type_str = field_type_str.strip()
            is_valid, field_type, error = parse_type_string(field_type_str)
            if not is_valid:
                return False, None, f"Invalid type for field '{field_name}': {error}"
            fields[field_name] = field_type

        if not fields:
            return False, None, "Dict type must have at least one field"

        return True, {'type': 'Dict', 'fields': fields}, None

    # Simple Dict without fields
    if type_str == 'Dict':
        return True, {'type': 'Dict'}, None

    # Check for common mistakes
    if type_str.startswith('list[') or type_str.startswith('array['):
        return False, None, f"Type should use 'List' not '{type_str.split('[')[0]}'"

    if type_str.startswith('dict[') or type_str.startswith('object['):
        return False, None, f"Type should use 'Dict' not '{type_str.split('[')[0]}'"

    if '{' in type_str or '}' in type_str:
        if not type_str.startswith('Dict[{'):
            return False, None, "Dict fields must be wrapped in 'Dict[{...}]'"

    # Unknown type format
    return False, None, f"Unrecognized type format: {type_str}"


def validate_type_representations(operators: List[Dict[str, Any]]) -> Tuple[bool, List[str], List[str]]:
    """
    Validate that all type representations in the pipeline are correctly formatted.

    Args:
        operators: List of operator dictionaries

    Returns:
        Tuple of (is_valid, errors, warnings)
    """
    errors = []
    warnings = []

    for op in operators:
        op_name = op.get('name', '<unnamed>')

        # Check input field types
        if 'input' in op and isinstance(op['input'], dict):
            if 'fields' in op['input'] and isinstance(op['input']['fields'], dict):
                for field_name, field_type in op['input']['fields'].items():
                    if isinstance(field_type, str):
                        is_valid, parsed, error = parse_type_string(field_type)
                        if not is_valid:
                            errors.append(f"Operator '{op_name}' input field '{field_name}': {error}")

        # Check output field types
        if 'output' in op and isinstance(op['output'], dict):
            for field_name, field_type in op['output'].items():
                if field_name == 'type':  # Skip special 'type' key
                    continue
                if isinstance(field_type, str):
                    is_valid, parsed, error = parse_type_string(field_type)
                    if not is_valid:
                        errors.append(f"Operator '{op_name}' output field '{field_name}': {error}")

    is_valid = len(errors) == 0
    return is_valid, errors, warnings


def check_type_compatibility(type1: str, type2: str) -> Tuple[bool, Optional[str]]:
    """
    Check if two types are compatible.

    Args:
        type1: First type string
        type2: Second type string

    Returns:
        Tuple of (is_compatible, reason_if_not)
    """
    # Unknown type is compatible with anything
    if type1 == 'Unknown' or type2 == 'Unknown':
        return True, None

    # Exact match
    if type1 == type2:
        return True, None

    # Parse both types
    is_valid1, parsed1, _ = parse_type_string(type1)
    is_valid2, parsed2, _ = parse_type_string(type2)

    if not is_valid1 or not is_valid2:
        return False, "Invalid type format"

    # Check base type compatibility
    base_type1 = parsed1.get('type')
    base_type2 = parsed2.get('type')

    if base_type1 != base_type2:
        # Array is compatible with List
        if {base_type1, base_type2} == {'Array', 'List'}:
            return True, None
        # Object is compatible with Dict
        if {base_type1, base_type2} == {'Object', 'Dict'}:
            return True, None
        return False, f"Type mismatch: {base_type1} vs {base_type2}"

    # For List types, check element type compatibility
    if base_type1 == 'List':
        elem1 = parsed1.get('element_type', {})
        elem2 = parsed2.get('element_type', {})
        elem_type1 = elem1.get('type', 'Unknown') if isinstance(elem1, dict) else 'Unknown'
        elem_type2 = elem2.get('type', 'Unknown') if isinstance(elem2, dict) else 'Unknown'

        if elem_type1 != elem_type2 and elem_type1 != 'Unknown' and elem_type2 != 'Unknown':
            return False, f"List element type mismatch: {elem_type1} vs {elem_type2}"

    # For Dict types, would need more complex field checking
    # For now, we'll consider them compatible if both are Dict

    return True, None


def validate_schema_consistency(operators: List[Dict[str, Any]], dataset_schema: Optional[Dict[str, Any]] = None) -> Tuple[bool, List[str], List[str]]:
    """
    Validate schema consistency through the pipeline.

    Args:
        operators: List of operator dictionaries
        dataset_schema: Optional dataset schema for strict mode

    Returns:
        Tuple of (is_valid, errors, warnings)
    """
    errors = []
    warnings = []
    strict_mode = dataset_schema is not None

    # Track available fields and their types through the pipeline
    available_fields = {}  # field_name -> type_string

    # Initialize with dataset schema fields if provided
    if dataset_schema and "fields" in dataset_schema:
        available_fields.update(dataset_schema["fields"])

    for i, op in enumerate(operators):
        op_name = op.get('name', f'<operator {i}>')

        # Check input fields against available fields
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
                        # Field doesn't exist in available fields
                        if strict_mode:
                            # In strict mode, this is an error
                            errors.append(
                                f"Operator '{op_name}' references field '{field_name}' "
                                f"which does not exist in dataset or upstream operators"
                            )
                        elif expected_type != 'Unknown':
                            # In normal mode, only warn for non-Unknown types
                            warnings.append(
                                f"Operator '{op_name}' references field '{field_name}' "
                                f"which may not exist in pipeline"
                            )

        # Update available fields with this operator's output
        if 'output' in op and isinstance(op['output'], dict):
            for field_name, field_type in op['output'].items():
                if field_name == 'type':  # Skip special keys
                    continue
                available_fields[field_name] = field_type

        # Special handling for certain operator types
        op_type = op.get('type', '')

        # Unnest operator exposes fields from nested structures
        if op_type == 'Unnest':
            unnest_key = op.get('properties', {}).get('unnest_key')
            if unnest_key and unnest_key in available_fields:
                # Try to extract nested fields from the type
                type_str = available_fields.get(unnest_key, '')
                # Parse for List[Dict[{fields}]] pattern
                dict_match = re.search(r'Dict\[\{(.+?)\}\]', type_str)
                if dict_match:
                    fields_str = dict_match.group(1)
                    field_pairs = re.findall(r'(\w+)\s*:\s*(\w+)', fields_str)
                    for field_name, field_type in field_pairs:
                        available_fields[field_name] = field_type

    is_valid = len(errors) == 0
    return is_valid, errors, warnings


def validate_dataset_compliance(operators: List[Dict[str, Any]], dataset_schema: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate strict compliance with dataset schema.

    In strict mode:
    - No fields with Unknown types are allowed
    - All fields must originate from dataset or be generated by an operator

    Args:
        operators: List of operator dictionaries
        dataset_schema: Dataset schema with field definitions

    Returns:
        Tuple of (is_valid, errors)
    """
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


def extract_fields_from_operator(operator: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
    """
    Extract all input and output fields from an operator.

    Args:
        operator: Operator dictionary

    Returns:
        Dictionary with 'input' and 'output' keys containing field mappings
    """
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
    """
    Perform comprehensive validation of an abstract pipeline.

    Args:
        operators: List of operator dictionaries
        verbose: Whether to include detailed information
        dataset_schema: Optional dataset schema for strict mode validation

    Returns:
        Validation report with is_valid, errors, warnings, and summary
    """
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

    # 5. In strict mode, validate dataset compliance
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
    """
    Format validation report for display.

    Args:
        report: Validation report dictionary
        format: Output format ('text' or 'json')

    Returns:
        Formatted string
    """
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