"""
DocETL Validation Implementation

Provides validation functions for DocETL pipelines using SchemaTracker.
"""

import yaml
import json
import sys
import argparse
from pathlib import Path
from typing import Dict, Any

try:
    from ..schema_tracking import DocETLSchemaTracker
except ImportError:
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from baselines.docetl_support.schema_tracking import DocETLSchemaTracker


def check_pipeline_file(pipeline_path: str, verbose: bool = False) -> Dict[str, Any]:
    """
    Check a DocETL pipeline file and return validation results.

    Compatibility function for static_checker.py migration.

    Args:
        pipeline_path: Path to pipeline YAML file
        verbose: Enable verbose logging

    Returns:
        dict: {"score": 1|0, "errors": [...], "warnings": [...]}
    """
    try:
        path = Path(pipeline_path)
        if not path.exists():
            return {
                "score": 0,
                "errors": [{
                    "type": "file_not_found",
                    "message": f"File not found: {pipeline_path}"
                }],
                "warnings": []
            }

        with open(pipeline_path, 'r') as f:
            pipeline_config = yaml.safe_load(f)

        if not isinstance(pipeline_config, dict):
            return {
                "score": 0,
                "errors": [{
                    "type": "yaml_structure_error",
                    "message": "Root element must be a dictionary"
                }],
                "warnings": []
            }

        # Get dataset path
        datasets = pipeline_config.get('datasets', {})
        if not datasets:
            return {
                "score": 0,
                "errors": [{
                    "type": "missing_dataset",
                    "message": "Pipeline has no datasets"
                }],
                "warnings": []
            }

        first_dataset_name = next(iter(datasets.keys()))
        first_dataset = datasets[first_dataset_name]
        dataset_path = first_dataset.get('path')

        if not dataset_path:
            return {
                "score": 0,
                "errors": [{
                    "type": "missing_dataset_path",
                    "message": "Dataset has no path"
                }],
                "warnings": []
            }

        # Resolve dataset path relative to pipeline file
        pipeline_dir = path.parent
        full_dataset_path = pipeline_dir / dataset_path

        if not full_dataset_path.exists():
            return {
                "score": 0,
                "errors": [{
                    "type": "dataset_not_found",
                    "message": f"Dataset file not found: {full_dataset_path}"
                }],
                "warnings": []
            }

        # Create schema tracker and validate
        tracker = DocETLSchemaTracker.from_dataset(str(full_dataset_path), verbose)
        operators = pipeline_config.get('operations', [])

        if not operators:
            return {
                "score": 0,
                "errors": [{
                    "type": "no_operators",
                    "message": "Pipeline has no operators"
                }],
                "warnings": []
            }

        # Validate pipeline
        passed, errors, warnings = tracker.validate_pipeline(operators, continue_on_error=True)

        # Format errors and warnings
        formatted_errors = []
        for error in errors:
            if isinstance(error, str):
                formatted_errors.append({"type": "validation_error", "message": error})
            elif isinstance(error, dict):
                formatted_errors.append(error)
            else:
                formatted_errors.append({"type": "validation_error", "message": str(error)})

        formatted_warnings = []
        for warning in warnings:
            if isinstance(warning, str):
                formatted_warnings.append({"type": "warning", "message": warning})
            elif isinstance(warning, dict):
                formatted_warnings.append(warning)
            else:
                formatted_warnings.append({"type": "warning", "message": str(warning)})

        return {
            "score": 1 if passed else 0,
            "errors": formatted_errors,
            "warnings": formatted_warnings
        }

    except Exception as e:
        return {
            "score": 0,
            "errors": [{
                "type": "validation_error",
                "message": f"Validation failed: {str(e)}"
            }],
            "warnings": []
        }


def check_pipeline_string(yaml_content: str, verbose: bool = False) -> Dict[str, Any]:
    """
    Check YAML content string and return validation results.

    Compatibility function for static_checker.py migration.

    Args:
        yaml_content: Pipeline YAML content as string
        verbose: Enable verbose logging

    Returns:
        dict: {"score": 1|0, "errors": [...], "warnings": [...]}
    """
    try:
        pipeline_config = yaml.safe_load(yaml_content)

        if not isinstance(pipeline_config, dict):
            return {
                "score": 0,
                "errors": [{
                    "type": "yaml_structure_error",
                    "message": "Root element must be a dictionary"
                }],
                "warnings": []
            }

        # For string validation, we need a dataset schema
        # Try to infer from pipeline or require it to be provided
        datasets = pipeline_config.get('datasets', {})
        if not datasets:
            return {
                "score": 0,
                "errors": [{
                    "type": "missing_dataset",
                    "message": "Pipeline has no datasets (cannot validate from string without dataset)"
                }],
                "warnings": []
            }

        # This is a limitation: we can't load dataset from string validation
        # Return a partial validation result
        return {
            "score": 0,
            "errors": [{
                "type": "unsupported_operation",
                "message": "String validation requires dataset access. Use check_pipeline_file instead."
            }],
            "warnings": []
        }

    except yaml.YAMLError as e:
        return {
            "score": 0,
            "errors": [{
                "type": "yaml_syntax_error",
                "message": f"Invalid YAML syntax: {str(e)}"
            }],
            "warnings": []
        }
    except Exception as e:
        return {
            "score": 0,
            "errors": [{
                "type": "validation_error",
                "message": f"Validation failed: {str(e)}"
            }],
            "warnings": []
        }


def main():
    """Command-line interface for DocETL pipeline validation."""
    parser = argparse.ArgumentParser(
        description='Validate DocETL pipeline configuration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s pipeline.yaml
  %(prog)s pipeline.yaml --verbose
  %(prog)s pipeline.yaml -v
        """
    )
    parser.add_argument(
        'pipeline_path',
        help='Path to the pipeline YAML file'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output'
    )

    args = parser.parse_args()

    # Run validation
    result = check_pipeline_file(args.pipeline_path, args.verbose)

    # Print results
    print(json.dumps(result, indent=2))

    # Exit with appropriate code
    sys.exit(0 if result['score'] == 1 else 1)


__all__ = [
    'check_pipeline_file',
    'check_pipeline_string',
    'DocETLSchemaTracker',
]


if __name__ == "__main__":
    main()
