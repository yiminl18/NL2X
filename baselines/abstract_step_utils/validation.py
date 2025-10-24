"""
Unified static validation utilities for abstract and base system pipelines.

Usage:
    from baselines.abstract_step_utils.validation import validate_pipeline_static

    # Validate a DocETL pipeline
    result = validate_pipeline_static("docetl", "pipeline.yaml", verbose=True)

    # Check result
    if result["passed"]:
        print("Validation passed!")
    else:
        print(f"Errors: {result['errors']}")
        print(f"Warnings: {result['warnings']}")
"""

import json
import logging
from typing import Any, Dict, List, Optional
from pathlib import Path

from ..abstract.support import BaseSystem


def validate_pipeline_static(
    system_name: str,
    pipeline_path: str,
    verbose: bool = False,
    logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """
    Unified static validation entry point for pipelines.

    This function validates pipelines across different systems using a unified interface.

    Args:
        system_name: System name - "abstract" | "docetl" | "lotus"
        pipeline_path: Path to the pipeline configuration file
        verbose: Whether to output verbose logs
        logger: Optional logger for structured output

    Returns:
        dict: Validation result with standardized format
            {
                "score": 1 | 0,        # 1 = passed, 0 = failed
                "passed": bool,        # Convenience field (score == 1)
                "errors": List[str],   # List of error messages
                "warnings": List[str]  # List of warning messages
            }
    """
    # Normalize system name to lowercase
    system_name = system_name.lower()

    # Map system names to BaseSystem enum
    system_map = {
        'docetl': BaseSystem.DOCETL,
        # Future: Add other systems
        # 'lotus': BaseSystem.LOTUS,
    }

    if system_name not in system_map and system_name != 'abstract':
        supported = ", ".join(list(system_map.keys()) + ['abstract'])
        raise ValueError(
            f"Unknown system '{system_name}'. Supported systems: {supported}"
        )

    # ANSI color codes
    ORANGE = '\033[33m'
    RESET = '\033[0m'

    # Log validation start
    if verbose and logger:
        logger.info(f"Starting {system_name} static validation for: {pipeline_path}")

    # Dispatch to appropriate validator
    if system_name == 'abstract':
        # Abstract layer pipeline (basic structural validation)
        result = _validate_abstract_pipeline(pipeline_path, verbose, logger)
    else:
        # Base system pipeline validation using factory function
        system = system_map[system_name]
        result = _validate_base_system_pipeline(system, pipeline_path, verbose)

    # Log validation result
    if verbose and logger:
        if result["passed"]:
            logger.info(f"✓ {system_name.capitalize()} validation passed")
        else:
            # Use orange color for error messages
            logger.error(f"{ORANGE}⚠ {system_name.capitalize()} validation found {len(result['errors'])} error(s){RESET}")
            for error in result["errors"]:
                # Extract message from error dict if available, otherwise use string representation
                error_msg = error.get('message', str(error)) if isinstance(error, dict) else str(error)
                logger.error(f"{ORANGE}  - {error_msg}{RESET}")
            if result.get("warnings"):
                logger.warning(f"Warnings ({len(result['warnings'])}): ")
                for warning in result["warnings"]:
                    # Extract message from warning dict if available, otherwise use string representation
                    warning_msg = warning.get('message', str(warning)) if isinstance(warning, dict) else str(warning)
                    logger.warning(f"  - {warning_msg}")

    return result


def _validate_base_system_pipeline(
    system: BaseSystem,
    pipeline_path: str,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Validate a base system pipeline using SchemaTracker directly.

    Args:
        system: Target base system
        pipeline_path: Path to pipeline file
        verbose: Whether to output verbose logs

    Returns:
        Validation result dict
    """
    try:
        # Load pipeline configuration
        pipeline_config = _load_pipeline_config(pipeline_path, system)

        if system == BaseSystem.DOCETL:
            # Import DocETL-specific modules
            from ..docetl_support.schema_tracking import DocETLSchemaTracker

            # Get dataset path
            dataset_path = _get_dataset_path(pipeline_config, pipeline_path)
            if not dataset_path:
                return {
                    "score": 0,
                    "passed": False,
                    "errors": [{"type": "missing_dataset", "message": "Cannot find dataset path in pipeline configuration"}],
                    "warnings": []
                }

            # Create schema tracker from dataset
            tracker = DocETLSchemaTracker.from_dataset(dataset_path, verbose)

            # Get operators
            operators = pipeline_config.get('operations', [])
            if not operators:
                return {
                    "score": 0,
                    "passed": False,
                    "errors": [{"type": "no_operators", "message": "Pipeline has no operators"}],
                    "warnings": []
                }

            # Validate pipeline
            passed, errors, warnings = tracker.validate_pipeline(operators, continue_on_error=True)

            # Format errors and warnings as dicts
            formatted_errors = [{"type": "schema_validation_error", "message": err} if isinstance(err, str) else err for err in errors]
            formatted_warnings = [{"type": "warning", "message": warn} if isinstance(warn, str) else warn for warn in warnings]

            return {
                "score": 1 if passed else 0,
                "passed": passed,
                "errors": formatted_errors,
                "warnings": formatted_warnings
            }
        else:
            raise ValueError(f"Unsupported base system: {system.value}")

    except FileNotFoundError:
        return {
            "score": 0,
            "passed": False,
            "errors": [{"type": "file_not_found", "message": f"Pipeline file not found: {pipeline_path}"}],
            "warnings": []
        }
    except Exception as e:
        return {
            "score": 0,
            "passed": False,
            "errors": [{"type": "validation_error", "message": f"Validation error: {str(e)}"}],
            "warnings": []
        }


def _validate_abstract_pipeline(
    pipeline_path: str,
    verbose: bool = False,
    logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """
    Validate an abstract layer pipeline structure.

    Abstract layer type checking has been removed in favor of base system validation.
    This now only validates basic pipeline structure.

    Args:
        pipeline_path: Path to abstract pipeline JSON file
        verbose: Whether to output verbose logs
        logger: Optional logger for output

    Returns:
        Validation result dict
    """
    try:
        # Load pipeline JSON
        with open(pipeline_path, 'r', encoding='utf-8') as f:
            pipeline_data = json.load(f)

        # Basic structural validation
        errors = []
        warnings = []

        # Check required fields
        if 'operators' not in pipeline_data:
            errors.append({"type": "missing_field", "message": "Pipeline missing 'operators' field"})
        else:
            operators = pipeline_data.get('operators', [])
            if not isinstance(operators, list):
                errors.append({"type": "type_error", "message": "'operators' must be a list"})
            elif len(operators) == 0:
                warnings.append({"type": "empty_operators", "message": "Pipeline has no operators"})
            else:
                # Validate each operator has required fields
                for i, op in enumerate(operators):
                    if not isinstance(op, dict):
                        errors.append({"type": "type_error", "message": f"Operator {i} is not a dictionary"})
                        continue
                    if 'name' not in op:
                        errors.append({"type": "missing_field", "message": f"Operator {i} missing 'name' field"})
                    if 'type' not in op:
                        errors.append({"type": "missing_field", "message": f"Operator {i} missing 'type' field"})

        # Check optional fields
        if 'name' not in pipeline_data:
            warnings.append({"type": "missing_field", "message": "Pipeline missing 'name' field"})

        if verbose and logger:
            if errors:
                logger.error(f"Abstract pipeline validation found {len(errors)} errors")
            if warnings:
                logger.warning(f"Abstract pipeline validation found {len(warnings)} warnings")

        score = 0 if errors else 1

        return {
            "score": score,
            "passed": score == 1,
            "errors": errors,
            "warnings": warnings
        }

    except FileNotFoundError:
        return {
            "score": 0,
            "passed": False,
            "errors": [{"type": "file_not_found", "message": f"Pipeline file not found: {pipeline_path}"}],
            "warnings": []
        }
    except json.JSONDecodeError as e:
        return {
            "score": 0,
            "passed": False,
            "errors": [{"type": "json_error", "message": f"Invalid JSON format: {str(e)}"}],
            "warnings": []
        }
    except Exception as e:
        return {
            "score": 0,
            "passed": False,
            "errors": [{"type": "validation_error", "message": f"Abstract validation error: {str(e)}"}],
            "warnings": []
        }


def _load_pipeline_config(pipeline_path: str, system: BaseSystem) -> Dict[str, Any]:
    """Load pipeline configuration based on system type."""
    path = Path(pipeline_path)

    if system == BaseSystem.DOCETL:
        # DocETL uses YAML
        import yaml
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    else:
        # Default to JSON
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)


def _get_dataset_path(pipeline_config: Dict[str, Any], pipeline_path: str) -> Optional[str]:
    """
    Extract dataset path from pipeline configuration.

    Args:
        pipeline_config: Pipeline configuration dictionary
        pipeline_path: Path to pipeline file (used to resolve relative paths)

    Returns:
        Absolute path to dataset file, or None if not found
    """
    datasets = pipeline_config.get('datasets', {})
    if not datasets:
        return None

    # Get first dataset
    first_dataset_name = next(iter(datasets.keys()))
    first_dataset = datasets[first_dataset_name]

    dataset_path = first_dataset.get('path')
    if not dataset_path:
        return None

    # Resolve path relative to pipeline file
    pipeline_dir = Path(pipeline_path).parent
    full_path = pipeline_dir / dataset_path

    if full_path.exists():
        return str(full_path.absolute())

    # Also try as absolute path
    if Path(dataset_path).exists():
        return str(Path(dataset_path).absolute())

    return None


def _format_errors(errors: List[Any]) -> List[Dict[str, Any]]:
    """Format ValidationError objects to dicts."""
    formatted = []
    for error in errors:
        if hasattr(error, 'type'):
            # ValidationError object
            error_dict = {
                "type": error.type,
                "message": error.message
            }
            if hasattr(error, 'operator') and error.operator:
                error_dict["operator"] = error.operator
            if hasattr(error, 'suggestions') and error.suggestions:
                error_dict["suggestions"] = error.suggestions
            formatted.append(error_dict)
        elif isinstance(error, dict):
            formatted.append(error)
        else:
            formatted.append({"type": "unknown_error", "message": str(error)})
    return formatted


def _format_warnings(warnings: List[Any]) -> List[Dict[str, Any]]:
    """Format ValidationWarning objects to dicts."""
    formatted = []
    for warning in warnings:
        if hasattr(warning, 'type'):
            # ValidationWarning object
            warning_dict = {
                "type": warning.type,
                "message": warning.message
            }
            if hasattr(warning, 'operator') and warning.operator:
                warning_dict["operator"] = warning.operator
            formatted.append(warning_dict)
        elif isinstance(warning, dict):
            formatted.append(warning)
        else:
            formatted.append({"type": "unknown_warning", "message": str(warning)})
    return formatted
