#!/usr/bin/env python3
"""
Unified static validation utilities for abstract and base system pipelines.
Usage:
    from baselines.abstract_step_utils.checker_utils import validate_pipeline_static

    # Validate an abstract pipeline
    result = validate_pipeline_static("abstract", "pipeline.json", verbose=True)

    # Validate a DocETL pipeline
    result = validate_pipeline_static("docetl", "pipeline.yaml", verbose=True)

    # Validate a Lotus pipeline
    result = validate_pipeline_static("lotus", "pipeline.py", verbose=True)

    # Check result
    if result["passed"]:
        print("Validation passed!")
    else:
        print(f"Errors: {result['errors']}")
        print(f"Warnings: {result['warnings']}")
"""

import json
import sys
import os
import logging
from typing import Any, Dict, List, Optional
from pathlib import Path


def _validate_abstract_pipeline(
    pipeline_path: str,
    verbose: bool = False,
    logger: Optional[logging.Logger] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Validate an abstract layer pipeline structure.

    Since abstract layer type checking has been removed in favor of base system
    validation, this now only validates basic pipeline structure.

    Args:
        pipeline_path: Path to abstract pipeline JSON file
        verbose: Whether to output verbose logs
        logger: Optional logger for output
        **kwargs: Additional arguments (e.g., dataset_schema)

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
            errors.append("Pipeline missing 'operators' field")
        else:
            operators = pipeline_data.get('operators', [])
            if not isinstance(operators, list):
                errors.append("'operators' must be a list")
            elif len(operators) == 0:
                warnings.append("Pipeline has no operators")
            else:
                # Validate each operator has required fields
                for i, op in enumerate(operators):
                    if not isinstance(op, dict):
                        errors.append(f"Operator {i} is not a dictionary")
                        continue
                    if 'name' not in op:
                        errors.append(f"Operator {i} missing 'name' field")
                    if 'type' not in op:
                        errors.append(f"Operator {i} missing 'type' field")

        # Check optional fields
        if 'name' not in pipeline_data:
            warnings.append("Pipeline missing 'name' field")

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
            "errors": [f"Pipeline file not found: {pipeline_path}"],
            "warnings": []
        }
    except json.JSONDecodeError as e:
        return {
            "score": 0,
            "passed": False,
            "errors": [f"Invalid JSON format: {str(e)}"],
            "warnings": []
        }
    except Exception as e:
        return {
            "score": 0,
            "passed": False,
            "errors": [f"Abstract validation error: {str(e)}"],
            "warnings": []
        }


def _validate_docetl_pipeline(
    pipeline_path: str,
    verbose: bool = False,
    logger: Optional[logging.Logger] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Validate a DocETL base system pipeline.

    Args:
        pipeline_path: Path to DocETL YAML file
        verbose: Whether to output verbose logs
        logger: Optional logger for output
        **kwargs: Additional arguments (e.g., enable_obfuscation)

    Returns:
        Validation result dict
    """
    try:
        # Add grader path to sys.path if needed
        grader_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'grader', 'docetl')
        if grader_dir not in sys.path:
            sys.path.insert(0, grader_dir)

        # Import DocETL static checker
        from baselines.grader.docetl.static_checker import check_pipeline_file

        # Run validation
        result = check_pipeline_file(pipeline_path)

        # Result is already in the correct format
        score = result.get('score', 0)
        return {
            "score": score,
            "passed": score == 1,
            "errors": result.get('errors', []),
            "warnings": result.get('warnings', [])
        }

    except FileNotFoundError:
        return {
            "score": 0,
            "passed": False,
            "errors": [f"Pipeline file not found: {pipeline_path}"],
            "warnings": []
        }
    except Exception as e:
        return {
            "score": 0,
            "passed": False,
            "errors": [f"DocETL validation error: {str(e)}"],
            "warnings": []
        }


def _validate_lotus_pipeline(
    pipeline_path: str,
    verbose: bool = False,
    logger: Optional[logging.Logger] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Validate a Lotus base system pipeline.

    Args:
        pipeline_path: Path to Lotus Python file
        verbose: Whether to output verbose logs
        logger: Optional logger for output
        **kwargs: Additional arguments

    Returns:
        Validation result dict
    """
    try:
        # Import Lotus static checker
        from ..grader.lotus.static_checker import check_lotus_pipeline

        # Run validation
        result = check_lotus_pipeline(pipeline_path=pipeline_path)

        # Standardize result format
        errors = result.get('errors', [])
        warnings = result.get('warnings', [])
        score = result.get('score', 0 if errors else 1)

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
            "errors": [f"Pipeline file not found: {pipeline_path}"],
            "warnings": []
        }
    except Exception as e:
        return {
            "score": 0,
            "passed": False,
            "errors": [f"Lotus validation error: {str(e)}"],
            "warnings": []
        }


def validate_pipeline_static(
    system_name: str,
    pipeline_path: str,
    verbose: bool = False,
    logger: Optional[logging.Logger] = None,
    **kwargs
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

    Raises:
        ValueError: If system_name is not recognized
    """
    # Normalize system name to lowercase
    system_name = system_name.lower()

    # Dispatch to appropriate validator
    validators = {
        'abstract': _validate_abstract_pipeline,
        'docetl': _validate_docetl_pipeline,
        'lotus': _validate_lotus_pipeline,
    }

    if system_name not in validators:
        supported = ", ".join(validators.keys())
        raise ValueError(
            f"Unknown system '{system_name}'. Supported systems: {supported}"
        )

    # Log validation start
    if verbose and logger:
        logger.info(f"Starting {system_name} static validation for: {pipeline_path}")

    # Run validation
    validator = validators[system_name]
    result = validator(pipeline_path, verbose=verbose, logger=logger, **kwargs)

    # ANSI color codes
    ORANGE = '\033[33m'
    RESET = '\033[0m'

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
            if result["warnings"]:
                logger.warning(f"Warnings ({len(result['warnings'])}): ")
                for warning in result["warnings"]:
                    # Extract message from warning dict if available, otherwise use string representation
                    warning_msg = warning.get('message', str(warning)) if isinstance(warning, dict) else str(warning)
                    logger.warning(f"  - {warning_msg}")

    return result
