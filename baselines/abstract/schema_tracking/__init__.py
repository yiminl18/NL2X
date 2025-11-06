"""
Schema Tracking Interface

Thin interface layer providing factory functions for system-specific schema trackers.
"""

from typing import Dict, Optional
from ..support import BaseSystem
from .interface import SchemaTrackerProtocol


def create_tracker(
    system: BaseSystem,
    initial_schema: Dict[str, str],
    verbose: bool = False
) -> SchemaTrackerProtocol:
    """
    Create system-specific schema tracker.

    Args:
        system: Target system (DOCETL, LOTUS, ABSTRACT, etc.)
        initial_schema: Initial dataset schema
        verbose: Enable verbose logging

    Returns:
        Schema tracker instance implementing SchemaTrackerProtocol

    Raises:
        ValueError: If system is not supported
    """
    if system == BaseSystem.DOCETL:
        from ...docetl_support.schema_tracking import DocETLSchemaTracker
        return DocETLSchemaTracker(initial_schema, verbose)
    elif system == BaseSystem.ABSTRACT:
        from ...abstract_support.schema_tracking import AbstractSchemaTracker
        return AbstractSchemaTracker(initial_schema, verbose)
    elif system == BaseSystem.LOTUS:
        raise NotImplementedError("LOTUS schema tracking not yet implemented")
    else:
        raise ValueError(f"Unsupported system: {system}")


def create_tracker_from_dataset(
    system: BaseSystem,
    dataset_path: str,
    verbose: bool = False
) -> SchemaTrackerProtocol:
    """
    Create schema tracker by inferring schema from dataset.

    Args:
        system: Target system (DOCETL, LOTUS, ABSTRACT, etc.)
        dataset_path: Path to dataset file
        verbose: Enable verbose logging

    Returns:
        Schema tracker instance with inferred schema

    Raises:
        ValueError: If system is not supported
        FileNotFoundError: If dataset file not found
    """
    if system == BaseSystem.DOCETL:
        from ...docetl_support.schema_tracking import DocETLSchemaTracker
        return DocETLSchemaTracker.from_dataset(dataset_path, verbose)
    elif system == BaseSystem.ABSTRACT:
        from ...abstract_support.schema_tracking import AbstractSchemaTracker
        return AbstractSchemaTracker.from_dataset(dataset_path, verbose)
    elif system == BaseSystem.LOTUS:
        raise NotImplementedError("LOTUS schema tracking not yet implemented")
    else:
        raise ValueError(f"Unsupported system: {system}")


__all__ = [
    'SchemaTrackerProtocol',
    'create_tracker',
    'create_tracker_from_dataset',
]
