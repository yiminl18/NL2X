"""
Abstract operators module for NL2X.

This module provides:
- Abstract operator definitions (ops)
- Pipeline class for building DAGs
- Conversion utilities for system-specific operators
- Data management layer for dataset transformations and path management
"""

from .ops import (
    Operator,
    Map,
    Filter,
    Reduce,
    Join,
    Rank,
    TopK,
    Resolve,
    Cluster,
    Extract,
    Split,
    Gather,
    Unnest,
    Sample,
    Index,
    Search,
    Project,
)

from .pipeline import Pipeline, PipelineNode

from .data_management import DatasetManager, SamplingConfig

__all__ = [
    "Operator",
    "Map",
    "Filter",
    "Reduce",
    "Join",
    "Rank",
    "TopK",
    "Resolve",
    "Cluster",
    "Extract",
    "Split",
    "Gather",
    "Unnest",
    "Sample",
    "Index",
    "Search",
    "Project",
    "Pipeline",
    "PipelineNode",
    "DatasetManager",
    "SamplingConfig",
]
