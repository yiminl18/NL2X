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

from .pipeline import Pipeline, PipelineNode, optimize_pipeline

from .optimizer import PipelineOptimizer

from .db import DatasetManager, SamplingConfig, DataSource

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
    "optimize_pipeline",
    "PipelineOptimizer",
    "DatasetManager",
    "SamplingConfig",
    "DataSource",
]
