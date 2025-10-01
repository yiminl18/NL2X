"""
Abstract operators module for NL2X.

This module provides:
- Abstract operator definitions (ops)
- Pipeline class for building DAGs
- Conversion utilities for system-specific operators
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

__all__ = [
    # Base operator
    "Operator",
    # Operators
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
    # Pipeline
    "Pipeline",
    "PipelineNode",
]
