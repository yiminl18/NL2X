"""
Abstract Pipeline Executors

This module provides system-specific executors for running abstract operators
and pipelines on different execution backends.

Available Executors:
- DocETLExecutor: Execute operators using DocETL system

Available Utilities:
- ExecutionResult: Container for execution results with metadata
"""

from .docetl import DocETLExecutor, ExecutionResult

__all__ = [
    'DocETLExecutor',
    'ExecutionResult'
]
