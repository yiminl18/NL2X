"""
Abstract Pipeline Executors

This module provides system-specific executors for running abstract operators
and pipelines on different execution backends.

Available Executors:
- BaseSystemExecutor: Abstract base class for all system executors
- DocETLExecutor: Execute operators using DocETL system
- LotusExecutor: Execute operators using Lotus system (stub)

Available Utilities:
- ExecutionResult: Container for execution results with metadata
"""

from .base_executor import BaseSystemExecutor, ExecutionResult
from .docetl_executor import DocETLExecutor
from .lotus_executor import LotusExecutor

__all__ = [
    'BaseSystemExecutor',
    'DocETLExecutor',
    'LotusExecutor',
    'ExecutionResult'
]
