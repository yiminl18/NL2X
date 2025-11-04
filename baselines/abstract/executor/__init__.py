"""
Abstract Procedure Executors

This module provides system-specific executors for running abstract operators
and procedures on different execution backends.

Available Executors:
- BaseSystemExecutor: Abstract base class for all system executors
- AbstractPythonExecutor: Execute PythonCode operators using abstract system
- DocETLExecutor: Execute operators using DocETL system
- LotusExecutor: Execute operators using Lotus system (stub)

Available Utilities:
- ExecutionResult: Container for execution results with metadata
"""

from .base_executor import BaseSystemExecutor, ExecutionResult
from .abstract_executor import AbstractPythonExecutor
from .docetl_executor import DocETLExecutor
from .lotus_executor import LotusExecutor

__all__ = [
    'BaseSystemExecutor',
    'AbstractPythonExecutor',
    'DocETLExecutor',
    'LotusExecutor',
    'ExecutionResult'
]
