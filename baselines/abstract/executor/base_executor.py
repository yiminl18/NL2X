"""
Base Executor Interface for Abstract Procedure Engine

This module provides the abstract base class that all system-specific executors
must implement. This enables the abstract layer to support multiple underlying
execution systems (DocETL, Lotus, etc.) in a unified way.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union
from pathlib import Path
from datetime import datetime


class ExecutionResult:
    """Container for execution results with metadata."""

    def __init__(self,
                 success: bool,
                 data: Optional[Any] = None,
                 error: Optional[str] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        self.success = success
        self.data = data
        self.error = error
        self.metadata = metadata or {}
        self.metadata['timestamp'] = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            'success': self.success,
            'data': self.data,
            'error': self.error,
            'metadata': self.metadata
        }


class BaseSystemExecutor(ABC):
    """
    Abstract base class for system-specific executors.

    All system executors (DocETL, Lotus, etc.) must inherit from this class
    and implement its abstract methods. This provides a uniform interface for
    the PipelineEngine to work with different execution backends.
    """

    def __init__(self,
                 verbose: bool = False,
                 cache_enabled: bool = True,
                 cache_dir: Optional[Union[str, Path]] = None,
                 data_manager: Optional[Any] = None):
        """Initialize base executor with caching configuration."""
        self.verbose = verbose
        self.cache_enabled = cache_enabled
        self.cache_dir = cache_dir
        self.data_manager = data_manager

    @abstractmethod
    def execute_operator(self,
                        operator: Any,
                        input_data: Any,
                        config: Optional[Dict[str, Any]] = None,
                        force_execute: bool = False) -> ExecutionResult:
        """Execute single abstract operator and return ExecutionResult."""
        pass

    @abstractmethod
    def get_system_name(self) -> str:
        """Get system identifier (e.g., 'docetl', 'lotus')."""
        pass

    def supports_operator_type(self, op_type: str) -> bool:
        """Check if executor supports given operator type. Default: True for all."""
        return True

    def execute_procedure(self,
                         procedure_config: Dict[str, Any],
                         input_data: Optional[Any] = None,
                         save_intermediates: bool = False,
                         intermediate_dir: Optional[str] = None) -> ExecutionResult:
        """Execute complete procedure. Optional method for systems with native procedure execution."""
        return ExecutionResult(
            success=False,
            error=f"{self.get_system_name()} executor does not support native procedure execution"
        )

    @abstractmethod
    def execute_original_procedure(self, procedure_path: Union[str, Path]) -> ExecutionResult:
        return ExecutionResult(
            success=False,
            error=f"{self.get_system_name()} executor does not support native original procedure execution"
        )