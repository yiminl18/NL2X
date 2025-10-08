"""
Base Executor Interface for Abstract Pipeline Engine

This module provides the abstract base class that all system-specific executors
must implement. This enables the abstract layer to support multiple underlying
execution systems (DocETL, Lotus, etc.) in a unified way.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
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
    the AbstractExecutor to work with different execution backends.
    """

    def __init__(self,
                 verbose: bool = False,
                 cache_enabled: bool = True,
                 cache_dir: Optional[Union[str, Path]] = None,
                 data_manager: Optional[Any] = None):
        """
        Initialize the base executor.

        Args:
            verbose: Enable verbose logging
            cache_enabled: Enable result caching
            cache_dir: Directory for cache storage
            data_manager: Optional DatasetManager for dataset transformations
        """
        self.verbose = verbose
        self.cache_enabled = cache_enabled
        self.cache_dir = cache_dir
        self.data_manager = data_manager

    @abstractmethod
    def execute_operator(self,
                        operator: Any,
                        input_data: Union[str, List[Dict], Path],
                        config: Optional[Dict[str, Any]] = None,
                        force_execute: bool = False) -> ExecutionResult:
        """
        Execute a single abstract operator.

        Args:
            operator: Abstract operator to execute
            input_data: Input data (file path, list of dicts, or Path object)
            config: Optional configuration (model, settings, etc.)
            force_execute: If True, bypass cache and re-execute

        Returns:
            ExecutionResult with output data and metadata
        """
        pass

    @abstractmethod
    def get_system_name(self) -> str:
        """
        Get the name of the execution system.

        Returns:
            String identifier for the system (e.g., 'docetl', 'lotus')
        """
        pass

    def supports_operator_type(self, op_type: str) -> bool:
        """
        Check if this executor supports a given operator type.

        Args:
            op_type: The operator type to check (e.g., 'map', 'filter', 'reduce')

        Returns:
            True if the operator type is supported, False otherwise

        Note:
            Default implementation returns True for all types. Override this
            method to provide system-specific operator support information.
        """
        return True

    def execute_pipeline(self,
                        pipeline_config: Dict[str, Any],
                        input_data: Optional[Union[str, Path]] = None,
                        save_intermediates: bool = False,
                        intermediate_dir: Optional[str] = None) -> ExecutionResult:
        """
        Execute a complete pipeline (optional method).

        This method is optional. If a system has native pipeline execution
        capabilities, it can override this method. Otherwise, the AbstractExecutor
        will execute the pipeline operator-by-operator using execute_operator().

        Args:
            pipeline_config: Complete pipeline configuration
            input_data: Optional override for input data path
            save_intermediates: Whether to save intermediate results
            intermediate_dir: Directory to save intermediate results

        Returns:
            ExecutionResult with final output and metadata
        """
        return ExecutionResult(
            success=False,
            error=f"{self.get_system_name()} executor does not support native pipeline execution"
        )
