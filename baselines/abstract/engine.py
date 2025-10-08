"""
Abstract Pipeline Execution Engine

This module provides a unified execution orchestrator for abstract operators and
pipelines. It routes operators to their appropriate system executors and manages
operator-by-operator execution flow.

Key Features:
- Execute single operators via system-specific executors
- Execute complete pipelines with operator-by-operator routing
- Execute pipeline ranges (subsets of operators)
- Support for multiple execution systems (DocETL, Lotus, etc.)
- Cache management across all system executors
- System-agnostic design: no hardcoded system-specific logic

Architecture:
The AbstractExecutor maintains a registry of system executors and routes each
operator to its appropriate backend based on the operator.source.system field.
This allows mixing operators from different systems in the same pipeline.
"""

from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import json
import os
import traceback

# Import abstract operator classes
from .ops.base import Operator
from .pipeline import Pipeline, PipelineNode

# Import executors
from .executor import DocETLExecutor, LotusExecutor, ExecutionResult

# Import data management
from .data_management import DatasetManager

# Import conversion utilities
try:
    from .convert.docetl import (
        dict_to_operator,
        operator_to_dict
    )
except ImportError:
    # Try relative import for standalone execution
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'convert'))
    from docetl import dict_to_operator, operator_to_dict


class AbstractExecutor:
    """Main executor for abstract operators and pipelines with caching support."""

    def __init__(self,
                 verbose: bool = False,
                 cache_enabled: bool = True,
                 cache_dir: Optional[Union[str, Path]] = None,
                 data_manager: Optional[DatasetManager] = None):
        """
        Initialize the AbstractExecutor.

        Args:
            verbose: Enable verbose logging
            cache_enabled: Enable result caching for operators
            cache_dir: Directory for cache storage (default: abstract/_cache)
            data_manager: Optional DatasetManager for dataset transformations and path management
        """
        self.verbose = verbose
        self.cache_enabled = cache_enabled
        self.cache_dir = cache_dir
        self.data_manager = data_manager

        # Initialize executors for different systems
        self.executors = {
            'docetl': DocETLExecutor(
                verbose=verbose,
                cache_enabled=cache_enabled,
                cache_dir=cache_dir,
                data_manager=data_manager
            ),
            'lotus': LotusExecutor(
                verbose=verbose,
                cache_enabled=cache_enabled,
                cache_dir=cache_dir,
                data_manager=data_manager
            )
        }

    def clear_cache(self, operator_hash: Optional[str] = None) -> int:
        """
        Clear cache entries across all executors.

        Args:
            operator_hash: Optional operator hash to clear specific cache.
                          If None, clears all cache.

        Returns:
            Total number of cache entries deleted
        """
        total_deleted = 0
        for executor in self.executors.values():
            if hasattr(executor, 'cache_manager'):
                deleted = executor.cache_manager.clear_cache(operator_hash)
                total_deleted += deleted
        return total_deleted

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics from all executors.

        Returns:
            Dictionary with cache statistics per executor
        """
        stats = {}
        for name, executor in self.executors.items():
            if hasattr(executor, 'cache_manager'):
                stats[name] = executor.cache_manager.get_cache_stats()
        return stats

    def disable_cache(self):
        """Temporarily disable caching for all executors."""
        self.cache_enabled = False
        for executor in self.executors.values():
            if hasattr(executor, 'cache_enabled'):
                executor.cache_enabled = False

    def enable_cache(self):
        """Re-enable caching for all executors."""
        self.cache_enabled = True
        for executor in self.executors.values():
            if hasattr(executor, 'cache_enabled'):
                executor.cache_enabled = True

    def inject_cache(self,
                    operator: Union[Operator, Dict[str, Any]],
                    input_data: Union[str, List[Dict], Path],
                    output_data: Any,
                    metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Manually inject a result into cache without execution.

        This allows bypassing specific operators with externally computed results,
        useful for integrating external processing or optimization steps.

        Args:
            operator: Abstract operator (Operator instance or dict)
            input_data: Input data that triggers this operator
            output_data: Output data to cache (externally computed)
            metadata: Optional metadata about the cached result

        Returns:
            True if injection successful, False otherwise

        Example:
            >>> executor = AbstractExecutor()
            >>> # Process data externally
            >>> external_result = custom_processing(input_data)
            >>> # Inject as cached output for operator
            >>> executor.inject_cache(
            ...     operator=my_operator,
            ...     input_data=input_to_operator,
            ...     output_data=external_result,
            ...     metadata={'source': 'external_processing'}
            ... )
        """
        # Convert dict to Operator if needed
        if isinstance(operator, dict):
            operator = dict_to_operator(operator)

        # Determine which system to use
        system = operator.source.get('system', 'docetl')

        if system not in self.executors:
            return False

        # Inject into cache using the system executor
        executor = self.executors[system]
        if hasattr(executor, 'cache_manager'):
            return executor.cache_manager.inject_cached_result(
                operator, input_data, output_data, metadata
            )

        return False

    def _load_pipeline_operators(self,
                                pipeline: Union[str, Path, List[Operator], Dict[str, Any]]) -> List[Operator]:
        """
        Load and normalize pipeline to list of Operator instances.

        Args:
            pipeline: Pipeline specification (JSON file path, operators list, or config dict)

        Returns:
            List of Operator instances

        Raises:
            ValueError: If no operators found or invalid format
        """
        # Load pipeline if it's a file path
        if isinstance(pipeline, (str, Path)):
            pipeline_path = Path(pipeline)
            with open(pipeline_path, 'r') as f:
                pipeline_data = json.load(f)
        elif isinstance(pipeline, list):
            # Already a list of operators
            return [op if isinstance(op, Operator) else dict_to_operator(op) for op in pipeline]
        else:
            pipeline_data = pipeline

        # Extract operators
        operators = pipeline_data.get('operators', [])

        if not operators:
            raise ValueError("No operators found in pipeline")

        # Convert dict operators to Operator instances
        return [dict_to_operator(op) if isinstance(op, dict) else op for op in operators]

    def execute_operator(self,
                        operator: Union[Operator, Dict[str, Any]],
                        input_data: Union[str, List[Dict], Path],
                        config: Optional[Dict[str, Any]] = None,
                        force_execute: bool = False) -> ExecutionResult:
        """
        Execute a single abstract operator.

        Args:
            operator: Abstract operator (Operator instance or dict)
            input_data: Input data (file path, list of dicts, or Path object)
            config: Optional execution configuration
            force_execute: If True, bypass cache and re-execute (default: False)

        Returns:
            ExecutionResult with output data and metadata

        Example:
            >>> executor = AbstractExecutor()
            >>> result = executor.execute_operator(
            ...     operator=my_operator,
            ...     input_data=[{"text": "example"}],
            ...     config={"default_model": "gpt-4o-mini"}
            ... )
            >>> if result.success:
            ...     print(result.data)
        """
        # Convert dict to Operator if needed
        if isinstance(operator, dict):
            operator = dict_to_operator(operator)

        # Determine which system to use
        system = operator.source.get('system', 'docetl')

        if system not in self.executors:
            return ExecutionResult(
                success=False,
                error=f"Executor for system '{system}' not available"
            )

        # Execute using appropriate executor
        return self.executors[system].execute_operator(operator, input_data, config or {}, force_execute=force_execute)

    def execute_pipeline(self,
                        pipeline: Union[str, Path, List[Operator], Dict[str, Any]],
                        input_data: Optional[Union[str, Path, List[Dict]]] = None,
                        config: Optional[Dict[str, Any]] = None,
                        save_intermediates: bool = False,
                        intermediate_dir: Optional[str] = None) -> ExecutionResult:
        """
        Execute a complete abstract pipeline.

        Executes operators sequentially, routing each to its appropriate system executor.
        This allows mixing operators from different systems in the same pipeline.

        Args:
            pipeline: Pipeline specification (JSON file path, operators list, or complete config dict)
            input_data: Optional input data (file path, list of dicts, or Path object)
            config: Optional execution configuration
            save_intermediates: If True, save intermediate results after each operator
            intermediate_dir: Directory to save intermediates (default: ./intermediates)

        Returns:
            ExecutionResult with final output and metadata

        Example:
            >>> executor = AbstractExecutor()
            >>> result = executor.execute_pipeline(
            ...     pipeline="abstract_pipeline.json",
            ...     input_data="data.json"
            ... )
            >>> if result.success:
            ...     print(f"Output: {result.data}")
            ...     print(f"Time: {result.metadata['execution_time']}")
            >>>
            >>> # With intermediate result saving
            >>> result = executor.execute_pipeline(
            ...     pipeline="abstract_pipeline.json",
            ...     input_data="data.json",
            ...     save_intermediates=True,
            ...     intermediate_dir="./my_results"
            ... )
        """
        try:
            # Load pipeline data from file if needed
            if isinstance(pipeline, (str, Path)):
                pipeline_path = Path(pipeline)
                with open(pipeline_path, 'r') as f:
                    pipeline_data = json.load(f)
            elif isinstance(pipeline, list):
                # Just a list of operators, no config data
                pipeline_data = None
            else:
                # Already a dict
                pipeline_data = pipeline

            # Load and normalize operators
            operator_instances = self._load_pipeline_operators(pipeline)

            # Get input data from pipeline config if not provided
            if input_data is None and pipeline_data is not None and isinstance(pipeline_data, dict):
                # First try to get from input_data field
                input_data = pipeline_data.get('input_data')

                # If not found, extract from datasets section
                if not input_data:
                    datasets = pipeline_data.get('datasets', {})
                    if datasets:
                        # Get the first dataset path (assuming it's the input dataset)
                        first_dataset = next(iter(datasets.values()), {})
                        input_data = first_dataset.get('path')
                        if input_data and self.verbose:
                            print(f"Using dataset path from config: {input_data}")

            # Execute operator-by-operator
            result = self.execute_pipeline_range(
                pipeline=operator_instances,
                input_data=input_data or [],
                start_index=0,
                end_index=None,
                config=config,
                save_intermediates=save_intermediates,
                intermediate_dir=intermediate_dir
            )

            # Save final output to specified path if pipeline config has output path
            if result.success and pipeline_data is not None:
                output_path = pipeline_data.get('pipeline', {}).get('output', {}).get('path')
                if output_path:
                    # Create directory if needed
                    output_path_obj = Path(output_path)
                    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

                    # Write final output
                    with open(output_path_obj, 'w') as f:
                        json.dump(result.data, f, indent=2)

                    if self.verbose:
                        print(f"Final output saved to: {output_path}")

                    # Update metadata
                    result.metadata['output_path'] = str(output_path)

            return result

        except Exception as e:
            return ExecutionResult(
                success=False,
                error=f"Pipeline execution failed: {str(e)}\n{traceback.format_exc()}"
            )

    def execute_pipeline_range(self,
                              pipeline: Union[str, Path, List[Operator], Dict[str, Any]],
                              input_data: Union[str, Path, List[Dict]],
                              start_index: int = 0,
                              end_index: Optional[int] = None,
                              config: Optional[Dict[str, Any]] = None,
                              save_intermediates: bool = False,
                              intermediate_dir: Optional[str] = None) -> ExecutionResult:
        """
        Execute a pipeline from start_index to end_index (inclusive).

        This allows executing a subset of operators in a pipeline, useful for:
        - Debugging specific pipeline segments
        - Re-executing only modified operators
        - Skipping expensive operators with cached/external results

        Args:
            pipeline: Pipeline specification (JSON file path, operators list, or complete config dict)
            input_data: Input data (file path, list of dicts, or Path object)
            start_index: Index of first operator to execute (0-based, inclusive)
            end_index: Index of last operator to execute (0-based, inclusive).
                      If None, executes to the end of pipeline.
            config: Optional execution configuration
            save_intermediates: If True, save intermediate results after each operator
            intermediate_dir: Directory to save intermediates (default: ./intermediates)

        Returns:
            ExecutionResult with output data from the last executed operator

        Example:
            >>> executor = AbstractExecutor()
            >>> # Execute only operators 1-3 (indices 1, 2, 3)
            >>> result = executor.execute_pipeline_range(
            ...     pipeline="abstract_pipeline.json",
            ...     input_data=[{"text": "example"}],
            ...     start_index=1,
            ...     end_index=3,
            ...     config={"default_model": "gpt-4o-mini"}
            ... )
            >>> # With intermediate result saving
            >>> result = executor.execute_pipeline_range(
            ...     pipeline="abstract_pipeline.json",
            ...     input_data=[{"text": "example"}],
            ...     save_intermediates=True,
            ...     intermediate_dir="./my_intermediates"
            ... )
        """
        try:
            # Load and normalize operators
            operators = self._load_pipeline_operators(pipeline)

            # Validate indices
            if start_index < 0 or start_index >= len(operators):
                return ExecutionResult(
                    success=False,
                    error=f"Invalid start_index {start_index}. Must be 0 <= start_index < {len(operators)}"
                )

            if end_index is None:
                end_index = len(operators) - 1

            if end_index < start_index or end_index >= len(operators):
                return ExecutionResult(
                    success=False,
                    error=f"Invalid end_index {end_index}. Must be {start_index} <= end_index < {len(operators)}"
                )

            # Setup intermediate directory if needed
            if save_intermediates:
                if not intermediate_dir:
                    intermediate_dir = os.path.join(os.getcwd(), 'intermediates')
                os.makedirs(intermediate_dir, exist_ok=True)
                if self.verbose:
                    print(f"Saving intermediates to: {intermediate_dir}")

            # Execute operators sequentially
            current_data = input_data

            for i in range(start_index, end_index + 1):
                operator = operators[i]

                if self.verbose:
                    print(f"Executing operator {i}: {operator.name} ({operator.type})")
                    # Print input data info
                    if isinstance(current_data, (str, Path)):
                        print(f"  Input: file path = {current_data}")
                    elif isinstance(current_data, list):
                        print(f"  Input: {len(current_data)} records")
                    else:
                        print(f"  Input: {type(current_data).__name__}")

                result = self.execute_operator(
                    operator=operator,
                    input_data=current_data,
                    config=config
                )

                if not result.success:
                    return ExecutionResult(
                        success=False,
                        error=f"Operator {i} ({operator.name}) failed: {result.error}",
                        metadata={
                            'failed_operator_index': i,
                            'failed_operator_name': operator.name
                        }
                    )

                # Use this operator's output as next operator's input
                current_data = result.data

                # Validate data before saving
                if not current_data:
                    if self.verbose:
                        print(f"  ⚠ WARNING: Operator {i} ({operator.name}) returned empty data")

                # Save intermediate result if requested
                if save_intermediates:
                    intermediate_file = os.path.join(intermediate_dir, f'operator_{i:02d}_{operator.name}.json')
                    with open(intermediate_file, 'w') as f:
                        json.dump(current_data, f, indent=2)
                    if self.verbose:
                        data_info = f"{len(current_data)} records" if isinstance(current_data, list) else type(current_data).__name__
                        print(f"  Saved intermediate result to {intermediate_file}")
                        print(f"    Data: {data_info}")

            # Return final result
            metadata = {
                'start_index': start_index,
                'end_index': end_index,
                'operators_executed': end_index - start_index + 1
            }
            if save_intermediates and intermediate_dir:
                metadata['intermediate_dir'] = intermediate_dir

            return ExecutionResult(
                success=True,
                data=current_data,
                metadata=metadata
            )

        except Exception as e:
            return ExecutionResult(
                success=False,
                error=f"Pipeline range execution failed: {str(e)}\n{traceback.format_exc()}"
            )
