"""
Abstract Pipeline Execution Engine

This module provides functionality to execute abstract operators and pipelines by
mapping them back to their original systems and executing them.

Features:
1. Execute single abstract operators with given input data
2. Execute complete abstract pipelines
3. Track and save intermediate results at each pipeline step
4. Support for multiple execution systems (DocETL, Lotus, etc.)
"""

from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import json
import yaml
import tempfile
import os
import traceback

# Import abstract operator classes
from .ops.base import Operator
from .pipeline import Pipeline, PipelineNode

# Import executors
from .executor import DocETLExecutor, ExecutionResult

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
                 cache_dir: Optional[Union[str, Path]] = None):
        """
        Initialize the AbstractExecutor.

        Args:
            verbose: Enable verbose logging
            cache_enabled: Enable result caching for operators
            cache_dir: Directory for cache storage (default: abstract/_cache)
        """
        self.verbose = verbose
        self.cache_enabled = cache_enabled
        self.cache_dir = cache_dir

        # Initialize executors with cache configuration
        self.executors = {
            'docetl': DocETLExecutor(
                verbose=verbose,
                cache_enabled=cache_enabled,
                cache_dir=cache_dir
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
                        input_data: Optional[Union[str, Path]] = None,
                        config: Optional[Dict[str, Any]] = None) -> ExecutionResult:
        """
        Execute a complete abstract pipeline.

        Args:
            pipeline: Pipeline specification (JSON file path, operators list, or complete config dict)
            input_data: Optional input data path (overrides pipeline config)
            config: Optional execution configuration

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
        """
        try:
            # Load pipeline if it's a file path
            if isinstance(pipeline, (str, Path)):
                pipeline_path = Path(pipeline)
                with open(pipeline_path, 'r') as f:
                    pipeline_data = json.load(f)
            elif isinstance(pipeline, list):
                # List of operators - need to wrap in full pipeline structure
                pipeline_data = {'operators': [operator_to_dict(op) if isinstance(op, Operator) else op
                                              for op in pipeline]}
            else:
                pipeline_data = pipeline

            # Convert abstract pipeline to DocETL format
            from .convert.docetl import abstract_json_to_yaml

            # Create temp file for DocETL pipeline
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(pipeline_data, f)
                temp_json = f.name

            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                temp_yaml = f.name

            try:
                # Convert to DocETL YAML
                abstract_json_to_yaml(temp_json, temp_yaml)

                # Load the converted YAML
                with open(temp_yaml, 'r') as f:
                    docetl_pipeline = yaml.safe_load(f)

                # Apply any config overrides
                if config:
                    if 'default_model' in config:
                        docetl_pipeline['default_model'] = config['default_model']
                    if 'system_prompt' in config:
                        docetl_pipeline['system_prompt'] = config['system_prompt']

                # Execute using DocETL executor
                return self.executors['docetl'].execute_pipeline(
                    pipeline_config=docetl_pipeline,
                    input_data=input_data,
                    save_intermediates=False
                )

            finally:
                # Clean up temp files
                if os.path.exists(temp_json):
                    os.remove(temp_json)
                if os.path.exists(temp_yaml):
                    os.remove(temp_yaml)

        except Exception as e:
            return ExecutionResult(
                success=False,
                error=f"Pipeline execution failed: {str(e)}\n{traceback.format_exc()}"
            )

    def execute_with_tracking(self,
                             pipeline: Union[str, Path, List[Operator], Dict[str, Any]],
                             input_data: Optional[Union[str, Path]] = None,
                             output_dir: Optional[str] = None,
                             config: Optional[Dict[str, Any]] = None) -> ExecutionResult:
        """
        Execute a pipeline with intermediate result tracking.

        Args:
            pipeline: Pipeline specification
            input_data: Optional input data path
            output_dir: Directory to save intermediate results (created if not exists)
            config: Optional execution configuration

        Returns:
            ExecutionResult with final output, intermediate results, and metadata

        Example:
            >>> executor = AbstractExecutor()
            >>> result = executor.execute_with_tracking(
            ...     pipeline="abstract_pipeline.json",
            ...     input_data="data.json",
            ...     output_dir="./results/intermediates"
            ... )
            >>> if result.success:
            ...     print(f"Final output: {result.data}")
            ...     print(f"Intermediates: {result.metadata['intermediate_results']}")
        """
        try:
            # Create output directory if needed
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                intermediate_dir = output_dir
            else:
                intermediate_dir = tempfile.mkdtemp(prefix='abstract_pipeline_')

            # Load pipeline
            if isinstance(pipeline, (str, Path)):
                pipeline_path = Path(pipeline)
                with open(pipeline_path, 'r') as f:
                    pipeline_data = json.load(f)
            elif isinstance(pipeline, list):
                pipeline_data = {'operators': [operator_to_dict(op) if isinstance(op, Operator) else op
                                              for op in pipeline]}
            else:
                pipeline_data = pipeline

            # Convert to DocETL format
            from .convert.docetl import abstract_json_to_yaml

            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(pipeline_data, f)
                temp_json = f.name

            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                temp_yaml = f.name

            try:
                # Convert to DocETL YAML
                abstract_json_to_yaml(temp_json, temp_yaml)

                # Load and configure
                with open(temp_yaml, 'r') as f:
                    docetl_pipeline = yaml.safe_load(f)

                # Apply config
                if config:
                    if 'default_model' in config:
                        docetl_pipeline['default_model'] = config['default_model']
                    if 'system_prompt' in config:
                        docetl_pipeline['system_prompt'] = config['system_prompt']

                # Execute with tracking
                result = self.executors['docetl'].execute_pipeline(
                    pipeline_config=docetl_pipeline,
                    input_data=input_data,
                    save_intermediates=True,
                    intermediate_dir=intermediate_dir
                )

                # Add intermediate directory to metadata
                if result.success and not output_dir:
                    # If we created a temp dir, include path in metadata
                    result.metadata['intermediate_dir'] = intermediate_dir

                return result

            finally:
                # Clean up temp files
                if os.path.exists(temp_json):
                    os.remove(temp_json)
                if os.path.exists(temp_yaml):
                    os.remove(temp_yaml)

        except Exception as e:
            return ExecutionResult(
                success=False,
                error=f"Tracked execution failed: {str(e)}\n{traceback.format_exc()}"
            )

    def execute_pipeline_range(self,
                              pipeline: Union[str, Path, List[Operator], Dict[str, Any]],
                              input_data: Union[str, Path, List[Dict]],
                              start_index: int = 0,
                              end_index: Optional[int] = None,
                              config: Optional[Dict[str, Any]] = None) -> ExecutionResult:
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
        """
        try:
            # Load pipeline
            if isinstance(pipeline, (str, Path)):
                pipeline_path = Path(pipeline)
                with open(pipeline_path, 'r') as f:
                    pipeline_data = json.load(f)
            elif isinstance(pipeline, list):
                pipeline_data = {'operators': [operator_to_dict(op) if isinstance(op, Operator) else op
                                              for op in pipeline]}
            else:
                pipeline_data = pipeline

            # Extract operators
            operators = pipeline_data.get('operators', [])

            if not operators:
                return ExecutionResult(
                    success=False,
                    error="No operators found in pipeline"
                )

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

            # Execute operators sequentially
            current_data = input_data

            for i in range(start_index, end_index + 1):
                op_dict = operators[i]
                operator = dict_to_operator(op_dict)

                if self.verbose:
                    print(f"Executing operator {i}: {operator.name} ({operator.type})")

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

            # Return final result
            return ExecutionResult(
                success=True,
                data=current_data,
                metadata={
                    'start_index': start_index,
                    'end_index': end_index,
                    'operators_executed': end_index - start_index + 1
                }
            )

        except Exception as e:
            return ExecutionResult(
                success=False,
                error=f"Pipeline range execution failed: {str(e)}\n{traceback.format_exc()}"
            )
