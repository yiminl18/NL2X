"""
Abstract Pipeline Execution Engine

Unified executor for abstract operators and pipelines. Routes operators to
appropriate system executors (DocETL, Lotus) based on operator.source.system.
Supports single operator execution, full pipelines, and pipeline ranges.
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
from .db import DatasetManager, DataSource

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
        """Initialize the AbstractExecutor with system executors and caching."""
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
        """Clear cache entries. If operator_hash is None, clears all cache."""
        total_deleted = 0
        for executor in self.executors.values():
            if hasattr(executor, 'cache_manager'):
                deleted = executor.cache_manager.clear_cache(operator_hash)
                total_deleted += deleted
        return total_deleted

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics from all executors."""
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
                    data_source: DataSource,
                    output_data: Any,
                    metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Manually inject externally computed result into cache for given operator."""
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
                operator, data_source, output_data, metadata
            )

        return False

    def execute_operator(self,
                        operator: Union[Operator, Dict[str, Any]],
                        data_source: DataSource,
                        config: Optional[Dict[str, Any]] = None,
                        force_execute: bool = False) -> ExecutionResult:
        """Execute single operator using appropriate system executor."""
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
        return self.executors[system].execute_operator(operator, data_source, config or {}, force_execute=force_execute)

    def execute_pipeline(self,
                        pipeline: Pipeline,
                        data_source: Optional[DataSource] = None,
                        config: Optional[Dict[str, Any]] = None,
                        save_intermediates: bool = False,
                        intermediate_dir: Optional[str] = None) -> ExecutionResult:
        """Execute complete pipeline, routing operators to appropriate system executors."""
        try:
            # If data_source not provided, try to load from pipeline
            if data_source is None:
                if pipeline.input_path and self.data_manager:
                    if self.verbose:
                        print(f"Loading dataset from pipeline.input_path: {pipeline.input_path}")
                    data_source = self.data_manager.load(pipeline.input_path)
                else:
                    # Neither data_source nor pipeline.input_path provided
                    raise ValueError(
                        "data_source is required for pipeline execution. "
                        "Either provide data_source parameter or set pipeline.input_path"
                    )

            # Execute operator-by-operator
            result = self.execute_pipeline_range(
                pipeline=pipeline,
                data_source=data_source,
                start_index=0,
                end_index=None,
                config=config,
                save_intermediates=save_intermediates,
                intermediate_dir=intermediate_dir
            )

            # Save final output if pipeline has output_path
            if result.success and pipeline.output_path:
                output_path_obj = Path(pipeline.output_path)
                output_path_obj.parent.mkdir(parents=True, exist_ok=True)

                with open(output_path_obj, 'w') as f:
                    json.dump(result.data, f, indent=2)

                if self.verbose:
                    print(f"Final output saved to: {pipeline.output_path}")

                result.metadata['output_path'] = pipeline.output_path

            return result

        except Exception as e:
            return ExecutionResult(
                success=False,
                error=f"Pipeline execution failed: {str(e)}\n{traceback.format_exc()}"
            )

    def execute_pipeline_range(self,
                              pipeline: Pipeline,
                              data_source: Optional[DataSource],
                              start_index: int = 0,
                              end_index: Optional[int] = None,
                              config: Optional[Dict[str, Any]] = None,
                              save_intermediates: bool = False,
                              intermediate_dir: Optional[str] = None) -> ExecutionResult:
        """Execute pipeline subset from start_index to end_index (inclusive)."""
        try:
            # Get operators from pipeline
            operators = pipeline.to_operators()

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
            current_source = data_source

            for i in range(start_index, end_index + 1):
                operator = operators[i]

                if self.verbose:
                    print(f"Executing operator {i}: {operator.name} ({operator.type})")
                    if current_source:
                        print(f"  Input: data source = {current_source.path}")

                result = self.execute_operator(
                    operator=operator,
                    data_source=current_source,
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

                # Get output data
                output_data = result.data

                # Validate data before saving
                if not output_data:
                    if self.verbose:
                        print(f"  ⚠ WARNING: Operator {i} ({operator.name}) returned empty data")

                # Save intermediate result and create new DataSource
                if i < end_index:
                    # Create new DataSource for next operator (using data manager)
                    if self.data_manager:
                        # Determine save path
                        if save_intermediates:
                            intermediate_file = os.path.join(
                                intermediate_dir,
                                f'operator_{i:02d}_{operator.name}.json'
                            )
                        else:
                            intermediate_file = None  # Will create temp file

                        # Create DataSource from output data
                        new_source = self.data_manager.create_from_data(
                            data=output_data,
                            path=intermediate_file,
                            metadata={'operator_index': i, 'operator_name': operator.name}
                        )

                        # Set parent and transformation
                        new_source.parent_id = current_source.id if current_source else None
                        new_source.transformation = f"operator_{i}_{operator.name}"

                        # Update parent's children
                        if current_source and current_source.id in self.data_manager.sources:
                            self.data_manager.sources[current_source.id].add_child(new_source.id)

                        current_source = new_source

                        if self.verbose:
                            if save_intermediates:
                                data_info = f"{len(output_data)} records" if isinstance(output_data, list) else type(output_data).__name__
                                print(f"  Saved intermediate result to {new_source.path}")
                                print(f"    Data: {data_info}")
                    else:
                        # No data manager - save to file manually
                        if save_intermediates:
                            intermediate_file = os.path.join(
                                intermediate_dir,
                                f'operator_{i:02d}_{operator.name}.json'
                            )
                            with open(intermediate_file, 'w') as f:
                                json.dump(output_data, f, indent=2)

                            if self.verbose:
                                data_info = f"{len(output_data)} records" if isinstance(output_data, list) else type(output_data).__name__
                                print(f"  Saved intermediate result to {intermediate_file}")
                                print(f"    Data: {data_info}")
                elif save_intermediates:
                    # Last operator - save if requested
                    if self.data_manager:
                        intermediate_file = os.path.join(
                            intermediate_dir,
                            f'operator_{i:02d}_{operator.name}.json'
                        )
                        with open(intermediate_file, 'w') as f:
                            json.dump(output_data, f, indent=2)

                        if self.verbose:
                            data_info = f"{len(output_data)} records" if isinstance(output_data, list) else type(output_data).__name__
                            print(f"  Saved final result to {intermediate_file}")
                            print(f"    Data: {data_info}")

            # Return final result
            metadata = {
                'start_index': start_index,
                'end_index': end_index,
                'operators_executed': end_index - start_index + 1
            }
            if save_intermediates and intermediate_dir:
                metadata['intermediate_dir'] = intermediate_dir
            if current_source:
                metadata['output_source_id'] = current_source.id

            return ExecutionResult(
                success=True,
                data=output_data,
                metadata=metadata
            )

        except Exception as e:
            return ExecutionResult(
                success=False,
                error=f"Pipeline range execution failed: {str(e)}\n{traceback.format_exc()}"
            )
