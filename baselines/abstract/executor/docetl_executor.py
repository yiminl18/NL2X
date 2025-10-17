"""
DocETL Executor for Abstract Pipeline Engine

This module provides DocETL-specific execution functionality for abstract operators
and pipelines. It handles conversion from abstract format to DocETL format and
manages execution using DocETL's DSLRunner.
"""

from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING
from pathlib import Path
import json
import yaml
import tempfile
import os
import time
import traceback

# Import base executor and result class
from .base_executor import BaseSystemExecutor, ExecutionResult

# Import abstract operator classes
from ..ops.base import Operator

# Import cache manager
from ..cache import OperatorCacheManager

# Import conversion utilities
try:
    from ..convert.docetl import abstract_to_docetl
    from ..convert.docetl.path_manager import (
        get_dataset_paths,
        set_dataset_paths,
        get_output_path,
        set_output_path,
        apply_path_mappings
    )
except ImportError:
    # Try alternative import for standalone execution
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'convert'))
    from docetl import abstract_to_docetl
    from docetl.path_manager import (
        get_dataset_paths,
        set_dataset_paths,
        get_output_path,
        set_output_path,
        apply_path_mappings
    )

# Use TYPE_CHECKING to avoid circular imports
if TYPE_CHECKING:
    from ..db import DatasetManager, DataSource


class DocETLExecutor(BaseSystemExecutor):
    """Executor for DocETL system-specific operations with caching support."""

    def __init__(self,
                 verbose: bool = False,
                 cache_enabled: bool = True,
                 cache_dir: Optional[Union[str, Path]] = None,
                 data_manager: Optional['DatasetManager'] = None):
        """
        Initialize DocETL executor.

        Args:
            verbose: Enable verbose logging
            cache_enabled: Enable result caching
            cache_dir: Directory for cache storage (default: abstract/_cache)
            data_manager: Optional DatasetManager for dataset transformations and path management
        """
        # Initialize base class
        super().__init__(verbose, cache_enabled, cache_dir, data_manager)

        self._docetl_available = self._check_docetl()

        self.cache_manager = OperatorCacheManager(
            cache_dir=cache_dir,
            enabled=cache_enabled
        )

    def _check_docetl(self) -> bool:
        """Check if DocETL is available for execution."""
        # try:
        from docetl.runner import DSLRunner
        return True
        # except ImportError:
        #     return False

    def get_system_name(self) -> str:
        """
        Get the name of the execution system.

        Returns:
            'docetl' as the system identifier
        """
        return 'docetl'

    def execute_operator(self,
                        operator: Operator,
                        data_source: 'DataSource',
                        config: Optional[Dict[str, Any]] = None,
                        force_execute: bool = False) -> ExecutionResult:
        """Execute single abstract operator using DocETL with caching support."""
        # Check cache first (unless force_execute is True)
        if self.cache_enabled:
            cached_result = self.cache_manager.get_cached_result(
                operator, data_source, force_execute=force_execute
            )
            if cached_result:
                cached_data = cached_result['output_data']
                if self.verbose:
                    print(f"✓ Cache hit for operator: {operator.name}")
                    data_info = f"{len(cached_data)} records" if isinstance(cached_data, list) else type(cached_data).__name__
                    print(f"  Cached data: {data_info}")

                return ExecutionResult(
                    success=True,
                    data=cached_data,
                    metadata={
                        **cached_result['metadata'],
                        'cache_hit': True,
                        'from_cache': True,
                        'operator_name': operator.name,
                        'operator_type': operator.type,
                        'system': 'docetl'
                    }
                )
            elif self.verbose:
                if force_execute:
                    print(f"🔄 Force execution for operator: {operator.name}")
                else:
                    print(f"✗ Cache miss for operator: {operator.name}")

        # Execute normally (cache miss, caching disabled, or force_execute)
        result = self._execute_operator_impl(operator, data_source, config)

        # Cache successful results (always cache, even for force_execute)
        if result.success and self.cache_enabled:
            self.cache_manager.set_cached_result(
                operator, data_source, result.data, result.metadata
            )
            if self.verbose:
                if force_execute:
                    print(f"✓ Updated cache for operator: {operator.name}")
                else:
                    print(f"✓ Cached result for operator: {operator.name}")

        # Add cache metadata
        result.metadata['cache_hit'] = False
        result.metadata['force_execute'] = force_execute

        return result

    def _execute_operator_impl(self,
                               operator: Operator,
                               data_source: 'DataSource',
                               config: Optional[Dict[str, Any]] = None) -> ExecutionResult:
        """Internal implementation of operator execution (without caching)."""
        if not self._docetl_available:
            return ExecutionResult(
                success=False,
                error="DocETL is not available. Please install docetl package."
            )

        try:
            from docetl.runner import DSLRunner

            docetl_op = abstract_to_docetl(operator)
            temp_pipeline = self._create_temp_pipeline(
                docetl_op, data_source, config or {}
            )

            start_time = time.time()

            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                yaml.dump(temp_pipeline, f, default_flow_style=False, sort_keys=False)
                temp_file = f.name
            # Print temp pipeline
            print(f"Temp pipeline: {temp_pipeline}")
            try:
                runner = DSLRunner.from_yaml(temp_file, max_threads=10)
                runner.load_run_save()

                output_path = temp_pipeline['pipeline']['output']['path']
                with open(output_path, 'r') as f:
                    output_data = json.load(f)

                execution_time = time.time() - start_time

                if os.path.exists(output_path):
                    os.remove(output_path)

                return ExecutionResult(
                    success=True,
                    data=output_data,
                    metadata={
                        'execution_time': execution_time,
                        'operator_name': operator.name,
                        'operator_type': operator.type,
                        'system': 'docetl'
                    }
                )

            finally:
                # Clean up temp files
                if os.path.exists(temp_file):
                    os.remove(temp_file)

        except Exception as e:
            return ExecutionResult(
                success=False,
                error=f"Execution failed: {str(e)}\n{traceback.format_exc()}",
                metadata={
                    'operator_name': operator.name,
                    'operator_type': operator.type
                }
            )

    def execute_pipeline(self,
                        pipeline_config: Dict[str, Any],
                        data_source: Optional['DataSource'] = None,
                        save_intermediates: bool = False,
                        intermediate_dir: Optional[str] = None) -> ExecutionResult:
        """Execute complete DocETL pipeline with optional DataSource override."""
        if not self._docetl_available:
            return ExecutionResult(
                success=False,
                error="DocETL is not available. Please install docetl package."
            )

        try:
            from docetl.runner import DSLRunner

            # Apply DatasetManager path mappings if available
            if self.data_manager:
                dataset_paths = get_dataset_paths(pipeline_config)
                dataset_path_mapping = {}

                for dataset_name, original_path in dataset_paths.items():
                    processed_path = self.data_manager.get_processed_path(original_path)
                    if processed_path:
                        dataset_path_mapping[dataset_name] = processed_path
                        if self.verbose:
                            print(f"Using processed dataset for {dataset_name}: {processed_path}")

                # Apply dataset path mappings
                if dataset_path_mapping:
                    pipeline_config = set_dataset_paths(pipeline_config, dataset_path_mapping, in_place=False)

                # Apply output path mapping if registered
                original_output_path = get_output_path(pipeline_config)
                if original_output_path:
                    new_output_path = self.data_manager.get_output_path(original_output_path)
                    if new_output_path:
                        pipeline_config = set_output_path(pipeline_config, new_output_path, in_place=True)
                        if self.verbose:
                            print(f"Using custom output path: {new_output_path}")

            # Override input data if provided (takes precedence over DatasetManager)
            if data_source:
                for dataset_name in pipeline_config.get('datasets', {}).keys():
                    pipeline_config['datasets'][dataset_name]['path'] = data_source.path

            # Configure intermediate directory if requested
            if save_intermediates and intermediate_dir:
                pipeline_config.setdefault('pipeline', {}).setdefault('output', {})
                pipeline_config['pipeline']['output']['intermediate_dir'] = intermediate_dir

            # Create temporary pipeline file
            start_time = time.time()

            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                yaml.dump(pipeline_config, f, default_flow_style=False, sort_keys=False)
                temp_file = f.name

            try:
                # Run the pipeline
                runner = DSLRunner.from_yaml(temp_file, max_threads=10)
                runner.load_run_save()

                # Load output
                output_path = pipeline_config['pipeline']['output']['path']
                with open(output_path, 'r') as f:
                    output_data = json.load(f)

                execution_time = time.time() - start_time

                # Collect intermediate results if saved
                intermediate_results = {}
                if save_intermediates and intermediate_dir:
                    intermediate_results = self._collect_intermediate_results(intermediate_dir)

                return ExecutionResult(
                    success=True,
                    data=output_data,
                    metadata={
                        'execution_time': execution_time,
                        'system': 'docetl',
                        'num_operations': len(pipeline_config.get('operations', [])),
                        'intermediate_results': intermediate_results,
                        'output_path': output_path
                    }
                )

            finally:
                # Clean up temp file
                if os.path.exists(temp_file):
                    os.remove(temp_file)

        except Exception as e:
            return ExecutionResult(
                success=False,
                error=f"Pipeline execution failed: {str(e)}\n{traceback.format_exc()}",
                metadata={'system': 'docetl'}
            )

    def _create_temp_pipeline(self,
                            docetl_op: Dict[str, Any],
                            data_source: 'DataSource',
                            config: Dict[str, Any]) -> Dict[str, Any]:
        """Create temporary pipeline for executing single operator."""
        # Use DataSource path directly
        input_path = data_source.path
        dataset_type = 'file'

        # Create output path
        output_path = tempfile.mktemp(suffix='.json')

        # Build pipeline
        pipeline = {
            'datasets': {
                'input_data': {
                    'type': dataset_type,
                    'path': input_path
                }
            },
            'operations': [docetl_op],
            'pipeline': {
                'steps': [
                    {
                        'name': 'exec_step',
                        'input': 'input_data',
                        'operations': [docetl_op['name']]
                    }
                ],
                'output': {
                    'type': 'file',
                    'path': output_path
                }
            }
        }

        # Add optional config
        if 'default_model' in config:
            pipeline['default_model'] = config['default_model']
        if 'system_prompt' in config:
            pipeline['system_prompt'] = config['system_prompt']

        return pipeline

    def _collect_intermediate_results(self, intermediate_dir: str) -> Dict[str, Any]:
        """
        Collect all intermediate results from checkpoint directory.

        Args:
            intermediate_dir: Directory containing intermediate checkpoints

        Returns:
            Dictionary mapping step_name -> operation_name -> data
        """
        intermediate_results = {}

        if not os.path.exists(intermediate_dir):
            return intermediate_results

        # Walk through intermediate directory
        for root, dirs, files in os.walk(intermediate_dir):
            for file in files:
                if file.endswith('.json'):
                    file_path = os.path.join(root, file)
                    step_name = os.path.basename(root)
                    op_name = file.replace('.json', '')

                    try:
                        with open(file_path, 'r') as f:
                            data = json.load(f)

                        if step_name not in intermediate_results:
                            intermediate_results[step_name] = {}
                        intermediate_results[step_name][op_name] = data
                    except:
                        pass

        return intermediate_results
