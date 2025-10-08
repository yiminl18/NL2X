"""
DocETL Executor for Abstract Pipeline Engine

This module provides DocETL-specific execution functionality for abstract operators
and pipelines. It handles conversion from abstract format to DocETL format and
manages execution using DocETL's DSLRunner.
"""

from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import json
import yaml
import tempfile
import os
import time
import traceback
from datetime import datetime

# Import abstract operator classes
from ..ops.base import Operator

# Import cache manager
from ..cache import OperatorCacheManager

# Import conversion utilities
try:
    from ..convert.docetl import abstract_to_docetl
except ImportError:
    # Try alternative import for standalone execution
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'convert'))
    from docetl import abstract_to_docetl


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


class DocETLExecutor:
    """Executor for DocETL system-specific operations with caching support."""

    def __init__(self,
                 verbose: bool = False,
                 cache_enabled: bool = True,
                 cache_dir: Optional[Union[str, Path]] = None):
        """
        Initialize DocETL executor.

        Args:
            verbose: Enable verbose logging
            cache_enabled: Enable result caching
            cache_dir: Directory for cache storage (default: abstract/_cache)
        """
        self.verbose = verbose
        self.cache_enabled = cache_enabled
        self._docetl_available = self._check_docetl()

        # Initialize cache manager
        self.cache_manager = OperatorCacheManager(
            cache_dir=cache_dir,
            enabled=cache_enabled
        )

    def _check_docetl(self) -> bool:
        """Check if DocETL is available for execution."""
        try:
            from docetl.runner import DSLRunner
            return True
        except ImportError:
            return False

    def execute_operator(self,
                        operator: Operator,
                        input_data: Union[str, List[Dict], Path],
                        config: Optional[Dict[str, Any]] = None,
                        force_execute: bool = False) -> ExecutionResult:
        """
        Execute a single abstract operator using DocETL with caching support.

        Checks cache first. If cache hit, returns cached result immediately.
        Otherwise executes normally and caches the result for future use.

        Args:
            operator: Abstract operator to execute
            input_data: Input data (file path, list of dicts, or Path object)
            config: Optional configuration (model, settings, etc.)
            force_execute: If True, bypass cache and re-execute (default: False)

        Returns:
            ExecutionResult with output data and metadata.
            Metadata includes 'cache_hit': True if result was from cache.

        Example:
            >>> executor = DocETLExecutor(cache_enabled=True)
            >>> # Normal execution (uses cache)
            >>> result = executor.execute_operator(
            ...     operator=my_operator,
            ...     input_data=[{"text": "example"}],
            ...     config={"default_model": "gpt-4o-mini"}
            ... )
            >>> # Force re-execution (bypasses cache)
            >>> result = executor.execute_operator(
            ...     operator=my_operator,
            ...     input_data=[{"text": "example"}],
            ...     force_execute=True
            ... )
        """
        # Check cache first (unless force_execute is True)
        if self.cache_enabled:
            cached_result = self.cache_manager.get_cached_result(
                operator, input_data, force_execute=force_execute
            )
            if cached_result:
                if self.verbose:
                    print(f"✓ Cache hit for operator: {operator.name}")

                return ExecutionResult(
                    success=True,
                    data=cached_result['output_data'],
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
        result = self._execute_operator_impl(operator, input_data, config)

        # Cache successful results (always cache, even for force_execute)
        if result.success and self.cache_enabled:
            self.cache_manager.set_cached_result(
                operator, input_data, result.data, result.metadata
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
                               input_data: Union[str, List[Dict], Path],
                               config: Optional[Dict[str, Any]] = None) -> ExecutionResult:
        """
        Internal implementation of operator execution (without caching).

        Args:
            operator: Abstract operator to execute
            input_data: Input data
            config: Optional configuration

        Returns:
            ExecutionResult with output data and metadata
        """
        if not self._docetl_available:
            return ExecutionResult(
                success=False,
                error="DocETL is not available. Please install docetl package."
            )

        try:
            from docetl.runner import DSLRunner

            # Convert abstract operator to DocETL format
            docetl_op = abstract_to_docetl(operator)

            # Create temporary pipeline
            temp_pipeline = self._create_temp_pipeline(
                docetl_op, input_data, config or {}
            )

            # Execute pipeline
            start_time = time.time()

            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                yaml.dump(temp_pipeline, f, default_flow_style=False, sort_keys=False)
                temp_file = f.name

            try:
                # Run the pipeline
                runner = DSLRunner.from_yaml(temp_file, max_threads=10)
                runner.load_run_save()

                # Load output
                output_path = temp_pipeline['pipeline']['output']['path']
                with open(output_path, 'r') as f:
                    output_data = json.load(f)

                execution_time = time.time() - start_time

                # Clean up output file
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
                        input_data: Optional[Union[str, Path]] = None,
                        save_intermediates: bool = False,
                        intermediate_dir: Optional[str] = None) -> ExecutionResult:
        """
        Execute a complete DocETL pipeline.

        Args:
            pipeline_config: Complete pipeline configuration (with datasets, operations, pipeline sections)
            input_data: Optional override for input data path
            save_intermediates: Whether to save intermediate results
            intermediate_dir: Directory to save intermediate results

        Returns:
            ExecutionResult with final output and metadata

        Example:
            >>> executor = DocETLExecutor()
            >>> result = executor.execute_pipeline(
            ...     pipeline_config=my_pipeline,
            ...     input_data="data.json",
            ...     save_intermediates=True,
            ...     intermediate_dir="./checkpoints"
            ... )
        """
        if not self._docetl_available:
            return ExecutionResult(
                success=False,
                error="DocETL is not available. Please install docetl package."
            )

        try:
            from docetl.runner import DSLRunner

            # Override input data if provided
            if input_data:
                for dataset_name in pipeline_config.get('datasets', {}).keys():
                    pipeline_config['datasets'][dataset_name]['path'] = str(input_data)

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
                            input_data: Union[str, List[Dict], Path],
                            config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a temporary pipeline for executing a single operator.

        Args:
            docetl_op: DocETL operator dictionary
            input_data: Input data (path or list)
            config: Configuration options

        Returns:
            Complete pipeline configuration dictionary
        """
        # Handle input data
        if isinstance(input_data, (str, Path)):
            input_path = str(input_data)
            dataset_type = 'file'
        else:
            # Save data to temp file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(input_data, f)
                input_path = f.name
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
