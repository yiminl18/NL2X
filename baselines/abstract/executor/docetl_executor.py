"""
DocETL Executor for Abstract Procedure Engine

This module provides DocETL-specific execution functionality for abstract operators
and procedures. It handles conversion from abstract format to DocETL format and
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
from ..convert.docetl import abstract_to_docetl
from ..convert.docetl.path_manager import (
    get_dataset_paths,
    set_dataset_paths,
    get_output_path,
    set_output_path,
    apply_path_mappings
)

# DatasetManager type is used but no longer exists in a separate module
# Using Any for type hints where needed


class DocETLExecutor(BaseSystemExecutor):
    """Executor for DocETL system-specific operations with caching support."""

    def __init__(self,
                 verbose: bool = False,
                 cache_manager: Optional[OperatorCacheManager] = None,
                 data_manager: Optional[Any] = None):
        """
        Initialize DocETL executor with shared cache manager.

        Args:
            verbose: Enable verbose logging
            cache_manager: Shared OperatorCacheManager instance (if None, caching disabled)
            data_manager: Optional DatasetManager for dataset transformations and path management
        """
        # Initialize base class with cache settings from cache_manager
        cache_enabled = cache_manager.enabled if cache_manager else False
        cache_dir = cache_manager.cache_dir if cache_manager else None
        super().__init__(verbose, cache_enabled, cache_dir, data_manager)

        # Use shared cache manager
        self.cache_manager = cache_manager

    def get_system_name(self) -> str:
        return 'docetl'

    def execute_operator(self,
                        operator: Operator,
                        input_data: Any,
                        config: Optional[Dict[str, Any]] = None,
                        force_execute: bool = False) -> ExecutionResult:
        """Execute single abstract operator using DocETL with caching support."""
        if self.cache_enabled:
            cached_result = self.cache_manager.get_cached_result(
                operator, input_data, force_execute=force_execute
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

        result = self._execute_operator_impl(operator, input_data, config)

        if result.success and self.cache_enabled:
            self.cache_manager.set_cached_result(
                operator, input_data, result.data, result.metadata
            )
            if self.verbose:
                if force_execute:
                    print(f"✓ Updated cache for operator: {operator.name}")
                else:
                    print(f"✓ Cached result for operator: {operator.name}")

        result.metadata['cache_hit'] = False
        result.metadata['force_execute'] = force_execute

        return result

    def _execute_operator_impl(self,
                               operator: Operator,
                               input_data: Any,
                               config: Optional[Dict[str, Any]] = None) -> ExecutionResult:
        """Internal implementation of operator execution (without caching)."""
        temp_input_file = None
        temp_yaml_file = None
        temp_output_file = None

        try:
            from docetl.runner import DSLRunner

            # Save input_data to temporary JSON file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(input_data, f, indent=2)
                temp_input_file = f.name

            # Convert operator and create temporary procedure
            docetl_op = abstract_to_docetl(operator)
            temp_procedure = self._create_temp_procedure(
                docetl_op, temp_input_file, config or {}
            )

            start_time = time.time()

            # Save procedure to temporary YAML file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                yaml.dump(temp_procedure, f, default_flow_style=False, sort_keys=False)
                temp_yaml_file = f.name

            if self.verbose:
                print(f"Temp procedure config: {temp_procedure}")

            # Execute procedure
            runner = DSLRunner.from_yaml(temp_yaml_file, max_threads=10)
            runner.load_run_save()

            # Read output
            temp_output_file = temp_procedure['pipeline']['output']['path']
            with open(temp_output_file, 'r') as f:
                output_data = json.load(f)

            execution_time = time.time() - start_time

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

        except Exception as e:
            return ExecutionResult(
                success=False,
                error=f"Execution failed: {str(e)}\n{traceback.format_exc()}",
                metadata={
                    'operator_name': operator.name,
                    'operator_type': operator.type
                }
            )
        finally:
            # Clean up all temporary files
            for temp_file in [temp_input_file, temp_yaml_file, temp_output_file]:
                if temp_file and os.path.exists(temp_file):
                    os.remove(temp_file)

    def execute_procedure(self,
                         procedure_config: Dict[str, Any],
                         input_data: Optional[Any] = None,
                         save_intermediates: bool = False,
                         intermediate_dir: Optional[str] = None) -> ExecutionResult:
        """Execute complete DocETL procedure (input_data parameter is ignored - uses paths from config)."""
        try:
            from docetl.runner import DSLRunner

            if self.data_manager:
                dataset_paths = get_dataset_paths(procedure_config)
                dataset_path_mapping = {}

                for dataset_name, original_path in dataset_paths.items():
                    processed_path = self.data_manager.get_processed_path(original_path)
                    if processed_path:
                        dataset_path_mapping[dataset_name] = processed_path
                        if self.verbose:
                            print(f"Using processed dataset for {dataset_name}: {processed_path}")

                if dataset_path_mapping:
                    procedure_config = set_dataset_paths(procedure_config, dataset_path_mapping, in_place=False)

                original_output_path = get_output_path(procedure_config)
                if original_output_path:
                    new_output_path = self.data_manager.get_output_path(original_output_path)
                    if new_output_path:
                        procedure_config = set_output_path(procedure_config, new_output_path, in_place=True)
                        if self.verbose:
                            print(f"Using custom output path: {new_output_path}")

            # Note: input_data parameter is ignored - DocETL uses file paths from procedure_config

            if save_intermediates and intermediate_dir:
                procedure_config.setdefault('pipeline', {}).setdefault('output', {})
                procedure_config['pipeline']['output']['intermediate_dir'] = intermediate_dir

            start_time = time.time()

            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                yaml.dump(procedure_config, f, default_flow_style=False, sort_keys=False)
                temp_file = f.name

            try:
                runner = DSLRunner.from_yaml(temp_file, max_threads=10)
                runner.load_run_save()

                output_path = procedure_config['pipeline']['output']['path']
                with open(output_path, 'r') as f:
                    output_data = json.load(f)

                execution_time = time.time() - start_time

                intermediate_results = {}
                if save_intermediates and intermediate_dir:
                    intermediate_results = self._collect_intermediate_results(intermediate_dir)

                return ExecutionResult(
                    success=True,
                    data=output_data,
                    metadata={
                        'execution_time': execution_time,
                        'system': 'docetl',
                        'num_operations': len(procedure_config.get('operations', [])),
                        'intermediate_results': intermediate_results,
                        'output_path': output_path
                    }
                )

            finally:
                if os.path.exists(temp_file):
                    os.remove(temp_file)

        except Exception as e:
            return ExecutionResult(
                success=False,
                error=f"Procedure execution failed: {str(e)}\n{traceback.format_exc()}",
                metadata={'system': 'docetl'}
            )

    def execute_original_procedure(self, procedure_path: Union[str, Path]) -> ExecutionResult:
        """
        Execute DocETL procedure from YAML file path (simple wrapper).
        """
        try:
            from docetl.runner import DSLRunner

            procedure_path = Path(procedure_path)

            if self.verbose:
                print(f"Executing DocETL procedure: {procedure_path}")

            start_time = time.time()

            # Execute procedure using DSLRunner
            runner = DSLRunner.from_yaml(str(procedure_path), max_threads=10)
            runner.load_run_save()

            execution_time = time.time() - start_time

            # Read procedure config to find output path
            with open(procedure_path, 'r') as f:
                procedure_config = yaml.safe_load(f)

            output_path = procedure_config.get('pipeline', {}).get('output', {}).get('path')

            if output_path and os.path.exists(output_path):
                with open(output_path, 'r') as f:
                    output_data = json.load(f)

                if self.verbose:
                    data_info = f"{len(output_data)} records" if isinstance(output_data, list) else type(output_data).__name__
                    print(f"Procedure executed successfully: {data_info}")

                return ExecutionResult(
                    success=True,
                    data=output_data,
                    metadata={
                        'execution_time': execution_time,
                        'system': 'docetl',
                        'output_path': output_path,
                        'procedure_path': str(procedure_path)
                    }
                )
            else:
                # Procedure executed but no output file found
                if self.verbose:
                    print(f"Warning: Procedure executed but output file not found: {output_path}")

                return ExecutionResult(
                    success=True,
                    data=[],
                    metadata={
                        'execution_time': execution_time,
                        'system': 'docetl',
                        'output_path': output_path,
                        'procedure_path': str(procedure_path),
                        'warning': 'Output file not found'
                    }
                )

        except Exception as e:
            error_msg = f"Procedure execution failed: {str(e)}\n{traceback.format_exc()}"
            if self.verbose:
                print(f"✗ {error_msg}")

            return ExecutionResult(
                success=False,
                error=error_msg,
                metadata={
                    'system': 'docetl',
                    'procedure_path': str(procedure_path)
                }
            )

    def build_procedure_config(
        self,
        operators: List[Dict[str, Any]],
        input_path: str,
        output_path: str,
        dataset_name: str = 'input',
        step_name: str = 'main',
        default_model: Optional[str] = 'gpt-4o-mini',
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Build DocETL procedure configuration dictionary.

        Args:
            operators: List of DocETL operator dictionaries
            input_path: Path to input dataset
            output_path: Path for output file
            dataset_name: Name for the dataset in config (default: 'input')
            step_name: Name for the procedure step (default: 'main')
            default_model: Default LLM model to use (default: 'gpt-4o-mini')
            system_prompt: Optional system prompt for LLM calls

        Returns:
            Complete DocETL procedure configuration dictionary
        """
        procedure_config = {
            'datasets': {
                dataset_name: {
                    'type': 'file',
                    'path': input_path
                }
            },
            'operations': operators,
            'pipeline': {
                'steps': [{
                    'name': step_name,
                    'input': dataset_name,
                    'operations': [op['name'] for op in operators]
                }],
                'output': {
                    'type': 'file',
                    'path': output_path
                }
            }
        }

        # Add optional configurations
        if default_model:
            procedure_config['default_model'] = default_model

        if system_prompt:
            procedure_config['system_prompt'] = system_prompt

        return procedure_config

    def _create_temp_procedure(self,
                              docetl_op: Dict[str, Any],
                              input_path: str,
                              config: Dict[str, Any]) -> Dict[str, Any]:
        """Create temporary procedure for executing single operator."""
        # Create temporary output file
        fd, output_path = tempfile.mkstemp(suffix='.json')
        os.close(fd)  # Close file descriptor

        # Use centralized procedure config builder
        procedure = self.build_procedure_config(
            operators=[docetl_op],
            input_path=input_path,
            output_path=output_path,
            dataset_name='input_data',
            step_name='exec_step',
            default_model=config.get('default_model'),
            system_prompt=config.get('system_prompt')
        )

        return procedure

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
