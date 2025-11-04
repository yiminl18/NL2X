"""
Abstract Executor for Abstract Procedure Engine

This module provides execution functionality for abstract system's native PythonCode
operators. It handles Python code execution in sandboxed subprocesses with caching support.
"""

from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING
from pathlib import Path
import json
import subprocess
import sys
import traceback

# Import base executor and result class
from .base_executor import BaseSystemExecutor, ExecutionResult

# Import abstract operator classes
from ..ops.base import Operator
from ..procedure import Procedure

# Import cache manager
from ..cache import OperatorCacheManager

# Import data I/O utilities
from ..utils.data_io import load_from_reference, save_to_reference, DataReference


class AbstractPythonExecutor(BaseSystemExecutor):
    """Executor for abstract system - PythonCode operators only."""

    def __init__(self,
                 verbose: bool = False,
                 cache_manager: Optional[OperatorCacheManager] = None,
                 data_manager: Optional[Any] = None):
        """
        Initialize Abstract executor with shared cache manager.

        Args:
            verbose: Enable verbose logging
            cache_manager: Shared OperatorCacheManager instance (if None, caching disabled)
            data_manager: Optional data manager (not used by AbstractPythonExecutor)
        """
        # Initialize base class with cache settings from cache_manager
        cache_enabled = cache_manager.enabled if cache_manager else False
        cache_dir = cache_manager.cache_dir if cache_manager else None
        super().__init__(verbose, cache_enabled, cache_dir, data_manager)

        # Use shared cache manager
        self.cache_manager = cache_manager

    def get_system_name(self) -> str:
        """Get the name of the execution system."""
        return 'abstract'

    def supports_operator_type(self, op_type: str) -> bool:
        """Check if executor supports given operator type. Only supports PythonCode."""
        return op_type == 'PythonCode'

    def execute_operator(self,
                        operator: Operator,
                        input_data: Any,
                        config: Optional[Dict[str, Any]] = None,
                        force_execute: bool = False) -> ExecutionResult:
        """Execute single PythonCode operator with caching support."""

        # Validate operator type
        if operator.type != 'PythonCode':
            return ExecutionResult(
                success=False,
                error=f"AbstractPythonExecutor only supports PythonCode operators, got: {operator.type}",
                metadata={
                    'operator_name': operator.name,
                    'operator_type': operator.type,
                    'system': 'abstract'
                }
            )

        # Extract parameters from config
        timeout = config.get('timeout', 300) if config else 300
        source_node_mapping = config.get('source_node_mapping') if config else None

        # Execute with internal method
        return self._execute_python_code(
            operator=operator,
            input_data=input_data,
            timeout=timeout,
            force_execute=force_execute,
            source_node_mapping=source_node_mapping
        )

    def execute_original_procedure(self, procedure_path: Union[str, Path]) -> ExecutionResult:
        """
        Execute abstract procedure from YAML file.

        Args:
            procedure_path: Path to procedure YAML file

        Returns:
            ExecutionResult with output data or error
        """
        try:
            # Load procedure
            procedure = Procedure.load(procedure_path)

            # Validate base_system
            if procedure.base_system != 'abstract':
                return ExecutionResult(
                    success=False,
                    error=f"Procedure base_system is '{procedure.base_system}', expected 'abstract'",
                    metadata={
                        'system': 'abstract',
                        'procedure_path': str(procedure_path)
                    }
                )

            # Validate all operators are PythonCode
            operators = procedure.to_operators()
            for i, op in enumerate(operators):
                if op.type != 'PythonCode':
                    return ExecutionResult(
                        success=False,
                        error=f"Operator {i} ({op.name}) has type '{op.type}', only PythonCode supported",
                        metadata={
                            'system': 'abstract',
                            'procedure_path': str(procedure_path)
                        }
                    )

            # Load input data
            if not procedure.data_sources:
                return ExecutionResult(
                    success=False,
                    error="Procedure has no data sources",
                    metadata={
                        'system': 'abstract',
                        'procedure_path': str(procedure_path)
                    }
                )

            if self.verbose:
                print(f"Loading input data from: {procedure.data_sources[0].ref}")

            input_data = load_from_reference(procedure.data_sources[0])

            # Execute operators sequentially
            current_data = input_data
            for i, operator in enumerate(operators):
                if self.verbose:
                    print(f"Executing operator {i+1}/{len(operators)}: {operator.name}")

                result = self.execute_operator(
                    operator=operator,
                    input_data=current_data,
                    config=None,
                    force_execute=False
                )

                if not result.success:
                    return ExecutionResult(
                        success=False,
                        error=f"Operator {i} ({operator.name}) failed: {result.error}",
                        metadata={
                            'system': 'abstract',
                            'procedure_path': str(procedure_path),
                            'failed_operator': operator.name
                        }
                    )

                current_data = result.data

            # Save output
            if procedure.outputs:
                if self.verbose:
                    print(f"Saving output to: {procedure.outputs[0].ref}")
                save_to_reference(current_data, procedure.outputs[0])

            return ExecutionResult(
                success=True,
                data=current_data,
                metadata={
                    'system': 'abstract',
                    'procedure_path': str(procedure_path),
                    'operators_executed': len(operators),
                    'output_path': procedure.outputs[0].ref if procedure.outputs else None
                }
            )

        except Exception as e:
            return ExecutionResult(
                success=False,
                error=f"Procedure execution failed: {str(e)}\n{traceback.format_exc()}",
                metadata={
                    'system': 'abstract',
                    'procedure_path': str(procedure_path)
                }
            )

    def _execute_python_code(self,
                            operator: Operator,
                            input_data: Any,
                            timeout: int = 300,
                            force_execute: bool = False,
                            source_node_mapping: Optional[Dict[str, Any]] = None) -> ExecutionResult:
        """
        Execute Python function in subprocess sandbox with caching support.

        The function must have signature: def process(sources)
        - sources: Dict with 'input_data' key containing previous operator output
        - For aggregate nodes: sources also contains keys by source node_id (e.g., 'sum_node', 'median_node')
        - Returns: Modified data with new fields

        Args:
            operator: PythonCode operator with 'function' in properties
            input_data: Input data from previous operator
            timeout: Subprocess timeout in seconds (default: 300)
            force_execute: If True, bypass cache and force execution
            source_node_mapping: Optional dict mapping source node_ids to their data (for aggregate nodes)

        Returns:
            ExecutionResult with output data or error
        """
        # Check cache first if enabled
        if self.cache_enabled and not force_execute:
            cached_result = self.cache_manager.get_cached_result(operator, input_data)
            if cached_result is not None:
                if self.verbose:
                    print(f"  ✓ Cache hit for PythonCode operator: {operator.name}")
                return ExecutionResult(
                    success=True,
                    data=cached_result['output_data'],
                    metadata={**cached_result['metadata'], 'cache_hit': True}
                )

        try:
            # Get function from operator properties
            function_code = operator.properties.get('function')
            if not function_code:
                return ExecutionResult(
                    success=False,
                    error="PythonCode operator missing 'function' in properties"
                )

            # Build sources dict with input_data as default
            sources = {
                'input_data': input_data
            }

            # Add source node mappings for aggregate nodes (keyed by node_id)
            if source_node_mapping:
                sources.update(source_node_mapping)
                if self.verbose:
                    print(f"  Multi-source execution: {list(source_node_mapping.keys())}")

            # Serialize sources to JSON
            try:
                sources_json = json.dumps(sources)
            except (TypeError, ValueError) as e:
                return ExecutionResult(
                    success=False,
                    error=f"Failed to serialize sources to JSON: {str(e)}"
                )

            # Create wrapper script
            wrapper_script = f"""
import json
import sys

# User-defined function
{function_code}

# Load sources from stdin
sources = json.load(sys.stdin)

# Execute function
try:
    result = process(sources)
    # Output result as JSON
    print(json.dumps(result))
except Exception as e:
    print(f"Function execution error: {{e}}", file=sys.stderr)
    sys.exit(1)
"""

            if self.verbose:
                print(f"  Executing Python function in subprocess (timeout: {timeout}s)")
                print(f"  Sources: {list(sources.keys())}")
                if isinstance(input_data, list):
                    print(f"  Input data: {len(input_data)} records")

            # Execute wrapper in subprocess
            try:
                result = subprocess.run(
                    [sys.executable, '-c', wrapper_script],
                    input=sources_json,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    check=False
                )
            except subprocess.TimeoutExpired:
                return ExecutionResult(
                    success=False,
                    error=f"Python function execution timed out after {timeout} seconds"
                )

            # Check for execution errors
            if result.returncode != 0:
                error_msg = result.stderr or "Unknown error (no stderr output)"
                return ExecutionResult(
                    success=False,
                    error=f"Python function execution failed (exit code {result.returncode}):\n{error_msg}"
                )

            # Parse output JSON
            try:
                output_data = json.loads(result.stdout)
            except json.JSONDecodeError as e:
                return ExecutionResult(
                    success=False,
                    error=f"Failed to parse output JSON: {str(e)}\nStdout: {result.stdout[:500]}"
                )

            if self.verbose:
                if isinstance(output_data, list):
                    print(f"  Output: {len(output_data)} records")
                else:
                    print(f"  Output: {type(output_data).__name__}")

            # Create result metadata
            result_metadata = {
                'operator_type': 'PythonCode',
                'operator_name': operator.name,
                'execution_time': None,
                'cache_hit': False
            }

            # Store in cache if enabled
            if self.cache_enabled:
                self.cache_manager.set_cached_result(
                    operator=operator,
                    input_data=input_data,
                    output_data=output_data,
                    metadata=result_metadata
                )
                if self.verbose:
                    print(f"  ✓ Result cached for PythonCode operator: {operator.name}")

            return ExecutionResult(
                success=True,
                data=output_data,
                metadata=result_metadata
            )

        except Exception as e:
            return ExecutionResult(
                success=False,
                error=f"PythonCode execution failed: {str(e)}\n{traceback.format_exc()}"
            )
