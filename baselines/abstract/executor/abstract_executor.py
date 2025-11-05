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


class AbstractExecutor(BaseSystemExecutor):
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
            data_manager: Optional data manager (not used by AbstractExecutor)
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

    def get_primary_data_format(self) -> str:
        """Get the primary data format supported by the executor (e.g., 'json', 'csv')."""
        return 'json'

    def supports_operator_type(self, op_type: str) -> bool:
        """Check if executor supports given operator type. Supports PythonCode, Convert, Sample, and Unnest."""
        return op_type in ('PythonCode', 'Convert', 'Sample', 'Unnest', 'unnest')

    def execute_operator(self,
                        operator: Operator,
                        input_data: Any,
                        config: Optional[Dict[str, Any]] = None,
                        force_execute: bool = False) -> ExecutionResult:
        """Execute PythonCode or Convert operator with caching support."""

        # Route to appropriate execution method based on operator type
        if operator.type == 'PythonCode':
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

        elif operator.type == 'Convert':
            # Execute format conversion
            return self._execute_convert(
                operator=operator,
                input_data=input_data,
                force_execute=force_execute
            )

        elif operator.type == 'Sample':
            # Execute sampling
            return self._execute_sample(
                operator=operator,
                input_data=input_data,
                force_execute=force_execute
            )

        elif operator.type in ('Unnest', 'unnest'):
            # Execute unnesting
            return self._execute_unnest(
                operator=operator,
                input_data=input_data,
                force_execute=force_execute
            )

        else:
            return ExecutionResult(
                success=False,
                error=f"AbstractExecutor only supports PythonCode, Convert, Sample, and Unnest operators, got: {operator.type}",
                metadata={
                    'operator_name': operator.name,
                    'operator_type': operator.type,
                    'system': 'abstract'
                }
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

            # Validate all operators are supported by AbstractExecutor
            operators = procedure.to_operators()
            for i, op in enumerate(operators):
                if not self.supports_operator_type(op.type):
                    return ExecutionResult(
                        success=False,
                        error=f"Operator {i} ({op.name}) has type '{op.type}', only PythonCode, Convert, Sample, and Unnest supported",
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

    def _execute_convert(self,
                        operator: Operator,
                        input_data: Any,
                        force_execute: bool = False) -> ExecutionResult:
        """
        Execute format conversion (CSV to JSON, JSON to CSV).

        Args:
            operator: Convert operator with source_format and target_format in properties
            input_data: Input data (list of dicts for JSON, file path for CSV)
            force_execute: If True, bypass cache and force execution

        Returns:
            ExecutionResult with converted data or error
        """
        # Check cache first if enabled
        if self.cache_enabled and not force_execute:
            cached_result = self.cache_manager.get_cached_result(operator, input_data)
            if cached_result is not None:
                if self.verbose:
                    print(f"  ✓ Cache hit for Convert operator: {operator.name}")
                return ExecutionResult(
                    success=True,
                    data=cached_result['output_data'],
                    metadata={**cached_result['metadata'], 'cache_hit': True}
                )

        try:
            # Get conversion parameters
            source_format = operator.properties.get('source_format')
            target_format = operator.properties.get('target_format')

            if not source_format or not target_format:
                return ExecutionResult(
                    success=False,
                    error="Convert operator missing 'source_format' or 'target_format' in properties"
                )

            if self.verbose:
                print(f"  Converting {source_format} → {target_format}")

            # Perform conversion based on formats
            if source_format == 'csv' and target_format == 'json':
                output_data = self._convert_csv_to_json(input_data)

            elif source_format == 'json' and target_format == 'csv':
                return ExecutionResult(
                    success=False,
                    error="JSON to CSV conversion not yet implemented"
                )

            else:
                return ExecutionResult(
                    success=False,
                    error=f"Unsupported conversion: {source_format} → {target_format}"
                )

            if self.verbose:
                if isinstance(output_data, list):
                    print(f"  Converted: {len(output_data)} records")

            # Create result metadata
            result_metadata = {
                'operator_type': 'Convert',
                'operator_name': operator.name,
                'source_format': source_format,
                'target_format': target_format,
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
                    print(f"  ✓ Result cached for Convert operator: {operator.name}")

            return ExecutionResult(
                success=True,
                data=output_data,
                metadata=result_metadata
            )

        except Exception as e:
            return ExecutionResult(
                success=False,
                error=f"Convert execution failed: {str(e)}\n{traceback.format_exc()}"
            )

    def _convert_csv_to_json(self, input_data: Any) -> List[Dict[str, Any]]:
        """
        Convert CSV data to JSON format.

        Args:
            input_data: CSV file path (str) or CSV-formatted data

        Returns:
            List of dictionaries (JSON format)
        """
        import pandas as pd

        # Handle file path input
        if isinstance(input_data, str):
            # Assume it's a file path
            if self.verbose:
                print(f"  Reading CSV file: {input_data}")
            df = pd.read_csv(input_data)
            return df.to_dict('records')

        # Handle DataFrame input
        elif isinstance(input_data, pd.DataFrame):
            return input_data.to_dict('records')

        # If already in JSON format, return as-is
        elif isinstance(input_data, list):
            return input_data

        else:
            raise ValueError(f"Unsupported input data type for CSV conversion: {type(input_data)}")

    def _execute_sample(self,
                       operator: Operator,
                       input_data: Any,
                       force_execute: bool = False) -> ExecutionResult:
        """
        Execute sampling operation (uniform random or first N records).

        Args:
            operator: Sample operator with method and samples in properties
            input_data: Input data (list of dicts)
            force_execute: If True, bypass cache and force execution

        Returns:
            ExecutionResult with sampled data or error
        """
        # Check cache first if enabled
        if self.cache_enabled and not force_execute:
            cached_result = self.cache_manager.get_cached_result(operator, input_data)
            if cached_result is not None:
                if self.verbose:
                    print(f"  ✓ Cache hit for Sample operator: {operator.name}")
                return ExecutionResult(
                    success=True,
                    data=cached_result['output_data'],
                    metadata={**cached_result['metadata'], 'cache_hit': True}
                )

        try:
            # Get sampling parameters
            method = operator.properties.get('method')
            samples = operator.properties.get('samples')
            random_state = operator.properties.get('random_state')

            if not method or samples is None:
                return ExecutionResult(
                    success=False,
                    error="Sample operator missing 'method' or 'samples' in properties"
                )

            # Validate input data is a list
            if not isinstance(input_data, list):
                return ExecutionResult(
                    success=False,
                    error=f"Sample operator expects list input, got: {type(input_data)}"
                )

            total_records = len(input_data)

            # Calculate actual number of samples
            if isinstance(samples, float):
                # Percentage (0.0 to 1.0)
                if not (0.0 < samples <= 1.0):
                    return ExecutionResult(
                        success=False,
                        error=f"Sample percentage must be between 0 and 1, got: {samples}"
                    )
                n_samples = int(total_records * samples)
            elif isinstance(samples, int):
                # Exact count
                if samples <= 0:
                    return ExecutionResult(
                        success=False,
                        error=f"Sample count must be positive, got: {samples}"
                    )
                n_samples = samples
            else:
                return ExecutionResult(
                    success=False,
                    error=f"Sample 'samples' must be int or float, got: {type(samples)}"
                )

            # Handle edge case: requested samples exceed available data
            if n_samples > total_records:
                if self.verbose:
                    print(f"  ⚠ Requested {n_samples} samples but only {total_records} records available")
                n_samples = total_records

            if self.verbose:
                print(f"  Sampling {n_samples} records from {total_records} using method '{method}'")

            # Perform sampling based on method
            if method == 'first':
                output_data = input_data[:n_samples]

            elif method == 'uniform':
                import random
                # Use random_state if provided
                if random_state is not None:
                    random.seed(random_state)

                # Random sampling without replacement
                indices = random.sample(range(total_records), n_samples)
                output_data = [input_data[i] for i in indices]

            else:
                return ExecutionResult(
                    success=False,
                    error=f"Unsupported sampling method: {method}. Use 'uniform' or 'first'"
                )

            if self.verbose:
                print(f"  Sampled: {len(output_data)} records")

            # Create result metadata
            result_metadata = {
                'operator_type': 'Sample',
                'operator_name': operator.name,
                'method': method,
                'requested_samples': samples,
                'actual_samples': n_samples,
                'total_records': total_records,
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
                    print(f"  ✓ Result cached for Sample operator: {operator.name}")

            return ExecutionResult(
                success=True,
                data=output_data,
                metadata=result_metadata
            )

        except Exception as e:
            return ExecutionResult(
                success=False,
                error=f"Sample execution failed: {str(e)}\n{traceback.format_exc()}"
            )

    def _execute_unnest(self,
                       operator: Operator,
                       input_data: Any,
                       force_execute: bool = False) -> ExecutionResult:
        """
        Execute unnest operation (expand arrays or flatten nested dicts).

        Args:
            operator: Unnest operator with unnest_key and depth in properties
            input_data: Input data (list of dicts)
            force_execute: If True, bypass cache and force execution

        Returns:
            ExecutionResult with unnested data or error
        """
        # Check cache first if enabled
        if self.cache_enabled and not force_execute:
            cached_result = self.cache_manager.get_cached_result(operator, input_data)
            if cached_result is not None:
                if self.verbose:
                    print(f"  ✓ Cache hit for Unnest operator: {operator.name}")
                return ExecutionResult(
                    success=True,
                    data=cached_result['output_data'],
                    metadata={**cached_result['metadata'], 'cache_hit': True}
                )

        try:
            import copy

            # Get unnesting parameters
            unnest_key = operator.properties.get('unnest_key')
            expand_fields = operator.properties.get('expand_fields')
            keep_empty = operator.properties.get('keep_empty', False)

            # Handle depth parameter (priority) vs recursive parameter (backward compatibility)
            depth = operator.properties.get('depth')
            if depth is None:
                # Fall back to recursive parameter
                recursive = operator.properties.get('recursive', False)
                depth = 999 if recursive else 1

            if not unnest_key:
                return ExecutionResult(
                    success=False,
                    error="Unnest operator missing 'unnest_key' in properties"
                )

            # Validate input data is a list
            if not isinstance(input_data, list):
                return ExecutionResult(
                    success=False,
                    error=f"Unnest operator expects list input, got: {type(input_data)}"
                )

            if self.verbose:
                print(f"  Unnesting field '{unnest_key}' with depth={depth}")

            # Helper function for recursive unnesting
            def unnest_recursive(item: Dict[str, Any], key: str, level: int = 0) -> List[Dict[str, Any]]:
                """Recursively unnest a single item."""
                # Check if key exists
                if key not in item:
                    available_keys = list(item.keys())
                    raise KeyError(
                        f"Unnest key '{key}' not found in record. "
                        f"Available keys: {available_keys}"
                    )

                value = item[key]

                # Reached depth limit
                if level >= depth:
                    return [item]

                # Handle empty/None values
                if value is None or (isinstance(value, (list, tuple, set, dict)) and len(value) == 0):
                    if keep_empty:
                        new_item = copy.deepcopy(item)
                        new_item[key] = None
                        return [new_item]
                    else:
                        return []

                # Handle list/tuple/set
                if isinstance(value, (list, tuple, set)):
                    result = []
                    for element in value:
                        new_item = copy.deepcopy(item)
                        new_item[key] = element
                        # Recursively unnest if we haven't reached depth limit
                        if level + 1 < depth:
                            result.extend(unnest_recursive(new_item, key, level + 1))
                        else:
                            result.append(new_item)
                    return result

                # Handle dict
                elif isinstance(value, dict):
                    new_item = copy.deepcopy(item)
                    # Determine which fields to expand
                    fields_to_expand = expand_fields if expand_fields else list(value.keys())
                    # Expand specified fields to top level
                    for field in fields_to_expand:
                        new_item[field] = value.get(field)
                    return [new_item]

                # Non-iterable value
                else:
                    if level == 0:
                        raise TypeError(
                            f"Value of unnest key '{key}' is not iterable (list, tuple, set, or dict). "
                            f"Got type: {type(value).__name__}"
                        )
                    # At deeper levels, just return the item as-is
                    return [item]

            # Apply unnesting to all records
            output_data = []
            for record in input_data:
                try:
                    unnested_records = unnest_recursive(record, unnest_key, level=0)
                    output_data.extend(unnested_records)
                except (KeyError, TypeError) as e:
                    # Re-raise with context about which record failed
                    return ExecutionResult(
                        success=False,
                        error=f"Unnest failed on record: {str(e)}"
                    )

            if self.verbose:
                print(f"  Unnested: {len(input_data)} records → {len(output_data)} records")

            # Create result metadata
            result_metadata = {
                'operator_type': 'Unnest',
                'operator_name': operator.name,
                'unnest_key': unnest_key,
                'depth': depth,
                'input_records': len(input_data),
                'output_records': len(output_data),
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
                    print(f"  ✓ Result cached for Unnest operator: {operator.name}")

            return ExecutionResult(
                success=True,
                data=output_data,
                metadata=result_metadata
            )

        except Exception as e:
            return ExecutionResult(
                success=False,
                error=f"Unnest execution failed: {str(e)}\n{traceback.format_exc()}"
            )
