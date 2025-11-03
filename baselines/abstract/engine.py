"""
Abstract Procedure Execution Engine

Unified executor for abstract operators and procedures. Routes operators to
appropriate system executors (DocETL, Lotus) based on operator.source.system.
Supports single operator execution, full procedures, and procedure ranges.
"""

from typing import Any, Dict, List, Optional, Union, Set
from pathlib import Path
import json
import os
import traceback
import subprocess  # For PythonCode operator execution
import sys
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import abstract operator classes
from .ops.base import Operator
from .procedure import Procedure

# Import executors
from .executor import DocETLExecutor, LotusExecutor, ExecutionResult

# Import conversion utilities
from .convert.docetl import dict_to_operator, operator_to_dict

# Import Pipeline and related classes
from .pipeline import Pipeline, DataReference, NodeType

# Import cache manager
from .cache import OperatorCacheManager


class AbstractExecutor:
    """Main executor for abstract operators and procedures with caching support."""

    def __init__(self,
                 verbose: bool = False,
                 cache_enabled: bool = True,
                 cache_dir: Optional[Union[str, Path]] = None):
        """Initialize the AbstractExecutor with unified cache manager and system executors."""
        self.verbose = verbose
        self.cache_enabled = cache_enabled
        self.cache_dir = cache_dir

        # Create unified cache manager shared by all executors
        self.cache_manager = OperatorCacheManager(
            cache_dir=cache_dir,
            enabled=cache_enabled
        )

        # Initialize executors for different systems with shared cache
        self.executors = {
            'docetl': DocETLExecutor(
                verbose=verbose,
                cache_manager=self.cache_manager
            ),
            'lotus': LotusExecutor(
                verbose=verbose,
                cache_manager=self.cache_manager
            )
        }

    def clear_cache(self, operator_hash: Optional[str] = None) -> int:
        """Clear cache entries. If operator_hash is None, clears all cache."""
        return self.cache_manager.clear_cache(operator_hash)

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics from unified cache manager."""
        return self.cache_manager.get_cache_stats()

    def disable_cache(self):
        """Temporarily disable caching."""
        self.cache_enabled = False
        self.cache_manager.enabled = False

    def enable_cache(self):
        """Re-enable caching."""
        self.cache_enabled = True
        self.cache_manager.enabled = True

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

    def _load_from_reference(self, ref: DataReference) -> List[Dict[str, Any]]:
        """
        Load data from a DataReference.

        Args:
            ref: DataReference to load (must be file type)

        Returns:
            List of dictionaries (raw data)

        Raises:
            ValueError: If reference is not a file or file doesn't exist
        """
        if ref.ref_type != "file":
            raise ValueError(f"Can only load from file references, got: {ref.ref_type}")

        file_path = Path(ref.ref)
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {ref.ref}")

        with open(file_path, 'r') as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError(f"Expected list data in {ref.ref}, got {type(data).__name__}")

        return data

    def _save_to_reference(self, data: List[Dict[str, Any]], ref: DataReference) -> None:
        """
        Save data to a DataReference.

        Args:
            data: Raw data to save
            ref: DataReference to save to (must be file type)

        Raises:
            ValueError: If reference is not a file
        """
        if ref.ref_type != "file":
            raise ValueError(f"Can only save to file references, got: {ref.ref_type}")

        file_path = Path(ref.ref)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)

    def execute_operator(self,
                        operator: Union[Operator, Dict[str, Any]],
                        input_data: Any,
                        config: Optional[Dict[str, Any]] = None,
                        force_execute: bool = False,
                        source_node_mapping: Optional[Dict[str, Any]] = None) -> ExecutionResult:
        """Execute single operator using appropriate system executor."""
        # Convert dict to Operator if needed
        if isinstance(operator, dict):
            operator = dict_to_operator(operator)

        # Route PythonCode operators to subprocess executor with cache support
        if operator.type == "PythonCode":
            return self._execute_python_code(operator, input_data, force_execute=force_execute, source_node_mapping=source_node_mapping)

        # Determine which system to use
        system = operator.source.get('system', 'docetl')

        if system not in self.executors:
            return ExecutionResult(
                success=False,
                error=f"Executor for system '{system}' not available"
            )

        # Execute using appropriate executor
        return self.executors[system].execute_operator(operator, input_data, config or {}, force_execute=force_execute)

    def execute_procedure(self,
                        procedure: Procedure,
                        input_data: Optional[Any] = None,
                        config: Optional[Dict[str, Any]] = None,
                        save_intermediates: bool = False,
                        intermediate_dir: Optional[str] = None,
                        override_data_sources: Optional[List[DataReference]] = None,
                        override_outputs: Optional[List[DataReference]] = None,
                        source_node_mapping: Optional[Dict[str, Any]] = None) -> ExecutionResult:
        """
        Execute complete procedure, routing operators to appropriate system executors.

        Args:
            procedure: Procedure to execute
            input_data: Input data (if None, loads from data_sources)
            config: Optional configuration
            save_intermediates: Whether to save intermediate results
            intermediate_dir: Directory for intermediates
            override_data_sources: Override procedure's data_sources (for pipeline use)
            override_outputs: Override procedure's outputs (for pipeline use)
            source_node_mapping: Optional dict mapping source node_ids to their data (for aggregate nodes)
        """
        try:
            # Use override_data_sources if provided, else procedure.data_sources
            effective_data_sources = override_data_sources if override_data_sources is not None else procedure.data_sources
            effective_outputs = override_outputs if override_outputs is not None else procedure.outputs

            # If input_data not provided, try to load from effective data sources
            if input_data is None:
                if effective_data_sources:
                    if self.verbose:
                        print(f"Loading dataset from data_sources: {effective_data_sources[0].ref}")
                    input_data = self._load_from_reference(effective_data_sources[0])
                else:
                    raise ValueError(
                        "input_data is required for procedure execution. "
                        "Either provide input_data parameter or set procedure.data_sources"
                    )

            # Execute operator-by-operator
            result = self.execute_procedure_range(
                procedure=procedure,
                input_data=input_data,
                start_index=0,
                end_index=None,
                config=config,
                save_intermediates=save_intermediates,
                intermediate_dir=intermediate_dir,
                source_node_mapping=source_node_mapping
            )

            # Save final output if effective outputs exist and are file references
            if result.success and effective_outputs:
                # Only save to file if output is a file reference (not a node reference)
                if effective_outputs[0].ref_type == "file":
                    if self.verbose:
                        print(f"Final output saved to: {effective_outputs[0].ref}")

                    self._save_to_reference(result.data, effective_outputs[0])
                    result.metadata['output_path'] = effective_outputs[0].ref
                elif self.verbose:
                    print(f"Skipping save (output is node reference): {effective_outputs[0].ref}")

            return result

        except Exception as e:
            return ExecutionResult(
                success=False,
                error=f"Procedure execution failed: {str(e)}\n{traceback.format_exc()}"
            )

    def execute_procedure_range(self,
                              procedure: Procedure,
                              input_data: Optional[Any],
                              start_index: int = 0,
                              end_index: Optional[int] = None,
                              config: Optional[Dict[str, Any]] = None,
                              save_intermediates: bool = False,
                              intermediate_dir: Optional[str] = None,
                              source_node_mapping: Optional[Dict[str, Any]] = None) -> ExecutionResult:
        """Execute procedure subset from start_index to end_index (inclusive)."""
        try:
            # Get operators from procedure
            operators = procedure.to_operators()

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

                # Pass source_node_mapping only for the first operator (if available)
                # This supports aggregate nodes where first operator needs multiple sources
                operator_source_mapping = source_node_mapping if i == start_index else None

                result = self.execute_operator(
                    operator=operator,
                    input_data=current_data,
                    config=config,
                    source_node_mapping=operator_source_mapping
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

                # Save intermediate result if requested
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

                # Update current data for next operator
                current_data = output_data

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
                data=output_data,
                metadata=metadata
            )

        except Exception as e:
            return ExecutionResult(
                success=False,
                error=f"Procedure range execution failed: {str(e)}\n{traceback.format_exc()}"
            )

    def _resolve_data_reference(self,
                               ref: DataReference,
                               outputs_registry: Dict[str, Any],
                               file_cache: Optional[Dict[str, Any]] = None,
                               file_cache_lock: Optional[Lock] = None) -> Any:
        """
        Resolve a DataReference to raw data (thread-safe when lock is provided).

        Args:
            ref: DataReference to resolve
            outputs_registry: Registry of node outputs (node_id -> raw data)
            file_cache: Optional cache of loaded file data
            file_cache_lock: Optional lock for thread-safe file cache access

        Returns:
            Resolved raw data (List[Dict])

        Raises:
            ValueError: If reference cannot be resolved
        """
        if ref.ref_type == "node":
            # Resolve node reference (read-only, safe in parallel)
            if ref.ref not in outputs_registry:
                raise ValueError(
                    f"Cannot resolve node reference '{ref.ref}': node has not been executed yet"
                )

            # Get node output (now a dict with output_name keys)
            node_output = outputs_registry[ref.ref]

            # Extract data for the requested output branch
            output_name = ref.output_name
            if output_name not in node_output:
                available = list(node_output.keys())
                raise ValueError(
                    f"Cannot resolve output '{output_name}' from node '{ref.ref}'. "
                    f"Available outputs: {available}"
                )

            return node_output[output_name]

        elif ref.ref_type == "file":
            # Check cache first (read-only)
            if file_cache and ref.ref in file_cache:
                return file_cache[ref.ref]

            # Load from file
            data = self._load_from_reference(ref)

            # Cache with lock if provided (thread-safe write)
            if file_cache is not None:
                if file_cache_lock:
                    with file_cache_lock:
                        # Double-check after acquiring lock (another thread may have loaded it)
                        if ref.ref not in file_cache:
                            file_cache[ref.ref] = data
                else:
                    # No lock provided (sequential execution)
                    file_cache[ref.ref] = data

            return data

        else:
            raise ValueError(f"Unknown DataReference type: {ref.ref_type}")

    def execute_pipeline(self,
                        pipeline: Pipeline,
                        initial_data: Optional[Dict[str, Any]] = None,
                        config: Optional[Dict[str, Any]] = None,
                        save_intermediates: bool = False,
                        intermediate_dir: Optional[str] = None) -> ExecutionResult:
        """
        Execute complete Pipeline DAG using topological sort.

        Args:
            pipeline: Pipeline object with DAG structure
            initial_data: Optional dict of file_path -> raw data for pre-loaded inputs
            config: Optional configuration for operator execution
            save_intermediates: Whether to save intermediate results
            intermediate_dir: Directory for intermediate results

        Returns:
            ExecutionResult with final output from FINAL node

        Raises:
            ValueError: If pipeline validation fails or execution error occurs
        """
        try:
            # Validate pipeline structure
            if self.verbose:
                print(f"Validating pipeline: {pipeline.name}")
            pipeline.validate()

            # Load all procedures
            if self.verbose:
                print(f"Loading procedures from pipeline nodes...")
            procedures = pipeline.load_procedures()

            # Check if parallel execution is enabled
            parallel_enabled = pipeline.properties.get('parallel_execution', False)
            max_workers = pipeline.properties.get('max_parallel_workers', 4)

            if parallel_enabled and max_workers > 1:
                # Use parallel execution
                if self.verbose:
                    print("Using parallel execution mode")

                # Group nodes into execution levels
                levels = self._group_nodes_into_levels(pipeline)

                # Setup intermediate directory
                if save_intermediates and not intermediate_dir:
                    intermediate_dir = os.path.join(os.getcwd(), 'pipeline_intermediates', pipeline.name)
                    os.makedirs(intermediate_dir, exist_ok=True)

                return self._execute_pipeline_parallel(
                    pipeline=pipeline,
                    procedures=procedures,
                    levels=levels,
                    max_workers=max_workers,
                    config=config,
                    save_intermediates=save_intermediates,
                    intermediate_dir=intermediate_dir
                )

            # Otherwise, use sequential execution (existing code)
            if self.verbose:
                print("Using sequential execution mode")

            # Perform topological sort
            execution_order = self._topological_sort(pipeline)
            if self.verbose:
                print(f"Execution order: {execution_order}")

            # Initialize registries
            outputs_registry: Dict[str, Any] = {}  # node_id -> output raw data
            file_cache: Dict[str, Any] = initial_data or {}

            # Setup intermediate directory
            if save_intermediates and not intermediate_dir:
                intermediate_dir = os.path.join(os.getcwd(), 'pipeline_intermediates', pipeline.name)
                os.makedirs(intermediate_dir, exist_ok=True)

            # Execute nodes in topological order
            for node_id in execution_order:
                node = pipeline.nodes[node_id]
                procedure = procedures[node_id]

                if self.verbose:
                    print(f"\n{'='*60}")
                    print(f"Executing node: {node_id} ({node.node_type.value})")
                    print(f"Procedure: {procedure.name}")
                    print(f"{'='*60}")

                # Resolve input data sources
                if len(node.data_sources) == 0:
                    raise ValueError(f"Node '{node_id}' has no data sources")

                # Initialize source_node_mapping for aggregate nodes
                source_node_mapping = None

                if len(node.data_sources) == 1:
                    # Single input
                    input_data = self._resolve_data_reference(
                        node.data_sources[0],
                        outputs_registry,
                        file_cache
                    )
                else:
                    # Multiple inputs (AGGREGATE node)
                    # Resolve all data sources and build node-based mapping
                    source_node_mapping = {}
                    all_resolved_data = []

                    for ds in node.data_sources:
                        resolved_data = self._resolve_data_reference(ds, outputs_registry, file_cache)
                        all_resolved_data.append(resolved_data)

                        # If data source is a node, key by node_id
                        if ds.ref_type == "node":
                            source_node_mapping[ds.ref] = resolved_data

                    # Use first data source as primary input_data (backwards compatible)
                    input_data = all_resolved_data[0]

                    if self.verbose:
                        print(f"  Multi-source node: {len(node.data_sources)} inputs from {list(source_node_mapping.keys())}")

                # Setup intermediate directory for this node
                node_intermediate_dir = None
                if save_intermediates and intermediate_dir:
                    node_intermediate_dir = os.path.join(intermediate_dir, node_id)
                    os.makedirs(node_intermediate_dir, exist_ok=True)

                # Check if this is a conditional loop
                condition_procedure_path = node.metadata.get('condition_procedure_path')

                if condition_procedure_path:
                    # Conditional loop execution
                    max_iterations = node.metadata.get('max_iterations', 10)
                    condition_field = node.metadata.get('condition_field', '_should_continue')
                    condition_aggregation = node.metadata.get('condition_aggregation', 'all')

                    # Load condition procedure
                    condition_procedure = Procedure.load(condition_procedure_path)

                    if self.verbose:
                        print(f"  Conditional loop: max_iterations={max_iterations}, condition_field='{condition_field}'")

                    iteration = 0
                    current_input_data = input_data

                    while iteration < max_iterations:
                        iteration += 1
                        if self.verbose:
                            print(f"  Loop iteration {iteration}/{max_iterations}")

                        # Execute data processing procedure
                        result = self.execute_procedure(
                            procedure=procedure,
                            input_data=current_input_data,
                            config=config,
                            save_intermediates=save_intermediates,
                            intermediate_dir=node_intermediate_dir,
                            override_data_sources=node.data_sources,
                            override_outputs=node.outputs,
                            source_node_mapping=source_node_mapping
                        )

                        if not result.success:
                            break

                        # Execute condition procedure to check if should continue
                        condition_result = self.execute_procedure(
                            procedure=condition_procedure,
                            input_data=result.data,
                            config=config,
                            save_intermediates=False,
                            intermediate_dir=None,
                            override_data_sources=None,
                            override_outputs=None,
                            source_node_mapping=None
                        )

                        if not condition_result.success:
                            if self.verbose:
                                print(f"  Condition procedure failed, stopping loop: {condition_result.error}")
                            break

                        # Evaluate condition
                        should_continue = self._evaluate_condition(
                            condition_result.data,
                            condition_field,
                            condition_aggregation
                        )

                        if not should_continue:
                            if self.verbose:
                                print(f"  Condition not met, stopping loop after {iteration} iterations")
                            break

                        # Continue loop: output becomes input for next iteration
                        current_input_data = result.data

                        if self.verbose:
                            print(f"  Condition met, continuing to next iteration")

                else:
                    # Check for fixed loop (any node type can loop)
                    iterations = node.metadata.get('iterations', 1)
                    is_fixed_loop = iterations > 1

                    # Execute procedure (with loop if needed)
                    current_input_data = input_data
                    for iteration in range(iterations):
                        if self.verbose and is_fixed_loop:
                            print(f"  Loop iteration {iteration + 1}/{iterations}")

                        result = self.execute_procedure(
                            procedure=procedure,
                            input_data=current_input_data,
                            config=config,
                            save_intermediates=save_intermediates,
                            intermediate_dir=node_intermediate_dir,
                            override_data_sources=node.data_sources,
                            override_outputs=node.outputs,
                            source_node_mapping=source_node_mapping
                        )

                        if not result.success:
                            break

                        # For loop nodes, output becomes input for next iteration
                        if is_fixed_loop and iteration < iterations - 1:
                            current_input_data = result.data

                if not result.success:
                    return ExecutionResult(
                        success=False,
                        error=f"Node '{node_id}' execution failed: {result.error}",
                        metadata={
                            'failed_node_id': node_id,
                            'failed_node_type': node.node_type.value,
                            'procedure_name': procedure.name
                        }
                    )

                # Save intermediate output if requested (for debugging)
                # Note: Final output to file is now handled by execute_procedure
                output_path = None
                if save_intermediates and intermediate_dir:
                    # Save intermediate node output for debugging
                    output_path = os.path.join(intermediate_dir, f"{node_id}_output.json")
                    with open(output_path, 'w') as f:
                        json.dump(result.data, f, indent=2)

                # Register output in memory for dependent nodes
                outputs_registry[node_id] = result.data

                if self.verbose:
                    data_info = f"{len(result.data)} records" if isinstance(result.data, list) else type(result.data).__name__
                    print(f"  ✓ Node '{node_id}' completed: {data_info}")
                    if output_path:
                        print(f"  Intermediate saved to: {output_path}")

            # Find FINAL nodes and collect their outputs
            final_nodes = [n for n in pipeline.nodes.values() if n.node_type == NodeType.FINAL]
            if len(final_nodes) < 1:
                return ExecutionResult(
                    success=False,
                    error=f"Pipeline should have at least 1 FINAL node, found {len(final_nodes)}"
                )

            # Collect outputs from all final nodes
            final_outputs = {}
            final_output_paths = {}
            for final_node in final_nodes:
                node_id = final_node.node_id
                output_data = outputs_registry.get(node_id)

                if output_data is None:
                    return ExecutionResult(
                        success=False,
                        error=f"Final node '{node_id}' did not produce output"
                    )

                final_outputs[node_id] = output_data
                final_output_paths[node_id] = final_node.outputs[0].ref if final_node.outputs else None

            if self.verbose:
                print(f"\n{'='*60}")
                print(f"Pipeline execution completed successfully")
                print(f"Final output from {len(final_nodes)} node(s): {list(final_outputs.keys())}")
                print(f"{'='*60}\n")

            # Return all final node outputs (for single final node, backwards compatible)
            return_data = final_outputs[final_nodes[0].node_id] if len(final_nodes) == 1 else final_outputs

            return ExecutionResult(
                success=True,
                data=return_data,
                metadata={
                    'pipeline_name': pipeline.name,
                    'nodes_executed': len(execution_order),
                    'final_node_ids': [n.node_id for n in final_nodes],
                    'final_output_paths': final_output_paths,
                    # Legacy fields for backwards compatibility (when single final node)
                    'final_node_id': final_nodes[0].node_id if len(final_nodes) == 1 else None,
                    'final_output_path': final_output_paths[final_nodes[0].node_id] if len(final_nodes) == 1 else None
                }
            )

        except Exception as e:
            return ExecutionResult(
                success=False,
                error=f"Pipeline execution failed: {str(e)}\n{traceback.format_exc()}"
            )

    def _evaluate_condition(
        self,
        condition_data: List[Dict[str, Any]],
        condition_field: str,
        aggregation: str = 'all'
    ) -> bool:
        """
        Evaluate condition from condition procedure output.

        Args:
            condition_data: Output from condition procedure
            condition_field: Field name to read boolean value from (e.g., '_should_continue')
            aggregation: How to aggregate multiple records ('all', 'any', 'first')

        Returns:
            True to continue loop, False to stop loop
        """
        if not condition_data:
            if self.verbose:
                print("  Condition evaluation: No data returned from condition procedure, stopping loop")
            return False  # No data = stop

        # Extract condition field values from all records
        values = []
        for record in condition_data:
            value = record.get(condition_field)
            if value is None:
                if self.verbose:
                    print(f"  Warning: Record missing condition field '{condition_field}', treating as False")
                values.append(False)
            else:
                values.append(bool(value))

        # Apply aggregation logic
        if aggregation == 'all':
            result = all(values)
            if self.verbose:
                print(f"  Condition evaluation (all): {values} → {result}")
        elif aggregation == 'any':
            result = any(values)
            if self.verbose:
                print(f"  Condition evaluation (any): {values} → {result}")
        elif aggregation == 'first':
            result = values[0] if values else False
            if self.verbose:
                print(f"  Condition evaluation (first): {values[0] if values else 'N/A'} → {result}")
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation}")

        return result

    def _topological_sort(self, pipeline: Pipeline) -> List[str]:
        """
        Perform topological sort on pipeline nodes.

        Args:
            pipeline: Pipeline object

        Returns:
            List of node IDs in execution order

        Raises:
            ValueError: If pipeline contains cycles
        """
        # Kahn's algorithm for topological sort
        in_degree = {node_id: len(node.parents) for node_id, node in pipeline.nodes.items()}
        queue = [node_id for node_id, degree in in_degree.items() if degree == 0]
        result = []

        while queue:
            # Process node with no dependencies
            current = queue.pop(0)
            result.append(current)

            # Reduce in-degree for children
            for child_id in pipeline.nodes[current].children:
                in_degree[child_id] -= 1
                if in_degree[child_id] == 0:
                    queue.append(child_id)

        # Check if all nodes were processed (no cycles)
        if len(result) != len(pipeline.nodes):
            raise ValueError("Pipeline contains cycles - cannot perform topological sort")

        return result

    def _group_nodes_into_levels(self, pipeline: Pipeline) -> List[List[str]]:
        """
        Group pipeline nodes into execution levels for parallel execution.

        Nodes in the same level have no dependencies on each other and can
        execute in parallel. Each level must complete before the next begins.
        AGGREGATE nodes naturally wait for all parents in previous levels.

        Args:
            pipeline: Pipeline object

        Returns:
            List of levels, where each level is a list of node IDs

        Raises:
            ValueError: If pipeline contains cycles
        """
        levels = []
        in_degree = {node_id: len(node.parents) for node_id, node in pipeline.nodes.items()}

        while in_degree:
            # Current level: all nodes with zero dependencies
            current_level = [nid for nid, deg in in_degree.items() if deg == 0]

            if not current_level:
                raise ValueError("Pipeline contains cycles - cannot group into levels")

            levels.append(current_level)

            # Remove current level nodes and update in-degrees
            for node_id in current_level:
                del in_degree[node_id]
                for child_id in pipeline.nodes[node_id].children:
                    if child_id in in_degree:
                        in_degree[child_id] -= 1

        return levels

    def _execute_node_worker(
        self,
        node_id: str,
        pipeline: Pipeline,
        procedures: Dict[str, 'Procedure'],
        outputs_registry: Dict[str, Any],
        outputs_lock: Lock,
        file_cache: Dict[str, Any],
        file_cache_lock: Lock,
        config: Optional[Dict[str, Any]],
        save_intermediates: bool,
        intermediate_dir: Optional[str]
    ) -> tuple:
        """
        Worker function to execute a single pipeline node in parallel.

        This function is designed to be thread-safe and executed in a ThreadPoolExecutor.
        It resolves inputs, executes the node's procedure, and stores results.

        Args:
            node_id: ID of the node to execute
            pipeline: Pipeline object
            procedures: Dict mapping node_id to Procedure objects
            outputs_registry: Shared dict for storing node outputs (thread-safe with lock)
            outputs_lock: Lock for writing to outputs_registry
            file_cache: Shared dict for caching file loads (thread-safe with lock)
            file_cache_lock: Lock for writing to file_cache
            config: Optional configuration for execution
            save_intermediates: Whether to save intermediate results
            intermediate_dir: Directory for intermediate results

        Returns:
            Tuple of (node_id, ExecutionResult)
        """
        try:
            node = pipeline.nodes[node_id]
            procedure = procedures[node_id]

            if self.verbose:
                print(f"[Thread] Executing node: {node_id} ({node.node_type.value})")

            # Resolve input data sources
            if len(node.data_sources) == 0:
                raise ValueError(f"Node '{node_id}' has no data sources")

            source_node_mapping = None

            if len(node.data_sources) == 1:
                # Single input
                input_data = self._resolve_data_reference(
                    node.data_sources[0],
                    outputs_registry,
                    file_cache,
                    file_cache_lock
                )
            else:
                # Multiple inputs (AGGREGATE node)
                source_node_mapping = {}
                all_resolved_data = []

                for ds in node.data_sources:
                    # Thread-safe read from registries
                    resolved_data = self._resolve_data_reference(
                        ds, outputs_registry, file_cache, file_cache_lock
                    )
                    all_resolved_data.append(resolved_data)

                    if ds.ref_type == "node":
                        source_node_mapping[ds.ref] = resolved_data

                input_data = all_resolved_data[0]

                if self.verbose:
                    print(f"  [Thread] Multi-source node: {len(node.data_sources)} inputs")

            # Setup intermediate directory
            node_intermediate_dir = None
            if save_intermediates and intermediate_dir:
                node_intermediate_dir = os.path.join(intermediate_dir, node_id)
                os.makedirs(node_intermediate_dir, exist_ok=True)

            # Check if this is a conditional loop
            condition_procedure_path = node.metadata.get('condition_procedure_path')

            if condition_procedure_path:
                # Conditional loop execution
                max_iterations = node.metadata.get('max_iterations', 10)
                condition_field = node.metadata.get('condition_field', '_should_continue')
                condition_aggregation = node.metadata.get('condition_aggregation', 'all')

                # Load condition procedure
                condition_procedure = Procedure.load(condition_procedure_path)

                if self.verbose:
                    print(f"  [Thread] Conditional loop: max_iterations={max_iterations}, condition_field='{condition_field}'")

                iteration = 0
                current_input_data = input_data

                while iteration < max_iterations:
                    iteration += 1
                    if self.verbose:
                        print(f"  [Thread] Loop iteration {iteration}/{max_iterations}")

                    # Execute data processing procedure
                    result = self.execute_procedure(
                        procedure=procedure,
                        input_data=current_input_data,
                        config=config,
                        save_intermediates=save_intermediates,
                        intermediate_dir=node_intermediate_dir,
                        override_data_sources=node.data_sources,
                        override_outputs=node.outputs,
                        source_node_mapping=source_node_mapping
                    )

                    if not result.success:
                        break

                    # Execute condition procedure to check if should continue
                    condition_result = self.execute_procedure(
                        procedure=condition_procedure,
                        input_data=result.data,
                        config=config,
                        save_intermediates=False,
                        intermediate_dir=None,
                        override_data_sources=None,
                        override_outputs=None,
                        source_node_mapping=None
                    )

                    if not condition_result.success:
                        if self.verbose:
                            print(f"  [Thread] Condition procedure failed, stopping loop: {condition_result.error}")
                        break

                    # Evaluate condition
                    should_continue = self._evaluate_condition(
                        condition_result.data,
                        condition_field,
                        condition_aggregation
                    )

                    if not should_continue:
                        if self.verbose:
                            print(f"  [Thread] Condition not met, stopping loop after {iteration} iterations")
                        break

                    # Continue loop: output becomes input for next iteration
                    current_input_data = result.data

                    if self.verbose:
                        print(f"  [Thread] Condition met, continuing to next iteration")

            else:
                # Check for fixed loop (any node type can loop)
                iterations = node.metadata.get('iterations', 1)
                is_fixed_loop = iterations > 1

                # Execute procedure (with loop if needed)
                current_input_data = input_data
                for iteration in range(iterations):
                    if self.verbose and is_fixed_loop:
                        print(f"  [Thread] Loop iteration {iteration + 1}/{iterations}")

                    result = self.execute_procedure(
                        procedure=procedure,
                        input_data=current_input_data,
                        config=config,
                        save_intermediates=save_intermediates,
                        intermediate_dir=node_intermediate_dir,
                        override_data_sources=node.data_sources,
                        override_outputs=node.outputs,
                        source_node_mapping=source_node_mapping
                    )

                    if not result.success:
                        break

                    # For loop nodes, output becomes input for next iteration
                    if is_fixed_loop and iteration < iterations - 1:
                        current_input_data = result.data

            if not result.success:
                return (node_id, result)

            # Thread-safe write to outputs_registry
            with outputs_lock:
                if node.node_type == NodeType.CONDITIONAL_ROUTING:
                    # Route records to different branches based on routing_field
                    routing_field = node.metadata.get('routing_field', '_route_to')
                    branch_outputs = {}

                    # Pre-initialize all declared output branches (ensures empty branches exist)
                    for output in node.outputs:
                        if output.ref_type == 'node':
                            branch_outputs[output.output_name] = []

                    # Route records to branches
                    for record in result.data:
                        # Get routing target from record
                        target = record.get(routing_field, 'default')

                        # Add record to target branch
                        if target in branch_outputs:
                            branch_outputs[target].append(record)
                        else:
                            # Route to default if target not in declared branches
                            if 'default' not in branch_outputs:
                                branch_outputs['default'] = []
                            branch_outputs['default'].append(record)
                            if self.verbose:
                                print(f"  [Warning] Record routed to undeclared branch '{target}', using 'default'")

                    outputs_registry[node_id] = branch_outputs

                    if self.verbose:
                        branch_info = ', '.join(f"{k}: {len(v)}" for k, v in branch_outputs.items())
                        print(f"  [Thread] Routed to branches: {branch_info}")
                else:
                    # Normal node: wrap in default key for consistent structure
                    outputs_registry[node_id] = {'default': result.data}

            # Save intermediate if requested
            if save_intermediates and intermediate_dir:
                output_path = os.path.join(intermediate_dir, f"{node_id}_output.json")
                with open(output_path, 'w') as f:
                    json.dump(result.data, f, indent=2)

            if self.verbose:
                data_info = f"{len(result.data)} records" if isinstance(result.data, list) else type(result.data).__name__
                print(f"  [Thread] ✓ Node '{node_id}' completed: {data_info}")

            return (node_id, result)

        except Exception as e:
            error_result = ExecutionResult(
                success=False,
                error=f"Node '{node_id}' execution failed: {str(e)}\n{traceback.format_exc()}"
            )
            return (node_id, error_result)

    def _execute_pipeline_parallel(
        self,
        pipeline: Pipeline,
        procedures: Dict[str, 'Procedure'],
        levels: List[List[str]],
        max_workers: int,
        config: Optional[Dict[str, Any]],
        save_intermediates: bool,
        intermediate_dir: Optional[str]
    ) -> ExecutionResult:
        """
        Execute pipeline nodes in parallel using level-based execution.

        Nodes are grouped into levels based on their dependencies. All nodes
        in a level execute in parallel, then the next level begins. This ensures
        AGGREGATE nodes receive all their inputs before execution.

        Args:
            pipeline: Pipeline object
            procedures: Dict mapping node_id to Procedure objects
            levels: List of execution levels (each level is a list of node IDs)
            max_workers: Maximum number of parallel threads
            config: Optional configuration
            save_intermediates: Whether to save intermediate results
            intermediate_dir: Directory for intermediate results

        Returns:
            ExecutionResult with data from final node(s) and execution metadata
        """
        # Initialize thread-safe registries
        outputs_registry: Dict[str, Any] = {}
        outputs_lock = Lock()
        file_cache: Dict[str, Any] = {}
        file_cache_lock = Lock()

        # Track failed nodes for best-effort execution
        failed_nodes: Set[str] = set()

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Parallel execution: {len(levels)} levels, max {max_workers} workers")
            print(f"{'='*60}\n")

        # Execute level by level
        for level_idx, level_nodes in enumerate(levels):
            if self.verbose:
                print(f"Level {level_idx + 1}/{len(levels)}: {len(level_nodes)} nodes - {level_nodes}")

            # Skip nodes whose parents failed (best-effort)
            executable_nodes = [
                nid for nid in level_nodes
                if not any(parent in failed_nodes for parent in pipeline.nodes[nid].parents)
            ]

            if not executable_nodes:
                if self.verbose:
                    print(f"  Skipping level {level_idx + 1} - all nodes depend on failed parents")
                continue

            # Execute all nodes in level in parallel
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        self._execute_node_worker,
                        node_id,
                        pipeline,
                        procedures,
                        outputs_registry,
                        outputs_lock,
                        file_cache,
                        file_cache_lock,
                        config,
                        save_intermediates,
                        intermediate_dir
                    ): node_id
                    for node_id in executable_nodes
                }

                # Collect results as they complete (best-effort)
                for future in as_completed(futures):
                    node_id, result = future.result()

                    if not result.success:
                        failed_nodes.add(node_id)
                        if self.verbose:
                            print(f"  ✗ Node '{node_id}' failed: {result.error}")
                        # Continue with other nodes (best-effort)

            if self.verbose:
                successful = len([n for n in executable_nodes if n not in failed_nodes])
                print(f"  Level {level_idx + 1} complete: {successful}/{len(executable_nodes)} successful\n")

        # Collect final outputs
        final_nodes = [n for n in pipeline.nodes.values() if n.node_type == NodeType.FINAL]

        if not final_nodes:
            return ExecutionResult(
                success=False,
                error="Pipeline has no FINAL nodes"
            )

        # Collect outputs from successful final nodes
        final_outputs = {}
        final_output_paths = {}
        successful_finals = []

        for final_node in final_nodes:
            node_id = final_node.node_id

            if node_id in failed_nodes:
                continue

            output_data = outputs_registry.get(node_id)
            if output_data is None:
                continue

            final_outputs[node_id] = output_data
            final_output_paths[node_id] = final_node.outputs[0].ref if final_node.outputs else None
            successful_finals.append(node_id)

        # Determine overall success
        all_finals_succeeded = len(successful_finals) == len(final_nodes)

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Parallel execution completed")
            print(f"Final outputs: {len(successful_finals)}/{len(final_nodes)} successful")
            if failed_nodes:
                print(f"Failed nodes: {list(failed_nodes)}")
            print(f"{'='*60}\n")

        # Return based on success
        if not successful_finals:
            return ExecutionResult(
                success=False,
                error=f"All final nodes failed. Failed nodes: {list(failed_nodes)}",
                metadata={
                    'failed_nodes': list(failed_nodes),
                    'pipeline_name': pipeline.name
                }
            )

        return_data = final_outputs[successful_finals[0]] if len(successful_finals) == 1 else final_outputs

        return ExecutionResult(
            success=all_finals_succeeded,
            data=return_data,
            metadata={
                'pipeline_name': pipeline.name,
                'execution_mode': 'parallel',
                'levels_executed': len(levels),
                'nodes_executed': len(outputs_registry),
                'nodes_failed': len(failed_nodes),
                'failed_nodes': list(failed_nodes) if failed_nodes else None,
                'final_node_ids': successful_finals,
                'final_output_paths': final_output_paths,
                # Legacy fields for backwards compatibility
                'final_node_id': successful_finals[0] if len(successful_finals) == 1 else None,
                'final_output_path': final_output_paths[successful_finals[0]] if len(successful_finals) == 1 else None
            }
        )
