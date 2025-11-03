"""
Abstract Procedure Execution Engine

Unified executor for abstract operators and procedures. Routes operators to
appropriate system executors (DocETL, Lotus) based on operator.source.system.
Supports single operator execution, full procedures, and procedure ranges.
"""

from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import json
import os
import traceback
import subprocess  # For PythonCode operator execution
import sys

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
                            force_execute: bool = False) -> ExecutionResult:
        """
        Execute Python function in subprocess sandbox with caching support.

        The function must have signature: def process(sources)
        - sources: Dict with 'input_data' key containing previous operator output
        - Returns: Modified data with new fields

        Args:
            operator: PythonCode operator with 'function' in properties
            input_data: Input data from previous operator
            timeout: Subprocess timeout in seconds (default: 300)
            force_execute: If True, bypass cache and force execution

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

            # Get additional sources (beyond input_data)
            additional_sources = operator.properties.get('sources', [])

            # Build sources dict (currently only supports input_data)
            # TODO: Support loading additional data sources
            sources = {
                'input_data': input_data
            }

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
                        force_execute: bool = False) -> ExecutionResult:
        """Execute single operator using appropriate system executor."""
        # Convert dict to Operator if needed
        if isinstance(operator, dict):
            operator = dict_to_operator(operator)

        # Route PythonCode operators to subprocess executor with cache support
        if operator.type == "PythonCode":
            return self._execute_python_code(operator, input_data, force_execute=force_execute)

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
                        override_outputs: Optional[List[DataReference]] = None) -> ExecutionResult:
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
                intermediate_dir=intermediate_dir
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
                              intermediate_dir: Optional[str] = None) -> ExecutionResult:
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
                               file_cache: Optional[Dict[str, Any]] = None) -> Any:
        """
        Resolve a DataReference to raw data.

        Args:
            ref: DataReference to resolve
            outputs_registry: Registry of node outputs (node_id -> raw data)
            file_cache: Optional cache of loaded file data

        Returns:
            Resolved raw data (List[Dict])

        Raises:
            ValueError: If reference cannot be resolved
        """
        if ref.ref_type == "node":
            # Resolve node reference
            if ref.ref not in outputs_registry:
                raise ValueError(
                    f"Cannot resolve node reference '{ref.ref}': node has not been executed yet"
                )
            return outputs_registry[ref.ref]

        elif ref.ref_type == "file":
            # Resolve file reference
            if file_cache and ref.ref in file_cache:
                return file_cache[ref.ref]

            # Load from file
            data = self._load_from_reference(ref)

            # Cache for future use
            if file_cache is not None:
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

                if len(node.data_sources) == 1:
                    # Single input
                    input_data = self._resolve_data_reference(
                        node.data_sources[0],
                        outputs_registry,
                        file_cache
                    )
                else:
                    # Multiple inputs (AGGREGATE node)
                    # For now, use the first data source as primary input
                    # TODO: Support proper aggregation of multiple inputs
                    if self.verbose:
                        print(f"  Note: Node has {len(node.data_sources)} inputs, using first as primary")
                    input_data = self._resolve_data_reference(
                        node.data_sources[0],
                        outputs_registry,
                        file_cache
                    )

                # Setup intermediate directory for this node
                node_intermediate_dir = None
                if save_intermediates and intermediate_dir:
                    node_intermediate_dir = os.path.join(intermediate_dir, node_id)
                    os.makedirs(node_intermediate_dir, exist_ok=True)

                # Execute procedure with node's data_sources and outputs as overrides
                result = self.execute_procedure(
                    procedure=procedure,
                    input_data=input_data,
                    config=config,
                    save_intermediates=save_intermediates,
                    intermediate_dir=node_intermediate_dir,
                    override_data_sources=node.data_sources,
                    override_outputs=node.outputs
                )

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

            # Find FINAL node and return its output
            final_nodes = [n for n in pipeline.nodes.values() if n.node_type == NodeType.FINAL]
            if len(final_nodes) != 1:
                return ExecutionResult(
                    success=False,
                    error=f"Pipeline should have exactly 1 FINAL node, found {len(final_nodes)}"
                )

            final_node_id = final_nodes[0].node_id
            final_output_data = outputs_registry.get(final_node_id)

            if final_output_data is None:
                return ExecutionResult(
                    success=False,
                    error=f"Final node '{final_node_id}' did not produce output"
                )

            # Get final output path
            final_node = final_nodes[0]
            final_output_path = final_node.outputs[0].ref if final_node.outputs else None

            if self.verbose:
                print(f"\n{'='*60}")
                print(f"Pipeline execution completed successfully")
                print(f"Final output from node: {final_node_id}")
                print(f"{'='*60}\n")

            return ExecutionResult(
                success=True,
                data=final_output_data,
                metadata={
                    'pipeline_name': pipeline.name,
                    'nodes_executed': len(execution_order),
                    'final_node_id': final_node_id,
                    'final_output_path': final_output_path
                }
            )

        except Exception as e:
            return ExecutionResult(
                success=False,
                error=f"Pipeline execution failed: {str(e)}\n{traceback.format_exc()}"
            )

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
