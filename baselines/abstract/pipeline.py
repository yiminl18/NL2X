"""
Pipeline orchestration layer for multi-procedure workflows.

A Pipeline contains references to one or more Procedure files and orchestrates
their execution using various patterns like map/reduce, chaining, routing, and loops.
"""

import yaml
import os
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Optional, Any, Union
from enum import Enum


class NodeType(Enum):
    """Node types in the pipeline DAG."""
    NORMAL = "normal"
    FANOUT = "fanout"
    AGGREGATE = "aggregate"
    FINAL = "final"
    CONDITIONAL_ROUTING = "conditional_routing"
    FIXED_LOOP = "fixed_loop"
    CONDITIONAL_LOOP = "conditional_loop"


class DataReference:
    """Represents a data reference (input source or output destination) for a pipeline node."""

    def __init__(self, ref_type: str, ref: str, output_name: str = 'default'):
        """
        Initialize a data reference.

        Args:
            ref_type: Type of reference - "node" or "file"
            ref: Reference target - node_id for nodes, file_path for files
            output_name: For node references, which output branch to read from (default: 'default')
        """
        if ref_type not in ("node", "file"):
            raise ValueError(f"ref_type must be 'node' or 'file', got: {ref_type}")

        self.ref_type = ref_type
        self.ref = ref
        self.output_name = output_name
        self.name = self._generate_name()

    def _generate_name(self) -> str:
        """Auto-generate reference name based on type and reference."""
        if self.ref_type == "node":
            return f"{self.ref}_source"
        else:  # file
            return Path(self.ref).name

    def to_dict(self) -> Dict[str, Any]:
        """Serialize DataReference to dictionary."""
        result = {
            "type": self.ref_type,
            "name": self.name,
            "reference": self.ref
        }
        if self.output_name != 'default':
            result["output_name"] = self.output_name
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DataReference':
        """Deserialize DataReference from dictionary."""
        output_name = data.get("output_name", "default")
        return cls(data["type"], data["reference"], output_name)

    def __repr__(self):
        if self.output_name != 'default':
            return f"DataReference(type='{self.ref_type}', name='{self.name}', ref='{self.ref}', output='{self.output_name}')"
        return f"DataReference(type='{self.ref_type}', name='{self.name}', ref='{self.ref}')"


class PipelineNode:
    """Represents a node in the pipeline DAG."""

    def __init__(self, procedure_path: str, node_id: str, node_type: NodeType, data_sources: List[DataReference], outputs: List[DataReference]):
        self.procedure_path = procedure_path
        self.node_id = node_id
        self.node_type = node_type
        self.data_sources = data_sources
        self.outputs = outputs
        self.parents: List[str] = []
        self.children: List[str] = []
        self.metadata: Dict[str, Any] = {}

    def __repr__(self):
        return f"PipelineNode(id={self.node_id}, type={self.node_type.value}, sources={len(self.data_sources)}, outputs={len(self.outputs)})"


class Pipeline:
    """
    Pipeline class for orchestrating multiple Procedures using a DAG structure.

    A Pipeline represents a DAG of Procedure nodes where edges define execution
    dependencies. Supports patterns:
    - Map/Reduce: Parallel fan-out and aggregation
    - Routing: Conditional execution based on rules
    - Chaining: Sequential or transformation-based connections
    - Loops: Iterative execution with conditions
    """

    def __init__(
        self,
        name: str = "",
        input_path: Optional[str] = None,
        output_path: Optional[str] = None,
        properties: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize a Pipeline.

        Args:
            name: Pipeline name
            input_path: Path to input data (may be deprecated, nodes specify sources)
            output_path: Path to output data
            properties: Metadata (query, base_system, default_model, created_at, etc.)
        """
        self.name = name
        self.input_path = input_path
        self.output_path = output_path
        self.properties = properties or {}
        # available_systems will be set by executor based on registered executors
        self.nodes: Dict[str, PipelineNode] = {}
        self.edges: Dict[str, List[str]] = defaultdict(list)

    @staticmethod
    def _resolve_path(path: str, base_dir: Path) -> str:
        """
        Resolve path relative to base_dir if relative, otherwise return as-is.

        Args:
            path: Path to resolve (can be absolute or relative)
            base_dir: Base directory for relative paths

        Returns:
            Absolute path as string
        """
        if not path:
            return path
        path_obj = Path(path)
        if path_obj.is_absolute():
            return str(path_obj)
        else:
            return str((base_dir / path_obj).resolve())

    def add_procedure(
        self,
        procedure_path: str,
        node_id: str,
        node_type: NodeType,
        data_sources: List[DataReference],
        outputs: List[DataReference],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add a procedure node to the pipeline DAG.

        Args:
            procedure_path: Path to procedure file (will be converted to absolute path)
            node_id: Unique node identifier
            node_type: Type of node (normal, fanout, aggregate, final)
            data_sources: List of DataReferences specifying where this node gets its input data
            outputs: List of DataReferences specifying where this node's output goes
            metadata: Optional node metadata

        Returns:
            The node_id of the added node

        Raises:
            ValueError: If node_id already exists
        """
        if node_id in self.nodes:
            raise ValueError(f"Node with id '{node_id}' already exists")

        # Convert to absolute path if not already
        abs_procedure_path = os.path.abspath(procedure_path)

        node = PipelineNode(abs_procedure_path, node_id, node_type, data_sources, outputs)
        if metadata:
            node.metadata = metadata
        self.nodes[node_id] = node
        return node_id

    def add_conditional_routing(
        self,
        procedure_path: str,
        node_id: str,
        data_sources: List[DataReference],
        output_branches: Dict[str, str],
        routing_field: str = '_route_to'
    ) -> str:
        """
        Add a conditional routing node to the pipeline.

        Args:
            procedure_path: Path to procedure file (must produce routing_field)
            node_id: Unique node identifier
            data_sources: List of DataReferences specifying input data
            output_branches: Dict mapping output_name -> target_node_id
            routing_field: Name of field in procedure output that contains routing target

        Returns:
            The node_id of the added node

        Raises:
            ValueError: If node_id already exists or output_branches has less than 2 entries
        """
        if len(output_branches) < 2:
            raise ValueError(f"ConditionalRouting node must have at least 2 output branches, got {len(output_branches)}")

        # Create outputs with output_name set for each branch
        outputs = [DataReference("node", target_node_id, output_name=branch_name)
                   for branch_name, target_node_id in output_branches.items()]

        # Set metadata with routing_field
        metadata = {'routing_field': routing_field}

        # Add node
        self.add_procedure(procedure_path, node_id, NodeType.CONDITIONAL_ROUTING,
                          data_sources, outputs, metadata)

        # Add edges to all target nodes
        for target_node_id in output_branches.values():
            if target_node_id in self.nodes:
                self.add_edge(node_id, target_node_id)

        return node_id

    def add_fixed_loop(
        self,
        procedure_path: str,
        node_id: str,
        data_sources: List[DataReference],
        outputs: List[DataReference],
        iterations: int,
        node_type: NodeType = NodeType.FIXED_LOOP
    ) -> str:
        """
        Add a fixed loop node to the pipeline. Executes the procedure N times,
        feeding the output of iteration i as input to iteration i+1.

        Args:
            procedure_path: Path to procedure file to loop
            node_id: Unique node identifier
            data_sources: List of DataReferences specifying input data
            outputs: List of DataReferences specifying where this node's output goes
            iterations: Number of times to execute the procedure (must be >= 1)
            node_type: Node type - can be FIXED_LOOP or combined with FANOUT/AGGREGATE

        Returns:
            The node_id of the added node

        Raises:
            ValueError: If node_id already exists or iterations < 1
        """
        if iterations < 1:
            raise ValueError(f"Fixed loop must have at least 1 iteration, got {iterations}")

        # Set metadata with iterations
        metadata = {'iterations': iterations}

        # Add node (allows combining FIXED_LOOP with FANOUT/AGGREGATE)
        self.add_procedure(procedure_path, node_id, node_type, data_sources, outputs, metadata)

        # Add edges to target nodes
        for output in outputs:
            if output.ref_type == "node" and output.ref in self.nodes:
                self.add_edge(node_id, output.ref)

        return node_id

    def add_conditional_loop(
        self,
        procedure_path: str,
        condition_procedure_path: str,
        node_id: str,
        data_sources: List[DataReference],
        outputs: List[DataReference],
        max_iterations: int,
        condition_field: str = '_should_continue',
        condition_aggregation: str = 'all',
        node_type: NodeType = NodeType.CONDITIONAL_LOOP
    ) -> str:
        """
        Add a conditional loop node to the pipeline. Executes the procedure repeatedly
        until a condition is met, with a maximum iteration limit.

        Args:
            procedure_path: Path to data processing procedure file
            condition_procedure_path: Path to condition evaluation procedure file
            node_id: Unique node identifier
            data_sources: List of DataReferences specifying input data
            outputs: List of DataReferences specifying where this node's output goes
            max_iterations: Maximum number of iterations (must be >= 1)
            condition_field: Field name to read from condition procedure output (default: '_should_continue')
            condition_aggregation: How to aggregate multiple records ('all', 'any', 'first')
            node_type: Node type - can be CONDITIONAL_LOOP or combined with FANOUT/AGGREGATE

        Returns:
            The node_id of the added node

        Raises:
            ValueError: If node_id already exists, max_iterations < 1, or invalid aggregation
        """
        if max_iterations < 1:
            raise ValueError(f"Conditional loop must have at least 1 max_iteration, got {max_iterations}")

        if condition_aggregation not in ('all', 'any', 'first'):
            raise ValueError(f"Invalid condition_aggregation: {condition_aggregation} (must be 'all', 'any', or 'first')")

        # Convert to absolute path if not already
        abs_condition_procedure_path = os.path.abspath(condition_procedure_path)

        # Set metadata with condition configuration
        metadata = {
            'condition_procedure_path': abs_condition_procedure_path,
            'max_iterations': max_iterations,
            'condition_field': condition_field,
            'condition_aggregation': condition_aggregation
        }

        # Add node
        self.add_procedure(procedure_path, node_id, node_type, data_sources, outputs, metadata)

        # Add edges to target nodes
        for output in outputs:
            if output.ref_type == "node" and output.ref in self.nodes:
                self.add_edge(node_id, output.ref)

        return node_id

    def add_edge(self, from_node_id: str, to_node_id: str):
        """
        Add directed edge between nodes.

        Args:
            from_node_id: Source node ID
            to_node_id: Target node ID

        Raises:
            ValueError: If node doesn't exist
        """
        if from_node_id not in self.nodes:
            raise ValueError(f"Source node '{from_node_id}' not found")
        if to_node_id not in self.nodes:
            raise ValueError(f"Target node '{to_node_id}' not found")

        self.edges[from_node_id].append(to_node_id)
        self.nodes[from_node_id].children.append(to_node_id)
        self.nodes[to_node_id].parents.append(from_node_id)

    def validate(self) -> None:
        """
        Validate pipeline structure and data flow consistency with type-specific rules.

        Checks:
        1. Pipeline must have at least 1 final node
        2. Type-specific validation (fanout, aggregate, final, normal)
        3. For data_sources with type='node': referenced node exists and edge exists
        4. For outputs with type='node': referenced node exists and edge exists

        Raises:
            ValueError: If validation fails with descriptive error message
        """
        errors = []

        # Check 1: Count final nodes
        final_nodes = [node_id for node_id, node in self.nodes.items() if node.node_type == NodeType.FINAL]
        if len(final_nodes) < 1:
            errors.append(f"Pipeline must have at least 1 final node, found: {len(final_nodes)}")

        for node_id, node in self.nodes.items():
            # Type-specific validation
            if node.node_type == NodeType.FANOUT:
                # FANOUT: Must have at least 2 outputs, all must be nodes
                if len(node.outputs) < 2:
                    errors.append(f"Fanout node '{node_id}' must have at least 2 outputs, has {len(node.outputs)}")
                for output in node.outputs:
                    if output.ref_type == "file":
                        errors.append(f"Fanout node '{node_id}' has file output '{output.name}' (only node outputs allowed for non-final nodes)")

            elif node.node_type == NodeType.AGGREGATE:
                # AGGREGATE: Must have at least 2 data_sources, all must be nodes
                if len(node.data_sources) < 2:
                    errors.append(f"Aggregate node '{node_id}' must have at least 2 data sources, has {len(node.data_sources)}")
                for data_source in node.data_sources:
                    if data_source.ref_type == "file":
                        errors.append(f"Aggregate node '{node_id}' has file data_source '{data_source.name}' (aggregates must use node sources)")
                # Aggregate outputs must be nodes
                for output in node.outputs:
                    if output.ref_type == "file":
                        errors.append(f"Aggregate node '{node_id}' has file output '{output.name}' (only node outputs allowed for non-final nodes)")

            elif node.node_type == NodeType.FINAL:
                # FINAL: Must have exactly 1 output, must be file
                if len(node.outputs) != 1:
                    errors.append(f"Final node '{node_id}' must have exactly 1 output, has {len(node.outputs)}")
                elif node.outputs[0].ref_type != "file":
                    errors.append(f"Final node '{node_id}' output must be a file, got node reference '{node.outputs[0].ref}'")

            elif node.node_type == NodeType.CONDITIONAL_ROUTING:
                # CONDITIONAL_ROUTING: Must have at least 2 outputs, all must be nodes, metadata must have routing_field
                if len(node.outputs) < 2:
                    errors.append(f"ConditionalRouting node '{node_id}' must have at least 2 outputs, has {len(node.outputs)}")
                for output in node.outputs:
                    if output.ref_type == "file":
                        errors.append(f"ConditionalRouting node '{node_id}' has file output '{output.name}' (only node outputs allowed for routing nodes)")
                if 'routing_field' not in node.metadata:
                    errors.append(f"ConditionalRouting node '{node_id}' must have 'routing_field' in metadata")

            elif node.node_type == NodeType.FIXED_LOOP:
                # FIXED_LOOP: Must have 'iterations' in metadata
                if 'iterations' not in node.metadata:
                    errors.append(f"FixedLoop node '{node_id}' must have 'iterations' in metadata")
                else:
                    iterations = node.metadata['iterations']
                    if not isinstance(iterations, int) or iterations < 1:
                        errors.append(f"FixedLoop node '{node_id}' iterations must be integer >= 1, got {iterations}")
                # Prohibit mixing with conditional loop
                if 'condition_procedure_path' in node.metadata:
                    errors.append(f"FixedLoop node '{node_id}' cannot have 'condition_procedure_path' (use CONDITIONAL_LOOP instead)")
                # Output validation: if not FINAL, outputs must be nodes
                for output in node.outputs:
                    if output.ref_type == "file":
                        # Check if this is also a FINAL node
                        is_final = any(f_id == node_id for f_id in final_nodes)
                        if not is_final:
                            errors.append(f"FixedLoop node '{node_id}' has file output '{output.name}' (only final loop nodes can output to files)")

            elif node.node_type == NodeType.CONDITIONAL_LOOP:
                # CONDITIONAL_LOOP: Must have condition configuration in metadata
                if 'condition_procedure_path' not in node.metadata:
                    errors.append(f"ConditionalLoop node '{node_id}' must have 'condition_procedure_path' in metadata")
                if 'max_iterations' not in node.metadata:
                    errors.append(f"ConditionalLoop node '{node_id}' must have 'max_iterations' in metadata")
                else:
                    max_iterations = node.metadata['max_iterations']
                    if not isinstance(max_iterations, int) or max_iterations < 1:
                        errors.append(f"ConditionalLoop node '{node_id}' max_iterations must be integer >= 1, got {max_iterations}")
                if 'condition_field' not in node.metadata:
                    errors.append(f"ConditionalLoop node '{node_id}' must have 'condition_field' in metadata")
                if 'condition_aggregation' in node.metadata:
                    aggregation = node.metadata['condition_aggregation']
                    if aggregation not in ('all', 'any', 'first'):
                        errors.append(f"ConditionalLoop node '{node_id}' condition_aggregation must be 'all', 'any', or 'first', got '{aggregation}'")
                # Prohibit mixing with fixed loop
                if 'iterations' in node.metadata:
                    errors.append(f"ConditionalLoop node '{node_id}' cannot have 'iterations' (use FIXED_LOOP instead)")
                # Output validation: if not FINAL, outputs must be nodes
                for output in node.outputs:
                    if output.ref_type == "file":
                        is_final = any(f_id == node_id for f_id in final_nodes)
                        if not is_final:
                            errors.append(f"ConditionalLoop node '{node_id}' has file output '{output.name}' (only final loop nodes can output to files)")

            elif node.node_type == NodeType.NORMAL:
                # NORMAL: All outputs must be nodes (not files)
                for output in node.outputs:
                    if output.ref_type == "file":
                        errors.append(f"Normal node '{node_id}' has file output '{output.name}' (only final nodes can output to files)")

            # Validate data_sources with type='node'
            for data_source in node.data_sources:
                if data_source.ref_type == "node":
                    source_node_id = data_source.ref

                    # Check referenced node exists
                    if source_node_id not in self.nodes:
                        errors.append(
                            f"Node '{node_id}' has data_source referencing non-existent node '{source_node_id}'"
                        )
                    # Check edge exists from source_node to current node
                    elif node_id not in self.edges.get(source_node_id, []):
                        errors.append(
                            f"Node '{node_id}' has data_source from '{source_node_id}' but no edge exists: {source_node_id} -> {node_id}"
                        )

            # Validate outputs with type='node'
            for output in node.outputs:
                if output.ref_type == "node":
                    target_node_id = output.ref

                    # Check referenced node exists
                    if target_node_id not in self.nodes:
                        errors.append(
                            f"Node '{node_id}' has output to non-existent node '{target_node_id}'"
                        )
                    # Check edge exists from current node to target node
                    elif target_node_id not in self.edges.get(node_id, []):
                        errors.append(
                            f"Node '{node_id}' has output to '{target_node_id}' but no edge exists: {node_id} -> {target_node_id}"
                        )

        # System compatibility validation
        warnings = []

        # Collect all required systems
        required_systems = set()
        for node_id, node in self.nodes.items():
            base_system = node.metadata.get('base_system')
            if not base_system:
                errors.append(
                    f"Node '{node_id}' missing 'base_system' in metadata. "
                    f"Procedure must declare base_system."
                )
            else:
                required_systems.add(base_system)

        # Get available systems (must exist, no default)
        available_systems = self.properties.get('available_systems')
        if available_systems is None:
            errors.append(
                "Pipeline properties missing 'available_systems'. "
                "This should be set automatically by AbstractExecutor."
            )
        else:
            available_systems = set(available_systems)

            # Check missing systems
            missing_systems = required_systems - available_systems
            if missing_systems:
                errors.append(
                    f"Pipeline requires unavailable systems: {sorted(missing_systems)}. "
                    f"Available systems: {sorted(available_systems)}"
                )

        # Cross-system flow detection (warnings)
        for from_node_id, to_node_ids in self.edges.items():
            from_system = self.nodes[from_node_id].metadata.get('base_system')
            for to_node_id in to_node_ids:
                to_system = self.nodes[to_node_id].metadata.get('base_system')
                if from_system and to_system and from_system != to_system:
                    warnings.append(
                        f"Cross-system data flow: "
                        f"{from_node_id} ({from_system}) -> {to_node_id} ({to_system}). "
                        f"Ensure data format compatibility."
                    )

        # Print warnings if any
        if warnings:
            print("⚠️  Pipeline validation warnings:")
            for warn in warnings:
                print(f"    {warn}")

        if errors:
            error_msg = "Pipeline validation failed:\n" + "\n".join(f"  - {err}" for err in errors)
            raise ValueError(error_msg)

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize Pipeline DAG to dictionary for YAML export.

        Returns:
            Dictionary with nodes, edges, node types, data sources, and outputs
        """
        return {
            "name": self.name,
            "input_path": self.input_path,
            "output_path": self.output_path,
            "properties": self.properties,
            "nodes": {
                node_id: {
                    "node_type": node.node_type.value,
                    "procedure_path": node.procedure_path,
                    "data_sources": [ds.to_dict() for ds in node.data_sources],
                    "outputs": [output.to_dict() for output in node.outputs],
                    "metadata": node.metadata
                }
                for node_id, node in self.nodes.items()
            },
            "edges": dict(self.edges),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], config_dir: Optional[Path] = None) -> 'Pipeline':
        """
        Deserialize Pipeline DAG from dictionary and validate.

        Args:
            data: Dictionary with pipeline data
            config_dir: Directory containing the pipeline config file (for resolving relative paths)

        Returns:
            Pipeline instance with DAG structure

        Raises:
            ValueError: If validation fails (missing fields, invalid references, etc.)
        """
        # Resolve output_path if provided
        output_path = data.get("output_path")
        if output_path and config_dir:
            output_path = cls._resolve_path(output_path, config_dir)

        pipeline = cls(
            name=data.get("name", ""),
            input_path=data.get("input_path"),
            output_path=output_path,
            properties=data.get("properties", {}),
        )

        # Load nodes
        for node_id, node_data in data.get("nodes", {}).items():
            procedure_path = node_data.get("procedure_path", "")
            # Resolve procedure_path relative to pipeline directory
            if procedure_path and config_dir:
                procedure_path = cls._resolve_path(procedure_path, config_dir)

            metadata = node_data.get("metadata", {})

            # Extract and cache base_system from procedure if not already in metadata
            if 'base_system' not in metadata and procedure_path:
                from .procedure import Procedure
                procedure = Procedure.load(procedure_path)
                metadata = metadata.copy()  # Don't modify original
                metadata['base_system'] = procedure.base_system

            # Resolve condition_procedure_path if present (for conditional loops)
            if 'condition_procedure_path' in metadata and config_dir:
                metadata = metadata.copy()  # Don't modify original
                metadata['condition_procedure_path'] = cls._resolve_path(
                    metadata['condition_procedure_path'],
                    config_dir
                )

            # Load node_type
            node_type_str = node_data.get("node_type")
            if not node_type_str:
                raise ValueError(f"Node '{node_id}' missing required 'node_type' field")
            try:
                node_type = NodeType(node_type_str)
            except ValueError:
                raise ValueError(f"Node '{node_id}' has invalid node_type: '{node_type_str}' (must be: normal, fanout, aggregate, or final)")

            # Load data_sources
            data_sources_list = node_data.get("data_sources")
            if not data_sources_list:
                raise ValueError(f"Node '{node_id}' missing required 'data_sources' field")
            data_sources = []
            for ds_dict in data_sources_list:
                # Resolve file paths relative to pipeline directory
                if ds_dict["type"] == "file" and config_dir:
                    ds_dict = ds_dict.copy()  # Don't modify original
                    ds_dict["reference"] = cls._resolve_path(ds_dict["reference"], config_dir)
                data_sources.append(DataReference.from_dict(ds_dict))

            # Load outputs
            outputs_list = node_data.get("outputs")
            if not outputs_list:
                raise ValueError(f"Node '{node_id}' missing required 'outputs' field")
            outputs = []
            for out_dict in outputs_list:
                # Resolve file paths relative to pipeline directory
                if out_dict["type"] == "file" and config_dir:
                    out_dict = out_dict.copy()  # Don't modify original
                    out_dict["reference"] = cls._resolve_path(out_dict["reference"], config_dir)
                outputs.append(DataReference.from_dict(out_dict))

            pipeline.add_procedure(procedure_path, node_id, node_type, data_sources, outputs, metadata)

        # Load edges
        for from_node_id, to_node_ids in data.get("edges", {}).items():
            for to_node_id in to_node_ids:
                pipeline.add_edge(from_node_id, to_node_id)

        # Validate pipeline structure
        pipeline.validate()

        return pipeline

    def save(self, filepath: Union[str, Path]) -> None:
        """
        Save Pipeline to YAML file.

        Args:
            filepath: Path where to save the pipeline YAML
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w', encoding='utf-8') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'Pipeline':
        """
        Load Pipeline from YAML file.

        Args:
            filepath: Path to pipeline YAML file

        Returns:
            Pipeline instance

        Raises:
            FileNotFoundError: If file doesn't exist
            yaml.YAMLError: If file is not valid YAML
        """
        filepath = Path(filepath).resolve()
        config_dir = filepath.parent

        if not filepath.exists():
            raise FileNotFoundError(f"Pipeline file not found: {filepath}")

        with open(filepath, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        return cls.from_dict(data, config_dir=config_dir)

    def load_procedures(self) -> Dict[str, 'Procedure']:
        """
        Load all Procedure objects from their stored paths.

        Returns:
            Dictionary mapping node_id -> Procedure object

        Raises:
            FileNotFoundError: If any procedure file doesn't exist
        """
        from .procedure import Procedure

        procedures = {}
        for node_id, node in self.nodes.items():
            procedure_path = Path(node.procedure_path)

            if not procedure_path.exists():
                raise FileNotFoundError(
                    f"Procedure file not found for node '{node_id}': {procedure_path}"
                )

            procedures[node_id] = Procedure.load(procedure_path)

        return procedures

    def __repr__(self):
        return f"Pipeline(name='{self.name}', nodes={len(self.nodes)}, edges={sum(len(children) for children in self.edges.values())})"
