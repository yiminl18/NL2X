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


class DataReference:
    """Represents a data reference (input source or output destination) for a pipeline node."""

    def __init__(self, ref_type: str, ref: str):
        """
        Initialize a data reference.

        Args:
            ref_type: Type of reference - "node" or "file"
            ref: Reference target - node_id for nodes, file_path for files
        """
        if ref_type not in ("node", "file"):
            raise ValueError(f"ref_type must be 'node' or 'file', got: {ref_type}")

        self.ref_type = ref_type
        self.ref = ref
        self.name = self._generate_name()

    def _generate_name(self) -> str:
        """Auto-generate reference name based on type and reference."""
        if self.ref_type == "node":
            return f"{self.ref}_source"
        else:  # file
            return Path(self.ref).name

    def to_dict(self) -> Dict[str, Any]:
        """Serialize DataReference to dictionary."""
        return {
            "type": self.ref_type,
            "name": self.name,
            "reference": self.ref
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DataReference':
        """Deserialize DataReference from dictionary."""
        return cls(data["type"], data["reference"])

    def __repr__(self):
        return f"DataReference(type='{self.ref_type}', name='{self.name}', ref='{self.ref}')"


class PipelineNode:
    """Represents a node in the pipeline DAG."""

    def __init__(self, procedure_path: str, node_id: str, data_source: DataReference, outputs: List[DataReference]):
        self.procedure_path = procedure_path
        self.node_id = node_id
        self.data_source = data_source
        self.outputs = outputs
        self.parents: List[str] = []
        self.children: List[str] = []
        self.metadata: Dict[str, Any] = {}

    def __repr__(self):
        return f"PipelineNode(id={self.node_id}, path={self.procedure_path}, source={self.data_source.name}, outputs={len(self.outputs)})"


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
        self.nodes: Dict[str, PipelineNode] = {}
        self.edges: Dict[str, List[str]] = defaultdict(list)

    def add_procedure(
        self,
        procedure_path: str,
        node_id: str,
        data_source: DataReference,
        outputs: List[DataReference],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add a procedure node to the pipeline DAG.

        Args:
            procedure_path: Path to procedure file (will be converted to absolute path)
            node_id: Unique node identifier
            data_source: DataReference specifying where this node gets its input data
            outputs: List of DataReferences specifying where this node's output goes (required, must have at least one)
            metadata: Optional node metadata

        Returns:
            The node_id of the added node

        Raises:
            ValueError: If node_id already exists or outputs is empty
        """
        if node_id in self.nodes:
            raise ValueError(f"Node with id '{node_id}' already exists")

        if not outputs:
            raise ValueError(f"Node '{node_id}' must have at least one output")

        # Convert to absolute path if not already
        abs_procedure_path = os.path.abspath(procedure_path)

        node = PipelineNode(abs_procedure_path, node_id, data_source, outputs)
        if metadata:
            node.metadata = metadata
        self.nodes[node_id] = node
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
        Validate pipeline structure and data flow consistency.

        Checks:
        1. All nodes have at least one output
        2. For data_source with type='node': referenced node exists and edge exists
        3. For outputs with type='node': referenced node exists and edge exists

        Raises:
            ValueError: If validation fails with descriptive error message
        """
        errors = []

        for node_id, node in self.nodes.items():
            # Check 1: All nodes must have at least one output
            if not node.outputs:
                errors.append(f"Node '{node_id}' has no outputs (at least one required)")

            # Check 2: Validate data_source with type='node'
            if node.data_source.ref_type == "node":
                source_node_id = node.data_source.ref

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

            # Check 3: Validate outputs with type='node'
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

        if errors:
            error_msg = "Pipeline validation failed:\n" + "\n".join(f"  - {err}" for err in errors)
            raise ValueError(error_msg)

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize Pipeline DAG to dictionary for YAML export.

        Returns:
            Dictionary with nodes, edges, data sources, and outputs
        """
        return {
            "name": self.name,
            "input_path": self.input_path,
            "output_path": self.output_path,
            "properties": self.properties,
            "nodes": {
                node_id: {
                    "procedure_path": node.procedure_path,
                    "data_source": node.data_source.to_dict(),
                    "outputs": [output.to_dict() for output in node.outputs],
                    "metadata": node.metadata
                }
                for node_id, node in self.nodes.items()
            },
            "edges": dict(self.edges),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Pipeline':
        """
        Deserialize Pipeline DAG from dictionary and validate.

        Args:
            data: Dictionary with pipeline data

        Returns:
            Pipeline instance with DAG structure

        Raises:
            ValueError: If validation fails (missing fields, invalid references, etc.)
        """
        pipeline = cls(
            name=data.get("name", ""),
            input_path=data.get("input_path"),
            output_path=data.get("output_path"),
            properties=data.get("properties", {}),
        )

        # Load nodes
        for node_id, node_data in data.get("nodes", {}).items():
            procedure_path = node_data.get("procedure_path", "")
            metadata = node_data.get("metadata", {})

            # Load data source
            data_source_dict = node_data.get("data_source")
            if not data_source_dict:
                raise ValueError(f"Node '{node_id}' missing required 'data_source' field")
            data_source = DataReference.from_dict(data_source_dict)

            # Load outputs
            outputs_list = node_data.get("outputs")
            if not outputs_list:
                raise ValueError(f"Node '{node_id}' missing required 'outputs' field")
            outputs = [DataReference.from_dict(out_dict) for out_dict in outputs_list]

            pipeline.add_procedure(procedure_path, node_id, data_source, outputs, metadata)

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
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Pipeline file not found: {filepath}")

        with open(filepath, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        return cls.from_dict(data)

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
