"""
Pipeline orchestration layer for multi-procedure workflows.

A Pipeline contains references to one or more Procedure files and orchestrates
their execution using various patterns like map/reduce, chaining, routing, and loops.
"""

import yaml
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Optional, Any, Union


class PipelineNode:
    """Represents a node in the pipeline DAG."""

    def __init__(self, procedure_path: str, node_id: str):
        self.procedure_path = procedure_path
        self.node_id = node_id
        self.parents: List[str] = []
        self.children: List[str] = []
        self.metadata: Dict[str, Any] = {}

    def __repr__(self):
        return f"PipelineNode(id={self.node_id}, path={self.procedure_path})"


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
        dataset_schema: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize a Pipeline.

        Args:
            name: Pipeline name
            input_path: Path to input data
            output_path: Path to output data
            properties: Metadata (query, base_system, default_model, created_at, etc.)
            dataset_schema: Input dataset schema in abstract format
        """
        self.name = name
        self.input_path = input_path
        self.output_path = output_path
        self.properties = properties or {}
        self.dataset_schema = dataset_schema or {}
        self.nodes: Dict[str, PipelineNode] = {}
        self.edges: Dict[str, List[str]] = defaultdict(list)

    def add_procedure(
        self,
        procedure_path: str,
        node_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add a procedure node to the pipeline DAG.

        Args:
            procedure_path: Absolute path to procedure file
            node_id: Unique node identifier
            metadata: Optional node metadata

        Returns:
            The node_id of the added node

        Raises:
            ValueError: If node_id already exists
        """
        if node_id in self.nodes:
            raise ValueError(f"Node with id '{node_id}' already exists")

        node = PipelineNode(procedure_path, node_id)
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

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize Pipeline DAG to dictionary for YAML export.

        Returns:
            Dictionary with nodes, edges
        """
        return {
            "name": self.name,
            "input_path": self.input_path,
            "output_path": self.output_path,
            "dataset_schema": self.dataset_schema,
            "properties": self.properties,
            "nodes": {
                node_id: {
                    "procedure_path": node.procedure_path,
                    "metadata": node.metadata
                }
                for node_id, node in self.nodes.items()
            },
            "edges": dict(self.edges),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Pipeline':
        """
        Deserialize Pipeline DAG from dictionary.

        Args:
            data: Dictionary with pipeline data

        Returns:
            Pipeline instance with DAG structure
        """
        pipeline = cls(
            name=data.get("name", ""),
            input_path=data.get("input_path"),
            output_path=data.get("output_path"),
            properties=data.get("properties", {}),
            dataset_schema=data.get("dataset_schema", {}),
        )

        # Load nodes
        for node_id, node_data in data.get("nodes", {}).items():
            procedure_path = node_data.get("procedure_path", "")
            metadata = node_data.get("metadata", {})
            pipeline.add_procedure(procedure_path, node_id, metadata)

        # Load edges
        for from_node_id, to_node_ids in data.get("edges", {}).items():
            for to_node_id in to_node_ids:
                pipeline.add_edge(from_node_id, to_node_id)

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

    def __repr__(self):
        return f"Pipeline(name='{self.name}', nodes={len(self.nodes)}, edges={sum(len(children) for children in self.edges.values())})"
