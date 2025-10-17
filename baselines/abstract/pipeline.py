from collections import defaultdict, deque
from typing import List, Dict, Set, Optional, Any, TYPE_CHECKING
from .ops.base import Operator

class PipelineNode:
    """Represents a node in the pipeline DAG."""

    def __init__(self, operator: Operator, node_id: str):
        self.operator = operator
        self.node_id = node_id
        self.parents: List[str] = []  # Parent node IDs (dependencies)
        self.children: List[str] = []  # Child node IDs (dependents)
        self.metadata: Dict[str, Any] = {}

    def __repr__(self):
        return f"PipelineNode(id={self.node_id}, type={self.operator.type})"


class Pipeline:
    """Pipeline class for building and managing a DAG of operators."""

    def __init__(self, name: str = "", input_path: Optional[str] = None,
                 output_path: Optional[str] = None, properties: Optional[Dict[str, Any]] = None,
                 dataset_schema: Optional[Dict[str, Any]] = None):
        self.name = name
        self.input_path = input_path  # Input data path
        self.output_path = output_path  # Output data path
        self.properties = properties or {}  # Other metadata from pipeline config
        self.dataset_schema = dataset_schema  # Dataset schema in abstract layer format
        self.nodes: Dict[str, PipelineNode] = {}
        self.edges: Dict[str, List[str]] = defaultdict(list)  # node_id -> [child_ids]

    @classmethod
    def from_operators(cls, operators: List[Operator], name: str = "",
                      input_path: Optional[str] = None,
                      output_path: Optional[str] = None,
                      properties: Optional[Dict[str, Any]] = None,
                      dataset_schema: Optional[Dict[str, Any]] = None) -> 'Pipeline':
        """Create linear pipeline from list of operators."""
        pipeline = cls(name=name, input_path=input_path,
                      output_path=output_path, properties=properties,
                      dataset_schema=dataset_schema)

        if not operators:
            return pipeline

        # Add all operators as nodes
        for i, op in enumerate(operators):
            node_id = f"op_{i}_{op.name}" if op.name else f"op_{i}"
            pipeline.add_operator(op, node_id)

        # Connect them linearly
        node_ids = list(pipeline.nodes.keys())
        for i in range(len(node_ids) - 1):
            pipeline.add_edge(node_ids[i], node_ids[i + 1])

        return pipeline

    def to_operators(self) -> List[Operator]:
        """Extract operators in topological execution order."""
        execution_order = self.get_execution_order()
        return [self.nodes[node_id].operator for node_id in execution_order]

    def add_operator(self, operator: Operator, node_id: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Add operator to pipeline with unique node_id."""
        if node_id in self.nodes:
            raise ValueError(f"Node with id '{node_id}' already exists in pipeline")

        node = PipelineNode(operator, node_id)
        if metadata:
            node.metadata = metadata

        self.nodes[node_id] = node
        return node_id

    def add_edge(self, from_node_id: str, to_node_id: str):
        """Add directed edge between nodes. Raises ValueError if creates cycle."""
        if from_node_id not in self.nodes:
            raise ValueError(f"Source node '{from_node_id}' not found in pipeline")
        if to_node_id not in self.nodes:
            raise ValueError(f"Target node '{to_node_id}' not found in pipeline")

        self.edges[from_node_id].append(to_node_id)
        self.nodes[from_node_id].children.append(to_node_id)
        self.nodes[to_node_id].parents.append(from_node_id)

        if self.has_cycle():
            self.edges[from_node_id].remove(to_node_id)
            self.nodes[from_node_id].children.remove(to_node_id)
            self.nodes[to_node_id].parents.remove(from_node_id)
            raise ValueError(f"Adding edge from '{from_node_id}' to '{to_node_id}' would create a cycle")

    def has_cycle(self) -> bool:
        """Check if pipeline contains cycle using DFS."""
        visited = set()
        rec_stack = set()

        def dfs(node_id: str) -> bool:
            visited.add(node_id)
            rec_stack.add(node_id)

            for child_id in self.edges.get(node_id, []):
                if child_id not in visited:
                    if dfs(child_id):
                        return True
                elif child_id in rec_stack:
                    return True

            rec_stack.remove(node_id)
            return False

        for node_id in self.nodes:
            if node_id not in visited:
                if dfs(node_id):
                    return True

        return False

    def get_execution_order(self) -> List[str]:
        """Get topological order of node IDs for execution."""
        if self.has_cycle():
            raise ValueError("Cannot get execution order: pipeline contains a cycle")

        # Calculate in-degree for each node
        in_degree = {node_id: len(node.parents) for node_id, node in self.nodes.items()}

        # Queue for nodes with in-degree 0
        queue = deque([node_id for node_id, degree in in_degree.items() if degree == 0])

        execution_order = []

        while queue:
            node_id = queue.popleft()
            execution_order.append(node_id)

            # Reduce in-degree for children
            for child_id in self.edges.get(node_id, []):
                in_degree[child_id] -= 1
                if in_degree[child_id] == 0:
                    queue.append(child_id)

        if len(execution_order) != len(self.nodes):
            raise ValueError("Cannot get execution order: pipeline contains a cycle")

        return execution_order

    def get_node(self, node_id: str) -> Optional[PipelineNode]:
        """Get a node by its ID."""
        return self.nodes.get(node_id)

    def get_roots(self) -> List[str]:
        """Get all root nodes (nodes with no parents)."""
        return [node_id for node_id, node in self.nodes.items() if not node.parents]

    def get_leaves(self) -> List[str]:
        """Get all leaf nodes (nodes with no children)."""
        return [node_id for node_id, node in self.nodes.items() if not node.children]

    def visualize(self) -> str:
        """Create text visualization of pipeline structure."""
        lines = [f"Pipeline: {self.name or '(unnamed)'}"]
        lines.append(f"Nodes: {len(self.nodes)}")
        lines.append(f"Edges: {sum(len(children) for children in self.edges.values())}")
        lines.append("")

        execution_order = self.get_execution_order()

        lines.append("Execution Order:")
        for i, node_id in enumerate(execution_order, 1):
            node = self.nodes[node_id]
            lines.append(f"  {i}. {node_id} ({node.operator.type})")

        lines.append("")
        lines.append("Dependencies:")
        for node_id, children in self.edges.items():
            if children:
                for child_id in children:
                    lines.append(f"  {node_id} -> {child_id}")

        return "\n".join(lines)

    def __repr__(self):
        return f"Pipeline(name='{self.name}', nodes={len(self.nodes)}, edges={sum(len(c) for c in self.edges.values())})"


def optimize_pipeline(pipeline: Pipeline) -> List[Pipeline]:
    """Optimize pipeline and return variants."""
    from .optimizer import PipelineOptimizer

    optimizer = PipelineOptimizer(pipeline)
    if optimizer.should_optimize():
        return optimizer.optimize()
    else:
        return [pipeline]