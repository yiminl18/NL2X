from collections import defaultdict, deque
from typing import List, Dict, Optional, Any
from .ops.base import Operator

class ProcedureNode:
    """Represents a node in the procedure DAG."""

    def __init__(self, operator: Operator, node_id: str):
        self.operator = operator
        self.node_id = node_id
        self.parents: List[str] = []  # Parent node IDs (dependencies)
        self.children: List[str] = []  # Child node IDs (dependents)
        self.metadata: Dict[str, Any] = {}

    def __repr__(self):
        return f"ProcedureNode(id={self.node_id}, type={self.operator.type})"


class Procedure:
    """Procedure class for building and managing a DAG of operators."""

    def __init__(self, name: str = "", input_path: Optional[str] = None,
                 output_path: Optional[str] = None, properties: Optional[Dict[str, Any]] = None,
                 dataset_schema: Optional[Dict[str, Any]] = None,
                 subtasks: Optional[List[str]] = None):
        self.name = name
        self.input_path = input_path  # Input data path
        self.output_path = output_path  # Output data path
        self.properties = properties or {}  # Other metadata from pipeline config
        self.dataset_schema = dataset_schema  # Dataset schema in abstract layer format
        self.subtasks = subtasks  # Subtasks descriptions for each operator
        self.nodes: Dict[str, ProcedureNode] = {}
        self.edges: Dict[str, List[str]] = defaultdict(list)  # node_id -> [child_ids]

    @classmethod
    def from_operators(cls, operators: List[Operator], name: str = "",
                      input_path: Optional[str] = None,
                      output_path: Optional[str] = None,
                      properties: Optional[Dict[str, Any]] = None,
                      dataset_schema: Optional[Dict[str, Any]] = None) -> 'Procedure':
        """Create linear procedure from list of operators."""
        procedure = cls(name=name, input_path=input_path,
                      output_path=output_path, properties=properties,
                      dataset_schema=dataset_schema)

        if not operators:
            return procedure

        # Add all operators as nodes
        for i, op in enumerate(operators):
            node_id = f"op_{i}_{op.name}" if op.name else f"op_{i}"
            procedure.add_operator(op, node_id)

        # Connect them linearly
        node_ids = list(procedure.nodes.keys())
        for i in range(len(node_ids) - 1):
            procedure.add_edge(node_ids[i], node_ids[i + 1])

        return procedure

    def to_operators(self) -> List[Operator]:
        """Extract operators in topological execution order."""
        execution_order = self.get_execution_order()
        return [self.nodes[node_id].operator for node_id in execution_order]

    def add_operator(self, operator: Operator, node_id: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Add operator to procedure with unique node_id."""
        if node_id in self.nodes:
            raise ValueError(f"Node with id '{node_id}' already exists in procedure")

        node = ProcedureNode(operator, node_id)
        if metadata:
            node.metadata = metadata

        self.nodes[node_id] = node
        return node_id

    def add_edge(self, from_node_id: str, to_node_id: str):
        """Add directed edge between nodes. Raises ValueError if creates cycle."""
        if from_node_id not in self.nodes:
            raise ValueError(f"Source node '{from_node_id}' not found in procedure")
        if to_node_id not in self.nodes:
            raise ValueError(f"Target node '{to_node_id}' not found in procedure")

        self.edges[from_node_id].append(to_node_id)
        self.nodes[from_node_id].children.append(to_node_id)
        self.nodes[to_node_id].parents.append(from_node_id)

        if self.has_cycle():
            self.edges[from_node_id].remove(to_node_id)
            self.nodes[from_node_id].children.remove(to_node_id)
            self.nodes[to_node_id].parents.remove(from_node_id)
            raise ValueError(f"Adding edge from '{from_node_id}' to '{to_node_id}' would create a cycle")

    def has_cycle(self) -> bool:
        """Check if procedure contains cycle using DFS."""
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
            raise ValueError("Cannot get execution order: procedure contains a cycle")

        in_degree = {node_id: len(node.parents) for node_id, node in self.nodes.items()}

        queue = deque([node_id for node_id, degree in in_degree.items() if degree == 0])

        execution_order = []

        while queue:
            node_id = queue.popleft()
            execution_order.append(node_id)

            for child_id in self.edges.get(node_id, []):
                in_degree[child_id] -= 1
                if in_degree[child_id] == 0:
                    queue.append(child_id)

        if len(execution_order) != len(self.nodes):
            raise ValueError("Cannot get execution order: procedure contains a cycle")

        return execution_order

    def get_node(self, node_id: str) -> Optional[ProcedureNode]:
        """Get a node by its ID."""
        return self.nodes.get(node_id)

    def get_roots(self) -> List[str]:
        """Get all root nodes (nodes with no parents)."""
        return [node_id for node_id, node in self.nodes.items() if not node.parents]

    def get_leaves(self) -> List[str]:
        """Get all leaf nodes (nodes with no children)."""
        return [node_id for node_id, node in self.nodes.items() if not node.children]

    def _reconnect_edges(self, removed_node_id: str):
        """
        Reconnects edges after removing a node by connecting its parents to its children.

        Args:
            removed_node_id: ID of the node being removed
        """
        node = self.nodes[removed_node_id]
        parents = node.parents.copy()
        children = node.children.copy()

        for parent_id in parents:
            self.edges[parent_id].remove(removed_node_id)
            self.nodes[parent_id].children.remove(removed_node_id)

        for child_id in children:
            self.nodes[child_id].parents.remove(removed_node_id)

        if removed_node_id in self.edges:
            del self.edges[removed_node_id]

        for parent_id in parents:
            for child_id in children:
                if child_id not in self.edges[parent_id]:
                    self.edges[parent_id].append(child_id)
                    self.nodes[parent_id].children.append(child_id)
                if parent_id not in self.nodes[child_id].parents:
                    self.nodes[child_id].parents.append(parent_id)

    def insert_operator(self, operator: Operator, position: int,
                       node_id: Optional[str] = None,
                       metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Insert an operator at a specific position in the execution order.

        Args:
            operator: Operator object to insert
            position: Index in execution order where to insert (supports negative indices)
            node_id: Optional custom node ID (auto-generated if None)
            metadata: Optional node metadata

        Returns:
            The node_id of the inserted operator

        Example:
            >>> procedure.insert_operator(filter_op, position=0)  # Insert at beginning
            >>> procedure.insert_operator(map_op, position=-1)   # Insert before last
        """
        execution_order = self.get_execution_order() if self.nodes else []
        n = len(execution_order)

        if position < 0:
            position = max(0, n + 1 + position)
        position = min(position, n)

        if node_id is None:
            base_name = operator.name if operator.name else "op"
            counter = 0
            while True:
                node_id = f"op_{position}_{base_name}_{counter}" if counter > 0 else f"op_{position}_{base_name}"
                if node_id not in self.nodes:
                    break
                counter += 1

        self.add_operator(operator, node_id, metadata)

        if n == 0:
            return node_id

        if position > 0:
            predecessor_id = execution_order[position - 1]

            if position < n:
                successor_id = execution_order[position]

                if successor_id in self.edges.get(predecessor_id, []):
                    self.edges[predecessor_id].remove(successor_id)
                    self.nodes[predecessor_id].children.remove(successor_id)
                    self.nodes[successor_id].parents.remove(predecessor_id)

            self.add_edge(predecessor_id, node_id)

        if position < n:
            successor_id = execution_order[position]
            self.add_edge(node_id, successor_id)

        return node_id

    def remove_operator(self, node_id: str) -> Operator:
        """
        Remove an operator from the procedure by node_id.

        Automatically reconnects edges: connects all parents to all children.
        Updates subtasks list if it exists.

        Args:
            node_id: ID of the node to remove

        Returns:
            The removed Operator object

        Raises:
            ValueError: If node_id not found

        Example:
            >>> removed_op = procedure.remove_operator("op_2_extract_age")
        """
        if node_id not in self.nodes:
            raise ValueError(f"Node '{node_id}' not found in procedure")

        node = self.nodes[node_id]
        operator = node.operator

        if self.subtasks:
            execution_order = self.get_execution_order()
            if node_id in execution_order:
                position = execution_order.index(node_id)
                if position < len(self.subtasks):
                    self.subtasks.pop(position)

        self._reconnect_edges(node_id)
        del self.nodes[node_id]

        return operator

    def remove_operator_at(self, position: int) -> Operator:
        """
        Remove an operator at a specific position in the execution order.

        Args:
            position: Index in execution order (supports negative indices)

        Returns:
            The removed Operator object

        Raises:
            IndexError: If position is out of range

        Example:
            >>> removed_op = procedure.remove_operator_at(1)  # Remove second operator
            >>> removed_op = procedure.remove_operator_at(-1) # Remove last operator
        """
        execution_order = self.get_execution_order()

        if not execution_order:
            raise IndexError("Cannot remove from empty procedure")

        try:
            node_id = execution_order[position]
        except IndexError:
            raise IndexError(f"Position {position} out of range for procedure with {len(execution_order)} operators")

        return self.remove_operator(node_id)

    def visualize(self) -> str:
        """Create text visualization of procedure structure."""
        lines = [f"Procedure: {self.name or '(unnamed)'}"]
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
        return f"Procedure(name='{self.name}', nodes={len(self.nodes)}, edges={sum(len(c) for c in self.edges.values())})"


def optimize_procedure(procedure: Procedure) -> List[Procedure]:
    """Optimize procedure and return variants."""
    from .optimizer import ProcedureOptimizer

    optimizer = ProcedureOptimizer(procedure)
    if optimizer.should_optimize():
        return optimizer.optimize()
    else:
        return [procedure]