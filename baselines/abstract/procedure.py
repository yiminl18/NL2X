from typing import List, Dict, Optional, Any
from .ops.base import Operator


class Procedure:
    """Linear sequence of operators for data transformation."""

    def __init__(self, name: str = "", input_path: Optional[str] = None,
                 output_path: Optional[str] = None, properties: Optional[Dict[str, Any]] = None,
                 dataset_schema: Optional[Dict[str, Any]] = None,
                 subtasks: Optional[List[str]] = None):
        self.name = name
        self.input_path = input_path
        self.output_path = output_path
        self.properties = properties or {}
        self.dataset_schema = dataset_schema
        self.subtasks = subtasks
        self.operators: List[Operator] = []

    @classmethod
    def from_operators(cls, operators: List[Operator], name: str = "",
                      input_path: Optional[str] = None,
                      output_path: Optional[str] = None,
                      properties: Optional[Dict[str, Any]] = None,
                      dataset_schema: Optional[Dict[str, Any]] = None) -> 'Procedure':
        """Create procedure from list of operators."""
        procedure = cls(name=name, input_path=input_path,
                      output_path=output_path, properties=properties,
                      dataset_schema=dataset_schema)
        procedure.operators = list(operators)
        return procedure

    def to_operators(self) -> List[Operator]:
        """Return operators in execution order."""
        return list(self.operators)

    def insert_operator(self, operator: Operator, position: int) -> None:
        """
        Insert an operator at a specific position.

        Args:
            operator: Operator object to insert
            position: Index where to insert (supports negative indices)
        """
        if position < 0:
            position = max(0, len(self.operators) + 1 + position)
        position = min(position, len(self.operators))
        self.operators.insert(position, operator)

    def remove_operator_at(self, position: int) -> Operator:
        """
        Remove an operator at a specific position.

        Args:
            position: Index in execution order (supports negative indices)

        Returns:
            The removed Operator object
        """
        if not self.operators:
            raise IndexError("Cannot remove from empty procedure")

        if self.subtasks and 0 <= position < len(self.subtasks):
            self.subtasks.pop(position)

        return self.operators.pop(position)

    def visualize(self) -> str:
        """Create text visualization of procedure structure."""
        lines = [f"Procedure: {self.name or '(unnamed)'}"]
        lines.append(f"Operators: {len(self.operators)}")
        lines.append("")
        lines.append("Execution Order:")
        for i, op in enumerate(self.operators, 1):
            lines.append(f"  {i}. {op.name or f'op_{i}'} ({op.type})")
        return "\n".join(lines)

    def __repr__(self):
        return f"Procedure(name='{self.name}', operators={len(self.operators)})"


def optimize_procedure(procedure: Procedure) -> List[Procedure]:
    """Optimize procedure and return variants."""
    from .optimizer import ProcedureOptimizer

    optimizer = ProcedureOptimizer(procedure)
    if optimizer.should_optimize():
        return optimizer.optimize()
    else:
        return [procedure]
