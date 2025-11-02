from typing import List, Dict, Optional, Any, Union
from pathlib import Path
import json

from .ops.base import Operator
from .pipeline import DataReference


class Procedure:
    """Linear sequence of operators for data transformation."""

    def __init__(self, name: str = "",
                 data_sources: Optional[List[DataReference]] = None,
                 outputs: Optional[List[DataReference]] = None,
                 properties: Optional[Dict[str, Any]] = None,
                 dataset_schema: Optional[Dict[str, Any]] = None,
                 subtasks: Optional[List[str]] = None):
        self.name = name
        self.data_sources = data_sources or []
        self.outputs = outputs or []
        self.properties = properties or {}
        self.dataset_schema = dataset_schema
        self.subtasks = subtasks
        self.operators: List[Operator] = []

    @classmethod
    def from_operators(cls, operators: List[Operator], name: str = "",
                      data_sources: Optional[List[DataReference]] = None,
                      outputs: Optional[List[DataReference]] = None,
                      properties: Optional[Dict[str, Any]] = None,
                      dataset_schema: Optional[Dict[str, Any]] = None) -> 'Procedure':
        """Create procedure from list of operators."""
        procedure = cls(name=name, data_sources=data_sources,
                      outputs=outputs, properties=properties,
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

    def validate(self) -> None:
        """
        Validate procedure structure.

        Raises:
            ValueError: If validation fails
        """
        if not self.data_sources:
            raise ValueError("Procedure must have at least one data source")
        if not self.outputs:
            raise ValueError("Procedure must have at least one output")

        # Validate all data sources are file references
        for ds in self.data_sources:
            if ds.ref_type != "file":
                raise ValueError(f"Procedure data sources must be files, got: {ds.ref_type}")

        # Validate all outputs are file references
        for out in self.outputs:
            if out.ref_type != "file":
                raise ValueError(f"Procedure outputs must be files, got: {out.ref_type}")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize Procedure to dictionary."""
        from .convert.docetl.converter import operator_to_dict

        return {
            "name": self.name,
            "data_sources": [ds.to_dict() for ds in self.data_sources],
            "outputs": [out.to_dict() for out in self.outputs],
            "properties": self.properties,
            "dataset_schema": self.dataset_schema,
            "subtasks": self.subtasks,
            "operators": [operator_to_dict(op) for op in self.operators]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Procedure':
        """Deserialize Procedure from dictionary."""
        from .convert.docetl.converter import dict_to_operator

        # Load data_sources
        data_sources = None
        if "data_sources" in data:
            data_sources = [DataReference.from_dict(ds) for ds in data["data_sources"]]

        # Load outputs
        outputs = None
        if "outputs" in data:
            outputs = [DataReference.from_dict(out) for out in data["outputs"]]

        procedure = cls(
            name=data.get("name", ""),
            data_sources=data_sources,
            outputs=outputs,
            properties=data.get("properties", {}),
            dataset_schema=data.get("dataset_schema"),
            subtasks=data.get("subtasks")
        )

        # Reconstruct operators
        operators_data = data.get("operators", [])
        procedure.operators = [dict_to_operator(op_dict) for op_dict in operators_data]

        return procedure

    def save(self, filepath: Union[str, Path]) -> None:
        """Save Procedure to JSON file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'Procedure':
        """Load Procedure from JSON file."""
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Procedure file not found: {filepath}")

        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return cls.from_dict(data)

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
