from typing import List, Dict, Optional, Any, Union
from pathlib import Path
import yaml

from .ops.base import Operator
from .utils.data_io import DataReference


class Procedure:
    """Linear sequence of operators for data transformation."""

    def __init__(self, name: str = "",
                 data_sources: Optional[List[DataReference]] = None,
                 outputs: Optional[List[DataReference]] = None,
                 properties: Optional[Dict[str, Any]] = None,
                 dataset_schema: Optional[Dict[str, Any]] = None,
                 subtasks: Optional[List[str]] = None,
                 base_system: str = 'docetl'):
        self.name = name
        self.data_sources = data_sources or []
        self.outputs = outputs or []
        self.properties = properties or {}
        self.dataset_schema = dataset_schema
        self.subtasks = subtasks
        self.base_system = base_system
        self.operators: List[Operator] = []

    @classmethod
    def from_operators(cls, operators: List[Operator], name: str = "",
                      data_sources: Optional[List[DataReference]] = None,
                      outputs: Optional[List[DataReference]] = None,
                      properties: Optional[Dict[str, Any]] = None,
                      dataset_schema: Optional[Dict[str, Any]] = None,
                      base_system: Optional[str] = None) -> 'Procedure':
        """Create procedure from list of operators.

        Args:
            operators: List of operators to include in procedure
            base_system: Base system for execution (e.g., 'docetl', 'lotus').
                        If None, inferred from first operator.

        Raises:
            ValueError: If operators use different systems (mixed-system procedures not allowed)
        """
        # Infer base_system from operators if not specified
        if base_system is None:
            if operators:
                base_system = operators[0].source.get('system', 'docetl')
            else:
                base_system = 'docetl'

        # Validate all operators use the same system
        if operators:
            mismatched = []
            for i, op in enumerate(operators):
                op_system = op.source.get('system', 'docetl')
                if op_system != base_system:
                    mismatched.append(f"  Operator {i} ({op.name}): expected '{base_system}', got '{op_system}'")

            if mismatched:
                raise ValueError(
                    f"Mixed-system procedures are not allowed. All operators must use system '{base_system}'.\n"
                    + "\n".join(mismatched)
                )

        procedure = cls(name=name, data_sources=data_sources,
                      outputs=outputs, properties=properties,
                      dataset_schema=dataset_schema, base_system=base_system)
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
        Validate procedure structure and system consistency.

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

        # Validate all operators use the same system as procedure.base_system
        if self.operators:
            mismatched = []
            for i, op in enumerate(self.operators):
                op_system = op.source.get('system', 'docetl')
                if op_system != self.base_system:
                    mismatched.append(
                        f"  Operator {i} ({op.name or op.type}): "
                        f"expected system '{self.base_system}', got '{op_system}'"
                    )

            if mismatched:
                raise ValueError(
                    f"Procedure system validation failed. All operators must use system '{self.base_system}'.\n"
                    + "\n".join(mismatched)
                )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize Procedure to dictionary."""
        from .convert.docetl.converter import operator_to_dict

        return {
            "name": self.name,
            "base_system": self.base_system,
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

        # Load base_system (REQUIRED, no default)
        base_system = data.get("base_system")
        if not base_system:
            raise ValueError(
                f"Procedure '{data.get('name', '<unnamed>')}' missing required field 'base_system'. "
                f"All procedures must declare their base system."
            )

        procedure = cls(
            name=data.get("name", ""),
            data_sources=data_sources,
            outputs=outputs,
            properties=data.get("properties", {}),
            dataset_schema=data.get("dataset_schema"),
            subtasks=data.get("subtasks"),
            base_system=base_system
        )

        # Reconstruct operators
        operators_data = data.get("operators", [])
        procedure.operators = [dict_to_operator(op_dict) for op_dict in operators_data]

        return procedure

    def save(self, filepath: Union[str, Path]) -> None:
        """Save Procedure to YAML file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w', encoding='utf-8') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'Procedure':
        """Load Procedure from YAML file."""
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Procedure file not found: {filepath}")

        with open(filepath, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        return cls.from_dict(data)

    def __repr__(self):
        return f"Procedure(name='{self.name}', base_system='{self.base_system}', operators={len(self.operators)})"


def optimize_procedure(procedure: Procedure) -> List[Procedure]:
    """Optimize procedure and return variants."""
    from .optimizer import ProcedureOptimizer

    optimizer = ProcedureOptimizer(procedure)
    if optimizer.should_optimize():
        return optimizer.optimize()
    else:
        return [procedure]
