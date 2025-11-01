"""User context and constraints management for interactive pipeline generation."""

from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class UserContext:
    """Manages user-provided context and constraints during pipeline generation."""

    operator_specific: Dict[int, List[str]] = field(default_factory=dict)

    def add_for_operator(self, position: int, constraint: str):
        """Add constraint for specific operator at position."""
        if not constraint:
            return
        if position not in self.operator_specific:
            self.operator_specific[position] = []
        if constraint not in self.operator_specific[position]:
            self.operator_specific[position].append(constraint)

    def get_for_operator(self, position: int) -> List[str]:
        """Get all constraints applicable to operator at position."""
        constraints = self.global_constraints.copy()
        constraints.extend(self.operator_specific.get(position, []))
        return constraints

    def format_for_prompt(self, position: int) -> str:
        """Format constraints as prompt section for given position."""
        constraints = self.get_for_operator(position)
        if not constraints:
            return ""

        section = "\n\nUSER CONSTRAINTS:\n"
        for i, constraint in enumerate(constraints, 1):
            section += f"{i}. {constraint}\n"
        section += "\nPlease ensure the generated configuration adheres to these constraints.\n"
        return section

    def has_constraints(self, position: int) -> bool:
        """Check if there are any constraints for given position."""
        return len(self.get_for_operator(position)) > 0
