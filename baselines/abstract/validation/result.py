"""
Validation Result Data Classes
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict


@dataclass
class ValidationError:
    """Validation error information."""

    type: str                           # Error type (e.g., 'missing_field', 'type_mismatch')
    operator: Optional[str] = None      # Operator name if applicable
    message: str = ""                   # Human-readable error message
    suggestions: Optional[List[str]] = None  # Suggested fixes


@dataclass
class ValidationWarning:
    """Validation warning information."""

    type: str                           # Warning type
    operator: Optional[str] = None      # Operator name if applicable
    message: str = ""                   # Human-readable warning message


@dataclass
class ValidationResult:
    """Result of static validation."""

    passed: bool                        # Whether validation passed
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[ValidationWarning] = field(default_factory=list)
    schema_history: Optional[List[Dict[str, str]]] = None  # Schema at each step

    @property
    def score(self) -> int:
        """Compatibility property for existing code (1 = passed, 0 = failed)."""
        return 1 if self.passed else 0
