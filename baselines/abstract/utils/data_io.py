"""
Shared data I/O utilities for abstract layer.
"""
from typing import List, Dict, Any
from pathlib import Path
import json


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


def load_from_reference(ref: DataReference) -> List[Dict[str, Any]]:
    """
    Load data from a DataReference.

    Args:
        ref: DataReference to load (must be file type)

    Returns:
        List of dictionaries (raw data)

    Raises:
        ValueError: If reference is not a file or file doesn't exist
    """
    if ref.ref_type != "file":
        raise ValueError(f"Can only load from file references, got: {ref.ref_type}")

    file_path = Path(ref.ref)
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {ref.ref}")

    with open(file_path, 'r') as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected list data in {ref.ref}, got {type(data).__name__}")

    return data


def save_to_reference(data: List[Dict[str, Any]], ref: DataReference) -> None:
    """
    Save data to a DataReference.

    Args:
        data: Raw data to save
        ref: DataReference to save to (must be file type)

    Raises:
        ValueError: If reference is not a file
    """
    if ref.ref_type != "file":
        raise ValueError(f"Can only save to file references, got: {ref.ref_type}")

    file_path = Path(ref.ref)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)
