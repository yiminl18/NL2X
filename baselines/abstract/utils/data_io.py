"""
Shared data I/O utilities for abstract layer.
"""
from typing import List, Dict, Any, Optional
from pathlib import Path
import json


def get_file_format(file_path: str) -> str:
    """
    Extract file format from file path based on extension.

    Args:
        file_path: Path to the file

    Returns:
        File format (e.g., 'json', 'csv', 'txt', 'parquet', 'xlsx')
        Returns 'unknown' if extension is missing or not recognized
    """
    path = Path(file_path)
    extension = path.suffix.lstrip('.').lower()

    if not extension:
        return 'unknown'

    # Map common extensions to format names
    known_formats = {'json', 'csv', 'txt', 'parquet', 'xlsx', 'xls'}

    if extension in known_formats:
        return extension

    return 'unknown'


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
        self.format = self._detect_format()  # Track original file format

    def _generate_name(self) -> str:
        """Auto-generate reference name based on type and reference."""
        if self.ref_type == "node":
            return f"{self.ref}_source"
        else:  # file
            return Path(self.ref).name

    def _detect_format(self) -> Optional[str]:
        """Detect file format for file references."""
        if self.ref_type == 'file':
            return get_file_format(self.ref)
        return None  # Node references don't have file format

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


def load_from_reference(ref: DataReference, verbose: bool = False) -> Any:
    """
    Load data from a DataReference with format detection.

    Preserves format semantics: even if CSV is converted to List[Dict],
    the original format is tracked in ref.format for validation purposes.

    Args:
        ref: DataReference to load (must be file type)
        verbose: If True, print warnings for non-list data

    Returns:
        Data in appropriate format (List[Dict] for CSV/JSON, or other types for future formats)

    Raises:
        ValueError: If reference is not a file or unsupported format
        FileNotFoundError: If file doesn't exist
    """
    if ref.ref_type != "file":
        raise ValueError(f"Can only load from file references, got: {ref.ref_type}")

    file_path = Path(ref.ref)
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {ref.ref}")

    # Detect file format
    file_format = get_file_format(str(file_path))

    # Load based on format
    if file_format == 'json':
        with open(file_path, 'r') as f:
            data = json.load(f)
    elif file_format == 'csv':
        import pandas as pd
        df = pd.read_csv(file_path)
        data = df.to_dict('records')
    else:
        raise ValueError(f"Unsupported file format: {file_format}. Supported formats: json, csv")

    # Optional warning for non-list data (for backwards compatibility awareness)
    if verbose and not isinstance(data, list):
        print(f"Warning: Loaded data from {ref.ref} is {type(data).__name__}, not list. "
              f"Some operators may require list input.")

    return data


def save_to_reference(data: Any, ref: DataReference) -> None:
    """
    Save data to a DataReference.

    Args:
        data: Data to save (any JSON-serializable type)
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
