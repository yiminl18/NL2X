"""
Data Management Module for Abstract Pipeline Layer

This module provides data management capabilities for the abstract layer to:
1. Read datasets before underlying system execution
2. Apply transformations (sampling, filtering, etc.)
3. Store processed datasets in temporary locations
4. Manage dataset path and output path modifications for underlying systems
5. Track multiple data sources with lineage (tree-based tracking)

The DatasetManager class acts as a proxy between the abstract layer and
underlying execution systems, allowing pre-processing and path management.
"""

from typing import Any, Dict, List, Optional, Union, Literal, Set
from pathlib import Path
import json
import tempfile
import os
import random
import shutil
from datetime import datetime
import uuid


class SamplingConfig:
    """Configuration for dataset sampling operations."""

    def __init__(
        self,
        mode: Literal["fixed", "ratio"] = "ratio",
        size: Optional[int] = None,
        ratio: Optional[float] = None,
        random_seed: Optional[int] = None
    ):
        """
        Initialize sampling configuration.

        Args:
            mode: Sampling mode - "fixed" for fixed-size or "ratio" for percentage-based
            size: Number of items to sample (required for mode="fixed")
            ratio: Ratio of items to sample, 0.0 to 1.0 (required for mode="ratio")
            random_seed: Random seed for reproducibility (optional)

        Raises:
            ValueError: If configuration is invalid
        """
        self.mode = mode
        self.size = size
        self.ratio = ratio
        self.random_seed = random_seed

        # Validate configuration
        if mode == "fixed" and size is None:
            raise ValueError("size must be specified for fixed sampling mode")
        if mode == "ratio" and ratio is None:
            raise ValueError("ratio must be specified for ratio sampling mode")
        if mode == "ratio" and (ratio <= 0.0 or ratio > 1.0):
            raise ValueError("ratio must be between 0.0 and 1.0")
        if mode == "fixed" and size is not None and size <= 0:
            raise ValueError("size must be positive")


class DataSource:
    """
    Represents a data source node in the lineage tree.

    Tracks:
    - Source identity and location
    - Parent-child relationships (transformations)
    - Metadata (creation time, size, transformation type)

    This enables:
    - Querying data lineage and history
    - Selecting specific sources for pipeline input
    - Tracking transformations across multiple pipelines
    """

    def __init__(
        self,
        path: Union[str, Path],
        source_id: Optional[str] = None,
        parent_id: Optional[str] = None,
        transformation: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a DataSource node.

        Args:
            path: Path to the data file
            source_id: Unique identifier (auto-generated if None)
            parent_id: ID of parent source (None for root sources)
            transformation: Type of transformation applied (e.g., "sample", "filter")
            metadata: Additional metadata about the source
        """
        self.id = source_id or str(uuid.uuid4())
        self.path = str(Path(path).resolve())
        self.parent_id = parent_id
        self.children_ids: Set[str] = set()
        self.transformation = transformation
        self.metadata = metadata or {}
        self.created_at = datetime.now().isoformat()

        # Auto-populate metadata
        if os.path.exists(self.path):
            self.metadata['file_size'] = os.path.getsize(self.path)
            self.metadata['file_mtime'] = os.path.getmtime(self.path)

    def add_child(self, child_id: str):
        """Add a child source ID to this node."""
        self.children_ids.add(child_id)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'id': self.id,
            'path': self.path,
            'parent_id': self.parent_id,
            'children_ids': list(self.children_ids),
            'transformation': self.transformation,
            'metadata': self.metadata,
            'created_at': self.created_at
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DataSource':
        """Deserialize from dictionary."""
        source = cls(
            path=data['path'],
            source_id=data['id'],
            parent_id=data.get('parent_id'),
            transformation=data.get('transformation'),
            metadata=data.get('metadata', {})
        )
        source.children_ids = set(data.get('children_ids', []))
        source.created_at = data.get('created_at', datetime.now().isoformat())
        return source

    def __repr__(self):
        return f"DataSource(id={self.id[:8]}..., path={self.path}, parent={self.parent_id[:8] if self.parent_id else None}...)"


class DatasetManager:
    """
    Manager for dataset operations in the abstract pipeline layer.

    Handles:
    - Loading datasets from various formats
    - Applying transformations (sampling, etc.)
    - Storing processed datasets in temporary or persistent locations
    - Tracking multiple data sources with lineage (tree structure)
    - Managing output paths for pipeline execution
    - Selecting specific sources for pipeline input

    Example:
        >>> manager = DatasetManager()
        >>> # Register and track a source
        >>> source_id = manager.register_source("data/input.json")
        >>> # Apply sampling (creates child source in tree)
        >>> processed_id = manager.sample_dataset(
        ...     source_id,
        ...     SamplingConfig(mode="ratio", ratio=0.1)
        ... )
        >>> # Get lineage
        >>> lineage = manager.get_source_lineage(processed_id)
        >>> # Use source in pipeline
        >>> data_path = manager.get_source_path(processed_id)
    """

    def __init__(
        self,
        temp_dir: Optional[Union[str, Path]] = None,
        keep_temp_files: bool = False,
        verbose: bool = False
    ):
        """
        Initialize the DatasetManager.

        Args:
            temp_dir: Directory for temporary processed datasets (default: system temp)
            keep_temp_files: If True, don't auto-cleanup temporary files
            verbose: Enable verbose logging
        """
        self.verbose = verbose
        self.keep_temp_files = keep_temp_files

        # Setup temporary directory
        if temp_dir:
            self.temp_dir = Path(temp_dir)
            self.temp_dir.mkdir(parents=True, exist_ok=True)
            self._temp_dir_obj = None
        else:
            self._temp_dir_obj = tempfile.TemporaryDirectory(prefix="abstract_data_")
            self.temp_dir = Path(self._temp_dir_obj.name)

        # Multi-source tracking registry
        self.sources: Dict[str, DataSource] = {}  # source_id -> DataSource
        self.path_to_source_id: Dict[str, str] = {}  # path -> source_id (for quick lookup)

        # Legacy support: Track path mappings: original_path -> processed_path
        self.path_mappings: Dict[str, str] = {}

        # Track processed files for cleanup
        self.processed_files: List[str] = []

        # Track output path modifications
        self.output_path_mapping: Dict[str, str] = {}

        if self.verbose:
            print(f"DatasetManager initialized with temp directory: {self.temp_dir}")

    def read_dataset(self, path: Union[str, Path]) -> List[Dict[str, Any]]:
        """
        Read a dataset from a file.

        Supports:
        - JSON (single array or line-delimited)
        - JSONL (JSON lines)

        Args:
            path: Path to the dataset file

        Returns:
            List of data items (dictionaries)

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is unsupported
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")

        if self.verbose:
            print(f"Reading dataset from: {path}")

        data = []

        # Try reading as JSON
        if path.suffix in ['.json', '.jsonl']:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    # Try to load as single JSON array
                    content = f.read().strip()
                    if content.startswith('['):
                        data = json.loads(content)
                    else:
                        # Try as JSONL (one JSON object per line)
                        f.seek(0)
                        for line in f:
                            line = line.strip()
                            if line:
                                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in file {path}: {e}")
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

        if self.verbose:
            print(f"Loaded {len(data)} items from {path}")

        return data

    def write_dataset(
        self,
        data: List[Dict[str, Any]],
        path: Optional[Union[str, Path]] = None,
        format: Literal["json", "jsonl"] = "json"
    ) -> str:
        """
        Write a dataset to a file.

        Args:
            data: List of data items to write
            path: Output path (if None, creates temp file)
            format: Output format - "json" or "jsonl"

        Returns:
            Path to the written file
        """
        if path is None:
            # Create temporary file
            suffix = '.json' if format == 'json' else '.jsonl'
            fd, path = tempfile.mkstemp(
                suffix=suffix,
                dir=self.temp_dir,
                prefix='dataset_'
            )
            os.close(fd)
            self.processed_files.append(str(path))
        else:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)

        path = str(path)

        with open(path, 'w', encoding='utf-8') as f:
            if format == 'json':
                json.dump(data, f, indent=2, ensure_ascii=False)
            elif format == 'jsonl':
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            else:
                raise ValueError(f"Unsupported format: {format}")

        if self.verbose:
            print(f"Wrote {len(data)} items to {path}")

        return path

    def sample_dataset(
        self,
        source_path: Union[str, Path],
        config: SamplingConfig,
        output_path: Optional[Union[str, Path]] = None
    ) -> str:
        """
        Sample a dataset according to the given configuration.

        Uses equal-weighted random sampling (each item has equal probability).

        Args:
            source_path: Path to the source dataset (or source_id if using multi-source tracking)
            config: Sampling configuration
            output_path: Optional output path (default: temp file)

        Returns:
            Path to the sampled dataset (or source_id if source was tracked)

        Example:
            >>> manager = DatasetManager()
            >>> # Sample 100 items
            >>> sampled = manager.sample_dataset(
            ...     "data.json",
            ...     SamplingConfig(mode="fixed", size=100, random_seed=42)
            ... )
            >>> # Sample 20% of data
            >>> sampled = manager.sample_dataset(
            ...     "data.json",
            ...     SamplingConfig(mode="ratio", ratio=0.2)
            ... )
        """
        # Check if source_path is a tracked source_id
        parent_source_id = None
        actual_source_path = source_path

        if isinstance(source_path, str) and source_path in self.sources:
            # It's a source_id
            parent_source_id = source_path
            actual_source_path = self.sources[source_path].path

        # Read source dataset
        data = self.read_dataset(actual_source_path)

        if len(data) == 0:
            if self.verbose:
                print("Warning: Source dataset is empty")
            output_path_str = self.write_dataset(data, output_path)
            # Still track as a transformation even if empty
            if parent_source_id:
                new_source = DataSource(
                    path=output_path_str,
                    parent_id=parent_source_id,
                    transformation="sample",
                    metadata={'config': config.__dict__, 'sample_size': 0, 'original_size': 0}
                )
                self._register_source_internal(new_source, parent_source_id)
                return new_source.id
            return output_path_str

        # Set random seed if specified
        if config.random_seed is not None:
            random.seed(config.random_seed)

        # Calculate sample size
        if config.mode == "fixed":
            sample_size = min(config.size, len(data))
        elif config.mode == "ratio":
            sample_size = max(1, int(len(data) * config.ratio))
        else:
            raise ValueError(f"Unknown sampling mode: {config.mode}")

        # Perform equal-weighted random sampling
        sampled_data = random.sample(data, sample_size)

        if self.verbose:
            print(f"Sampled {sample_size} items from {len(data)} (mode={config.mode})")

        # Write sampled dataset
        output_path_str = self.write_dataset(sampled_data, output_path)

        # Track the mapping (legacy)
        source_path_str = str(Path(actual_source_path).resolve())
        self.path_mappings[source_path_str] = output_path_str

        # If parent is tracked, create child source in tree
        if parent_source_id:
            new_source = DataSource(
                path=output_path_str,
                parent_id=parent_source_id,
                transformation="sample",
                metadata={
                    'config': config.__dict__,
                    'sample_size': sample_size,
                    'original_size': len(data)
                }
            )
            self._register_source_internal(new_source, parent_source_id)
            if self.verbose:
                print(f"Registered sampled source: {new_source.id}")
            return new_source.id

        return output_path_str

    def apply_transformation(
        self,
        source_path: Union[str, Path],
        transformation: str,
        **kwargs
    ) -> str:
        """
        Apply a transformation to a dataset.

        Currently supported transformations:
        - "sample": Sample the dataset (requires sampling_config)

        Args:
            source_path: Path to source dataset
            transformation: Type of transformation
            **kwargs: Transformation-specific parameters

        Returns:
            Path to transformed dataset
        """
        if transformation == "sample":
            sampling_config = kwargs.get('sampling_config')
            if not isinstance(sampling_config, SamplingConfig):
                raise ValueError("sampling_config must be a SamplingConfig instance")
            return self.sample_dataset(source_path, sampling_config, kwargs.get('output_path'))
        else:
            raise ValueError(f"Unknown transformation: {transformation}")

    def get_processed_path(self, original_path: Union[str, Path]) -> Optional[str]:
        """
        Get the processed path for an original dataset path.

        Args:
            original_path: Original dataset path

        Returns:
            Processed path if exists, None otherwise
        """
        original_path_str = str(Path(original_path).resolve())
        return self.path_mappings.get(original_path_str)

    def get_path_mappings(self) -> Dict[str, str]:
        """
        Get all path mappings (original -> processed).

        Returns:
            Dictionary of path mappings
        """
        return self.path_mappings.copy()

    def register_output_path(self, original_path: str, new_path: str):
        """
        Register an output path modification.

        Args:
            original_path: Original output path from pipeline config
            new_path: New output path to use
        """
        self.output_path_mapping[original_path] = new_path
        if self.verbose:
            print(f"Registered output path mapping: {original_path} -> {new_path}")

    def get_output_path(self, original_path: str) -> Optional[str]:
        """
        Get the modified output path for an original path.

        Args:
            original_path: Original output path

        Returns:
            Modified path if registered, None otherwise
        """
        return self.output_path_mapping.get(original_path)

    def load_output(self, path: Union[str, Path]) -> List[Dict[str, Any]]:
        """
        Load output data from a path.

        Args:
            path: Path to output file

        Returns:
            List of output items
        """
        return self.read_dataset(path)

    # ===== Multi-Source Tracking Methods =====

    def register_source(
        self,
        path: Union[str, Path],
        source_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Register a data source for tracking.

        Args:
            path: Path to the data file
            source_id: Optional custom ID (auto-generated if None)
            metadata: Optional metadata about the source

        Returns:
            Source ID

        Example:
            >>> manager = DatasetManager()
            >>> source_id = manager.register_source("data/input.json")
            >>> print(f"Registered: {source_id}")
        """
        path_str = str(Path(path).resolve())

        # Check if already registered by path
        if path_str in self.path_to_source_id:
            existing_id = self.path_to_source_id[path_str]
            if self.verbose:
                print(f"Source already registered: {existing_id}")
            return existing_id

        # Create new source
        source = DataSource(
            path=path_str,
            source_id=source_id,
            parent_id=None,
            transformation=None,
            metadata=metadata
        )

        self.sources[source.id] = source
        self.path_to_source_id[path_str] = source.id

        if self.verbose:
            print(f"Registered new source: {source.id} at {path_str}")

        return source.id

    def _register_source_internal(self, source: DataSource, parent_id: Optional[str] = None):
        """Internal method to register a source and update parent-child relationships."""
        self.sources[source.id] = source
        self.path_to_source_id[source.path] = source.id

        # Update parent's children list
        if parent_id and parent_id in self.sources:
            self.sources[parent_id].add_child(source.id)

    def get_source(self, source_id: str) -> Optional[DataSource]:
        """
        Get a data source by ID.

        Args:
            source_id: Source ID

        Returns:
            DataSource object or None if not found
        """
        return self.sources.get(source_id)

    def get_source_by_path(self, path: Union[str, Path]) -> Optional[DataSource]:
        """
        Get a data source by path.

        Args:
            path: Path to the data file

        Returns:
            DataSource object or None if not found
        """
        path_str = str(Path(path).resolve())
        source_id = self.path_to_source_id.get(path_str)
        return self.sources.get(source_id) if source_id else None

    def get_source_path(self, source_id: str) -> Optional[str]:
        """
        Get the path for a source ID.

        Args:
            source_id: Source ID

        Returns:
            Path to the data file or None if not found
        """
        source = self.sources.get(source_id)
        return source.path if source else None

    def get_all_sources(self) -> Dict[str, DataSource]:
        """
        Get all registered sources.

        Returns:
            Dictionary of source_id -> DataSource
        """
        return self.sources.copy()

    def get_root_sources(self) -> List[DataSource]:
        """
        Get all root sources (sources with no parent).

        Returns:
            List of root DataSource objects
        """
        return [source for source in self.sources.values() if source.parent_id is None]

    def get_source_lineage(self, source_id: str) -> List[DataSource]:
        """
        Get the lineage (ancestry path) of a source from root to current.

        Args:
            source_id: Source ID

        Returns:
            List of DataSource objects from root to current source

        Example:
            >>> lineage = manager.get_source_lineage(processed_id)
            >>> for i, source in enumerate(lineage):
            ...     print(f"  {i}: {source.transformation or 'original'} -> {source.path}")
        """
        if source_id not in self.sources:
            return []

        lineage = []
        current_id = source_id

        # Walk backwards to root
        while current_id:
            source = self.sources[current_id]
            lineage.insert(0, source)  # Insert at beginning to maintain order
            current_id = source.parent_id

        return lineage

    def get_source_descendants(self, source_id: str) -> List[DataSource]:
        """
        Get all descendants of a source (breadth-first).

        Args:
            source_id: Source ID

        Returns:
            List of descendant DataSource objects

        Example:
            >>> descendants = manager.get_source_descendants(root_id)
            >>> print(f"Found {len(descendants)} derived sources")
        """
        if source_id not in self.sources:
            return []

        descendants = []
        queue = [source_id]

        while queue:
            current_id = queue.pop(0)
            source = self.sources[current_id]

            for child_id in source.children_ids:
                if child_id in self.sources:
                    descendants.append(self.sources[child_id])
                    queue.append(child_id)

        return descendants

    def get_source_children(self, source_id: str) -> List[DataSource]:
        """
        Get immediate children of a source.

        Args:
            source_id: Source ID

        Returns:
            List of child DataSource objects
        """
        if source_id not in self.sources:
            return []

        source = self.sources[source_id]
        return [self.sources[child_id] for child_id in source.children_ids if child_id in self.sources]

    def export_lineage_tree(self, source_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Export the lineage tree structure as a dictionary.

        Args:
            source_id: Optional source ID to export subtree (default: all sources)

        Returns:
            Dictionary representation of the lineage tree
        """
        if source_id:
            # Export subtree
            if source_id not in self.sources:
                return {}
            return self._export_subtree(source_id)
        else:
            # Export all roots and their subtrees
            roots = self.get_root_sources()
            return {
                'roots': [self._export_subtree(root.id) for root in roots]
            }

    def _export_subtree(self, source_id: str) -> Dict[str, Any]:
        """Recursively export a source subtree."""
        if source_id not in self.sources:
            return {}

        source = self.sources[source_id]
        children = [self._export_subtree(child_id) for child_id in source.children_ids]

        return {
            **source.to_dict(),
            'children': children
        }

    def cleanup(self):
        """Clean up temporary files and directories."""
        if self.keep_temp_files:
            if self.verbose:
                print("Keeping temporary files (keep_temp_files=True)")
            return

        # Remove processed files
        for file_path in self.processed_files:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    if self.verbose:
                        print(f"Removed temporary file: {file_path}")
            except Exception as e:
                if self.verbose:
                    print(f"Warning: Failed to remove {file_path}: {e}")

        # Clean up temp directory if we created it
        if self._temp_dir_obj is not None:
            try:
                self._temp_dir_obj.cleanup()
                if self.verbose:
                    print(f"Cleaned up temporary directory: {self.temp_dir}")
            except Exception as e:
                if self.verbose:
                    print(f"Warning: Failed to cleanup temp directory: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup resources."""
        self.cleanup()
        return False

    def __del__(self):
        """Destructor - cleanup resources."""
        try:
            self.cleanup()
        except:
            pass  # Ignore errors during cleanup in destructor
