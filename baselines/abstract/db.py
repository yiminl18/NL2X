"""Data management with immutable DataSource and lineage tracking."""

from typing import Any, Dict, List, Optional, Union, Literal, Set
from pathlib import Path
import json
import tempfile
import os
import random
from datetime import datetime
import uuid


class SamplingConfig:
    """Configuration for dataset sampling (fixed size or ratio-based)."""

    def __init__(
        self,
        mode: Literal["fixed", "ratio"] = "ratio",
        size: Optional[int] = None,
        ratio: Optional[float] = None,
        random_seed: Optional[int] = None
    ):
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
    """Immutable data source with lineage tracking (all transformations return new instances)."""

    def __init__(
        self,
        path: Union[str, Path],
        data: Optional[Any] = None,
        data_type: str = "List[Dict]",
        source_id: Optional[str] = None,
        parent_id: Optional[str] = None,
        transformation: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.id = source_id or str(uuid.uuid4())
        self.path = str(Path(path).resolve())
        self.data = data
        self.data_type = data_type
        self.parent_id = parent_id
        self.children_ids: Set[str] = set()
        self.transformation = transformation
        self.metadata = metadata or {}
        self.created_at = datetime.now().isoformat()

        # Auto-populate metadata
        if os.path.exists(self.path):
            self.metadata['file_size'] = os.path.getsize(self.path)
            self.metadata['file_mtime'] = os.path.getmtime(self.path)

        # Add data metadata
        if data is not None:
            if isinstance(data, list):
                self.metadata['data_count'] = len(data)
            self.metadata['data_loaded'] = True
        else:
            self.metadata['data_loaded'] = False

    @classmethod
    def from_path(
        cls,
        path: Union[str, Path],
        data_type: str = "List[Dict]",
        metadata: Optional[Dict[str, Any]] = None
    ) -> 'DataSource':
        """Create DataSource from file path, automatically loading data."""
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")

        # Load data based on data_type
        if data_type == "List[Dict]":
            # Load JSON data
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content.startswith('['):
                    data = json.loads(content)
                else:
                    # Try as JSONL
                    data = []
                    f.seek(0)
                    for line in f:
                        line = line.strip()
                        if line:
                            data.append(json.loads(line))
        else:
            raise ValueError(f"Unsupported data_type: {data_type}")

        return cls(
            path=path,
            data=data,
            data_type=data_type,
            metadata=metadata
        )

    @classmethod
    def from_data(
        cls,
        data: Any,
        path: Optional[Union[str, Path]] = None,
        data_type: str = "List[Dict]",
        metadata: Optional[Dict[str, Any]] = None,
        temp_dir: Optional[Union[str, Path]] = None
    ) -> 'DataSource':
        """Create DataSource from in-memory data, optionally saving to file."""
        # If no path provided, create a temporary file
        if path is None:
            if temp_dir is None:
                temp_dir = tempfile.gettempdir()
            else:
                temp_dir = Path(temp_dir)
                temp_dir.mkdir(parents=True, exist_ok=True)

            # Create temp file
            fd, path = tempfile.mkstemp(
                suffix='.json',
                dir=temp_dir,
                prefix='datasource_'
            )
            os.close(fd)

            # Write data to file
            with open(path, 'w', encoding='utf-8') as f:
                if data_type == "List[Dict]":
                    json.dump(data, f, indent=2, ensure_ascii=False)
                else:
                    raise ValueError(f"Unsupported data_type for writing: {data_type}")
        else:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)

            # Write data to specified path
            with open(path, 'w', encoding='utf-8') as f:
                if data_type == "List[Dict]":
                    json.dump(data, f, indent=2, ensure_ascii=False)
                else:
                    raise ValueError(f"Unsupported data_type for writing: {data_type}")

        return cls(
            path=path,
            data=data,
            data_type=data_type,
            metadata=metadata
        )

    def sample(
        self,
        config: SamplingConfig,
        output_path: Optional[Union[str, Path]] = None,
        temp_dir: Optional[Union[str, Path]] = None
    ) -> 'DataSource':
        """Create sampled DataSource (immutable transformation)."""
        if self.data is None:
            raise ValueError("Cannot sample: data not loaded in DataSource")

        if self.data_type != "List[Dict]":
            raise ValueError(f"Sampling not supported for data_type: {self.data_type}")

        data = self.data

        if len(data) == 0:
            # Return empty DataSource
            return DataSource.from_data(
                data=[],
                path=output_path,
                data_type=self.data_type,
                metadata={'config': config.__dict__, 'sample_size': 0, 'original_size': 0},
                temp_dir=temp_dir
            )

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

        # Perform sampling
        sampled_data = random.sample(data, sample_size)

        # Create new DataSource
        new_source = DataSource.from_data(
            data=sampled_data,
            path=output_path,
            data_type=self.data_type,
            metadata={
                'config': config.__dict__,
                'sample_size': sample_size,
                'original_size': len(data)
            },
            temp_dir=temp_dir
        )

        # Set parent relationship
        new_source.parent_id = self.id
        new_source.transformation = "sample"

        return new_source

    def add_child(self, child_id: str):
        """Add child ID to this DataSource's children set."""
        self.children_ids.add(child_id)

    def __repr__(self):
        return f"DataSource(id={self.id[:8]}..., path={self.path}, parent={self.parent_id[:8] if self.parent_id else None}...)"


class DatasetManager:
    """Manager for dataset loading, transformations, and lineage tracking."""

    def __init__(
        self,
        temp_dir: Optional[Union[str, Path]] = None,
        keep_temp_files: bool = False,
        verbose: bool = False
    ):
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
        self.sources: Dict[str, DataSource] = {}
        self.path_to_source_id: Dict[str, str] = {}

        # Legacy support for DocETL executor
        self.path_mappings: Dict[str, str] = {}
        self.output_path_mapping: Dict[str, str] = {}

        # Track temp files for cleanup
        self.temp_files: List[str] = []

        if self.verbose:
            print(f"DatasetManager initialized with temp directory: {self.temp_dir}")

    def load(
        self,
        path: Union[str, Path],
        data_type: str = "List[Dict]",
        metadata: Optional[Dict[str, Any]] = None
    ) -> DataSource:
        """Load data file as DataSource (simplified API)."""
        path_str = str(Path(path).resolve())

        # Check if already loaded
        if path_str in self.path_to_source_id:
            existing_id = self.path_to_source_id[path_str]
            if self.verbose:
                print(f"Source already loaded: {existing_id}")
            return self.sources[existing_id]

        # Create DataSource from path (loads data automatically)
        data_source = DataSource.from_path(path, data_type=data_type, metadata=metadata)

        # Register for tracking
        self.sources[data_source.id] = data_source
        self.path_to_source_id[path_str] = data_source.id

        if self.verbose:
            print(f"Loaded new source: {data_source.id} at {path_str}")

        return data_source

    def create_from_data(
        self,
        data: Any,
        path: Optional[Union[str, Path]] = None,
        data_type: str = "List[Dict]",
        metadata: Optional[Dict[str, Any]] = None
    ) -> DataSource:
        """Create DataSource from in-memory data."""
        # Create DataSource from data
        data_source = DataSource.from_data(
            data=data,
            path=path,
            data_type=data_type,
            metadata=metadata,
            temp_dir=self.temp_dir
        )

        # Register for tracking
        self.sources[data_source.id] = data_source
        self.path_to_source_id[data_source.path] = data_source.id

        # Track temp file for cleanup if path was auto-generated
        if path is None:
            self.temp_files.append(data_source.path)

        if self.verbose:
            print(f"Created new source from data: {data_source.id}")

        return data_source

    def get_source(self, source_id: str) -> Optional[DataSource]:
        """Get a data source by ID."""
        return self.sources.get(source_id)

    def get_source_lineage(self, source_id: str) -> List[DataSource]:
        """Get the lineage (ancestry path) of a source from root to current."""
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

    def _register_source_internal(self, source: DataSource, parent_id: Optional[str] = None):
        """Internal method to register a source and update parent-child relationships."""
        self.sources[source.id] = source
        self.path_to_source_id[source.path] = source.id

        # Update parent's children list
        if parent_id and parent_id in self.sources:
            self.sources[parent_id].add_child(source.id)

    def get_processed_path(self, original_path: Union[str, Path]) -> Optional[str]:
        """Get processed path for original dataset path (legacy support for DocETL)."""
        original_path_str = str(Path(original_path).resolve())
        return self.path_mappings.get(original_path_str)

    def get_output_path(self, original_path: str) -> Optional[str]:
        """Get modified output path for original path (legacy support for DocETL)."""
        return self.output_path_mapping.get(original_path)

    def cleanup(self):
        """Clean up temporary files and directories."""
        if self.keep_temp_files:
            if self.verbose:
                print("Keeping temporary files (keep_temp_files=True)")
            return

        # Remove temp files
        for file_path in self.temp_files:
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
