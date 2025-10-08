"""
Data Management Module for Abstract Pipeline Layer

This module provides data management capabilities for the abstract layer to:
1. Read datasets before underlying system execution
2. Apply transformations (sampling, filtering, etc.)
3. Store processed datasets in temporary locations
4. Manage dataset path and output path modifications for underlying systems

The DatasetManager class acts as a proxy between the abstract layer and
underlying execution systems, allowing pre-processing and path management.
"""

from typing import Any, Dict, List, Optional, Union, Literal
from pathlib import Path
import json
import tempfile
import os
import random
import shutil
from datetime import datetime


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


class DatasetManager:
    """
    Manager for dataset operations in the abstract pipeline layer.

    Handles:
    - Loading datasets from various formats
    - Applying transformations (sampling, etc.)
    - Storing processed datasets in temporary or persistent locations
    - Tracking original and processed dataset paths
    - Managing output paths for pipeline execution

    Example:
        >>> manager = DatasetManager()
        >>> # Apply sampling
        >>> processed_path = manager.sample_dataset(
        ...     "data/input.json",
        ...     SamplingConfig(mode="ratio", ratio=0.1)
        ... )
        >>> # Use processed path in pipeline
        >>> manager.get_processed_path("data/input.json")
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

        # Track path mappings: original_path -> processed_path
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
            source_path: Path to the source dataset
            config: Sampling configuration
            output_path: Optional output path (default: temp file)

        Returns:
            Path to the sampled dataset

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
        # Read source dataset
        data = self.read_dataset(source_path)

        if len(data) == 0:
            if self.verbose:
                print("Warning: Source dataset is empty")
            return self.write_dataset(data, output_path)

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

        # Track the mapping
        source_path_str = str(Path(source_path).resolve())
        self.path_mappings[source_path_str] = output_path_str

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
