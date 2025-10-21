#!/usr/bin/env python3
"""
File management utilities for abstract step pipeline generation.

This module provides centralized file saving and path management to reduce code duplication.
"""

import os
import json
import yaml
import logging
from typing import Any, Dict, Optional


def _create_base_filename(query: str, suffix: str = "") -> str:
    """
    Create a base filename from query and optional suffix.

    Args:
        query: The query string
        suffix: Optional suffix to append

    Returns:
        Safe filename string
    """
    # Simple filename generation - clean the query for use as filename
    safe_query = "".join(c for c in query[:50] if c.isalnum() or c in (' ', '-', '_')).rstrip()
    safe_query = safe_query.replace(' ', '_')

    if suffix:
        return f"{safe_query}_{suffix}"
    return safe_query


class PipelineFileManager:
    """
    Centralized manager for pipeline file operations.

    Handles:
    - Path generation and caching
    - JSON/YAML file saving with consistent formatting
    - Automatic logging of file operations
    """

    def __init__(
        self,
        base_dirs: Dict[str, str],
        logger: Optional[logging.Logger] = None,
        verbose: bool = False
    ):
        """
        Initialize the file manager.

        Args:
            base_dirs: Dictionary of base directory paths
                Example: {'pipeline_output': '/path/to/output', ...}
            logger: Optional logger for file operation logging
            verbose: Whether to output verbose logs
        """
        self.base_dirs = base_dirs
        self.logger = logger
        self.verbose = verbose
        self._path_cache = {}

    def _get_path(
        self,
        query: str,
        file_type: str,
        subdir_key: str,
        ext: str = '.json'
    ) -> str:
        """
        Generate and cache file path.

        Args:
            query: User query (used for filename generation)
            file_type: File type identifier (e.g., 'pipeline', 'output')
            subdir_key: Key for base directory in self.base_dirs
            ext: File extension

        Returns:
            Full file path
        """
        cache_key = f"{subdir_key}:{file_type}:{ext}"

        if cache_key in self._path_cache:
            return self._path_cache[cache_key]

        base_dir = self.base_dirs.get(subdir_key)
        if not base_dir:
            raise ValueError(f"Unknown subdirectory key: {subdir_key}")

        filename = f"{_create_base_filename(query, file_type)}{ext}"
        path = os.path.join(base_dir, filename)

        self._path_cache[cache_key] = path
        return path

    def _log(self, message: str, level: str = 'info'):
        """Internal logging with automatic verbose check."""
        if self.verbose and self.logger:
            getattr(self.logger, level)(message)

    def _save_file(self, path: str, data: Any, format: str = 'json'):
        """
        Save data to file with appropriate formatter.

        Args:
            path: File path to save to
            data: Data to save
            format: Format type ('json' or 'yaml')
        """
        try:
            with open(path, 'w', encoding='utf-8') as f:
                if format == 'json':
                    json.dump(data, f, indent=2, ensure_ascii=False)
                elif format == 'yaml':
                    yaml.dump(data, f, default_flow_style=False)
                else:
                    raise ValueError(f"Unsupported format: {format}")

            self._log(f"Saved {format.upper()} file to {path}")

        except Exception as e:
            self._log(f"Failed to save {format.upper()} file to {path}: {e}", level='error')
            raise

    def save_json(
        self,
        data: Any,
        query: str,
        file_type: str,
        subdir_key: str
    ) -> str:
        """
        Save data as JSON file.

        Args:
            data: Data to save (must be JSON serializable)
            query: User query (for filename generation)
            file_type: File type identifier (e.g., 'pipeline', 'output')
            subdir_key: Base directory key ('pipeline_output', 'abstract_pipeline', etc.)

        Returns:
            Path to saved file

        Example:
            >>> path = file_manager.save_json(
            ...     pipeline_dict,
            ...     query="Extract names",
            ...     file_type='pipeline',
            ...     subdir_key='abstract_pipeline'
            ... )
        """
        path = self._get_path(query, file_type, subdir_key, ext='.json')
        self._save_file(path, data, format='json')
        return path

    def save_yaml(
        self,
        data: Any,
        query: str,
        file_type: str,
        subdir_key: str
    ) -> str:
        """
        Save data as YAML file.

        Args:
            data: Data to save (must be YAML serializable)
            query: User query (for filename generation)
            file_type: File type identifier (e.g., 'pipeline_temp')
            subdir_key: Base directory key ('pipeline_output', etc.)

        Returns:
            Path to saved file

        Example:
            >>> path = file_manager.save_yaml(
            ...     docetl_config,
            ...     query="Extract names",
            ...     file_type='pipeline_temp',
            ...     subdir_key='pipeline_output'
            ... )
        """
        path = self._get_path(query, file_type, subdir_key, ext='.yaml')
        self._save_file(path, data, format='yaml')
        return path

    def get_output_path(self, query: str, file_type: str = 'output') -> str:
        """
        Get output file path without saving.

        Useful for generating output paths before execution.

        Args:
            query: User query
            file_type: File type identifier

        Returns:
            Output file path
        """
        return self._get_path(query, file_type, 'pipeline_output', ext='.json')

    def clear_cache(self):
        """Clear the path cache."""
        self._path_cache.clear()
