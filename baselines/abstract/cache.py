"""
Pipeline Cache Manager for Abstract Operators

This module provides caching functionality for abstract operator executions.
It caches operator configuration-input-output mappings to avoid redundant executions.

Cache Key Design:
- Operator hash: SHA-512 of operator config (excluding name and source.name)
- Input hash: SHA-512 of normalized input data
- Cache key: {operator_hash}_{input_hash}.json
"""

from typing import Any, Dict, Optional, Union, TYPE_CHECKING
from pathlib import Path
import json
import hashlib
import os
from datetime import datetime
import copy

# Import abstract operator classes
from .ops.base import Operator

if TYPE_CHECKING:
    from .db import DataSource

 
class OperatorCacheManager:
    """Manager for caching abstract operator execution results."""

    CACHE_VERSION = "1.0"

    def __init__(self, cache_dir: Optional[Union[str, Path]] = None, enabled: bool = True):
        """Initialize cache manager with optional cache directory."""
        if cache_dir is None:
            # Default to abstract/_cache directory
            cache_dir = os.path.join(
                os.path.dirname(__file__),
                '_cache'
            )

        self.cache_dir = Path(cache_dir)
        self.enabled = enabled

        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.stats = {
            'hits': 0,
            'misses': 0,
            'writes': 0
        }

    def _compute_operator_hash(self, operator: Operator) -> str:
        """Compute SHA-512 hash of operator configuration (excludes name fields)."""
        config_dict = {
            'type': operator.type,
            'source_system': operator.source.get('system', ''),
            'properties': operator.properties,
            'input': operator.input,
            'output': operator.output
        }

        config_json = json.dumps(config_dict, sort_keys=True, ensure_ascii=False)
        hash_obj = hashlib.sha512(config_json.encode('utf-8'))
        return hash_obj.hexdigest()

    def _compute_input_hash(self, data_source: 'DataSource') -> str:
        """Compute SHA-512 hash of input data from DataSource."""
        # Use data from DataSource directly if available
        if data_source.data is not None:
            data_to_hash = data_source.data
        else:
            # Fallback: read from file if data not loaded
            file_path = Path(data_source.path)
            if file_path.exists():
                with open(file_path, 'r') as f:
                    try:
                        data_to_hash = json.load(f)
                    except json.JSONDecodeError:
                        # If not JSON, hash the raw content
                        f.seek(0)
                        data_to_hash = f.read()
            else:
                # File doesn't exist, hash the path itself
                data_to_hash = data_source.path

        # Convert to normalized JSON string
        if isinstance(data_to_hash, str):
            input_json = data_to_hash
        else:
            input_json = json.dumps(data_to_hash, sort_keys=True, ensure_ascii=False)

        # Compute SHA-512 hash
        hash_obj = hashlib.sha512(input_json.encode('utf-8'))
        return hash_obj.hexdigest()

    def _get_cache_key(self, operator: Operator, data_source: 'DataSource') -> str:
        """Generate cache key from operator and data source."""
        operator_hash = self._compute_operator_hash(operator)[:32]
        input_hash = self._compute_input_hash(data_source)[:32]
        return f"{operator_hash}_{input_hash}"

    def _get_cache_path(self, cache_key: str) -> Path:
        """Get file path for cache entry using operator hash subdirectory."""
        # Extract operator hash (first part before underscore)
        operator_hash = cache_key.split('_')[0]

        # Use first 16 chars as subdirectory
        subdir = operator_hash[:16]

        # Create subdirectory path
        cache_subdir = self.cache_dir / subdir
        cache_subdir.mkdir(parents=True, exist_ok=True)

        # Return full cache file path
        return cache_subdir / f"{cache_key}.json"

    def get_cached_result(self,
                         operator: Operator,
                         data_source: 'DataSource',
                         force_execute: bool = False) -> Optional[Dict[str, Any]]:
        """Retrieve cached result if available."""
        # Force execution bypasses cache
        if force_execute:
            return None

        if not self.enabled:
            return None

        try:
            cache_key = self._get_cache_key(operator, data_source)
            cache_path = self._get_cache_path(cache_key)

            if not cache_path.exists():
                self.stats['misses'] += 1
                return None

            # Load cached data
            with open(cache_path, 'r', encoding='utf-8') as f:
                cached_entry = json.load(f)

            # Verify cache version
            if cached_entry.get('version') != self.CACHE_VERSION:
                # Cache version mismatch, invalidate
                self.stats['misses'] += 1
                return None

            # Verify full hashes match (prevent collisions on truncated filenames)
            current_operator_hash = self._compute_operator_hash(operator)
            current_input_hash = self._compute_input_hash(data_source)

            if (cached_entry.get('operator_hash') != current_operator_hash or
                cached_entry.get('input_hash') != current_input_hash):
                # Hash mismatch - this is a collision on the truncated filename
                self.stats['misses'] += 1
                return None

            # Extract result
            result = {
                'output_data': cached_entry['output_data'],
                'metadata': cached_entry['metadata']
            }

            self.stats['hits'] += 1
            return result

        except Exception:
            # If any error occurs during cache read, treat as cache miss
            self.stats['misses'] += 1
            return None

    def set_cached_result(self,
                         operator: Operator,
                         data_source: 'DataSource',
                         output_data: Any,
                         metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Store execution result in cache."""
        if not self.enabled:
            return False

        try:
            cache_key = self._get_cache_key(operator, data_source)
            cache_path = self._get_cache_path(cache_key)

            # Compute hashes
            operator_hash = self._compute_operator_hash(operator)
            input_hash = self._compute_input_hash(data_source)

            # Prepare cache entry
            cache_entry = {
                'version': self.CACHE_VERSION,
                'operator_hash': operator_hash,
                'input_hash': input_hash,
                'operator_config': {
                    'type': operator.type,
                    'source': copy.deepcopy(operator.source),
                    'properties': copy.deepcopy(operator.properties),
                    'input': copy.deepcopy(operator.input),
                    'output': copy.deepcopy(operator.output)
                },
                'input_data': self._serialize_input_data(data_source),
                'output_data': output_data,
                'metadata': {
                    **(metadata or {}),
                    'cached_at': datetime.now().isoformat(),
                    'original_operator_name': operator.name
                }
            }

            # Write to cache file
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache_entry, f, indent=2, ensure_ascii=False)

            self.stats['writes'] += 1
            return True

        except Exception:
            # If cache write fails, don't crash - just log and continue
            return False

    def inject_cached_result(self,
                             operator: Operator,
                             data_source: 'DataSource',
                             output_data: Any,
                             metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Manually inject result into cache without execution."""
        # Use the same logic as set_cached_result
        # This ensures consistency in cache format
        custom_metadata = metadata or {}
        custom_metadata['injected'] = True
        custom_metadata['injection_time'] = datetime.now().isoformat()

        return self.set_cached_result(operator, data_source, output_data, custom_metadata)

    def _serialize_input_data(self, data_source: 'DataSource') -> Any:
        """Serialize input data from DataSource for storage in cache."""
        # Use data from DataSource if available
        if data_source.data is not None:
            data = data_source.data
            # Store first few items as sample for list data
            if isinstance(data, list):
                return {
                    '_type': 'data_source',
                    'path': data_source.path,
                    'source_id': data_source.id,
                    'data_type': data_source.data_type,
                    'sample': data[:5] if len(data) > 5 else data,
                    'total_items': len(data)
                }
            else:
                return {
                    '_type': 'data_source',
                    'path': data_source.path,
                    'source_id': data_source.id,
                    'data_type': data_source.data_type,
                    'data': data
                }
        else:
            # Fallback: try to read from file
            file_path = Path(data_source.path)
            if file_path.exists():
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    # Store first few items as sample
                    if isinstance(data, list):
                        return {
                            '_type': 'data_source',
                            'path': data_source.path,
                            'source_id': data_source.id,
                            'data_type': data_source.data_type,
                            'sample': data[:5] if len(data) > 5 else data,
                            'total_items': len(data)
                        }
                    else:
                        return {
                            '_type': 'data_source',
                            'path': data_source.path,
                            'source_id': data_source.id,
                            'data_type': data_source.data_type,
                            'data': data
                        }
                except:
                    return {
                        '_type': 'data_source',
                        'path': data_source.path,
                        'source_id': data_source.id,
                        'data_type': data_source.data_type
                    }
            else:
                return {
                    '_type': 'data_source',
                    'path': data_source.path,
                    'source_id': data_source.id,
                    'data_type': data_source.data_type,
                    'exists': False
                }

    def clear_cache(self, operator_hash: Optional[str] = None) -> int:
        """Clear cache entries (all if operator_hash=None, else specific operator)."""
        if not self.enabled or not self.cache_dir.exists():
            return 0

        deleted_count = 0

        if operator_hash:
            # Clear specific operator cache
            subdir = operator_hash[:16]
            cache_subdir = self.cache_dir / subdir

            if cache_subdir.exists():
                # Delete all cache files starting with this operator hash
                for cache_file in cache_subdir.glob(f"{operator_hash}_*.json"):
                    cache_file.unlink()
                    deleted_count += 1

                # Remove subdir if empty
                try:
                    cache_subdir.rmdir()
                except OSError:
                    pass  # Directory not empty, keep it
        else:
            # Clear all cache
            for subdir in self.cache_dir.iterdir():
                if subdir.is_dir() and subdir.name != '.gitignore':
                    for cache_file in subdir.glob("*.json"):
                        cache_file.unlink()
                        deleted_count += 1

                    # Remove subdir
                    try:
                        subdir.rmdir()
                    except OSError:
                        pass

        return deleted_count

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics including hits, misses, and total size."""
        stats = self.stats.copy()

        # Calculate hit rate
        total_requests = stats['hits'] + stats['misses']
        stats['hit_rate'] = stats['hits'] / total_requests if total_requests > 0 else 0.0

        # Count total cache entries
        total_entries = 0
        total_size = 0

        if self.cache_dir.exists():
            for subdir in self.cache_dir.iterdir():
                if subdir.is_dir():
                    for cache_file in subdir.glob("*.json"):
                        total_entries += 1
                        total_size += cache_file.stat().st_size

        stats['total_entries'] = total_entries
        stats['cache_size_mb'] = total_size / (1024 * 1024)

        return stats

    def get_cache_info(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get information about specific cache entry by key."""
        if not self.enabled:
            return None

        try:
            cache_path = self._get_cache_path(cache_key)

            if not cache_path.exists():
                return None

            with open(cache_path, 'r', encoding='utf-8') as f:
                cached_entry = json.load(f)

            return {
                'version': cached_entry.get('version'),
                'operator_hash': cached_entry.get('operator_hash'),
                'input_hash': cached_entry.get('input_hash'),
                'cached_at': cached_entry.get('metadata', {}).get('cached_at'),
                'original_operator_name': cached_entry.get('metadata', {}).get('original_operator_name'),
                'file_path': str(cache_path),
                'file_size': cache_path.stat().st_size
            }

        except Exception:
            return None
