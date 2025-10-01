"""
LLM response cache for DocETL step-by-step pipeline generation.
Caches responses based on prompt hash to avoid redundant API calls.
"""

import os
import json
import hashlib
import glob
from datetime import datetime
from typing import Optional


class LLMCache:
    """
    Simple cache for LLM responses using prompt hash as key.
    """

    def __init__(self, cache_dir: str):
        """
        Initialize cache with specified directory.
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def _get_cache_key(self, prompt: str) -> str:
        """
        Generate SHA256 hash key from prompt.
        """
        return hashlib.sha256(prompt.encode()).hexdigest()

    def get(self, prompt: str, force_regenerate: bool = False) -> Optional[str]:
        """
        Get cached response if exists.

        Args:
            prompt: The prompt to look up in cache
            force_regenerate: If True, bypass cache and return None to trigger regeneration

        Returns:
            Cached response string, or None if not cached or force_regenerate is True
        """
        if force_regenerate:
            print(f"🔄 Force regeneration requested, bypassing cache")
            return None

        cache_key = self._get_cache_key(prompt)
        # Search for cache files matching pattern: *_{cache_key}.json
        pattern = os.path.join(self.cache_dir, f"*_{cache_key}.json")
        matching_files = glob.glob(pattern)

        if matching_files:
            # Use the most recent file if multiple exist
            cache_file = max(matching_files, key=os.path.getmtime)
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                    print(f"🔄 Using cached response (hash: {cache_key}...)")
                    return cache_data['response']
            except (json.JSONDecodeError, IOError) as e:
                # Cache file corrupted, will regenerate
                print(f"⚠️ Cache file corrupted, regenerating: {e}")
                return None
        return None

    def set(self, prompt: str, response: str):
        """
        Save response to cache.
        """
        cache_key = self._get_cache_key(prompt)
        # Generate timestamp prefix in format YYYYMMDD_HHMMSS
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cache_file = os.path.join(self.cache_dir, f"{timestamp}_{cache_key}.json")

        cache_data = {
            'prompt': prompt,
            'response': response,
            'timestamp': datetime.now().isoformat()
        }

        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)
        except IOError as e:
            print(f"⚠️ Failed to save cache: {e}")