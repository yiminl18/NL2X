"""
LLM response cache for DocETL step-by-step pipeline generation.
Caches responses based on prompt hash to avoid redundant API calls.
"""

import os
import json
import hashlib
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
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")

        if os.path.exists(cache_file):
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
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")

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