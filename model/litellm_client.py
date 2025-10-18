#!/usr/bin/env python3
"""
LiteLLM Client with integrated caching support.
Provides a unified interface for LLM calls with automatic response caching.
"""

import os
import json
import hashlib
import glob
from typing import Dict, List, Any, Optional, Union, Tuple
from litellm import completion, RateLimitError
import time
from pathlib import Path
from datetime import datetime


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
                    print(f"🔄 Using cached response (hash: {cache_key[:8]}...)")
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


# Initialize global LLM cache
_cache_dir = Path(__file__).parent.parent / "llm_cache"
_global_cache = LLMCache(str(_cache_dir))


def _get_azure_config():
    api_key = os.getenv("AZURE_API_KEY")

    if not api_key:
        raise ValueError("AZURE_API_KEY environment variable is not set")

    config = {
        "api_key": api_key,
        "api_base": os.getenv("AZURE_API_BASE", "https://text-db.openai.azure.com/"),
        "api_version": os.getenv("AZURE_API_VERSION", "2025-01-01-preview"),
        "model": "azure/gpt-4o"
    }

    return config


def llm_call(
    messages: Union[str, List[Dict[str, Any]]],
    schema: Optional[Dict[str, Any]] = None,
    system_prompt: str = "",
    max_tokens: int = 4000,
    temperature: float = 0.3,
    top_p: float = 1.0,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    response_format: Optional[Dict[str, Any]] = None,
    bypass_cache: bool = False,
    return_usage: bool = False
) -> Union[str, Tuple[str, Any]]:
    """
    Call LLM with caching support.

    Args:
        messages: Either a string prompt or a list of message dictionaries
        schema: JSON schema for structured output (will be converted to response_format)
        system_prompt: System prompt to prepend
        max_tokens: Maximum tokens for response
        temperature: Response randomness (0-1)
        top_p: Nucleus sampling parameter
        frequency_penalty: Frequency penalty
        presence_penalty: Presence penalty
        response_format: Response format for structured output (takes precedence over schema)
        bypass_cache: If True, bypass cache and force regeneration
        return_usage: If True, return (content, usage) tuple instead of just content

    Returns:
        str: The response content, or (content, usage) if return_usage=True
    """
    config = _get_azure_config()

    # Convert string prompt to messages format
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]

    # Build full messages with system prompt
    full_messages = []
    if system_prompt:
        full_messages.append({"role": "system", "content": system_prompt})
    full_messages.extend(messages)

    # Build params
    params = {
        "model": config["model"],
        "messages": full_messages,
        "api_key": config["api_key"],
        "api_base": config["api_base"],
        "api_version": config["api_version"],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
        "frequency_penalty": frequency_penalty,
        "presence_penalty": presence_penalty
    }

    # Handle response format (response_format takes precedence over schema)
    if response_format:
        params["response_format"] = response_format
    elif schema:
        params["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "output",
                "strict": False,
                "schema": schema
            }
        }

    # Generate cache key based on params (excluding api credentials)
    cache_key_data = {
        "messages": full_messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "frequency_penalty": frequency_penalty,
        "presence_penalty": presence_penalty,
        "schema": schema,
        "response_format": response_format
    }
    cache_key_str = json.dumps(cache_key_data, sort_keys=True)

    # Check cache (unless bypassing)
    if not bypass_cache:
        cached_response = _global_cache.get(cache_key_str, force_regenerate=False)
        if cached_response:
            # Cache hit - parse and return
            try:
                cache_data = json.loads(cached_response)
                content = cache_data["content"]
                if return_usage:
                    # Return cached usage if available, otherwise create dummy usage
                    usage = cache_data.get("usage", {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0
                    })
                    # Convert dict to object-like structure for compatibility
                    class Usage:
                        def __init__(self, data):
                            self.prompt_tokens = data.get("prompt_tokens", 0)
                            self.completion_tokens = data.get("completion_tokens", 0)
                            self.total_tokens = data.get("total_tokens", 0)
                    return content, Usage(usage)
                return content
            except (json.JSONDecodeError, KeyError):
                # Cache corrupted, will regenerate
                pass

    # Call LLM with retry logic
    rate_limited_attempt = 0
    max_attempts = 6

    while rate_limited_attempt < max_attempts:
        try:
            response = completion(**params)
            content = response.choices[0].message.content

            # Save to cache
            cache_data = {
                "content": content,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            }
            _global_cache.set(cache_key_str, json.dumps(cache_data))

            if return_usage:
                return content, response.usage
            return content

        except RateLimitError:
            backoff_time = 4 * (2**rate_limited_attempt)
            max_backoff = 120
            sleep_time = min(backoff_time, max_backoff)

            print(f"Rate limit hit. Retrying in {sleep_time:.2f} seconds...")
            time.sleep(sleep_time)
            rate_limited_attempt += 1

        except Exception as e:
            raise Exception(f"LLM call failed: {str(e)}")

    raise Exception("Rate limit hit too many times")


