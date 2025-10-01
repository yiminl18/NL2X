#!/usr/bin/env python3
"""
LiteLLM Client for DocETL Step-by-Step
Based on docetl/optimizers/utils.py implementation
"""

import os
from typing import Dict, List, Any, Optional
from litellm import completion, RateLimitError
import time


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


def llm_call(messages: List[Dict[str, str]],
             schema: Optional[Dict[str, Any]] = None,
             system_prompt: str = "",
             max_tokens: int = 4000,
             temperature: float = 0.3) -> str:
    config = _get_azure_config()

    full_messages = []
    if system_prompt:
        full_messages.append({"role": "system", "content": system_prompt})
    full_messages.extend(messages)

    params = {
        "model": config["model"],
        "messages": full_messages,
        "api_key": config["api_key"],
        "api_base": config["api_base"],
        "api_version": config["api_version"],
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    if schema:
        # Don't set additionalProperties at root level, let the schema define it
        params["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "output",
                "strict": False,  # Disable strict mode to allow flexible schemas
                "schema": schema
            }
        }

    rate_limited_attempt = 0
    max_attempts = 6

    while rate_limited_attempt < max_attempts:
        try:
            response = completion(**params)
            return response.choices[0].message.content

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


