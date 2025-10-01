"""
Shared logging utilities for DocETL baseline implementations.
This module provides unified functions for saving prompts, messages, validations, and intermediate results.
"""

import os
import json
from datetime import datetime
from typing import Any, Dict, List


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


def _write_json(data: Any, filepath: str) -> None:
    """
    Write data to JSON file with proper formatting.

    Args:
        data: Data to write
        filepath: Path to write to
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_prompt(output_dir: str, filename_base: str, prompt: str, query: str, attempt: int = 0) -> str:
    """
    Save prompt to persistent directory.

    Args:
        output_dir: Directory to save to
        filename_base: Base filename (from data_processor._create_base_filename if available)
        prompt: Prompt text to save
        query: Original query
        attempt: Attempt number (deprecated, now included in filename_base)

    Returns:
        Path to saved file
    """
    filename = f"{filename_base}.txt"
    filepath = os.path.join(output_dir, filename)

    os.makedirs(output_dir, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(prompt)

    return filepath


def save_messages(output_dir: str, filename_base: str, messages: List[Dict], query: str) -> str:
    """
    Save complete message history (prompts and responses only).

    Args:
        output_dir: Directory to save to
        filename_base: Base filename
        messages: List of message dictionaries (user prompts and assistant responses)
        query: Original query

    Returns:
        Path to saved file
    """
    filename = f"{filename_base}_messages.json"
    filepath = os.path.join(output_dir, filename)

    messages_data = {
        "query": query,
        "timestamp": datetime.now().isoformat(),
        "total_messages": len(messages),
        "messages": messages  # Only save prompt+response dialogue
    }

    _write_json(messages_data, filepath)
    return filepath


def save_validation(output_dir: str, filename_base: str, validation_prompt: str,
                   validation_result: Dict, query: str) -> str:
    """
    Save validation prompt and result.

    Args:
        output_dir: Directory to save to
        filename_base: Base filename
        validation_prompt: Validation prompt text
        validation_result: Validation result dictionary
        query: Original query

    Returns:
        Path to saved file
    """
    filename = f"{filename_base}_validation.json"
    filepath = os.path.join(output_dir, filename)

    validation_data = {
        "query": query,
        "timestamp": datetime.now().isoformat(),
        "validation_prompt": validation_prompt,
        "validation_result": validation_result
    }

    _write_json(validation_data, filepath)
    return filepath


def save_step_output(output_dir: str, filename_base: str, step_name: str, content: Any,
                    query: str, attempt: int = 0) -> str:
    """
    Save intermediate step output.

    Args:
        output_dir: Directory to save to
        filename_base: Base filename
        step_name: Name of the step
        content: Step output content
        query: Original query
        attempt: Attempt number

    Returns:
        Path to saved file
    """
    filename = f"{filename_base}_attempt{attempt}_{step_name}.json"
    filepath = os.path.join(output_dir, filename)

    step_data = {
        "query": query,
        "attempt": attempt,
        "step": step_name,
        "timestamp": datetime.now().isoformat(),
        "content": content
    }

    _write_json(step_data, filepath)
    return filepath


def get_filename_base(data_processor, query: str, suffix: str = "") -> str:
    """
    Get filename base using data_processor if available, fallback to simple generation.

    Args:
        data_processor: Data processor object with _create_base_filename method
        query: Query string
        suffix: Optional suffix

    Returns:
        Base filename string
    """
    if hasattr(data_processor, '_create_base_filename'):
        return data_processor._create_base_filename(query, suffix)
    else:
        return _create_base_filename(query, suffix)