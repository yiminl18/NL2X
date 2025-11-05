"""
Split Operator - Text chunking

This operator splits long text fields into smaller chunks.
Supports token-based splitting (using tiktoken) or delimiter-based splitting.
"""

from typing import Dict, Any
from .base import Operator


class Split(Operator):
    """
    Operator that splits long text into chunks.

    Supports:
    - Token count splitting (using tiktoken)
    - Delimiter splitting (e.g., by paragraphs, sentences)

    Output fields:
    - _chunk_text: The chunk content
    - _split_id: Document unique ID (UUID)
    - _chunk_index: Chunk index (0-based)
    - _total_chunks: Total number of chunks for this document

    Usage:
        op = Split()
        op.name = "split_content"
        op.properties = {
            "split_key": "content",
            "method": "token_count",
            "method_kwargs": {"num_tokens": 500}
        }
    """

    def __init__(self):
        super().__init__()
        self.type = "Split"
        self.source = {
            "system": "abstract",
            "name": "Split"
        }

    @staticmethod
    def get_json_schema() -> Dict[str, Any]:
        """
        JSON schema for Split operator generation.

        The operator splits text into manageable chunks for processing.

        Returns:
            JSON schema dictionary for structured LLM output
        """
        return {
            "type": "object",
            "properties": {
                "split_key": {
                    "type": "string",
                    "description": "Field name containing the text to split"
                },
                "method": {
                    "type": "string",
                    "enum": ["token_count", "delimiter"],
                    "description": (
                        "Split method: 'token_count' for token-based splitting using tiktoken, "
                        "'delimiter' for delimiter-based splitting"
                    )
                },
                "method_kwargs": {
                    "type": "object",
                    "description": (
                        "Method-specific parameters. "
                        "For token_count: {'num_tokens': int, 'model': str (optional, default gpt-4o)}. "
                        "For delimiter: {'delimiter': str, 'num_splits_to_group': int (optional, default 1)}"
                    )
                },
                "input": {
                    "type": "object",
                    "description": "Input schema (documentation only)"
                },
                "output": {
                    "type": "object",
                    "description": "Output schema (documentation only)"
                }
            },
            "required": ["split_key", "method", "method_kwargs"]
        }
