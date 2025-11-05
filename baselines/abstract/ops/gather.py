"""
Gather Operator - Add context to chunks

This operator adds surrounding context to split chunks.
Works in conjunction with Split operator to provide peripheral context.
"""

from typing import Dict, Any
from .base import Operator


class Gather(Operator):
    """
    Operator that adds context from adjacent chunks.

    Designed to work with Split operator output.
    Expects records with _chunk_text, _split_id, and _chunk_index fields.

    Output fields:
    - _chunk_with_context: Chunk with surrounding context included

    Usage:
        op = Gather()
        op.name = "gather_context"
        op.properties = {
            "context_size": 1  # Include 1 chunk before and after
        }
    """

    def __init__(self):
        super().__init__()
        self.type = "Gather"
        self.source = {
            "system": "abstract",
            "name": "Gather"
        }

    @staticmethod
    def get_json_schema() -> Dict[str, Any]:
        """
        JSON schema for Gather operator generation.

        The operator adds peripheral context to chunks from Split operator.

        Returns:
            JSON schema dictionary for structured LLM output
        """
        return {
            "type": "object",
            "properties": {
                "context_size": {
                    "type": "integer",
                    "description": (
                        "Number of chunks to include before and after current chunk (default: 1). "
                        "context_size=1 means include 1 previous and 1 next chunk."
                    ),
                    "default": 1
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
            "required": []
        }
