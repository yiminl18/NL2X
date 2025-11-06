"""
Convert Operator - Data format conversion

This operator converts data between different formats (CSV, JSON, etc.).
It's useful for cross-system data flow where format compatibility is needed.
"""

from typing import Dict, Any
from .base import Operator


class ConvertOperator(Operator):
    """
    Operator that converts data between different formats.

    Currently supports:
    - CSV to JSON conversion
    - HTML to JSON conversion

    Usage:
        op = ConvertOperator()
        op.name = "csv_to_json"
        op.properties = {
            "source_format": "csv",
            "target_format": "json"
        }
    """

    def __init__(self):
        super().__init__()
        self.type = "Convert"
        self.source = {
            "system": "abstract",
            "name": "Convert"
        }

    @staticmethod
    def get_json_schema() -> Dict[str, Any]:
        """
        JSON schema for Convert operator generation.

        The operator converts data from source_format to target_format.
        Input and output schemas are for documentation/tracking purposes.

        Returns:
            JSON schema dictionary for structured LLM output
        """
        return {
            "type": "object",
            "properties": {
                "source_format": {
                    "type": "string",
                    "enum": ["csv", "html"],
                    "description": "Source data format to convert from"
                },
                "target_format": {
                    "type": "string",
                    "enum": ["json"],
                    "description": "Target data format to convert to"
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
            "required": ["source_format", "target_format"]
        }
