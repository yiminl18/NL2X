from typing import Dict, Any
from .base import Operator


class Unnest(Operator):
    def __init__(self):
        super().__init__()
        self.type = "unnest"

    @staticmethod
    def get_json_schema() -> Dict[str, Any]:
        """
        Get the JSON schema for Unnest operator.

        Returns:
            JSON schema dictionary for structured LLM output
        """
        return {
            "type": "object",
            "properties": {
                "unnest_key": {
                    "type": "string",
                    "description": "The field name containing the array or nested structure to expand"
                },
                "recursive": {
                    "type": "boolean",
                    "description": "Whether to recursively unnest nested structures"
                },
                "depth": {
                    "type": "integer",
                    "description": "Maximum depth for recursive unnesting"
                },
            },
            "required": ["unnest_key", "recursive", "depth"]
        }

