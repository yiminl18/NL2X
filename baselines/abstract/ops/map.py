from typing import Dict, Any
from .base import Operator


class Map(Operator):
    def __init__(self):
        super().__init__()
        self.type = "map"

    @staticmethod
    def get_json_schema() -> Dict[str, Any]:
        """
        Get the JSON schema for Map operator.

        Returns:
            JSON schema dictionary for structured LLM output
        """
        return {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Jinja2 template for the transformation. Use {{ input.field_name }} to reference fields."
                },
                "output": {
                    "type": "object",
                    "description": "Output schema with NEW fields. Examples: {\"text\": \"String\", \"names\": \"List[String]\", \"data\": \"Dict[{count: Integer}]\"}"
                },
            },
            "required": ["prompt", "output"]
        }
