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
                "input": {
                    "type": "object",
                    "properties": {
                        "fields": {
                            "type": "object",
                            "description": "Input field types, e.g., {\"text\": \"String\", \"id\": \"Integer\"}"
                        }
                    },
                    "required": ["fields"]
                },
                "output": {
                    "type": "object",
                    "description": "Output schema with ALL fields (preserved + new). IMPORTANT: For List types, MUST specify element type as 'List[ElementType]' (e.g., 'List[String]', 'List[Integer]', 'List[Dict[{name: String}]]'). For Dict types, can use 'Dict' or 'Dict[{field1: Type1, field2: Type2}]'. Examples: {\"text\": \"String\", \"names\": \"List[String]\", \"data\": \"Dict[{count: Integer}]\"}"
                },
                "properties": {
                    "type": "object",
                    "description": "Additional configuration (optional)"
                }
            },
            "required": ["prompt", "input", "output"]
        }
