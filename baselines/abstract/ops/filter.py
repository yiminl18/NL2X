from typing import Dict, Any
from .base import Operator


class Filter(Operator):
    def __init__(self):
        super().__init__()
        self.type = "filter"

    @staticmethod
    def get_json_schema() -> Dict[str, Any]:
        """
        Get the JSON schema for Filter operator.

        Returns:
            JSON schema dictionary for structured LLM output
        """
        return {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Jinja2 template for the filter condition. Use {{ input.field_name }} to reference fields."
                },
                "input": {
                    "type": "object",
                    "properties": {
                        "fields": {
                            "type": "object",
                            "description": "Input field types used in the condition"
                        }
                    },
                    "required": ["fields"]
                },
                "output": {
                    "type": "object",
                    "description": "Output schema (same as all available fields). Type specifications: String, Integer, Float, Boolean, List[Type], Dict[{field: Type}]."
                },
                "properties": {
                    "type": "object",
                    "description": "Additional configuration (optional)"
                }
            },
            "required": ["prompt", "input", "output"]
        }
