from typing import Dict, Any
from .base import Operator


class Reduce(Operator):
    def __init__(self):
        super().__init__()
        self.type = "reduce"

    @staticmethod
    def get_json_schema() -> Dict[str, Any]:
        """
        Get the JSON schema for Reduce operator.

        Returns:
            JSON schema dictionary for structured LLM output
        """
        return {
            "type": "object",
            "properties": {
                "reduce_key": {
                    "type": "string",
                    "description": "Field name to group by (must exist in available fields)"
                },
                "prompt": {
                    "type": "string",
                    "description": "Jinja2 template for aggregation. Use {{ inputs }} to reference grouped records."
                },
                "output": {
                    "type": "object",
                    "description": "Output schema including reduce_key and aggregated fields. Use complete types: 'List[String]', 'Dict[{avg: Float, max: Float}]' etc."
                },
            },
            "required": ["reduce_key", "prompt", "output"]
        }
