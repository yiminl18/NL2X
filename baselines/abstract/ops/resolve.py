from typing import Dict, Any
from .base import Operator


class Resolve(Operator):
    def __init__(self):
        super().__init__()
        self.type = "resolve"

    @staticmethod
    def get_json_schema() -> Dict[str, Any]:
        """
        Get the JSON schema for Resolve operator.

        Returns:
            JSON schema dictionary for structured LLM output
        """
        return {
            "type": "object",
            "properties": {
                "comparison_prompt": {"type": "string"},
                "resolution_prompt": {"type": "string"},
                "output": {"type": "object"},
            },
            "required": ["comparison_prompt", "resolution_prompt", "output"]
        }
