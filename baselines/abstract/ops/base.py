from typing import Dict, Any


class Operator:
    def __init__(self):
        self.name = ""          # Name of the operator from the abstract pipeline perspective
        self.type = ""          # Type of the abstract operator (e.g., map, filter, reduce)
        self.source = {
            "system": "",       # System where the operator is acutally from before is transformed into the abstract operator
            "name": ""          # Name of the operator in the system where it is acutally from
        }
        self.properties = {}    # Properties of the abstract operator (e.g., prompt, reduce_key ...)
        # The following fields should be linked to type and field tracking system
        self.input = {}         # Input schema of the abstract operator
        self.output = {}        # Output schema of the abstract operator

    @staticmethod
    def get_json_schema() -> Dict[str, Any]:
        """
        Get the default/generic JSON schema for operators.

        This is the fallback schema for operators that don't have a specific schema defined.

        Returns:
            JSON schema dictionary for structured LLM output
        """
        return {
            "type": "object",
            "properties": {
                "prompt": {"type": "string"},
                "input": {
                    "type": "object",
                    "properties": {
                        "fields": {"type": "object"}
                    }
                },
                "output": {"type": "object"},
                "properties": {"type": "object"}
            },
            "required": []
        }