"""
PythonCode Operator - Function-based execution

This operator executes a Python function within a procedure.
The function receives data sources as a dict parameter and returns new fields.
"""

from typing import Dict, Any, Optional, List
from .base import Operator


class PythonCodeOperator(Operator):
    """
    Operator that executes a Python function in a subprocess sandbox.

    The function signature must be: def process(sources)
    - sources: Dict with 'input_data' key containing previous operator output,
               plus any additional sources specified in the operator
    - Returns: Dict of new fields to merge into input_data records

    Usage:
        op = PythonCodeOperator()
        op.name = "calculate_average"
        op.properties = {
            "function": "def process(sources):\n    data = sources['input_data']\n    ..."
        }
    """

    def __init__(self):
        super().__init__()
        self.type = "PythonCode"
        self.source = {
            "system": "abstract",
            "name": "PythonCode"
        }

    @staticmethod
    def get_json_schema() -> Dict[str, Any]:
        """
        JSON schema for PythonCode operator generation.

        The function field should contain a Python function definition:
        - Function signature: def process(sources)
        - sources param: Dict with 'input_data' key (list of records from previous operator)
        - Returns: Dict of new fields to add to each record

        Example:
            def process(sources):
                data = sources['input_data']
                for record in data:
                    # Calculate new field
                    record['new_field'] = compute(record['existing_field'])
                return data

        Returns:
            JSON schema dictionary for structured LLM output
        """
        return {
            "type": "object",
            "properties": {
                "function": {
                    "type": "string",
                    "description": (
                        "Python function with signature 'def process(sources)'. "
                        "The function receives sources dict with 'input_data' key containing "
                        "list of records. Must return the modified data list with new fields added."
                    )
                },
                "sources": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Optional list of additional data source names beyond 'input_data'. "
                        "These will be available in sources dict."
                    )
                },
                "input": {
                    "type": "object",
                    "description": "Input schema (documentation only)"
                },
                "output": {
                    "type": "object",
                    "description": "Output schema with new fields (documentation only)"
                }
            },
            "required": ["function"]
        }
