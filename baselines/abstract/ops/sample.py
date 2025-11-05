"""
Sample Operator - Random or sequential sampling

This operator selects a subset of records from the input data.
Supports uniform random sampling or taking first N records.
"""

from typing import Dict, Any
from .base import Operator


class Sample(Operator):
    """
    Operator that samples a subset of records.

    Supports:
    - Uniform random sampling
    - First N records

    Usage:
        op = Sample()
        op.name = "sample_100"
        op.properties = {
            "method": "uniform",
            "samples": 100,
            "random_state": 42
        }
    """

    def __init__(self):
        super().__init__()
        self.type = "Sample"
        self.source = {
            "system": "abstract",
            "name": "Sample"
        }

    @staticmethod
    def get_json_schema() -> Dict[str, Any]:
        """
        JSON schema for Sample operator generation.

        The operator samples a subset of records from input data.
        Supports exact count or percentage sampling.

        Returns:
            JSON schema dictionary for structured LLM output
        """
        return {
            "type": "object",
            "properties": {
                "method": {
                    "type": "string",
                    "enum": ["uniform", "first"],
                    "description": (
                        "Sampling method: 'uniform' for random sampling with replacement=False, "
                        "'first' for taking first N records"
                    )
                },
                "samples": {
                    "type": ["integer", "number"],
                    "description": (
                        "Number of samples to select. "
                        "Can be integer (exact count, e.g., 100) or "
                        "float between 0 and 1 (percentage, e.g., 0.1 for 10%)"
                    )
                },
                "random_state": {
                    "type": "integer",
                    "description": "Random seed for reproducibility (optional, only used with uniform method)"
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
            "required": ["method", "samples"]
        }
