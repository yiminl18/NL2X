"""
Project Operator - Field projection

This operator selects specific fields from records, removing all others.
Simple field selection without transformation.
"""

from typing import Dict, Any
from .base import Operator


class Project(Operator):
    """
    Operator that projects (selects) specific fields from records.

    Keeps only specified fields, removing all others.
    Does not perform field transformation or renaming.

    Usage:
        op = Project()
        op.name = "select_fields"
        op.properties = {
            "fields": ["name", "email", "age"]
        }
    """

    def __init__(self):
        super().__init__()
        self.type = "Project"
        self.source = {
            "system": "abstract",
            "name": "Project"
        }

    @staticmethod
    def get_json_schema() -> Dict[str, Any]:
        """
        JSON schema for Project operator generation.

        The operator selects specific fields from each record.

        Returns:
            JSON schema dictionary for structured LLM output
        """
        return {
            "type": "object",
            "properties": {
                "fields": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "List of field names to keep in the output. "
                        "All other fields will be removed. "
                        "If a field does not exist in a record, it is skipped (no error)."
                    )
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
            "required": ["fields"]
        }
