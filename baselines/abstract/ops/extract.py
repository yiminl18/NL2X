from typing import Dict, Any
from .base import Operator


class Extract(Operator):
    def __init__(self):
        super().__init__()
        self.type = "extract"

    @staticmethod
    def get_json_schema() -> Dict[str, Any]:
        """
        Get the JSON schema for Extract operator.

        Returns:
            JSON schema dictionary for structured LLM output
        """
        return {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Jinja2 template describing what to extract. Use {{ input.field }} for document fields."
                },
                "document_keys": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of field names containing the documents to extract from"
                },
                "input": {
                    "type": "object",
                    "properties": {
                        "fields": {
                            "type": "object",
                            "description": "Input field types including document fields"
                        }
                    },
                    "required": ["fields"]
                },
                "output": {
                    "type": "object",
                    "description": "Output schema with extracted fields. CRITICAL: Specify complete types. For lists extracted from documents, use 'List[String]' not just 'List'. Examples: {\"document_name\": \"List[String]\", \"parties\": \"List[String]\", \"agreement_date\": \"List[String]\"}"
                },
                "properties": {
                    "type": "object",
                    "description": "Additional configuration (optional)"
                }
            },
            "required": ["prompt", "document_keys", "input", "output"]
        }
