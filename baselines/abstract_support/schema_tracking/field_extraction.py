"""
Field extraction from Python code.

Extracts field references from Python functions for validation.
"""
import re
from typing import Set


def extract_fields_from_python_code(code: str) -> Set[str]:
    """
    Extract field names referenced in Python code.

    Looks for common patterns:
    - record['field_name']
    - item['field_name']
    - data.get('field_name')
    - sources['input_data'][0]['field_name']

    Args:
        code: Python code string

    Returns:
        Set of field names found in the code
    """
    fields = set()

    # Pattern 1: dict['key'] or dict.get('key')
    # Matches: record['field'], item['field'], data.get('field')
    pattern1 = r"['\"]([a-zA-Z_][a-zA-Z0-9_]*)['\"]"
    matches = re.findall(pattern1, code)

    # Filter out common Python keywords and sources
    excluded = {'input_data', 'sources', 'record', 'item', 'data', 'result', 'output',
                'str', 'int', 'float', 'list', 'dict', 'True', 'False', 'None'}

    for match in matches:
        if match not in excluded:
            fields.add(match)

    # Pattern 2: record.field_name (attribute access)
    pattern2 = r"\w+\.([a-zA-Z_][a-zA-Z0-9_]*)"
    attr_matches = re.findall(pattern2, code)
    for match in attr_matches:
        if match not in excluded and not match.startswith('_'):
            fields.add(match)

    return fields
