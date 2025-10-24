"""
DocETL Field Extraction Utilities

Extract field references from DocETL operator configurations.
"""

import re
from typing import Set, Dict, Any


def extract_fields_from_jinja2(template: str) -> Set[str]:
    """
    Extract field references from Jinja2 templates.

    Args:
        template: Jinja2 template string

    Returns:
        Set of field names referenced in template

    Examples:
        {{ input.name }} -> {'name'}
        {{ input["field"] }} -> {'field'}
        {{ inputs[0].field }} -> {'field'}
    """
    if not template:
        return set()

    fields = set()

    # {{ input.field }}
    input_pattern = r'\{\{\s*input\.(\w+)'
    fields.update(re.findall(input_pattern, template))

    # {{ input["field"] }}
    input_bracket_pattern = r'\{\{\s*input\["([^"]+)"\]'
    fields.update(re.findall(input_bracket_pattern, template))

    # {{ input['field'] }}
    input_bracket_single_pattern = r"\{\{\s*input\['([^']+)'\]"
    fields.update(re.findall(input_bracket_single_pattern, template))

    # {{ input1.field }} or {{ input2.field }}
    input_n_pattern = r'\{\{\s*input[12]\.(\w+)'
    fields.update(re.findall(input_n_pattern, template))

    # {{ input1["field"] }} or {{ input2["field"] }}
    input_n_bracket_pattern = r'\{\{\s*input[12]\["([^"]+)"\]'
    fields.update(re.findall(input_n_bracket_pattern, template))

    # {{ inputs[0].field }}
    inputs_idx_pattern = r'\{\{\s*inputs\[\d+\]\.(\w+)'
    fields.update(re.findall(inputs_idx_pattern, template))

    # {% for item in inputs %}{{ item.field }}
    for_loop_pattern = r'\{%\s*for\s+(\w+)\s+in\s+inputs\s*%\}'
    loop_vars = re.findall(for_loop_pattern, template)

    for var in loop_vars:
        escaped_var = re.escape(var)
        # {{ var.field }}
        var_pattern = r'\{\{?\s*' + escaped_var + r'\.(\w+)'
        fields.update(re.findall(var_pattern, template))
        # {{ var["field"] }}
        var_bracket_pattern = r'\{\{?\s*' + escaped_var + r'\["([^"]+)"\]'
        fields.update(re.findall(var_bracket_pattern, template))

    # {% for entry in field %}
    for_entry_pattern = r'\{%\s*for\s+(\w+)\s+in\s+\w+\s*%\}'
    entry_vars = re.findall(for_entry_pattern, template)
    for var in entry_vars:
        if var not in loop_vars:
            escaped_var = re.escape(var)
            var_pattern = r'\{\{?\s*' + escaped_var + r'\.(\w+)'
            fields.update(re.findall(var_pattern, template))

    # {{ items[0].field }} for reduce operations
    items_pattern = r'\{\{\s*items\[\d+\]\.(\w+)'
    fields.update(re.findall(items_pattern, template))

    # {{ items[0]["field"] }}
    items_bracket_pattern = r'\{\{\s*items\[\d+\]\["([^"]+)"\]'
    fields.update(re.findall(items_bracket_pattern, template))

    # {{ items[0]['field'] }}
    items_bracket_single_pattern = r"\{\{\s*items\[\d+\]\['([^']+)'\]"
    fields.update(re.findall(items_bracket_single_pattern, template))

    return fields


def is_direct_list_access(prompt: str, field: str) -> bool:
    """
    Check if field is accessed directly without indexing or loops.

    Args:
        prompt: Jinja2 template string
        field: Field name to check

    Returns:
        True if field is accessed directly (invalid for list fields)

    Examples:
        >>> is_direct_list_access("{{ input.medications }}", "medications")
        True
        >>> is_direct_list_access("{{ input.medications[0] }}", "medications")
        False
        >>> is_direct_list_access("{% for m in input.medications %}{{ m }}{% endfor %}", "medications")
        False
    """
    if not prompt:
        return False

    escaped_field = re.escape(field)

    # Direct access patterns: {{ input.field }}, {{ input1.field }}, {{ input2.field }}
    direct_patterns = [
        rf'\{{\{{\s*input\.{escaped_field}\s*\}}\}}',
        rf'\{{\{{\s*input1\.{escaped_field}\s*\}}\}}',
        rf'\{{\{{\s*input2\.{escaped_field}\s*\}}\}}',
        rf'\{{\{{\s*inputs\[\d+\]\.{escaped_field}\s*\}}\}}'
    ]

    for pattern in direct_patterns:
        if re.search(pattern, prompt):
            # Check if followed by indexing or in a loop
            indexed_pattern = pattern.rstrip(r'\}\}\}\}') + r'\['
            loop_pattern = r'\{%\s*for\s+\w+\s+in\s+\w*\.?' + escaped_field + r'\s*%\}'

            if not re.search(indexed_pattern, prompt) and not re.search(loop_pattern, prompt):
                return True

    return False


def extract_fields_from_python_code(code: str) -> Set[str]:
    """
    Extract field references from Python code.

    Args:
        code: Python code string

    Returns:
        Set of field names referenced in code

    Examples:
        doc["field"] -> {'field'}
        doc.field -> {'field'}
        input.get("field") -> {'field'}
    """
    if not code:
        return set()

    fields = set()

    # doc["field"] or doc['field']
    doc_bracket_pattern = r'doc\[[\'"]([\w_]+)[\'"]\]'
    fields.update(re.findall(doc_bracket_pattern, code))

    # doc.get("field")
    doc_get_pattern = r'doc\.get\([\'"]([^\'",]+)[\'"]'
    fields.update(re.findall(doc_get_pattern, code))

    # doc.field (but not doc.get)
    doc_dot_pattern = r'doc\.(?!get\b)(\w+)'
    fields.update(re.findall(doc_dot_pattern, code))

    # input["field"] or input['field']
    input_bracket_pattern = r'input\[[\'"]([\w_]+)[\'"]\]'
    fields.update(re.findall(input_bracket_pattern, code))

    # input.field
    input_dot_pattern = r'input\.(\w+)'
    fields.update(re.findall(input_dot_pattern, code))

    # Remove common method names that aren't fields
    fields.discard('get')
    fields.discard('keys')
    fields.discard('values')
    fields.discard('items')

    return fields
