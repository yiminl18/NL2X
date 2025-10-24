"""
DocETL Complex Type Parsing

Parse and analyze DocETL complex type structures like list[{field: type}], Dict{...}, etc.
"""

from typing import Dict, Any, Set, Optional, List


def parse_complex_type(type_str: str) -> Dict[str, Any]:
    """
    Parse complex type strings to extract nested field information.

    Args:
        type_str: Type string (e.g., 'list[{name: str, age: int}]', 'dict', 'str')

    Returns:
        Dict with structure:
        {
            'is_complex': bool,
            'base_type': 'list' | 'dict' | 'simple',
            'nested_fields': Set[str],      # Field names within complex type
            'nested_types': Dict[str, str]  # Field name -> type mapping
        }

    Examples:
        >>> parse_complex_type('list[{name: str, age: int}]')
        {'is_complex': True, 'base_type': 'list',
         'nested_fields': {'name', 'age'},
         'nested_types': {'name': 'str', 'age': 'int'}}
    """
    result = {
        'is_complex': False,
        'base_type': 'simple',
        'nested_fields': set(),
        'nested_types': {}
    }

    if not isinstance(type_str, str):
        return result

    type_str = type_str.strip()

    # Check for list[{...}] pattern (list of dicts with fields)
    if type_str.startswith('list[{') and type_str.endswith('}]'):
        result['is_complex'] = True
        result['base_type'] = 'list'

        # Extract content between list[{ and }]
        content = type_str[6:-2]

        # Parse field definitions
        parts = content.split(',')
        for part in parts:
            part = part.strip()
            if ':' in part:
                field_name, field_type = part.split(':', 1)
                field_name = field_name.strip()
                field_type = field_type.strip()
                result['nested_fields'].add(field_name)
                result['nested_types'][field_name] = field_type

    # Check for simple list[type] pattern
    elif type_str.startswith('list[') and type_str.endswith(']'):
        result['is_complex'] = True
        result['base_type'] = 'list'
        # Simple list, no nested fields to extract

    # Check for dict type (generic dict without specific fields)
    elif type_str == 'dict':
        result['is_complex'] = True
        result['base_type'] = 'dict'

    # Check for {field: type, ...} pattern (dict with fields)
    elif type_str.startswith('{') and type_str.endswith('}'):
        result['is_complex'] = True
        result['base_type'] = 'dict'

        # Extract content between { and }
        content = type_str[1:-1]

        # Parse field definitions
        parts = content.split(',')
        for part in parts:
            part = part.strip()
            if ':' in part:
                field_name, field_type = part.split(':', 1)
                field_name = field_name.strip()
                field_type = field_type.strip()
                result['nested_fields'].add(field_name)
                result['nested_types'][field_name] = field_type

    return result


def find_field_in_complex_types(field_name: str, schema: Dict[str, str]) -> Optional[Dict[str, Any]]:
    """
    Find if a field exists within complex types in the schema.

    Args:
        field_name: The field name to search for
        schema: Current schema (field -> type mapping)

    Returns:
        Dict with location info if found, None otherwise:
        {
            'parent_field': str,       # Name of parent field containing this field
            'parent_type': str,        # Type of parent field
            'field_type': str,         # Type of nested field
            'base_type': 'list' | 'dict',  # Type of complex container
            'access_suggestions': List[str]  # Suggested ways to access the field
        }

    Examples:
        >>> schema = {'medications': 'list[{name: str, dose: str}]'}
        >>> find_field_in_complex_types('name', schema)
        {'parent_field': 'medications', 'parent_type': 'list[{...}]',
         'base_type': 'list', 'access_suggestions': [...]}
    """
    for parent_field, parent_type in schema.items():
        # Parse the parent type to check for nested fields
        type_info = parse_complex_type(parent_type)

        if type_info['is_complex'] and field_name in type_info['nested_fields']:
            # Found the field within this complex type
            access_suggestions = generate_access_suggestions(
                field_name, parent_field, type_info['base_type']
            )

            return {
                'parent_field': parent_field,
                'parent_type': parent_type,
                'field_type': type_info['nested_types'].get(field_name, 'unknown'),
                'base_type': type_info['base_type'],
                'access_suggestions': access_suggestions
            }

    return None


def generate_access_suggestions(field_name: str, parent_field: str, base_type: str) -> List[str]:
    """
    Generate suggestions for accessing nested fields in complex types.

    Args:
        field_name: The nested field name
        parent_field: The parent field name
        base_type: Type of complex container ('list' or 'dict')

    Returns:
        List of suggestion strings

    Examples:
        >>> generate_access_suggestions('name', 'medications', 'list')
        ['1. Use an Unnest operator to expand the medications field first',
         '2. Access via index in Jinja2: {{ input.medications[0].name }}',
         '3. Use a for loop in Jinja2: {% for item in input.medications %}{{ item.name }}{% endfor %}']
    """
    suggestions = []

    if base_type == 'list':
        # List of dicts - suggest unnest or indexed access
        suggestions.append(
            f"1. Use an Unnest operator to expand the '{parent_field}' field first"
        )
        suggestions.append(
            f"2. Access via index in Jinja2: {{{{ input.{parent_field}[0].{field_name} }}}}"
        )
        suggestions.append(
            f"3. Use a for loop in Jinja2: {{% for item in input.{parent_field} %}}{{{{ item.{field_name} }}}}{{% endfor %}}"
        )

    elif base_type == 'dict':
        # Dict with fields - suggest dotted access
        suggestions.append(
            f"1. Access via dotted notation in Jinja2: {{{{ input.{parent_field}.{field_name} }}}}"
        )
        suggestions.append(
            f"2. Use an Unnest operator to expand the '{parent_field}' field if needed"
        )

    return suggestions


def parse_list_schema_fields(field_def: str) -> Set[str]:
    """
    Parse a list schema definition to extract nested field names.

    Args:
        field_def: List schema string like 'list[{name: str, age: int}]'

    Returns:
        Set of field names

    Examples:
        >>> parse_list_schema_fields('list[{name: str, age: int}]')
        {'name', 'age'}
    """
    fields = set()

    if isinstance(field_def, str) and field_def.startswith('list[{') and field_def.endswith('}]'):
        # Extract the content between list[{ and }]
        content = field_def[6:-2]  # Remove 'list[{' and '}]'

        # Split by commas and extract field names
        parts = content.split(',')
        for part in parts:
            if ':' in part:
                field_name = part.split(':')[0].strip()
                fields.add(field_name)

    return fields
