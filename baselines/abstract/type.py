"""
Abstract Layer Type System

Unified type system independent of specific data processing systems.
Provides type parsing, serialization, compatibility checking, and validation.
"""

from typing import Any, Dict, Optional, Set, Tuple
import re


class TypeSystem:
    """Abstract layer unified type system."""
    BASIC_TYPES: Set[str] = {'String', 'Integer', 'Float', 'Boolean', 'Unknown', 'Null'}
    COMPLEX_TYPES: Set[str] = {'List', 'Dict', 'Array', 'Object'}
    ALL_TYPES: Set[str] = BASIC_TYPES | COMPLEX_TYPES
    TYPE_ALIASES: Dict[str, str] = {'Array': 'List', 'Object': 'Dict'}

    @classmethod
    def normalize_type_name(cls, type_name: str) -> str:
        """Normalize type name using aliases."""
        return cls.TYPE_ALIASES.get(type_name, type_name)

    @classmethod
    def is_valid_type_name(cls, type_name: str) -> bool:
        """Check if type name is valid."""
        return type_name in cls.ALL_TYPES

    @classmethod
    def is_basic_type(cls, type_name: str) -> bool:
        """Check if type is basic/scalar."""
        return type_name in cls.BASIC_TYPES

    @classmethod
    def is_complex_type(cls, type_name: str) -> bool:
        """Check if type is complex/container."""
        normalized = cls.normalize_type_name(type_name)
        return normalized in {'List', 'Dict'}


def parse_type_string(type_str: str) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
    """Parse and validate abstract layer type string."""
    if not type_str:
        return False, None, "Empty type string"

    type_str = type_str.strip()

    if type_str in TypeSystem.BASIC_TYPES:
        return True, {'type': type_str}, None

    if type_str in TypeSystem.TYPE_ALIASES:
        normalized = TypeSystem.TYPE_ALIASES[type_str]
        return True, {'type': normalized}, None

    list_match = re.match(r'^List\[(.+)\]$', type_str)
    if list_match:
        element_type_str = list_match.group(1)
        is_valid, element_type, error = parse_type_string(element_type_str)
        if not is_valid:
            return False, None, f"Invalid List element type: {error}"
        return True, {'type': 'List', 'element_type': element_type}, None

    dict_match = re.match(r'^Dict\{(.+)\}$', type_str)
    if dict_match:
        fields_str = dict_match.group(1)
        fields = {}

        field_pairs = re.findall(r'(\w+)\s*:\s*([^,}]+)', fields_str)
        for field_name, field_type_str in field_pairs:
            field_type_str = field_type_str.strip()
            is_valid, field_type, error = parse_type_string(field_type_str)
            if not is_valid:
                return False, None, f"Invalid type for field '{field_name}': {error}"
            fields[field_name] = field_type

        if not fields:
            return False, None, "Dict type must have at least one field"

        return True, {'type': 'Dict', 'fields': fields}, None

    if type_str == 'Dict':
        return False, None, "Dict must specify fields. Use 'Dict{field: Type, ...}' instead of 'Dict'"

    if type_str == 'List':
        return False, None, "List must specify element type. Use 'List[ElementType]' instead of 'List'"

    if type_str.startswith('list[') or type_str.startswith('array['):
        return False, None, f"Type should use 'List' (capitalized) not '{type_str.split('[')[0]}'"

    if type_str.startswith('dict[') or type_str.startswith('object['):
        return False, None, f"Type should use 'Dict' (capitalized) not '{type_str.split('[')[0]}'"

    if '{' in type_str or '}' in type_str:
        if not type_str.startswith('Dict{'):
            return False, None, "Dict fields must be wrapped in 'Dict{...}'"

    return False, None, f"Unrecognized type format: {type_str}"


def serialize_type_dict(type_dict: Dict[str, Any]) -> str:
    """Convert type dictionary to type string."""
    if not type_dict:
        return "Unknown"

    type_name = type_dict.get('type', 'Unknown')

    if type_name in TypeSystem.BASIC_TYPES:
        return type_name

    if type_name == 'List':
        element_type = type_dict.get('element_type')
        if not element_type:
            raise ValueError("List type must specify element_type. Use 'List[ElementType]' instead of 'List'")
        element_str = serialize_type_dict(element_type)
        return f"List[{element_str}]"

    if type_name == 'Dict':
        fields = type_dict.get('fields', {})
        if not fields:
            raise ValueError("Dict type must specify fields. Use 'Dict{field: Type, ...}' instead of 'Dict'")
        field_strs = []
        for field_name, field_type in fields.items():
            field_type_str = serialize_type_dict(field_type)
            field_strs.append(f"{field_name}: {field_type_str}")
        return f"Dict{{{', '.join(field_strs)}}}"

    return type_name


def check_type_compatibility(type1: str, type2: str) -> Tuple[bool, Optional[str]]:
    """Check if two types are compatible."""
    # Unknown type is compatible with anything
    if type1 == 'Unknown' or type2 == 'Unknown':
        return True, None

    # Exact match
    if type1 == type2:
        return True, None

    # Parse both types
    is_valid1, parsed1, _ = parse_type_string(type1)
    is_valid2, parsed2, _ = parse_type_string(type2)

    if not is_valid1 or not is_valid2:
        return False, "Invalid type format"

    # Get base types
    base_type1 = parsed1.get('type')
    base_type2 = parsed2.get('type')

    # Normalize using aliases
    base_type1 = TypeSystem.normalize_type_name(base_type1)
    base_type2 = TypeSystem.normalize_type_name(base_type2)

    if base_type1 != base_type2:
        return False, f"Type mismatch: {base_type1} vs {base_type2}"

    # For List types, check element type compatibility
    if base_type1 == 'List':
        elem1 = parsed1.get('element_type', {})
        elem2 = parsed2.get('element_type', {})
        elem_type1 = elem1.get('type', 'Unknown') if isinstance(elem1, dict) else 'Unknown'
        elem_type2 = elem2.get('type', 'Unknown') if isinstance(elem2, dict) else 'Unknown'

        if elem_type1 != elem_type2 and elem_type1 != 'Unknown' and elem_type2 != 'Unknown':
            return False, f"List element type mismatch: {elem_type1} vs {elem_type2}"

    # For Dict types, both being Dict is considered compatible
    # More detailed field-level checking can be added if needed

    return True, None


def validate_type_syntax(type_str: str) -> Tuple[bool, Optional[str]]:
    """Validate type string syntax."""
    is_valid, _, error = parse_type_string(type_str)
    return is_valid, error


def get_element_type(list_type_str: str) -> Optional[str]:
    """Extract element type from List type string."""
    is_valid, parsed, _ = parse_type_string(list_type_str)
    if not is_valid or parsed.get('type') != 'List':
        return None

    element_type = parsed.get('element_type', {})
    return serialize_type_dict(element_type)


def get_dict_fields(dict_type_str: str) -> Optional[Dict[str, str]]:
    """Extract field definitions from Dict type string."""
    is_valid, parsed, _ = parse_type_string(dict_type_str)
    if not is_valid or parsed.get('type') != 'Dict':
        return None

    fields = parsed.get('fields', {})
    if not fields:
        return {}

    # Convert field type dicts to strings
    return {name: serialize_type_dict(type_dict) for name, type_dict in fields.items()}


def normalize_type(type_str: str) -> str:
    """Normalize type string to canonical form."""
    is_valid, parsed, _ = parse_type_string(type_str)
    if not is_valid:
        return type_str  # Return as-is if invalid

    return serialize_type_dict(parsed)
