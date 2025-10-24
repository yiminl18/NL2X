"""
DocETL Type Mapper

Type mapper for converting between DocETL and abstract layer types.
"""

from typing import Any, Dict
import re
from ...abstract.type import serialize_type_dict


class DocETLTypeMapper:
    """Maps DocETL type strings to abstract layer types."""

    DOCETL_TO_ABSTRACT = {
        "str": "String",
        "string": "String",
        "int": "Integer",
        "integer": "Integer",
        "number": "Integer",
        "float": "Float",
        "bool": "Boolean",
        "boolean": "Boolean",
        "dict": "Dict",
        "object": "Dict",
        "array": "List",
        "list": "List"
    }

    @classmethod
    def parse_docetl_type(cls, type_str: str) -> str:
        """
        Parse simple DocETL type to abstract layer type name.

        Args:
            type_str: DocETL type string (e.g., 'str', 'int', 'list[str]')

        Returns:
            Abstract type name (e.g., 'String', 'Integer', 'List')
        """
        if not type_str:
            return "Unknown"

        type_str = str(type_str).strip()

        # Handle list types (simple check)
        if type_str.startswith("list[") or type_str.startswith("List["):
            return "List"

        # Map basic types
        lower_type = type_str.lower()
        if lower_type in cls.DOCETL_TO_ABSTRACT:
            return cls.DOCETL_TO_ABSTRACT[lower_type]

        # If it contains dictionary-like structure
        if "{" in type_str and "}" in type_str:
            return "Dict"

        return "Unknown"

    @classmethod
    def parse_docetl_schema(cls, schema_str: str) -> Dict[str, Any]:
        """
        Parse DocETL schema to abstract layer type dict.

        Args:
            schema_str: DocETL schema string (e.g., 'list[{name: str, age: int}]')

        Returns:
            Type dictionary with structure: {'type': ..., 'element_type': ..., 'fields': ...}
        """
        if not schema_str:
            return {'type': 'Unknown'}

        schema_str = str(schema_str).strip()

        # Handle list types with nested structures
        if schema_str.startswith("list[") or schema_str.startswith("List["):
            inner_match = re.search(r'[Ll]ist\[(.+)\]$', schema_str)
            if inner_match:
                element_str = inner_match.group(1).strip()

                # Check if element is a dict/object
                if element_str.startswith("{") and element_str.endswith("}"):
                    fields_dict = {}
                    field_pairs = re.findall(r'(\w+)\s*:\s*(\w+)', element_str)
                    for field_name, field_type in field_pairs:
                        fields_dict[field_name] = cls.parse_docetl_schema(field_type)

                    return {
                        'type': 'List',
                        'element_type': {
                            'type': 'Dict',
                            'fields': fields_dict
                        }
                    }
                else:
                    return {
                        'type': 'List',
                        'element_type': cls.parse_docetl_schema(element_str)
                    }

        # Handle dict/object types
        if schema_str.startswith("{") and schema_str.endswith("}"):
            fields_dict = {}
            field_pairs = re.findall(r'(\w+)\s*:\s*(\w+)', schema_str)
            for field_name, field_type in field_pairs:
                fields_dict[field_name] = cls.parse_docetl_schema(field_type)
            return {'type': 'Dict', 'fields': fields_dict}

        # Handle basic types using the mapping
        lower_type = schema_str.lower()
        if lower_type in cls.DOCETL_TO_ABSTRACT:
            return {'type': cls.DOCETL_TO_ABSTRACT[lower_type]}

        return {'type': 'Unknown'}

    @classmethod
    def to_abstract_type_string(cls, docetl_schema: str) -> str:
        """
        Convert DocETL schema to abstract layer type string.

        Args:
            docetl_schema: DocETL schema string

        Returns:
            Abstract type string (e.g., 'String', 'List[String]', 'Dict{name: String}')
        """
        type_dict = cls.parse_docetl_schema(docetl_schema)
        return serialize_type_dict(type_dict)

    @classmethod
    def from_abstract_type_string(cls, abstract_type: str) -> str:
        """
        Convert abstract layer type string to DocETL type string.

        Args:
            abstract_type: Abstract type string (e.g., 'String', 'List[String]', 'Dict{name: String}')

        Returns:
            DocETL type string (e.g., 'str', 'list[str]', '{name: str}')

        Examples:
            'String' -> 'str'
            'Integer' -> 'int'
            'List[String]' -> 'list[str]'
            'Dict' -> 'dict'
            'Dict{name: String, age: Integer}' -> '{name: str, age: int}'
            'List[Dict{name: String, age: Integer}]' -> 'list[{name: str, age: int}]'
        """
        if not abstract_type:
            return 'str'  # Default to string

        abstract_type = abstract_type.strip()

        # Basic type mappings (reverse of DOCETL_TO_ABSTRACT)
        ABSTRACT_TO_DOCETL = {
            'String': 'str',
            'Integer': 'int',
            'Float': 'float',
            'Boolean': 'bool',
            'Dict': 'dict',
            'Unknown': 'str'  # Default unknown to string
        }

        # Handle basic types
        if abstract_type in ABSTRACT_TO_DOCETL:
            return ABSTRACT_TO_DOCETL[abstract_type]

        # Handle List types: List[ElementType]
        if abstract_type.startswith('List[') and abstract_type.endswith(']'):
            # Extract element type
            element_type_start = abstract_type.index('[') + 1
            element_type_end = abstract_type.rindex(']')
            element_type = abstract_type[element_type_start:element_type_end].strip()

            # Special case: List[Dict{...}] -> list[{...}]
            if element_type.startswith('Dict{') and element_type.endswith('}'):
                # Extract fields from Dict{field1: Type1, field2: Type2, ...}
                fields_start = element_type.index('{') + 1
                fields_end = element_type.rindex('}')
                fields_str = element_type[fields_start:fields_end].strip()

                if fields_str:
                    # Parse fields: "field1: Type1, field2: Type2"
                    docetl_fields = []
                    # Split by comma, but be careful with nested types
                    field_pairs = []
                    current_field = ""
                    bracket_depth = 0

                    for char in fields_str + ',':
                        if char in '[{':
                            bracket_depth += 1
                            current_field += char
                        elif char in ']}':
                            bracket_depth -= 1
                            current_field += char
                        elif char == ',' and bracket_depth == 0:
                            if current_field.strip():
                                field_pairs.append(current_field.strip())
                            current_field = ""
                        else:
                            current_field += char

                    # Convert each field
                    for field_pair in field_pairs:
                        if ':' in field_pair:
                            field_name, field_type = field_pair.split(':', 1)
                            field_name = field_name.strip()
                            field_type = field_type.strip()

                            # Recursively convert field type
                            docetl_field_type = cls.from_abstract_type_string(field_type)
                            docetl_fields.append(f"{field_name}: {docetl_field_type}")

                    if docetl_fields:
                        return f"list[{{{', '.join(docetl_fields)}}}]"

                # Fallback if parsing failed
                return 'list[dict]'

            # Regular list type
            docetl_element_type = cls.from_abstract_type_string(element_type)
            return f'list[{docetl_element_type}]'

        # Handle Dict types with fields: Dict{...}
        if abstract_type.startswith('Dict{') and abstract_type.endswith('}'):
            # Extract fields from Dict{field1: Type1, field2: Type2, ...}
            fields_start = abstract_type.index('{') + 1
            fields_end = abstract_type.rindex('}')
            fields_str = abstract_type[fields_start:fields_end].strip()

            if fields_str:
                # Parse fields: "field1: Type1, field2: Type2"
                docetl_fields = []
                # Split by comma, but be careful with nested types
                field_pairs = []
                current_field = ""
                bracket_depth = 0

                for char in fields_str + ',':
                    if char in '[{':
                        bracket_depth += 1
                        current_field += char
                    elif char in ']}':
                        bracket_depth -= 1
                        current_field += char
                    elif char == ',' and bracket_depth == 0:
                        if current_field.strip():
                            field_pairs.append(current_field.strip())
                        current_field = ""
                    else:
                        current_field += char

                # Convert each field
                for field_pair in field_pairs:
                    if ':' in field_pair:
                        field_name, field_type = field_pair.split(':', 1)
                        field_name = field_name.strip()
                        field_type = field_type.strip()

                        # Recursively convert field type
                        docetl_field_type = cls.from_abstract_type_string(field_type)
                        docetl_fields.append(f"{field_name}: {docetl_field_type}")

                if docetl_fields:
                    return f"{{{', '.join(docetl_fields)}}}"

            # Fallback if parsing failed
            return 'dict'

        # Unknown format, default to string
        return 'str'
