"""
Type tracking utilities for DocETL pipeline generation.

This module provides a comprehensive type system for tracking data types
and field structures throughout the DocETL pipeline generation process.
Uses simple dict-based storage for efficiency and simplicity.
"""

from typing import Any, Dict, List, Optional, Union


class TypeSystem:
    def __init__(self):
        self.root_fields: Dict[str, Dict[str, Any]] = {}

    def add_field(self, field_path: str, type_dict: Dict[str, Any]) -> None:
        """Add a field with its type to the system."""
        self.root_fields[field_path] = type_dict

    def get_field_type(self, field_path: str) -> Optional[Dict[str, Any]]:
        """Get the type of a specific field."""
        return self.root_fields.get(field_path)

    def get_all_fields(self) -> List[str]:
        """Get all field names in the type system."""
        return list(self.root_fields.keys())

    def get_fields_with_types(self) -> Dict[str, str]:
        """Get all fields with their string type representations."""
        return {field: type_dict_to_string(type_dict)
                for field, type_dict in self.root_fields.items()}

    def copy(self) -> 'TypeSystem':
        """Create a deep copy of the type system."""
        new_system = TypeSystem()
        # Simple dict copy since we're using plain dicts
        for field, type_dict in self.root_fields.items():
            new_system.root_fields[field] = dict(type_dict)
        return new_system

    def format_for_prompt(self) -> str:
        """Format the type system for inclusion in LLM prompts."""
        if not self.root_fields:
            return "No fields available"

        lines = []
        for field, type_dict in sorted(self.root_fields.items()):
            lines.append(f"  {field}: {type_dict_to_string(type_dict)}")

        return "Available fields:\n" + "\n".join(lines)

    def to_nested_dict(self) -> Dict[str, Any]:
        """Convert type system to nested dictionary representation."""
        result = {}

        for field_path, type_dict in self.root_fields.items():
            # Split field path and create nested structure
            parts = field_path.split('.')
            current = result

            for i, part in enumerate(parts):
                if i == len(parts) - 1:
                    # Last part, assign the type
                    current[part] = type_dict_to_string(type_dict)
                else:
                    # Intermediate part, create nested dict
                    if part not in current:
                        current[part] = {}
                    current = current[part]

        return result


def type_dict_to_string(type_dict: Dict[str, Any]) -> str:
    """Convert a type dictionary to human-readable string representation."""
    type_name = type_dict.get('type', 'Unknown')

    if type_name in ['String', 'Integer', 'Float', 'Boolean', 'Null', 'Unknown']:
        return type_name

    elif type_name == 'List':
        element_type = type_dict.get('element_type')
        if element_type:
            return f"List[{type_dict_to_string(element_type)}]"
        return "List[Unknown]"

    elif type_name == 'Dict':
        fields = type_dict.get('fields', {})
        if fields:
            field_strs = []
            for field_name, field_type in sorted(fields.items()):
                field_str = f"{field_name}: {type_dict_to_string(field_type)}"
                field_strs.append(field_str)
            return f"Dict[{', '.join(field_strs)}]"
        return "Dict[Unknown]"

    return type_name


def infer_type_from_value(value: Any) -> Dict[str, Any]:
    """
    Infer type dictionary from a Python value.

    Args:
        value: The value to analyze

    Returns:
        Dict representing the inferred type
    """
    if value is None:
        return {'type': 'Null'}

    elif isinstance(value, bool):
        return {'type': 'Boolean'}

    elif isinstance(value, int):
        return {'type': 'Integer'}

    elif isinstance(value, float):
        return {'type': 'Float'}

    elif isinstance(value, str):
        return {'type': 'String'}

    elif isinstance(value, list):
        if value:
            # Analyze element types - for simplicity, use the first element's type
            element_type = infer_type_from_value(value[0])
            return {'type': 'List', 'element_type': element_type}
        return {'type': 'List', 'element_type': {'type': 'Unknown'}}

    elif isinstance(value, dict):
        fields = {}
        for key, val in value.items():
            if isinstance(key, str):
                fields[key] = infer_type_from_value(val)
        return {'type': 'Dict', 'fields': fields}

    else:
        return {'type': 'Unknown'}


def extract_type_system(dataset_samples: Union[List[Dict], Dict[str, Any]]) -> TypeSystem:
    """
    Extract type system from dataset samples.

    Args:
        dataset_samples: Dataset samples to analyze

    Returns:
        TypeSystem representing the dataset structure
    """
    type_system = TypeSystem()

    def analyze_item(item: Dict[str, Any], prefix: str = "") -> None:
        """Recursively analyze an item to extract field types."""
        if not isinstance(item, dict):
            return

        for key, value in item.items():
            field_path = f"{prefix}.{key}" if prefix else key
            type_dict = infer_type_from_value(value)

            # Add to type system
            existing_type = type_system.get_field_type(field_path)
            if existing_type is None:
                type_system.add_field(field_path, type_dict)
            else:
                # Merge types if different (handle type variations)
                merged_type = merge_type_dicts(existing_type, type_dict)
                type_system.add_field(field_path, merged_type)

            # For nested dictionaries, recursively analyze
            if isinstance(value, dict):
                analyze_item(value, field_path)

    # Handle different input formats
    if isinstance(dataset_samples, list):
        for item in dataset_samples:
            analyze_item(item)
    elif isinstance(dataset_samples, dict):
        # Check if this is a dict of datasets or a single item
        if all(isinstance(v, (list, dict)) for v in dataset_samples.values()):
            # Multiple datasets
            for dataset in dataset_samples.values():
                if isinstance(dataset, list):
                    for item in dataset:
                        analyze_item(item)
                else:
                    analyze_item(dataset)
        else:
            # Single item
            analyze_item(dataset_samples)

    return type_system


def merge_type_dicts(type1: Dict[str, Any], type2: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two type dictionaries to handle type variations.

    Args:
        type1: First type dict
        type2: Second type dict

    Returns:
        Merged type dict
    """
    if type1 == type2:
        return type1

    # If one is null and the other isn't, use the non-null type
    if type1.get('type') == 'Null':
        return type2
    if type2.get('type') == 'Null':
        return type1

    # If both are lists, merge element types
    if (type1.get('type') == 'List' and type2.get('type') == 'List'):
        element1 = type1.get('element_type', {'type': 'Unknown'})
        element2 = type2.get('element_type', {'type': 'Unknown'})
        merged_element = merge_type_dicts(element1, element2)
        return {'type': 'List', 'element_type': merged_element}

    # If both are dicts, merge fields
    if (type1.get('type') == 'Dict' and type2.get('type') == 'Dict'):
        fields1 = type1.get('fields', {})
        fields2 = type2.get('fields', {})
        merged_fields = {}

        all_fields = set(fields1.keys()).union(set(fields2.keys()))
        for field in all_fields:
            if field in fields1 and field in fields2:
                merged_fields[field] = merge_type_dicts(fields1[field], fields2[field])
            elif field in fields1:
                merged_fields[field] = fields1[field]
            else:
                merged_fields[field] = fields2[field]

        return {'type': 'Dict', 'fields': merged_fields}

    # For different basic types, default to unknown
    return {'type': 'Unknown'}


def apply_map_operator_transformation(type_system: TypeSystem, operator: Dict[str, Any]) -> TypeSystem:
    """Apply map operator transformation to type system."""
    new_system = type_system.copy()

    if 'output' in operator and 'schema' in operator['output']:
        schema = operator['output']['schema']
        if isinstance(schema, dict):
            for field_name, field_type in schema.items():
                # Infer type from schema field type description
                type_dict = infer_type_from_schema_type(field_type)
                new_system.add_field(field_name, type_dict)

    return new_system


def apply_extract_operator_transformation(type_system: TypeSystem, operator: Dict[str, Any]) -> TypeSystem:
    """Apply extract operator transformation to type system."""
    new_system = type_system.copy()

    if 'document_keys' in operator and operator['document_keys']:
        for doc_key in operator['document_keys']:
            suffix = operator.get('extraction_key_suffix', f"_extracted_{operator.get('name', 'extract')}")
            extracted_field = f"{doc_key}{suffix}"
            # Extract operations typically produce strings
            new_system.add_field(extracted_field, {'type': 'String'})

    return new_system


def apply_unnest_operator_transformation(type_system: TypeSystem, operator: Dict[str, Any]) -> TypeSystem:
    """Apply unnest operator transformation to type system."""
    new_system = type_system.copy()

    if 'unnest_key' in operator and operator['unnest_key']:
        unnest_key = operator['unnest_key']
        unnest_type = type_system.get_field_type(unnest_key)

        if unnest_type:
            if unnest_type.get('type') == 'List':
                # For list unnest: preserve element types as new fields
                element_type = unnest_type.get('element_type')
                if element_type and element_type.get('type') == 'Dict':
                    # Add all fields from the nested dict
                    fields = element_type.get('fields', {})
                    for field_name, field_type in fields.items():
                        new_system.add_field(field_name, field_type)

            elif unnest_type.get('type') == 'Dict':
                # For dict unnest: flatten all nested fields to top level
                fields = unnest_type.get('fields', {})
                for field_name, field_type in fields.items():
                    new_system.add_field(field_name, field_type)

    return new_system


def apply_reduce_operator_transformation(type_system: TypeSystem, operator: Dict[str, Any]) -> TypeSystem:
    """Apply reduce operator transformation to type system."""
    new_system = type_system.copy()

    # Add reduce_key as a grouping field (typically remains the same type)
    if 'reduce_key' in operator and operator['reduce_key']:
        reduce_key = operator['reduce_key']
        existing_type = type_system.get_field_type(reduce_key)
        if existing_type:
            new_system.add_field(reduce_key, existing_type)

    # Add output schema fields
    if 'output' in operator and 'schema' in operator['output']:
        schema = operator['output']['schema']
        if isinstance(schema, dict):
            for field_name, field_type in schema.items():
                type_dict = infer_type_from_schema_type(field_type)
                new_system.add_field(field_name, type_dict)

    return new_system


def apply_split_operator_transformation(type_system: TypeSystem, _: Dict[str, Any]) -> TypeSystem:
    """Apply split operator transformation to type system."""
    new_system = type_system.copy()

    # Split operations add these standard fields
    new_system.add_field('_split_id', {'type': 'String'})
    new_system.add_field('_split_index', {'type': 'Integer'})

    return new_system


def apply_gather_operator_transformation(type_system: TypeSystem, operator: Dict[str, Any]) -> TypeSystem:
    """Apply gather operator transformation to type system."""
    new_system = type_system.copy()

    if 'output_key' in operator:
        # Gather typically produces a list of strings
        list_type = {'type': 'List', 'element_type': {'type': 'String'}}
        new_system.add_field(operator['output_key'], list_type)

    return new_system


def apply_operator_transformation(type_system: TypeSystem, operator: Dict[str, Any]) -> TypeSystem:
    """
    Apply operator transformation to type system.

    Args:
        type_system: Current type system
        operator: Operator configuration

    Returns:
        Updated type system
    """
    op_type = operator.get('type', '')

    if op_type == 'map':
        return apply_map_operator_transformation(type_system, operator)
    elif op_type == 'extract':
        return apply_extract_operator_transformation(type_system, operator)
    elif op_type == 'unnest':
        return apply_unnest_operator_transformation(type_system, operator)
    elif op_type == 'reduce':
        return apply_reduce_operator_transformation(type_system, operator)
    elif op_type == 'split':
        return apply_split_operator_transformation(type_system, operator)
    elif op_type == 'gather':
        return apply_gather_operator_transformation(type_system, operator)
    elif op_type in ['filter', 'rank', 'resolve', 'cluster', 'sample', 'topk']:
        # These operators don't change the type system structure
        return type_system.copy()
    elif op_type in ['code_map', 'code_filter']:
        # For code operators, we'd need more sophisticated analysis
        # For now, preserve existing fields
        return type_system.copy()

    return type_system.copy()


def infer_type_from_schema_type(schema_type: Any) -> Dict[str, Any]:
    """
    Infer type dict from schema type description.

    Args:
        schema_type: Schema type description (could be string or dict)

    Returns:
        Inferred type dict
    """
    if isinstance(schema_type, str):
        schema_type_lower = schema_type.lower()
        if any(keyword in schema_type_lower for keyword in ['str', 'string', 'text']):
            return {'type': 'String'}
        elif any(keyword in schema_type_lower for keyword in ['int', 'integer', 'number']):
            return {'type': 'Integer'}
        elif any(keyword in schema_type_lower for keyword in ['float', 'decimal', 'double']):
            return {'type': 'Float'}
        elif any(keyword in schema_type_lower for keyword in ['bool', 'boolean']):
            return {'type': 'Boolean'}
        elif any(keyword in schema_type_lower for keyword in ['list', 'array']):
            return {'type': 'List', 'element_type': {'type': 'Unknown'}}
        elif any(keyword in schema_type_lower for keyword in ['dict', 'object', 'map']):
            return {'type': 'Dict', 'fields': {}}

    # Default to string for unknown types
    return {'type': 'String'}


def format_available_fields_with_types(type_system: TypeSystem) -> str:
    """
    Format available fields with types for LLM prompts.

    Args:
        type_system: The current type system

    Returns:
        Formatted string for use in prompts
    """
    return type_system.format_for_prompt()