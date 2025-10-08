"""
Type tracking utilities for DocETL pipeline generation.

This module provides a comprehensive type system for tracking data types
and field structures throughout the DocETL pipeline generation process.
Uses simple dict-based storage for efficiency and simplicity.
"""

from typing import Any, Dict, List, Optional, Set, Union


class TypeSystem:
    def __init__(self):
        self.root_fields: Dict[str, Dict[str, Any]] = {}
        # Track field sources (which operator introduced each field)
        self.field_sources: Dict[str, str] = {}  # field_path -> operator_name
        # Track field readers (which operators use each field)
        self.field_readers: Dict[str, Set[str]] = {}  # field_path -> set of operator_names
        # Track field history (all operations that affected each field)
        self.field_history: Dict[str, List[Dict[str, Any]]] = {}  # field_path -> [operations]
        # Track overall operation history
        self.operation_history: List[Dict[str, Any]] = []

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
        # Copy all fields with their types
        for field, type_dict in self.root_fields.items():
            new_system.root_fields[field] = dict(type_dict)
        # Copy field sources
        new_system.field_sources = dict(self.field_sources)
        # Deep copy field readers
        for field, readers in self.field_readers.items():
            new_system.field_readers[field] = set(readers)
        # Deep copy field history
        for field, history in self.field_history.items():
            new_system.field_history[field] = [dict(op) for op in history]
        # Deep copy operation history
        new_system.operation_history = [dict(op) for op in self.operation_history]
        return new_system

    def format_for_prompt(self) -> str:
        """Format the type system for inclusion in LLM prompts."""
        if not self.root_fields:
            return "No fields available"

        lines = []
        for field, type_dict in sorted(self.root_fields.items()):
            lines.append(f"  {field}: {type_dict_to_string(type_dict)}")

        return "Available fields:\n" + "\n".join(lines)

    def add_field_with_source(self, field_path: str, type_dict: Dict[str, Any],
                              source_operator: str) -> None:
        """Add a field with its type and source operator to the system."""
        self.root_fields[field_path] = type_dict
        self.field_sources[field_path] = source_operator

        # Initialize field history with creation event
        if field_path not in self.field_history:
            self.field_history[field_path] = []
            self.field_history[field_path].append({
                'operator': source_operator,
                'action': 'created',
                'type': type_dict_to_string(type_dict)
            })

    def get_field_source(self, field_path: str) -> Optional[str]:
        """Get the source operator for a field."""
        return self.field_sources.get(field_path)

    def get_fields_by_source(self, operator_name: str) -> List[str]:
        """Get all fields introduced by a specific operator."""
        return [field for field, source in self.field_sources.items()
                if source == operator_name]

    def get_field_history(self, field_path: str) -> List[Dict[str, Any]]:
        """Get the complete history of operations for a field."""
        return self.field_history.get(field_path, [])

    def record_field_usage(self, field_path: str, operator_name: str) -> None:
        """Record that a field was used by an operator."""
        # Add to readers set
        if field_path not in self.field_readers:
            self.field_readers[field_path] = set()
        self.field_readers[field_path].add(operator_name)

        # Add to field history
        if field_path in self.field_history:
            self.field_history[field_path].append({
                'operator': operator_name,
                'action': 'consumed',
                'timestamp': len(self.operation_history)
            })

    def get_field_lineage(self, field_path: str) -> Dict[str, Any]:
        """Get complete lineage information for a field."""
        return {
            'field': field_path,
            'type': type_dict_to_string(self.root_fields.get(field_path, {})),
            'source': self.field_sources.get(field_path),
            'consumers': list(self.field_readers.get(field_path, set())),
            'history': self.field_history.get(field_path, [])
        }

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

    def get_field_consumers(self, field_path: str) -> List[str]:
        """Get list of operators that consume/read a specific field."""
        return list(self.field_readers.get(field_path, set()))

    def get_unused_fields(self) -> List[str]:
        """Get list of fields that are created but never used."""
        unused = []
        for field_path in self.root_fields:
            if field_path not in self.field_readers or not self.field_readers[field_path]:
                unused.append(field_path)
        return unused

    def get_data_flow(self) -> Dict[str, Any]:
        """Get complete data flow information."""
        flow = {
            'fields': {},
            'operations': self.operation_history
        }

        for field_path in self.root_fields:
            flow['fields'][field_path] = {
                'type': type_dict_to_string(self.root_fields[field_path]),
                'source': self.field_sources.get(field_path),
                'consumers': list(self.field_readers.get(field_path, set()))
            }

        return flow


def type_dict_to_string(type_dict: Dict[str, Any]) -> str:
    """Convert a type dictionary to human-readable string representation."""
    type_name = type_dict.get('type', 'Unknown')

    if type_name in ['String', 'Integer', 'Float', 'Boolean', 'Null', 'Unknown']:
        return type_name

    elif type_name == 'List':
        if type_dict.get('element_type'):
            return f"List[{type_dict_to_string(type_dict['element_type'])}]"
        return "List[Unknown]"

    elif type_name == 'Dict':
        if type_dict.get('fields'):
            field_strs = []
            for field_name, field_type in sorted(type_dict['fields'].items()):
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
    operator_name = operator.get('name', f"map_{operator.get('type', 'unknown')}")

    # Track which fields this operator uses
    used_fields = parse_operator_used_fields(operator)
    for field in used_fields:
        new_system.record_field_usage(field, operator_name)

    produced_fields = []
    if 'output_schema' in operator:
        if isinstance(operator['output_schema'], dict):
            for field_name, field_type in operator['output_schema'].items():
                # Infer type from schema field type description
                type_dict = infer_type_from_schema_type(field_type)
                new_system.add_field_with_source(field_name, type_dict, operator_name)
                produced_fields.append(field_name)

    # Record operation in history with both produced and consumed fields
    new_system.operation_history.append({
        'operator': operator_name,
        'type': 'map',
        'produced_fields': produced_fields,
        'consumed_fields': list(used_fields),
        'timestamp': len(new_system.operation_history)
    })

    return new_system


def apply_extract_operator_transformation(type_system: TypeSystem, operator: Dict[str, Any]) -> TypeSystem:
    """Apply extract operator transformation to type system."""
    new_system = type_system.copy()
    operator_name = operator.get('name', f"extract_{operator.get('type', 'unknown')}")

    # Track which fields this operator uses
    used_fields = parse_operator_used_fields(operator)
    for field in used_fields:
        new_system.record_field_usage(field, operator_name)

    produced_fields = []
    if 'document_keys' in operator and operator['document_keys']:
        for doc_key in operator['document_keys']:
            # Ground truth: suffix is always "_extracted_findings"
            extracted_field = f"{doc_key}_extracted_findings"
            # Extract operations typically produce strings
            new_system.add_field_with_source(extracted_field, {'type': 'String'}, operator_name)
            produced_fields.append(extracted_field)

    # Record operation in history with both produced and consumed fields
    new_system.operation_history.append({
        'operator': operator_name,
        'type': 'extract',
        'produced_fields': produced_fields,
        'consumed_fields': list(used_fields),
        'timestamp': len(new_system.operation_history)
    })

    return new_system


def apply_unnest_operator_transformation(type_system: TypeSystem, operator: Dict[str, Any]) -> TypeSystem:
    """Apply unnest operator transformation to type system."""
    new_system = type_system.copy()
    operator_name = operator.get('name', f"unnest_{operator.get('type', 'unknown')}")

    # Track which fields this operator uses
    used_fields = parse_operator_used_fields(operator)
    for field in used_fields:
        new_system.record_field_usage(field, operator_name)

    produced_fields = []
    unnest_key = operator.get('unnest_key')

    if unnest_key:
        unnest_type = type_system.get_field_type(unnest_key)

        if unnest_type:
            if unnest_type.get('type') == 'List':
                # Ground truth: For list unnest, change list[T] to T
                element_type = unnest_type.get('element_type')
                if element_type:
                    if element_type.get('type') == 'Dict':
                        # list[{name: str, age: int}] becomes name: str, age: int
                        for field_name, field_type in element_type.get('fields', {}).items():
                            new_system.add_field_with_source(field_name, field_type, operator_name)
                            produced_fields.append(field_name)
                    else:
                        # list[str] becomes the unnest_key field with type str
                        new_system.add_field_with_source(unnest_key, element_type, operator_name)
                        produced_fields.append(unnest_key)

            elif unnest_type.get('type') == 'Dict':
                # Ground truth: For dict unnest, flatten all nested fields to top level
                # The unnest_key field itself is removed, fields are promoted
                for field_name, field_type in unnest_type.get('fields', {}).items():
                    new_system.add_field_with_source(field_name, field_type, operator_name)
                    produced_fields.append(field_name)

        # Ground truth: REMOVE the unnest_key field from the type system
        if unnest_key in new_system.root_fields:
            del new_system.root_fields[unnest_key]

    # Record operation in history with both produced and consumed fields
    new_system.operation_history.append({
        'operator': operator_name,
        'type': 'unnest',
        'produced_fields': produced_fields,
        'consumed_fields': list(used_fields),
        'unnest_key': unnest_key,
        'timestamp': len(new_system.operation_history)
    })

    return new_system


def apply_reduce_operator_transformation(type_system: TypeSystem, operator: Dict[str, Any]) -> TypeSystem:
    """Apply reduce operator transformation to type system."""
    new_system = type_system.copy()
    operator_name = operator.get('name', f"reduce_{operator.get('type', 'unknown')}")

    # Track which fields this operator uses
    used_fields = parse_operator_used_fields(operator)
    for field in used_fields:
        new_system.record_field_usage(field, operator_name)

    produced_fields = []
    # Add reduce_key as a grouping field (typically remains the same type)
    if 'reduce_key' in operator and operator['reduce_key']:
        existing_type = type_system.get_field_type(operator['reduce_key'])
        if existing_type:
            # Preserve the field with its original source
            existing_source = type_system.get_field_source(operator['reduce_key']) or 'original'
            new_system.add_field_with_source(operator['reduce_key'], existing_type, existing_source)

    # Add output schema fields
    if 'output_schema' in operator:
        if isinstance(operator['output_schema'], dict):
            for field_name, field_type in operator['output_schema'].items():
                type_dict = infer_type_from_schema_type(field_type)
                new_system.add_field_with_source(field_name, type_dict, operator_name)
                produced_fields.append(field_name)

    # Record operation in history with both produced and consumed fields
    new_system.operation_history.append({
        'operator': operator_name,
        'type': 'reduce',
        'produced_fields': produced_fields,
        'consumed_fields': list(used_fields),
        'timestamp': len(new_system.operation_history)
    })

    return new_system


def apply_resolve_operator_transformation(type_system: TypeSystem, operator: Dict[str, Any]) -> TypeSystem:
    """Apply resolve operator transformation to type system."""
    new_system = type_system.copy()
    operator_name = operator.get('name', f"resolve_{operator.get('type', 'unknown')}")

    # Track which fields this operator uses
    used_fields = parse_operator_used_fields(operator)
    for field in used_fields:
        new_system.record_field_usage(field, operator_name)

    produced_fields = []
    # Ground truth: Resolve uses output.schema fields like Map
    if 'output_schema' in operator:
        if isinstance(operator['output_schema'], dict):
            for field_name, field_type in operator['output_schema'].items():
                type_dict = infer_type_from_schema_type(field_type)
                new_system.add_field_with_source(field_name, type_dict, operator_name)
                produced_fields.append(field_name)

    # Record operation in history with both produced and consumed fields
    new_system.operation_history.append({
        'operator': operator_name,
        'type': 'resolve',
        'produced_fields': produced_fields,
        'consumed_fields': list(used_fields),
        'timestamp': len(new_system.operation_history)
    })

    return new_system


def apply_split_operator_transformation(type_system: TypeSystem, operator: Dict[str, Any]) -> TypeSystem:
    """Apply split operator transformation to type system."""
    new_system = type_system.copy()
    operator_name = operator.get('name', f"split_{operator.get('type', 'unknown')}")

    # Track which fields this operator uses
    used_fields = parse_operator_used_fields(operator)
    for field in used_fields:
        new_system.record_field_usage(field, operator_name)

    # Ground truth: Split operations add:
    # 1. {split_key}_chunk: string
    # 2. {op_name}_id: string (unique identifier for each original document)
    # 3. {op_name}_chunk_num: integer (sequential number of chunk)
    produced_fields = []

    if 'split_key' in operator and operator['split_key']:
        split_key = operator['split_key']
        chunk_field = f"{split_key}_chunk"
        new_system.add_field_with_source(chunk_field, {'type': 'String'}, operator_name)
        produced_fields.append(chunk_field)

    id_field = f"{operator_name}_id"
    chunk_num_field = f"{operator_name}_chunk_num"
    new_system.add_field_with_source(id_field, {'type': 'String'}, operator_name)
    new_system.add_field_with_source(chunk_num_field, {'type': 'Integer'}, operator_name)
    produced_fields.extend([id_field, chunk_num_field])

    # Record operation in history with both produced and consumed fields
    new_system.operation_history.append({
        'operator': operator_name,
        'type': 'split',
        'produced_fields': produced_fields,
        'consumed_fields': list(used_fields),
        'timestamp': len(new_system.operation_history)
    })

    return new_system


def apply_gather_operator_transformation(type_system: TypeSystem, operator: Dict[str, Any]) -> TypeSystem:
    """Apply gather operator transformation to type system."""
    new_system = type_system.copy()
    operator_name = operator.get('name', f"gather_{operator.get('type', 'unknown')}")

    # Track which fields this operator uses
    used_fields = parse_operator_used_fields(operator)
    for field in used_fields:
        new_system.record_field_usage(field, operator_name)

    produced_fields = []
    # Ground truth: Gather produces {content_key}_rendered field
    if 'content_key' in operator and operator['content_key']:
        rendered_field = f"{operator['content_key']}_rendered"
        # Gather typically produces a string
        new_system.add_field_with_source(rendered_field, {'type': 'String'}, operator_name)
        produced_fields.append(rendered_field)

    # Record operation in history with both produced and consumed fields
    new_system.operation_history.append({
        'operator': operator_name,
        'type': 'gather',
        'produced_fields': produced_fields,
        'consumed_fields': list(used_fields),
        'timestamp': len(new_system.operation_history)
    })

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
    elif op_type == 'resolve':
        # Ground truth: Resolve uses output.schema like Map
        return apply_resolve_operator_transformation(type_system, operator)

    elif op_type in ['filter', 'sample', 'topk', 'join']:
        # These operators don't change the type system structure but may use fields
        new_system = type_system.copy()
        operator_name = operator.get('name', f"{op_type}_{len(type_system.operation_history)}")

        # Track field usage
        used_fields = parse_operator_used_fields(operator)
        for field in used_fields:
            new_system.record_field_usage(field, operator_name)

        # Record operation in history
        new_system.operation_history.append({
            'operator': operator_name,
            'type': op_type,
            'produced_fields': [],  # These operators don't produce new fields
            'consumed_fields': list(used_fields),
            'timestamp': len(new_system.operation_history)
        })
        return new_system

    elif op_type == 'rank':
        # Ground truth: Rank adds _rank field (with underscore prefix) to each item
        new_system = type_system.copy()
        operator_name = operator.get('name', f"rank_{len(type_system.operation_history)}")

        # Track field usage
        used_fields = parse_operator_used_fields(operator)
        for field in used_fields:
            new_system.record_field_usage(field, operator_name)

        # Add _rank field (with underscore prefix)
        new_system.add_field_with_source('_rank', {'type': 'Integer'}, operator_name)

        # Record operation in history
        new_system.operation_history.append({
            'operator': operator_name,
            'type': 'rank',
            'produced_fields': ['_rank'],
            'consumed_fields': list(used_fields),
            'timestamp': len(new_system.operation_history)
        })
        return new_system

    elif op_type == 'cluster':
        # Ground truth: Cluster creates a dict with summary_schema fields + distance field
        new_system = type_system.copy()
        operator_name = operator.get('name', f"cluster_{len(type_system.operation_history)}")

        # Track field usage
        used_fields = parse_operator_used_fields(operator)
        for field in used_fields:
            new_system.record_field_usage(field, operator_name)

        produced_fields = []
        # Default output_key is "clusters" if not specified
        output_key = operator.get('output_key', 'clusters')

        # Build the cluster output type: dict with summary_schema fields + distance
        cluster_fields = {}

        # Add fields from summary_schema
        if 'summary_schema' in operator and isinstance(operator['summary_schema'], dict):
            for field_name, field_type in operator['summary_schema'].items():
                cluster_fields[field_name] = infer_type_from_schema_type(field_type)

        # Add distance field
        cluster_fields['distance'] = {'type': 'Float'}

        # Create the dict type
        cluster_type = {'type': 'Dict', 'fields': cluster_fields}
        new_system.add_field_with_source(output_key, cluster_type, operator_name)
        produced_fields.append(output_key)

        # Record operation in history
        new_system.operation_history.append({
            'operator': operator_name,
            'type': 'cluster',
            'produced_fields': produced_fields,
            'consumed_fields': list(used_fields),
            'timestamp': len(new_system.operation_history)
        })
        return new_system

    elif op_type in ['code_map', 'code_filter']:
        # For code operators, we'd need more sophisticated analysis
        # For now, track field usage from prompts but can't detect output fields
        new_system = type_system.copy()
        operator_name = operator.get('name', f"{op_type}_{len(type_system.operation_history)}")

        # Track field usage from prompts
        used_fields = parse_operator_used_fields(operator)
        for field in used_fields:
            new_system.record_field_usage(field, operator_name)

        # Record operation in history
        new_system.operation_history.append({
            'operator': operator_name,
            'type': op_type,
            'produced_fields': [],  # Can't detect without code analysis
            'consumed_fields': list(used_fields),
            'timestamp': len(new_system.operation_history)
        })
        return new_system

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


def parse_operator_used_fields(operator: Dict[str, Any]) -> Dict[str, str]:
    """
    Parse fields used/consumed by an operator and their usage types.

    Args:
        operator: Operator configuration dict

    Returns:
        Dict mapping field names to usage types
    """
    used_fields = {}
    op_type = operator.get('type', '')

    # Extract fields from prompt
    if 'prompt' in operator and isinstance(operator['prompt'], str):
        import re
        field_refs = re.findall(r'\{\{\s*(?:input\.)?(\w+)\s*\}\}', operator['prompt'])
        for field_ref in field_refs:
            if field_ref != 'inputs':  # Skip the special 'inputs' variable
                used_fields[field_ref] = 'in_prompt'

    # Extract fields from comparison_prompt (for resolve)
    if 'comparison_prompt' in operator and isinstance(operator['comparison_prompt'], str):
        import re
        field_refs = re.findall(r'\{\{\s*(?:input\.)?(\w+)\s*\}\}', operator['comparison_prompt'])
        for field_ref in field_refs:
            if field_ref != 'inputs':
                used_fields[field_ref] = 'in_comparison'

    # Extract fields from resolution_prompt (for resolve)
    if 'resolution_prompt' in operator and isinstance(operator['resolution_prompt'], str):
        import re
        field_refs = re.findall(r'\{\{\s*(?:input\.)?(\w+)\s*\}\}', operator['resolution_prompt'])
        for field_ref in field_refs:
            if field_ref != 'inputs':
                used_fields[field_ref] = 'in_resolution'

    # Operator-specific field usage
    if op_type == 'reduce' and 'reduce_key' in operator:
        if operator['reduce_key'] and operator['reduce_key'] != "TO_BE_GENERATED":
            used_fields[operator['reduce_key']] = 'as_reduce_key'

    elif op_type == 'split' and 'split_key' in operator:
        if operator['split_key'] and operator['split_key'] != "TO_BE_GENERATED":
            used_fields[operator['split_key']] = 'as_split_key'

    elif op_type == 'unnest' and 'unnest_key' in operator:
        if operator['unnest_key'] and operator['unnest_key'] != "TO_BE_GENERATED":
            used_fields[operator['unnest_key']] = 'as_unnest_key'

    elif op_type == 'gather' and 'content_key' in operator:
        if operator['content_key'] and operator['content_key'] != "TO_BE_GENERATED":
            used_fields[operator['content_key']] = 'as_content_key'
        if 'doc_id_key' in operator:
            if operator['doc_id_key'] and operator['doc_id_key'] != "TO_BE_GENERATED":
                used_fields[operator['doc_id_key']] = 'as_doc_id_key'
        if 'order_key' in operator:
            if operator['order_key'] and operator['order_key'] != "TO_BE_GENERATED":
                used_fields[operator['order_key']] = 'as_order_key'

    elif op_type == 'rank' and 'input_keys' in operator:
        if isinstance(operator.get('input_keys', []), list):
            for key in operator['input_keys']:
                if key and key != "TO_BE_GENERATED":
                    used_fields[key] = 'as_rank_input'

    elif op_type == 'cluster' and 'embedding_keys' in operator:
        if isinstance(operator.get('embedding_keys', []), list):
            for key in operator['embedding_keys']:
                if key and key != "TO_BE_GENERATED":
                    used_fields[key] = 'as_embedding_key'

    elif op_type == 'topk' and 'keys' in operator:
        if isinstance(operator.get('keys', []), list):
            for key in operator['keys']:
                if key and key != "TO_BE_GENERATED":
                    used_fields[key] = 'as_topk_key'

    elif op_type == 'extract' and 'document_keys' in operator:
        if isinstance(operator.get('document_keys', []), list):
            for key in operator['document_keys']:
                if key and key != "TO_BE_GENERATED":
                    used_fields[key] = 'as_document_key'

    return used_fields


def parse_operator_output_fields(operator: Dict[str, Any]) -> List[str]:
    """
    Parse output fields from an operator's configuration.
    Replaces _parse_operator_output_fields from docetl_step.py.

    Args:
        operator: Operator configuration dict

    Returns:
        List of field names that this operator will add to the data
    """
    output_fields = []
    op_type = operator.get('type', '')

    if op_type == 'extract':
        # Ground truth: Extract always uses "_extracted_findings" suffix
        if 'document_keys' in operator and operator['document_keys'] != "TO_BE_GENERATED":
            if isinstance(operator['document_keys'], list) and operator['document_keys']:
                for doc_key in operator['document_keys']:
                    output_fields.append(f"{doc_key}_extracted_findings")
            else:
                output_fields.append(f"src_extracted_findings")
        return output_fields

    # Handle operators with output schema
    if 'output_schema' in operator:
        if isinstance(operator['output_schema'], dict):
            output_fields.extend(operator['output_schema'].keys())
        elif isinstance(operator['output_schema'], str):
            import re
            matches = re.findall(r'(\w+)\s*:\s*\w+', operator['output_schema'])
            output_fields.extend(matches)

    # Handle specific operator types
    if op_type == 'split':
        # Ground truth: Split creates {split_key}_chunk, {op_name}_id, {op_name}_chunk_num
        operator_name = operator.get('name', 'split')
        if 'split_key' in operator and operator['split_key']:
            output_fields.append(f"{operator['split_key']}_chunk")
        output_fields.append(f"{operator_name}_id")
        output_fields.append(f"{operator_name}_chunk_num")

    elif op_type == 'gather':
        # Ground truth: Gather creates {content_key}_rendered
        if 'content_key' in operator and operator['content_key']:
            output_fields.append(f"{operator['content_key']}_rendered")

    elif op_type == 'rank':
        # Ground truth: Rank adds _rank field
        output_fields.append('_rank')

    elif op_type == 'cluster':
        # Ground truth: Cluster adds output_key field (default "clusters")
        output_key = operator.get('output_key', 'clusters')
        output_fields.append(output_key)

    elif op_type == 'unnest' and 'unnest_key' in operator:
        # Unnest expands existing fields, handled by apply_unnest_operator_transformation
        pass

    return output_fields


def validate_operator_inputs(type_system: TypeSystem, operator: Dict[str, Any]) -> tuple:
    """
    Validate that an operator's referenced fields exist in the type system.

    Args:
        type_system: Current type system with available fields
        operator: Operator configuration to validate

    Returns:
        Tuple of (is_valid: bool, errors: List[str])
    """
    errors = []
    available_fields = type_system.get_all_fields()
    op_type = operator.get('type', '')

    # Check prompt field references
    if 'prompt' in operator and isinstance(operator['prompt'], str):
        import re
        field_refs = re.findall(r'\{\{\s*(?:input\.)?(\w+)\s*\}\}', operator['prompt'])
        for field_ref in field_refs:
            if field_ref not in available_fields and field_ref != 'inputs':
                errors.append(f"Prompt references unavailable field: {field_ref}")

    # Operator-specific field checks
    field_checks = {
        'reduce': ('reduce_key',),
        'split': ('split_key',),
        'unnest': ('unnest_key',),
        'gather': ('content_key', 'doc_id_key'),
    }

    if op_type in field_checks:
        for field_param in field_checks[op_type]:
            if field_param in operator:
                field_value = operator[field_param]
                if field_value != "TO_BE_GENERATED" and field_value not in available_fields:
                    errors.append(f"{op_type} operation references unavailable {field_param}: {field_value}")

    # Check array field references
    array_field_checks = {
        'rank': 'input_keys',
        'cluster': 'embedding_keys',
        'topk': 'keys',
    }

    if op_type in array_field_checks:
        field_param = array_field_checks[op_type]
        if field_param in operator and isinstance(operator[field_param], list):
            for field_value in operator[field_param]:
                if field_value not in available_fields:
                    errors.append(f"{op_type} operation references unavailable {field_param}: {field_value}")

    return len(errors) == 0, errors


def format_available_fields_with_types(type_system: TypeSystem) -> str:
    """
    Format available fields with types for LLM prompts.

    Args:
        type_system: The current type system

    Returns:
        Formatted string for use in prompts
    """
    return type_system.format_for_prompt()