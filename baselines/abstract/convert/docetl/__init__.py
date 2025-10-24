"""
This module provides utilities to convert between DocETL operators and abstract operators.
"""

# Import all public functions from converter
from .converter import (
    DOCETL_TO_ABSTRACT_TYPE_MAP,
    docetl_to_abstract,
    docetl_pipeline_to_abstract,
    abstract_to_docetl,
    yaml_to_abstract_pipeline,
    operator_to_dict,
    dict_to_operator,
    yaml_to_abstract_json,
    abstract_json_to_yaml,
    LiteralDumper
)

# Import schema functions and type mapper from docetl_support
from ....docetl_support.schema_tracking.field_extraction import (
    extract_fields_from_jinja2 as _extract_fields_from_jinja2,
    extract_fields_from_python_code as _extract_fields_from_python_code
)
from ....docetl_support.schema_tracking.type_mapper import DocETLTypeMapper
from ....docetl_support.schema_tracking import DocETLSchemaTracker

# Create compatibility functions for the old API
def _extract_input_schema(docetl_operator, field_types=None, dataset_schema=None):
    """Extract input schema - compatibility wrapper for DocETLSchemaTracker."""
    tracker = DocETLSchemaTracker({})
    return tracker.get_input_schema(docetl_operator, field_types, dataset_schema)

def _extract_output_schema(docetl_operator, cumulative_field_types=None):
    """Extract output schema - compatibility wrapper for DocETLSchemaTracker."""
    tracker = DocETLSchemaTracker({})
    return tracker.get_output_schema(docetl_operator, cumulative_field_types)

# Import path management APIs (keep local for now)
from .path_manager import (
    get_dataset_paths,
    set_dataset_paths,
    set_dataset_path,
    get_output_path,
    set_output_path,
    get_intermediate_dir,
    set_intermediate_dir,
    apply_path_mappings,
    get_all_file_paths
)

# Import validation functions from docetl_support
from ....docetl_support.validation import (
    check_pipeline_file,
    check_pipeline_string,
)

__all__ = [
    # Main conversion functions
    'docetl_to_abstract',
    'abstract_to_docetl',
    'docetl_pipeline_to_abstract',

    # YAML/Pipeline conversion
    'yaml_to_abstract_pipeline',

    # Serialization
    'operator_to_dict',
    'dict_to_operator',

    # Standalone conversion
    'yaml_to_abstract_json',
    'abstract_json_to_yaml',

    # Constants and utilities
    'DOCETL_TO_ABSTRACT_TYPE_MAP',
    'LiteralDumper',

    # Schema functions and type mapping (advanced)
    '_extract_fields_from_jinja2',
    '_extract_fields_from_python_code',
    'DocETLTypeMapper',
    '_extract_input_schema',
    '_extract_output_schema',

    # Path management APIs
    'get_dataset_paths',
    'set_dataset_paths',
    'set_dataset_path',
    'get_output_path',
    'set_output_path',
    'get_intermediate_dir',
    'set_intermediate_dir',
    'apply_path_mappings',
    'get_all_file_paths',

    # Validation APIs
    'check_pipeline_file',
    'check_pipeline_string',
]
