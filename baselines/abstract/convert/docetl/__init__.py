"""
DocETL Converter Module

This module provides utilities to convert between DocETL operators and abstract operators.
It is organized into three sub-modules:

- schema: Type and schema tracking utilities
- converter: Main conversion logic and YAML/JSON handling
- path_manager: Path management APIs for dataset and output paths
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

# Import schema functions and type mapper
from .schema import (
    _extract_fields_from_jinja2,
    _extract_fields_from_python_code,
    DocETLTypeMapper,
    _extract_input_schema,
    _extract_output_schema
)

# Import path management APIs
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
]
