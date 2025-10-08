"""
DocETL Converter Module

This module provides utilities to convert between DocETL operators and abstract operators.
It is organized into two sub-modules:

- schema: Type and schema tracking utilities
- converter: Main conversion logic and YAML/JSON handling

All public functions are re-exported here for backward compatibility.
"""

# Import all public functions from converter
from .converter import (
    DOCETL_TO_ABSTRACT_TYPE_MAP,
    docetl_to_abstract,
    docetl_pipeline_to_abstract,
    abstract_to_docetl,
    yaml_to_abstract_pipeline,
    yaml_to_abstract_pipeline_dag,
    convert_yaml_or_operators,
    operator_to_dict,
    dict_to_operator,
    yaml_to_abstract_json,
    abstract_json_to_yaml,
    LiteralDumper
)

# Import schema functions (these are typically used internally, but exposed for advanced usage)
from .schema import (
    _extract_fields_from_jinja2,
    _extract_fields_from_python_code,
    _parse_docetl_type,
    _parse_docetl_schema_to_type_dict,
    _type_dict_to_docetl_string,
    _extract_input_schema,
    _extract_output_schema
)

__all__ = [
    # Main conversion functions
    'docetl_to_abstract',
    'abstract_to_docetl',
    'docetl_pipeline_to_abstract',

    # YAML/Pipeline conversion
    'yaml_to_abstract_pipeline',
    'yaml_to_abstract_pipeline_dag',
    'convert_yaml_or_operators',

    # Serialization
    'operator_to_dict',
    'dict_to_operator',

    # Standalone conversion
    'yaml_to_abstract_json',
    'abstract_json_to_yaml',

    # Constants and utilities
    'DOCETL_TO_ABSTRACT_TYPE_MAP',
    'LiteralDumper',

    # Schema functions (advanced)
    '_extract_fields_from_jinja2',
    '_extract_fields_from_python_code',
    '_parse_docetl_type',
    '_parse_docetl_schema_to_type_dict',
    '_type_dict_to_docetl_string',
    '_extract_input_schema',
    '_extract_output_schema',
]
