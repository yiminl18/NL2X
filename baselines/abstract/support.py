"""
Base System Support
"""

from typing import Set, Dict, List, Tuple
from enum import Enum


class BaseSystem(Enum):
    DOCETL = "docetl"


ALL_ABSTRACT_OPERATORS = {
    'Map', 'Filter', 'Reduce', 'Resolve', 'Join',
    'Rank', 'TopK', 'Extract', 'Cluster',
    'Split', 'Gather', 'Unnest', 'Sample',
    'Index', 'Project', 'Search'
}

BASE_SYSTEM_SUPPORT: Dict[BaseSystem, Set[str]] = {
    BaseSystem.DOCETL: {
        'Map', 'Filter', 'Reduce', 'Resolve', 'Join',
        'Rank', 'TopK', 'Extract', 'Cluster',
        'Split', 'Gather', 'Unnest', 'Sample'
    },
}

BASE_SYSTEM_UNSUPPORTED: Dict[BaseSystem, Set[str]] = {
    BaseSystem.DOCETL: {'Index', 'Project', 'Search'}
}

OPERATOR_DESCRIPTIONS = {
    'Map': 'Transform each record into one or more fields independently using an LLM.',
    'Filter': 'Select records based on condition',
    'Reduce': 'Aggregate records by grouping key',
    'Resolve': 'Merge/deduplicate similar records',
    'Join': 'Combine two datasets based on key',
    'Rank': 'Order records by criteria',
    'TopK': 'Select top K records',
    'Extract': 'Verbatim extract span from text. Cannot analyze or transform the text. Not applicable for multi-field extraction.',
    'Cluster': 'Group similar records into clusters',
    'Split': 'Split records into smaller chunks',
    'Gather': 'Reassemble previously split chunks',
    'Unnest': 'Flatten complex structures (List[...], Dict{...}).',
    'Sample': 'Sample a subset of records',
    'Index': 'Create searchable index for records',
    'Project': 'Select specific fields from records',
    'Search': 'Search indexed records by query'
}


def get_supported_operators(base_system: BaseSystem) -> Set[str]:
    """
    Get the set of operators supported by the specified base system.

    Args:
        base_system: Base system enum value

    Returns:
        Set of supported operator types
    """
    return BASE_SYSTEM_SUPPORT.get(base_system, set()).copy()


def is_operator_supported(operator_type: str, base_system: BaseSystem) -> bool:
    """
    Check if an operator is supported by the base system.

    Args:
        operator_type: Operator type (e.g., 'Map', 'Filter')
        base_system: Base system enum value

    Returns:
        True if supported, False otherwise

    Example:
        >>> is_operator_supported('Map', BaseSystem.DOCETL)
        True
        >>> is_operator_supported('Index', BaseSystem.DOCETL)
        False
    """
    return operator_type in BASE_SYSTEM_SUPPORT.get(base_system, set())


def get_unsupported_operators(base_system: BaseSystem) -> Set[str]:
    """
    Get the set of operators NOT supported by the specified base system.

    Args:
        base_system: Base system enum value

    Returns:
        Set of unsupported operator types

    Example:
        >>> unsupported = get_unsupported_operators(BaseSystem.DOCETL)
        >>> 'Index' in unsupported
        True
    """
    return BASE_SYSTEM_UNSUPPORTED.get(base_system, set()).copy()


def validate_pipeline_compatibility(
    operator_types: List[str],
    base_system: BaseSystem
) -> Tuple[bool, List[str]]:
    """
    Validate whether all operators in a pipeline are compatible with the base system.

    Args:
        operator_types: List of operator types
        base_system: Base system enum value

    Returns:
        (is_compatible, unsupported_operators)
        - is_compatible: True if all operators supported
        - unsupported_operators: List of unsupported operator types

    Example:
        >>> is_valid, unsupported = validate_pipeline_compatibility(
        ...     ['Map', 'Filter', 'Reduce'], BaseSystem.DOCETL
        ... )
        >>> is_valid
        True
        >>> is_valid, unsupported = validate_pipeline_compatibility(
        ...     ['Map', 'Index', 'Filter'], BaseSystem.DOCETL
        ... )
        >>> is_valid
        False
        >>> 'Index' in unsupported
        True
    """
    supported = get_supported_operators(base_system)
    unsupported = [op for op in operator_types if op not in supported]
    return len(unsupported) == 0, unsupported


def format_supported_operators_for_prompt(base_system: BaseSystem) -> str:
    """
    Format supported operator list for LLM prompts.

    Args:
        base_system: Base system enum value

    Returns:
        Formatted string

    Example:
        >>> prompt_str = format_supported_operators_for_prompt(BaseSystem.DOCETL)
        >>> 'Map' in prompt_str
        True
        >>> 'DocETL' in prompt_str
        True
    """
    supported = sorted(get_supported_operators(base_system))
    return f"Available operators for {base_system.value}: {', '.join(supported)}"


def get_operator_description(operator_type: str) -> str:
    """
    Get the description for an operator type.

    Args:
        operator_type: Operator type

    Returns:
        Operator description string

    Example:
        >>> desc = get_operator_description('Map')
        >>> 'Transform' in desc
        True
    """
    return OPERATOR_DESCRIPTIONS.get(operator_type, f"{operator_type} operator")


def format_operators_with_descriptions(base_system: BaseSystem) -> str:
    """
    Format operators with descriptions for LLM prompts.

    Args:
        base_system: Base system enum value

    Returns:
        Formatted string with one operator and description per line

    Example:
        >>> prompt_text = format_operators_with_descriptions(BaseSystem.DOCETL)
        >>> '- Map:' in prompt_text
        True
        >>> '- Index:' in prompt_text  # DocETL doesn't support
        False
    """
    supported = sorted(get_supported_operators(base_system))
    lines = []
    for op_type in supported:
        desc = get_operator_description(op_type)
        lines.append(f"- {op_type}: {desc}")
    return "\n".join(lines)


def get_base_system_info(base_system: BaseSystem) -> Dict[str, any]:
    """
    Get complete information about a base system.

    Args:
        base_system: Base system enum value

    Returns:
        Dictionary containing support information

    Example:
        >>> info = get_base_system_info(BaseSystem.DOCETL)
        >>> info['name']
        'docetl'
        >>> len(info['supported_operators'])
        13
    """
    return {
        'name': base_system.value,
        'supported_operators': get_supported_operators(base_system),
        'unsupported_operators': get_unsupported_operators(base_system),
        'operator_count': len(get_supported_operators(base_system))
    }


def filter_unsupported_operators(
    operators: List[Dict[str, str]],
    base_system: BaseSystem
) -> List[Dict[str, str]]:
    """
    Filter out unsupported operators from a list.

    Args:
        operators: List of operators, each containing a 'type' field
        base_system: Base system enum value

    Returns:
        Filtered list of operators

    Example:
        >>> ops = [
        ...     {'type': 'Map', 'purpose': '...'},
        ...     {'type': 'Index', 'purpose': '...'},
        ...     {'type': 'Filter', 'purpose': '...'}
        ... ]
        >>> filtered = filter_unsupported_operators(ops, BaseSystem.DOCETL)
        >>> len(filtered)
        2
        >>> filtered[0]['type']
        'Map'
    """
    supported = get_supported_operators(base_system)
    return [op for op in operators if op.get('type') in supported]
