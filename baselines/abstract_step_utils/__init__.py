"""
Abstract Step Utilities

Utility module for abstract_step baseline, providing prompts and schema tracking
for abstract layer pipeline generation. Uses existing converter from
baselines/abstract/convert/docetl/ for DocETL conversion.
"""

from .prompts import (
    get_abstract_operator_prompt,
    ABSTRACT_MAP_PROMPT,
    ABSTRACT_FILTER_PROMPT,
    ABSTRACT_REDUCE_PROMPT
)

from .schema_tracker import (
    format_fields_for_prompt
)

from .instructions import (
    OPERATOR_SELECTION_RULES,
    MAX_INSTRUCTIONS,
    DEFAULT_IMPORTANCE_SCORE,
    format_selection_rules,
    get_operator_selection_rules,
    get_top_rules_by_score,
    update_rule_score,
    get_rule_scores
)

__all__ = [
    # Prompts
    'get_abstract_operator_prompt',
    'ABSTRACT_MAP_PROMPT',
    'ABSTRACT_FILTER_PROMPT',
    'ABSTRACT_REDUCE_PROMPT',

    # Schema formatting
    'format_fields_for_prompt',

    # Dynamic instructions
    'OPERATOR_SELECTION_RULES',
    'MAX_INSTRUCTIONS',
    'DEFAULT_IMPORTANCE_SCORE',
    'format_selection_rules',
    'get_operator_selection_rules',
    'get_top_rules_by_score',
    'update_rule_score',
    'get_rule_scores',
]
