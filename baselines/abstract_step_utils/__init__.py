"""
Abstract Step Utilities

Utility module for abstract_step baseline, providing prompts and type system
for abstract layer pipeline generation. Uses existing converter from
baselines/abstract/convert/docetl/ for DocETL conversion.
"""

from .prompt_abstract import (
    ABSTRACT_OPERATOR_SELECTION_PROMPT,
    get_operator_selection_prompt,
    get_abstract_operator_prompt,
    ABSTRACT_MAP_PROMPT,
    ABSTRACT_FILTER_PROMPT,
    ABSTRACT_REDUCE_PROMPT
)

from .type_utils_abstract import (
    AbstractTypeSystem,
    infer_type_from_sample,
    format_fields_for_prompt
)

from .dynamic_inst import (
    OPERATOR_SELECTION_RULES,
    MAX_INSTRUCTIONS,
    DEFAULT_IMPORTANCE_SCORE,
    format_selection_rules,
    get_operator_selection_rules,
    get_top_rules_by_score,
    update_rule_score,
    get_rule_scores
)

from ..docetl_step_utils.ui import DocETLUserInterface

__all__ = [
    # Prompts
    'ABSTRACT_OPERATOR_SELECTION_PROMPT',
    'get_operator_selection_prompt',
    'get_abstract_operator_prompt',
    'ABSTRACT_MAP_PROMPT',
    'ABSTRACT_FILTER_PROMPT',
    'ABSTRACT_REDUCE_PROMPT',

    # Type system
    'AbstractTypeSystem',
    'infer_type_from_sample',
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

    # UI
    'DocETLUserInterface'
]
