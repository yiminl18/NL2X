"""
Dynamic Instruction Management

This module manages the rules and instructions for operator selection.
It provides a centralized location for maintaining operator selection rules,
making them easier to update and potentially customize dynamically in the future.

Each rule has an importance score that can be used for future dynamic rule selection.
"""

from typing import List, Dict, Union


# =============================================================================
# Configuration Constants
# =============================================================================

# Maximum number of instructions to include in prompts
MAX_INSTRUCTIONS = 20

# Default importance score for new rules
DEFAULT_IMPORTANCE_SCORE = 1


# =============================================================================
# Operator Selection Rules
# =============================================================================

OPERATOR_SELECTION_RULES: List[Dict[str, Union[str, Dict, int]]] = [
    {
        "rule": "Only select operators that are actually needed",
        "score": 1
    },
    {
        "rule": "Consider the data flow between operators",
        "score": 1
    },
    {
        "rule": "For aggregation tasks, use Reduce",
        "score": 1
    },
    {
        "rule": "For deduplication, use Resolve",
        "score": 1
    },
    {
        "rule": "For splitting long text, use Split followed by Gather if context is needed",
        "score": 1
    },
    {
        "rule": "Use Unnest when you need to expand arrays or nested structures",
        "score": 1
    },
    {
        "rule": {
            "title": "Map and Extract have similar functions - do not use both together",
            "items": [
                "Map: Multiple fields, analysis, transformation, structured data",
                "Extract: SINGLE string field ONLY, cannot specify the output schema (verbatim text, no analysis/transformation)",
                "Conclusion: Use Map unless you are exactly sure the needed information is continuous verbatim text and only ONE field is needed"
            ]
        },
        "score": 1
    },
    {
        "rule": "Chain operators logically - outputs of one operator should match inputs expected by the next",
        "score": 1
    }
]


# =============================================================================
# Formatting Functions
# =============================================================================

def format_selection_rules(rules: List[Dict] = None) -> str:
    """
    Format operator selection rules as a numbered list.

    The score field is used internally but not displayed in the formatted output.

    Args:
        rules: List of rule dictionaries. If None, uses OPERATOR_SELECTION_RULES.

    Returns:
        str: Formatted rules text ready for insertion into prompts
    """
    if rules is None:
        rules = OPERATOR_SELECTION_RULES

    lines = []
    rule_number = 1

    for rule_entry in rules:
        rule_content = rule_entry["rule"]
        # score = rule_entry["score"]  # Available for future use, not displayed

        if isinstance(rule_content, str):
            # Simple string rule
            lines.append(f"{rule_number}. {rule_content}")
            rule_number += 1
        elif isinstance(rule_content, dict):
            # Complex rule with title and sub-items
            title = rule_content.get("title", "")
            items = rule_content.get("items", [])

            lines.append(f"{rule_number}. {title}")
            for item in items:
                lines.append(f"   - {item}")
            rule_number += 1

    return "\n".join(lines)


# =============================================================================
# Rule Selection Functions
# =============================================================================

def get_top_rules_by_score(max_rules: int = MAX_INSTRUCTIONS) -> List[Dict]:
    """
    Get the most important rules based on their scores.

    This function can be used in the future for dynamic rule selection,
    where only the most relevant rules are included based on context.

    Args:
        max_rules: Maximum number of rules to return (default: MAX_INSTRUCTIONS)

    Returns:
        List of rule dictionaries sorted by score (highest first)
    """
    sorted_rules = sorted(
        OPERATOR_SELECTION_RULES,
        key=lambda x: x["score"],
        reverse=True
    )
    return sorted_rules[:max_rules]


def get_operator_selection_rules() -> str:
    return format_selection_rules()


# =============================================================================
# Rule Management Functions
# =============================================================================

def update_rule_score(rule_index: int, new_score: int) -> None:
    """
    Update the importance score of a specific rule.

    This function can be used to dynamically adjust rule importance
    based on empirical effectiveness or user feedback.

    Args:
        rule_index: Index of the rule in OPERATOR_SELECTION_RULES
        new_score: New importance score to assign
    """
    if 0 <= rule_index < len(OPERATOR_SELECTION_RULES):
        OPERATOR_SELECTION_RULES[rule_index]["score"] = new_score
    else:
        raise IndexError(f"Rule index {rule_index} out of range")


def get_rule_scores() -> Dict[int, int]:
    """
    Get a dictionary mapping rule indices to their scores.

    Returns:
        Dict[int, int]: Mapping of rule index to score

    Example output:
        {0: 1, 1: 1, 2: 1, ..., 6: 2, ...}
    """
    return {i: rule["score"] for i, rule in enumerate(OPERATOR_SELECTION_RULES)}
