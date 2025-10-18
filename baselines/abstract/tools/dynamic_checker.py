#!/usr/bin/env python3
"""
Dynamic checker for abstraction layer pipelines using LLM-based evaluation.
Implements a 3-step methodology with separation of concerns:
1. Pipeline -> Question (P -> Q')
2. Intent Alignment (Q vs Q' -> S_IntentAlign)
3. Constraint Adherence (Q vs P -> S_ConstraintAdherence)

Final score: S_Sem = w_intent * S_IntentAlign + w_constraint * S_ConstraintAdherence

Usage:
    from dynamic_checker import check_pipeline_dynamic
    import json

    # Load pipeline from JSON file
    with open("abstract_pipeline.json", "r") as f:
        pipeline_data = json.load(f)

    # Check pipeline operators against question
    result = check_pipeline_dynamic(
        question="Report the average number of reported identity thefts...",
        operators=pipeline_data['operators']
    )
"""

import json
import sys
import os
from typing import Dict, Any, Optional, List
from string import Template
from pathlib import Path
from datetime import datetime

# Import litellm client
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from model.litellm_client import llm_call


# ============================================================================
# JSON Schema Definitions for Structured Output
# ============================================================================

# Schema for Step 1: Pipeline -> Question
STEP1_SCHEMA = {
    "type": "object",
    "properties": {
        "inferred_query": {
            "type": "string",
            "description": "A clear, natural language question representing the pipeline's main strategic goal."
        },
        "analysis_components": {
            "type": "object",
            "properties": {
                "main_goal": {
                    "type": "string",
                    "description": "The primary objective of the pipeline"
                },
                "data_sources_used": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of dataset names or descriptions"
                },
                "key_processing_steps": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Summary of major transformation or filtering steps"
                }
            },
            "required": ["main_goal", "data_sources_used", "key_processing_steps"],
            "additionalProperties": False
        }
    },
    "required": ["inferred_query", "analysis_components"],
    "additionalProperties": False
}

# Schema for Step 2: Intent Alignment
STEP2_SCHEMA = {
    "type": "object",
    "properties": {
        "evaluation_summary": {
            "type": "object",
            "properties": {
                "core_task_domain": {
                    "type": "object",
                    "properties": {
                        "score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "rationale": {"type": "string"}
                    },
                    "required": ["score", "rationale"],
                    "additionalProperties": False
                },
                "filtering_selection": {
                    "type": "object",
                    "properties": {
                        "score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "rationale": {"type": "string"}
                    },
                    "required": ["score", "rationale"],
                    "additionalProperties": False
                },
                "transformation_calculation": {
                    "type": "object",
                    "properties": {
                        "score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "rationale": {"type": "string"}
                    },
                    "required": ["score", "rationale"],
                    "additionalProperties": False
                },
                "output_format_constraints": {
                    "type": "object",
                    "properties": {
                        "score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "rationale": {"type": "string"}
                    },
                    "required": ["score", "rationale"],
                    "additionalProperties": False
                }
            },
            "required": ["core_task_domain", "filtering_selection", "transformation_calculation", "output_format_constraints"],
            "additionalProperties": False
        },
        "intent_alignment_score": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "description": "The weighted average of the dimensional scores"
        }
    },
    "required": ["evaluation_summary", "intent_alignment_score"],
    "additionalProperties": False
}

# Schema for Step 3: Constraint Adherence
STEP3_SCHEMA = {
    "type": "object",
    "properties": {
        "constraint_verification_list": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "constraint_description": {
                        "type": "string",
                        "description": "Description of the constraint extracted from the query"
                    },
                    "verification_status": {
                        "type": "string",
                        "enum": ["VERIFIED", "PARTIALLY_VERIFIED", "INCORRECTLY_IMPLEMENTED", "NOT_FOUND"]
                    },
                    "evidence": {
                        "type": "string",
                        "description": "The specific part of the pipeline that implements or fails to implement this constraint"
                    },
                    "rationale": {
                        "type": "string",
                        "description": "Explanation of why the status was assigned"
                    },
                    "score": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "description": "1.0 for VERIFIED, 0.5 for PARTIALLY_VERIFIED, 0.0 for INCORRECTLY_IMPLEMENTED or NOT_FOUND"
                    }
                },
                "required": ["constraint_description", "verification_status", "evidence", "rationale", "score"],
                "additionalProperties": False
            }
        },
        "constraint_adherence_score": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "description": "The simple average of the scores from the constraint verification list"
        }
    },
    "required": ["constraint_verification_list", "constraint_adherence_score"],
    "additionalProperties": False
}


# ============================================================================
# Prompt Templates
# ============================================================================

PIPELINE_TO_QUESTION_PROMPT = Template("""
# [Role]
You are an expert data scientist and system analyst. Your task is to reverse-engineer a data analysis pipeline written in an abstraction layer format (JSON with operators) to infer the high-level natural language question it was designed to answer.

# [Task]
Carefully analyze the provided pipeline definition. Synthesize its purpose into a concise and precise natural language question (we will call this Q'). Your focus should be on the overall strategic goal, not the low-level implementation details.

# [Analysis Instructions]
1.  **Data Source Analysis:** Identify the primary data sources and their conceptual meaning.
2.  **Operator Sequence Analysis:** Examine the sequence of operations to understand the main data flow and transformations. What is the core analytical objective (e.g., averaging, filtering, joining)?
3.  **Identify Intent-Defining Constraints:** Extract only the most critical constraints that define the core scope of the analysis, such as the primary entities being analyzed (e.g., "metropolitan areas") or fundamental filtering criteria (e.g., "with populations over a million").
4.  **Synthesize High-Level Question:** Combine your analysis into a coherent natural language question. **It is acceptable if minor implementation details, such as specific rounding rules, text normalization methods, or precise calculation formulas (e.g., linear interpolation), are summarized generally or omitted.**

# [Pipeline Definition to Analyze]
Here is the abstraction layer pipeline (JSON format):
$pipeline_json_definition
""")

INTENT_ALIGNMENT_CHECK_PROMPT = Template("""
# [Role]
You are a meticulous and impartial evaluator. Your task is to assess the high-level semantic alignment between an Original Query (Q) and an Inferred Query (Q') that represents a pipeline's strategic goal.

# [Task]
Compare Q and Q' across several strategic dimensions. For each dimension, provide a similarity score from 0.0 (complete mismatch) to 1.0 (perfect alignment) and a brief justification. Your evaluation should focus on the core intent rather than minor implementation details.

# [Evaluation Dimensions & Weights]
1.  **Core Task & Domain Alignment (Weight: 50%):**
    - Does Q' correctly identify the primary analytical task (e.g., averaging, summing) and the conceptual domain (e.g., identity theft, population statistics)? This is the most critical dimension.
    - **Score (0.0 - 1.0):**
    - **Rationale:**

2.  **Data Filtering & Selection Criteria Alignment (Weight: 30%):**
    - Does Q' accurately capture the fundamental filtering conditions that define the dataset's scope (e.g., focusing on large cities, specific timeframes)?
    - **Score (0.0 - 1.0):**
    - **Rationale:**

3.  **Data Transformation Alignment (Weight: 15%):**
    - Does Q' generally reflect the major data transformations mentioned in Q (e.g., the concept of estimating a future value, even if the exact formula isn't mentioned)?
    - **Score (0.0 - 1.0):**
    - **Rationale:**

4.  **Minor Constraints & Output Format Alignment (Weight: 5%):**
    - Does Q' capture any high-level instructions about the output? Discrepancies in minor details (like exact rounding) should only be lightly penalized.
    - **Score (0.0 - 1.0):**
    - **Rationale:**

# [Queries to Compare]
## Original Query (Q):
$original_query

## Inferred Query (Q'):
$inferred_query_from_prompt_1
""")

CONSTRAINT_CHECK_PROMPT = Template("""
# [Role]
You are a meticulous Quality Assurance (QA) analyst and code reviewer. Your sole task is to verify if a given abstraction layer pipeline correctly implements all the constraints specified in a natural language query.

# [Task]
1.  **Deconstruct the Query:** First, carefully read the Original Query (Q) and create a checklist of every explicit and implicit constraint. A constraint is any specific instruction that limits, directs, or formats the data, calculation, or output (e.g., numerical comparisons, rounding rules, specific formulas, text normalization steps).
2.  **Audit the Pipeline:** For each item on your checklist, meticulously scan the entire Pipeline Definition (P) to find the operator configuration or prompt snippet that is supposed to implement it.
3.  **Evaluate Implementation:** Judge whether the implementation is correct, partially correct, incorrect, or missing entirely.

# [Instructions]
- Be precise. Refer to specific operator names, properties, or parts of prompts from the pipeline as evidence for your judgment.
- Do not evaluate the overall strategic logic of the pipeline. Your focus is strictly on its adherence to the specific constraints you have identified from the query.

# [Inputs]
## Original Query (Q):
$original_query

## Pipeline Definition (P) - Abstraction Layer JSON:
$pipeline_json_definition
""")


# Default weights for combining scores
DEFAULT_INTENT_WEIGHT = 0.6  # Strategic alignment weight
DEFAULT_CONSTRAINT_WEIGHT = 0.4  # Tactical constraint weight


class AbstractDynamicChecker:
    """Dynamic checker using LLM-based evaluation for abstraction layer pipelines."""

    def __init__(self, intent_weight: float = DEFAULT_INTENT_WEIGHT,
                 constraint_weight: float = DEFAULT_CONSTRAINT_WEIGHT,
                 save_intermediate: bool = True,
                 use_mock: bool = False):
        """
        Initialize the dynamic checker.

        Args:
            intent_weight: Weight for intent alignment score (0.0-1.0)
            constraint_weight: Weight for constraint adherence score (0.0-1.0)
            save_intermediate: Whether to save intermediate responses to files
            use_mock: Whether to use mock responses for testing (default: False)
        """
        self.intent_weight = intent_weight
        self.constraint_weight = constraint_weight
        self.save_intermediate = save_intermediate
        self.use_mock = use_mock
        self.intermediate_dir = None

        if self.save_intermediate:
            # Create checker_intermediate directory if it doesn't exist
            self.intermediate_base_dir = Path("checker_intermediate")
            self.intermediate_base_dir.mkdir(exist_ok=True)

        # Validate weights
        if abs(intent_weight + constraint_weight - 1.0) > 1e-6:
            raise ValueError("Intent weight and constraint weight must sum to 1.0")

    def _save_intermediate(self, filename: str, content: Any):
        """Save intermediate content to file."""
        if not self.save_intermediate or not self.intermediate_dir:
            return

        file_path = self.intermediate_dir / filename

        # Handle different content types
        if isinstance(content, (dict, list)):
            content_str = json.dumps(content, indent=2, ensure_ascii=False)
        else:
            content_str = str(content)

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content_str)

        print(f"  Saved intermediate: {file_path}")

    def check(self, question: str, operators: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Check pipeline against question using 3-step methodology.

        Args:
            question: Original natural language query
            operators: List of operator dictionaries (from pipeline JSON)

        Returns:
            Dict containing semantic score and detailed step results
        """
        try:
            # Create session-specific intermediate directory
            if self.save_intermediate:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                self.intermediate_dir = self.intermediate_base_dir / timestamp
                self.intermediate_dir.mkdir(exist_ok=True)

                # Save the question
                self._save_intermediate("question.txt", question)

            # Convert operators list to JSON string for LLM
            pipeline_content = json.dumps({"operators": operators}, indent=2, ensure_ascii=False)

            # Save pipeline content
            if self.save_intermediate:
                self._save_intermediate("pipeline.json", pipeline_content)

            # Step 1: Pipeline -> Question (P -> Q')
            step1_result = self._pipeline_to_question(pipeline_content)

            if "error" in step1_result:
                return {
                    "semantic_score": 0.0,
                    "error": f"Step 1 failed: {step1_result['error']}",
                    "step_results": {"step1_pipeline_to_question": step1_result}
                }

            # Step 2: Intent Alignment (Q vs Q' -> S_IntentAlign)
            step2_result = self._intent_alignment_check(question, step1_result["inferred_query"])

            if "error" in step2_result:
                return {
                    "semantic_score": 0.0,
                    "intent_alignment_score": 0.0,
                    "error": f"Step 2 failed: {step2_result['error']}",
                    "step_results": {
                        "step1_pipeline_to_question": step1_result,
                        "step2_intent_alignment": step2_result
                    }
                }

            # Step 3: Constraint Adherence (Q vs P -> S_ConstraintAdherence)
            step3_result = self._constraint_check(question, pipeline_content)

            if "error" in step3_result:
                return {
                    "semantic_score": 0.0,
                    "intent_alignment_score": step2_result["intent_alignment_score"],
                    "constraint_adherence_score": 0.0,
                    "error": f"Step 3 failed: {step3_result['error']}",
                    "step_results": {
                        "step1_pipeline_to_question": step1_result,
                        "step2_intent_alignment": step2_result,
                        "step3_constraint_adherence": step3_result
                    }
                }

            # Calculate final semantic score
            intent_score = step2_result["intent_alignment_score"]
            constraint_score = step3_result["constraint_adherence_score"]
            semantic_score = self.intent_weight * intent_score + self.constraint_weight * constraint_score

            final_result = {
                "semantic_score": semantic_score,
                "intent_alignment_score": intent_score,
                "constraint_adherence_score": constraint_score,
                "weights": {
                    "intent_weight": self.intent_weight,
                    "constraint_weight": self.constraint_weight
                },
                "step_results": {
                    "step1_pipeline_to_question": step1_result,
                    "step2_intent_alignment": step2_result,
                    "step3_constraint_adherence": step3_result
                }
            }

            # Save final result
            if self.save_intermediate:
                self._save_intermediate("final_result.json", final_result)
                print(f"\nAll intermediate files saved to: {self.intermediate_dir}")

            return final_result

        except Exception as e:
            return {
                "semantic_score": 0.0,
                "error": f"Checker error: {str(e)}",
                "step_results": {}
            }

    def _pipeline_to_question(self, pipeline_content: str) -> Dict[str, Any]:
        """Step 1: Reverse-engineer pipeline to infer intended question."""
        print("Step 1: Pipeline -> Question (P -> Q')")
        try:
            prompt = PIPELINE_TO_QUESTION_PROMPT.substitute(pipeline_json_definition=pipeline_content)

            # Save prompt
            if self.save_intermediate:
                self._save_intermediate("step1_prompt.txt", prompt)

            response = self._call_llm(prompt, STEP1_SCHEMA)

            # Save raw response
            if self.save_intermediate:
                self._save_intermediate("step1_response_raw.txt", response)

            # Parse JSON response
            result = json.loads(response)

            # Save parsed result
            if self.save_intermediate:
                self._save_intermediate("step1_result.json", result)

            return result

        except json.JSONDecodeError as e:
            error_result = {"error": f"Failed to parse LLM response as JSON: {str(e)}"}
            if self.save_intermediate:
                self._save_intermediate("step1_error.json", error_result)
            return error_result
        except Exception as e:
            error_result = {"error": f"Step 1 error: {str(e)}"}
            if self.save_intermediate:
                self._save_intermediate("step1_error.json", error_result)
            return error_result

    def _intent_alignment_check(self, original_query: str, inferred_query: str) -> Dict[str, Any]:
        """Step 2: Check alignment between original and inferred queries."""
        print("Step 2: Intent Alignment (Q vs Q' -> S_IntentAlign)")
        try:
            prompt = INTENT_ALIGNMENT_CHECK_PROMPT.substitute(
                original_query=original_query,
                inferred_query_from_prompt_1=inferred_query
            )

            # Save prompt
            if self.save_intermediate:
                self._save_intermediate("step2_prompt.txt", prompt)

            response = self._call_llm(prompt, STEP2_SCHEMA)

            # Save raw response
            if self.save_intermediate:
                self._save_intermediate("step2_response_raw.txt", response)

            # Parse JSON response
            result = json.loads(response)

            # Save parsed result
            if self.save_intermediate:
                self._save_intermediate("step2_result.json", result)

            return result

        except json.JSONDecodeError as e:
            error_result = {"error": f"Failed to parse LLM response as JSON: {str(e)}"}
            if self.save_intermediate:
                self._save_intermediate("step2_error.json", error_result)
            return error_result
        except Exception as e:
            error_result = {"error": f"Step 2 error: {str(e)}"}
            if self.save_intermediate:
                self._save_intermediate("step2_error.json", error_result)
            return error_result

    def _constraint_check(self, original_query: str, pipeline_content: str) -> Dict[str, Any]:
        """Step 3: Check if pipeline correctly implements all constraints."""
        print("Step 3: Constraint Adherence (Q vs P -> S_ConstraintAdherence)")
        try:
            prompt = CONSTRAINT_CHECK_PROMPT.substitute(
                original_query=original_query,
                pipeline_json_definition=pipeline_content
            )

            # Save prompt
            if self.save_intermediate:
                self._save_intermediate("step3_prompt.txt", prompt)

            response = self._call_llm(prompt, STEP3_SCHEMA)

            # Save raw response
            if self.save_intermediate:
                self._save_intermediate("step3_response_raw.txt", response)

            # Parse JSON response
            result = json.loads(response)

            # Save parsed result
            if self.save_intermediate:
                self._save_intermediate("step3_result.json", result)

            return result

        except json.JSONDecodeError as e:
            error_result = {"error": f"Failed to parse LLM response as JSON: {str(e)}"}
            if self.save_intermediate:
                self._save_intermediate("step3_error.json", error_result)
            return error_result
        except Exception as e:
            error_result = {"error": f"Step 3 error: {str(e)}"}
            if self.save_intermediate:
                self._save_intermediate("step3_error.json", error_result)
            return error_result

    def _call_llm(self, prompt: str, schema: Dict[str, Any]) -> str:
        """Call LLM with the given prompt and schema."""
        if self.use_mock:
            # Return mock JSON responses for testing
            if "Pipeline Definition to Analyze" in prompt or "Here is the abstraction layer pipeline" in prompt:
                # Mock response for Step 1 (Pipeline -> Question)
                return json.dumps({
                    "inferred_query": "What is the average number of reported identity thefts for metropolitan areas with population over one million?",
                    "analysis_components": {
                        "main_goal": "Calculate average identity thefts for large metropolitan areas",
                        "data_sources_used": ["identity_theft_data"],
                        "key_processing_steps": [
                            "Extract metropolitan area statistics from input data",
                            "Filter areas with population over 1 million",
                            "Calculate average identity thefts"
                        ]
                    }
                })
            elif "Original Query (Q):" in prompt and "Inferred Query (Q'):" in prompt:
                # Mock response for Step 2 (Intent Alignment)
                return json.dumps({
                    "evaluation_summary": {
                        "core_task_domain": {"score": 0.9, "rationale": "Both queries focus on identity theft statistics for metropolitan areas"},
                        "filtering_selection": {"score": 0.85, "rationale": "Both specify population threshold of one million"},
                        "transformation_calculation": {"score": 0.8, "rationale": "Both require averaging calculation"},
                        "output_format_constraints": {"score": 0.7, "rationale": "Output format generally aligned"}
                    },
                    "intent_alignment_score": 0.83
                })
            elif "constraint_verification_list" in prompt or "Pipeline Definition (P)" in prompt:
                # Mock response for Step 3 (Constraint Check)
                return json.dumps({
                    "constraint_verification_list": [
                        {
                            "constraint_description": "Metropolitan areas must be larger than one million in population",
                            "verification_status": "VERIFIED",
                            "evidence": "Filter operator with population threshold",
                            "rationale": "Code correctly implements > 1 million population filter",
                            "score": 1.0
                        },
                        {
                            "constraint_description": "Calculate average identity thefts",
                            "verification_status": "VERIFIED",
                            "evidence": "Reduce operation with averaging logic",
                            "rationale": "Reduce operation correctly calculates average",
                            "score": 1.0
                        }
                    ],
                    "constraint_adherence_score": 1.0
                })
            else:
                return json.dumps({"mock_response": "Unable to determine prompt type"})

        # Real LLM call with schema
        try:
            response = llm_call(
                messages=prompt,
                schema=schema,
                max_tokens=4000,
                temperature=0.0,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0
            )
            return response
        except Exception as e:
            raise RuntimeError(f"LLM call failed: {str(e)}")


def check_pipeline_dynamic(question: str,
                          operators: List[Dict[str, Any]],
                          intent_weight: float = DEFAULT_INTENT_WEIGHT,
                          constraint_weight: float = DEFAULT_CONSTRAINT_WEIGHT,
                          save_intermediate: bool = True,
                          use_mock: bool = False) -> Dict[str, Any]:
    """
    Convenience function to check pipeline semantics against a question.

    Args:
        question: Natural language query to evaluate against
        operators: List of operator dictionaries (from pipeline JSON)
        intent_weight: Weight for strategic alignment (default: 0.6)
        constraint_weight: Weight for constraint adherence (default: 0.4)
        save_intermediate: Whether to save intermediate responses to files
        use_mock: Whether to use mock responses for testing (default: False)

    Returns:
        Dict with semantic_score and detailed results
    """
    checker = AbstractDynamicChecker(intent_weight, constraint_weight, save_intermediate, use_mock)
    return checker.check(question, operators)


def main():
    """Command-line interface for the dynamic checker."""
    if len(sys.argv) < 3:
        print("Usage: python3 dynamic_checker.py <question> <pipeline_json_path> [intent_weight] [--no-save-intermediate] [--use-mock]")
        print("Example: python3 dynamic_checker.py 'What is the average?' abstract_pipeline.json 0.8")
        print("Example with mock: python3 dynamic_checker.py 'What is the average?' abstract_pipeline.json 0.8 --use-mock")
        sys.exit(1)

    question = sys.argv[1]
    pipeline_file = sys.argv[2]
    intent_weight = DEFAULT_INTENT_WEIGHT
    save_intermediate = True
    use_mock = False

    # Parse arguments
    i = 3
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == '--no-save-intermediate':
            save_intermediate = False
        elif arg == '--use-mock':
            use_mock = True
        elif not arg.startswith('--'):
            # Assume it's the intent weight
            try:
                intent_weight = float(arg)
            except ValueError:
                print(f"Error: Invalid intent weight '{arg}'. Must be a float.")
                sys.exit(1)
        i += 1

    constraint_weight = 1.0 - intent_weight

    # Load pipeline file
    pipeline_path = Path(pipeline_file)
    if not pipeline_path.exists():
        print(f"Error: Pipeline file not found: {pipeline_file}", file=sys.stderr)
        sys.exit(1)

    try:
        with open(pipeline_path, 'r', encoding='utf-8') as f:
            pipeline_data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in pipeline file: {e}", file=sys.stderr)
        sys.exit(1)

    # Extract operators
    if 'operators' not in pipeline_data:
        print("Error: Pipeline file must contain 'operators' field", file=sys.stderr)
        sys.exit(1)

    operators = pipeline_data['operators']

    try:
        result = check_pipeline_dynamic(
            question=question,
            operators=operators,
            intent_weight=intent_weight,
            constraint_weight=constraint_weight,
            save_intermediate=save_intermediate,
            use_mock=use_mock
        )

        print(json.dumps(result, indent=2))

        # Exit with appropriate code
        semantic_score = result.get("semantic_score", 0.0)
        sys.exit(0 if semantic_score > 0.5 else 1)

    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
