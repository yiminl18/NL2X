#!/usr/bin/env python3
"""
Dynamic comparator for abstraction layer pipelines using LLM-based evaluation.
Implements a 4-step methodology to determine pipeline equivalence:
1. Pipeline -> Intent (P1 -> I1, P2 -> I2)
2. Intent Alignment (I1 vs I2 -> S_IntentAlign)
3. Implementation Equivalence (P1 vs P2 -> S_ImplEquiv)
4. Cross-Verification (P1 -> I2, P2 -> I1 -> S_CrossVerif)

Final score: S_Equiv = w_intent * S_IntentAlign + w_impl * S_ImplEquiv + w_cross * S_CrossVerif

Usage:
    from dynamic_comparator import compare_pipelines_dynamic
    import json

    # Load pipelines
    with open("pipeline1.json", "r") as f:
        data1 = json.load(f)
    with open("pipeline2.json", "r") as f:
        data2 = json.load(f)

    # Compare pipelines
    result = compare_pipelines_dynamic(
        operators1=data1['operators'],
        operators2=data2['operators']
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

# Schema for Step 1: Pipeline -> Intent
STEP1_SCHEMA = {
    "type": "object",
    "properties": {
        "pipeline1_intent": {
            "type": "string",
            "description": "Clear description of Pipeline 1's intended functionality"
        },
        "pipeline2_intent": {
            "type": "string",
            "description": "Clear description of Pipeline 2's intended functionality"
        },
        "analysis": {
            "type": "object",
            "properties": {
                "pipeline1_goal": {"type": "string"},
                "pipeline1_key_steps": {"type": "array", "items": {"type": "string"}},
                "pipeline2_goal": {"type": "string"},
                "pipeline2_key_steps": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["pipeline1_goal", "pipeline1_key_steps", "pipeline2_goal", "pipeline2_key_steps"],
            "additionalProperties": False
        }
    },
    "required": ["pipeline1_intent", "pipeline2_intent", "analysis"],
    "additionalProperties": False
}

# Schema for Step 2: Intent Alignment
STEP2_SCHEMA = {
    "type": "object",
    "properties": {
        "dimensional_scores": {
            "type": "object",
            "properties": {
                "core_task_similarity": {
                    "type": "object",
                    "properties": {
                        "score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "rationale": {"type": "string"}
                    },
                    "required": ["score", "rationale"],
                    "additionalProperties": False
                },
                "domain_alignment": {
                    "type": "object",
                    "properties": {
                        "score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "rationale": {"type": "string"}
                    },
                    "required": ["score", "rationale"],
                    "additionalProperties": False
                },
                "analytical_goal": {
                    "type": "object",
                    "properties": {
                        "score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "rationale": {"type": "string"}
                    },
                    "required": ["score", "rationale"],
                    "additionalProperties": False
                }
            },
            "required": ["core_task_similarity", "domain_alignment", "analytical_goal"],
            "additionalProperties": False
        },
        "intent_alignment_score": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "description": "Weighted average: 0.5*core_task + 0.3*domain + 0.2*analytical"
        }
    },
    "required": ["dimensional_scores", "intent_alignment_score"],
    "additionalProperties": False
}

# Schema for Step 3: Implementation Equivalence
STEP3_SCHEMA = {
    "type": "object",
    "properties": {
        "dimensional_scores": {
            "type": "object",
            "properties": {
                "data_transformation": {
                    "type": "object",
                    "properties": {
                        "score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "rationale": {"type": "string"}
                    },
                    "required": ["score", "rationale"],
                    "additionalProperties": False
                },
                "filter_criteria": {
                    "type": "object",
                    "properties": {
                        "score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "rationale": {"type": "string"}
                    },
                    "required": ["score", "rationale"],
                    "additionalProperties": False
                },
                "aggregation_method": {
                    "type": "object",
                    "properties": {
                        "score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "rationale": {"type": "string"}
                    },
                    "required": ["score", "rationale"],
                    "additionalProperties": False
                },
                "edge_case_handling": {
                    "type": "object",
                    "properties": {
                        "score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "rationale": {"type": "string"}
                    },
                    "required": ["score", "rationale"],
                    "additionalProperties": False
                }
            },
            "required": ["data_transformation", "filter_criteria", "aggregation_method", "edge_case_handling"],
            "additionalProperties": False
        },
        "implementation_equivalence_score": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "description": "Weighted average: 0.4*transform + 0.3*filter + 0.2*aggregation + 0.1*edge"
        },
        "key_differences": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of key implementation differences"
        }
    },
    "required": ["dimensional_scores", "implementation_equivalence_score", "key_differences"],
    "additionalProperties": False
}

# Schema for Step 4: Cross-Verification
STEP4_SCHEMA = {
    "type": "object",
    "properties": {
        "p1_implements_p2_intent": {
            "type": "boolean",
            "description": "Can Pipeline 1 achieve Pipeline 2's intent?"
        },
        "p2_implements_p1_intent": {
            "type": "boolean",
            "description": "Can Pipeline 2 achieve Pipeline 1's intent?"
        },
        "bidirectional_equivalence": {
            "type": "boolean",
            "description": "Both directions are equivalent"
        },
        "asymmetries": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of asymmetric behaviors"
        },
        "cross_verification_score": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "description": "1.0 if bidirectional, 0.5 if unidirectional, 0.0 if no equivalence"
        }
    },
    "required": ["p1_implements_p2_intent", "p2_implements_p1_intent", "bidirectional_equivalence", "asymmetries", "cross_verification_score"],
    "additionalProperties": False
}


# ============================================================================
# Prompt Templates
# ============================================================================

PIPELINE_TO_INTENT_PROMPT = Template("""
# [Role]
You are an expert data scientist analyzing two data processing pipelines to understand their intended functionality.

# [Task]
Carefully analyze both pipeline definitions and infer the high-level intent of each. Focus on:
- What is the core analytical task?
- What domain/type of data is being processed?
- What is the ultimate goal or output?

# [Pipeline Definitions]

## Pipeline 1 (JSON format):
$pipeline1_json

## Pipeline 2 (JSON format):
$pipeline2_json

# [Instructions]
For each pipeline, provide:
1. A concise intent statement (1-2 sentences)
2. Analysis components: main goal and key processing steps
""")

INTENT_ALIGNMENT_PROMPT = Template("""
# [Role]
You are a meticulous evaluator comparing the high-level intents of two data processing pipelines.

# [Task]
Compare the intents across multiple dimensions and provide similarity scores.

# [Intents to Compare]

## Pipeline 1 Intent:
$pipeline1_intent

## Pipeline 2 Intent:
$pipeline2_intent

# [Evaluation Dimensions]

1. **Core Task Similarity (Weight: 50%)**
   - Do both pipelines perform the same fundamental task? (e.g., both "calculate average", both "filter and group")
   - Score 0.0-1.0

2. **Domain Alignment (Weight: 30%)**
   - Do both pipelines work on the same type/domain of data? (e.g., both process "identity theft statistics")
   - Score 0.0-1.0

3. **Analytical Goal (Weight: 20%)**
   - Do both pipelines aim for the same analytical outcome? (e.g., both produce "summary statistics")
   - Score 0.0-1.0

For each dimension, provide a score and brief rationale.
Calculate the final intent_alignment_score as the weighted average.
""")

IMPLEMENTATION_EQUIVALENCE_PROMPT = Template("""
# [Role]
You are a QA analyst verifying if two pipelines are functionally equivalent at the implementation level.

# [Task]
Compare the implementation details of both pipelines to determine if they produce equivalent results.

# [Pipeline Definitions]

## Pipeline 1:
$pipeline1_json

## Pipeline 2:
$pipeline2_json

# [Evaluation Dimensions]

1. **Data Transformation Logic (Weight: 40%)**
   - Do both pipelines apply the same transformations to the data?
   - Even if using different operators, do they achieve the same effect?
   - Score 0.0-1.0

2. **Filter/Selection Criteria (Weight: 30%)**
   - Are the filtering conditions identical or equivalent?
   - Score 0.0-1.0

3. **Aggregation/Computation Method (Weight: 20%)**
   - Do both use the same aggregation/calculation approach?
   - Score 0.0-1.0

4. **Edge Case Handling (Weight: 10%)**
   - Do both handle edge cases (nulls, empty data, etc.) similarly?
   - Score 0.0-1.0

Calculate implementation_equivalence_score as weighted average.
List key differences between implementations.
""")

CROSS_VERIFICATION_PROMPT = Template("""
# [Role]
You are a verification expert checking for asymmetric behavior between two pipelines.

# [Task]
Determine if the pipelines are bidirectionally equivalent:
- Can Pipeline 1 achieve what Pipeline 2 intends?
- Can Pipeline 2 achieve what Pipeline 1 intends?

# [Intents]

Pipeline 1 Intent: $pipeline1_intent
Pipeline 2 Intent: $pipeline2_intent

# [Pipeline Implementations]

## Pipeline 1:
$pipeline1_json

## Pipeline 2:
$pipeline2_json

# [Instructions]
Answer the following:
1. p1_implements_p2_intent: true/false
2. p2_implements_p1_intent: true/false
3. bidirectional_equivalence: true if both are true
4. asymmetries: list any one-way capabilities
5. cross_verification_score: 1.0 if bidirectional, 0.5 if unidirectional, 0.0 otherwise
""")


# Default weights
DEFAULT_INTENT_WEIGHT = 0.4
DEFAULT_IMPLEMENTATION_WEIGHT = 0.4
DEFAULT_CROSS_WEIGHT = 0.2
DEFAULT_EQUIVALENCE_THRESHOLD = 0.7


class AbstractDynamicComparator:
    """Dynamic comparator using LLM-based evaluation for abstraction layer pipelines."""

    def __init__(self,
                 intent_weight: float = DEFAULT_INTENT_WEIGHT,
                 implementation_weight: float = DEFAULT_IMPLEMENTATION_WEIGHT,
                 cross_verification_weight: float = DEFAULT_CROSS_WEIGHT,
                 equivalence_threshold: float = DEFAULT_EQUIVALENCE_THRESHOLD,
                 save_intermediate: bool = True,
                 use_mock: bool = False):
        """
        Initialize the dynamic comparator.

        Args:
            intent_weight: Weight for intent alignment score (0.0-1.0)
            implementation_weight: Weight for implementation equivalence score (0.0-1.0)
            cross_verification_weight: Weight for cross-verification score (0.0-1.0)
            equivalence_threshold: Threshold for determining equivalence (default: 0.7)
            save_intermediate: Whether to save intermediate responses to files
            use_mock: Whether to use mock responses for testing
        """
        self.intent_weight = intent_weight
        self.implementation_weight = implementation_weight
        self.cross_verification_weight = cross_verification_weight
        self.equivalence_threshold = equivalence_threshold
        self.save_intermediate = save_intermediate
        self.use_mock = use_mock
        self.intermediate_dir = None

        if self.save_intermediate:
            self.intermediate_base_dir = Path("comparator_intermediate")
            self.intermediate_base_dir.mkdir(exist_ok=True)

        # Validate weights
        total_weight = intent_weight + implementation_weight + cross_verification_weight
        if abs(total_weight - 1.0) > 1e-6:
            raise ValueError("Sum of weights must equal 1.0")

    def _save_intermediate(self, filename: str, content: Any):
        """Save intermediate content to file."""
        if not self.save_intermediate or not self.intermediate_dir:
            return

        file_path = self.intermediate_dir / filename

        if isinstance(content, (dict, list)):
            content_str = json.dumps(content, indent=2, ensure_ascii=False)
        else:
            content_str = str(content)

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content_str)

        print(f"  Saved intermediate: {file_path}")

    def compare(self, operators1: List[Dict[str, Any]], operators2: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compare two pipelines using 4-step LLM methodology.

        Args:
            operators1: First pipeline operators list
            operators2: Second pipeline operators list

        Returns:
            Dict containing equivalence score and detailed step results
        """
        try:
            # Create session-specific intermediate directory
            if self.save_intermediate:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                self.intermediate_dir = self.intermediate_base_dir / timestamp
                self.intermediate_dir.mkdir(exist_ok=True)

            # Convert operators lists to JSON strings for LLM
            pipeline1_json = json.dumps({"operators": operators1}, indent=2, ensure_ascii=False)
            pipeline2_json = json.dumps({"operators": operators2}, indent=2, ensure_ascii=False)

            # Save pipelines
            if self.save_intermediate:
                self._save_intermediate("pipeline1.json", pipeline1_json)
                self._save_intermediate("pipeline2.json", pipeline2_json)

            # Step 1: Pipeline -> Intent
            step1_result = self._pipeline_to_intent(pipeline1_json, pipeline2_json)
            if "error" in step1_result:
                return self._build_error_result("Step 1 failed", step1_result, {})

            # Step 2: Intent Alignment
            step2_result = self._intent_alignment(
                step1_result["pipeline1_intent"],
                step1_result["pipeline2_intent"]
            )
            if "error" in step2_result:
                return self._build_error_result("Step 2 failed", step1_result, step2_result)

            # Step 3: Implementation Equivalence
            step3_result = self._implementation_equivalence(pipeline1_json, pipeline2_json)
            if "error" in step3_result:
                return self._build_error_result("Step 3 failed", step1_result, step2_result, step3_result)

            # Step 4: Cross-Verification
            step4_result = self._cross_verification(
                step1_result["pipeline1_intent"],
                step1_result["pipeline2_intent"],
                pipeline1_json,
                pipeline2_json
            )
            if "error" in step4_result:
                return self._build_error_result("Step 4 failed", step1_result, step2_result, step3_result, step4_result)

            # Calculate final equivalence score
            intent_score = step2_result["intent_alignment_score"]
            impl_score = step3_result["implementation_equivalence_score"]
            cross_score = step4_result["cross_verification_score"]

            equivalence_score = (
                self.intent_weight * intent_score +
                self.implementation_weight * impl_score +
                self.cross_verification_weight * cross_score
            )

            are_equivalent = equivalence_score >= self.equivalence_threshold

            # Generate recommendation
            recommendation = self._generate_recommendation(
                equivalence_score, are_equivalent, step3_result.get("key_differences", [])
            )

            final_result = {
                "equivalence_score": equivalence_score,
                "are_equivalent": are_equivalent,
                "intent_alignment_score": intent_score,
                "implementation_equivalence_score": impl_score,
                "cross_verification_score": cross_score,
                "weights": {
                    "intent_weight": self.intent_weight,
                    "implementation_weight": self.implementation_weight,
                    "cross_verification_weight": self.cross_verification_weight
                },
                "equivalence_threshold": self.equivalence_threshold,
                "key_differences": step3_result.get("key_differences", []),
                "recommendation": recommendation,
                "step_results": {
                    "step1_pipeline_to_intent": step1_result,
                    "step2_intent_alignment": step2_result,
                    "step3_implementation_equivalence": step3_result,
                    "step4_cross_verification": step4_result
                }
            }

            if self.save_intermediate:
                self._save_intermediate("final_result.json", final_result)
                print(f"\nAll intermediate files saved to: {self.intermediate_dir}")

            return final_result

        except Exception as e:
            return {
                "equivalence_score": 0.0,
                "are_equivalent": False,
                "error": f"Comparator error: {str(e)}",
                "step_results": {}
            }

    def _build_error_result(self, error_msg: str, *step_results) -> Dict[str, Any]:
        """Build error result with partial step results."""
        return {
            "equivalence_score": 0.0,
            "are_equivalent": False,
            "error": error_msg,
            "step_results": {f"step{i+1}": result for i, result in enumerate(step_results) if result}
        }

    def _generate_recommendation(self, score: float, are_equiv: bool, differences: List[str]) -> str:
        """Generate human-readable recommendation."""
        if are_equiv:
            if score >= 0.9:
                return "Pipelines are highly equivalent and can be considered functionally identical."
            else:
                return f"Pipelines are functionally equivalent (score: {score:.2f}) despite minor implementation differences."
        else:
            if score >= 0.5:
                return f"Pipelines are partially equivalent (score: {score:.2f}). Key differences: {'; '.join(differences[:2])}"
            else:
                return f"Pipelines are not equivalent (score: {score:.2f}). They serve different purposes or use fundamentally different approaches."

    def _pipeline_to_intent(self, pipeline1_json: str, pipeline2_json: str) -> Dict[str, Any]:
        """Step 1: Infer intents from both pipelines."""
        print("Step 1: Pipeline -> Intent (P1 -> I1, P2 -> I2)")
        try:
            prompt = PIPELINE_TO_INTENT_PROMPT.substitute(
                pipeline1_json=pipeline1_json,
                pipeline2_json=pipeline2_json
            )

            if self.save_intermediate:
                self._save_intermediate("step1_prompt.txt", prompt)

            response = self._call_llm(prompt, STEP1_SCHEMA)

            if self.save_intermediate:
                self._save_intermediate("step1_response_raw.txt", response)

            result = json.loads(response)

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

    def _intent_alignment(self, intent1: str, intent2: str) -> Dict[str, Any]:
        """Step 2: Check alignment between intents."""
        print("Step 2: Intent Alignment (I1 vs I2 -> S_IntentAlign)")
        try:
            prompt = INTENT_ALIGNMENT_PROMPT.substitute(
                pipeline1_intent=intent1,
                pipeline2_intent=intent2
            )

            if self.save_intermediate:
                self._save_intermediate("step2_prompt.txt", prompt)

            response = self._call_llm(prompt, STEP2_SCHEMA)

            if self.save_intermediate:
                self._save_intermediate("step2_response_raw.txt", response)

            result = json.loads(response)

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

    def _implementation_equivalence(self, pipeline1_json: str, pipeline2_json: str) -> Dict[str, Any]:
        """Step 3: Check implementation-level equivalence."""
        print("Step 3: Implementation Equivalence (P1 vs P2 -> S_ImplEquiv)")
        try:
            prompt = IMPLEMENTATION_EQUIVALENCE_PROMPT.substitute(
                pipeline1_json=pipeline1_json,
                pipeline2_json=pipeline2_json
            )

            if self.save_intermediate:
                self._save_intermediate("step3_prompt.txt", prompt)

            response = self._call_llm(prompt, STEP3_SCHEMA)

            if self.save_intermediate:
                self._save_intermediate("step3_response_raw.txt", response)

            result = json.loads(response)

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

    def _cross_verification(self, intent1: str, intent2: str,
                           pipeline1_json: str, pipeline2_json: str) -> Dict[str, Any]:
        """Step 4: Cross-verify bidirectional equivalence."""
        print("Step 4: Cross-Verification (P1 -> I2, P2 -> I1)")
        try:
            prompt = CROSS_VERIFICATION_PROMPT.substitute(
                pipeline1_intent=intent1,
                pipeline2_intent=intent2,
                pipeline1_json=pipeline1_json,
                pipeline2_json=pipeline2_json
            )

            if self.save_intermediate:
                self._save_intermediate("step4_prompt.txt", prompt)

            response = self._call_llm(prompt, STEP4_SCHEMA)

            if self.save_intermediate:
                self._save_intermediate("step4_response_raw.txt", response)

            result = json.loads(response)

            if self.save_intermediate:
                self._save_intermediate("step4_result.json", result)

            return result

        except json.JSONDecodeError as e:
            error_result = {"error": f"Failed to parse LLM response as JSON: {str(e)}"}
            if self.save_intermediate:
                self._save_intermediate("step4_error.json", error_result)
            return error_result
        except Exception as e:
            error_result = {"error": f"Step 4 error: {str(e)}"}
            if self.save_intermediate:
                self._save_intermediate("step4_error.json", error_result)
            return error_result

    def _call_llm(self, prompt: str, schema: Dict[str, Any]) -> str:
        """Call LLM with the given prompt and schema."""
        if self.use_mock:
            # Return mock responses for testing
            return json.dumps({"mock": "response"})

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


def compare_pipelines_dynamic(
    operators1: List[Dict[str, Any]],
    operators2: List[Dict[str, Any]],
    intent_weight: float = DEFAULT_INTENT_WEIGHT,
    implementation_weight: float = DEFAULT_IMPLEMENTATION_WEIGHT,
    cross_verification_weight: float = DEFAULT_CROSS_WEIGHT,
    equivalence_threshold: float = DEFAULT_EQUIVALENCE_THRESHOLD,
    save_intermediate: bool = True,
    use_mock: bool = False
) -> Dict[str, Any]:
    """
    Convenience function to compare pipeline equivalence.

    Args:
        operators1: First pipeline operators list
        operators2: Second pipeline operators list
        intent_weight: Weight for intent alignment (default: 0.4)
        implementation_weight: Weight for implementation equivalence (default: 0.4)
        cross_verification_weight: Weight for cross-verification (default: 0.2)
        equivalence_threshold: Threshold for determining equivalence (default: 0.7)
        save_intermediate: Whether to save intermediate responses to files
        use_mock: Whether to use mock responses for testing

    Returns:
        Dict with equivalence_score and detailed results
    """
    comparator = AbstractDynamicComparator(
        intent_weight,
        implementation_weight,
        cross_verification_weight,
        equivalence_threshold,
        save_intermediate,
        use_mock
    )
    return comparator.compare(operators1, operators2)


def main():
    """Command-line interface for the dynamic comparator."""
    if len(sys.argv) < 3:
        print("Usage: python3 dynamic_comparator.py <pipeline1_json_path> <pipeline2_json_path> [intent_weight] [impl_weight] [cross_weight] [--no-save-intermediate] [--use-mock]")
        print("Example: python3 dynamic_comparator.py pipeline1.json pipeline2.json 0.4 0.4 0.2")
        sys.exit(1)

    pipeline1_file = sys.argv[1]
    pipeline2_file = sys.argv[2]

    intent_weight = DEFAULT_INTENT_WEIGHT
    impl_weight = DEFAULT_IMPLEMENTATION_WEIGHT
    cross_weight = DEFAULT_CROSS_WEIGHT
    save_intermediate = True
    use_mock = False

    # Parse arguments
    i = 3
    weight_count = 0
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == '--no-save-intermediate':
            save_intermediate = False
        elif arg == '--use-mock':
            use_mock = True
        elif not arg.startswith('--'):
            try:
                weight = float(arg)
                if weight_count == 0:
                    intent_weight = weight
                elif weight_count == 1:
                    impl_weight = weight
                elif weight_count == 2:
                    cross_weight = weight
                weight_count += 1
            except ValueError:
                print(f"Error: Invalid weight '{arg}'. Must be a float.")
                sys.exit(1)
        i += 1

    # Load pipelines
    try:
        with open(pipeline1_file, 'r', encoding='utf-8') as f:
            pipeline1_data = json.load(f)
        with open(pipeline2_file, 'r', encoding='utf-8') as f:
            pipeline2_data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON: {e}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"Error: File not found: {e}", file=sys.stderr)
        sys.exit(1)

    if 'operators' not in pipeline1_data or 'operators' not in pipeline2_data:
        print("Error: Pipeline files must contain 'operators' field", file=sys.stderr)
        sys.exit(1)

    operators1 = pipeline1_data['operators']
    operators2 = pipeline2_data['operators']

    try:
        result = compare_pipelines_dynamic(
            operators1=operators1,
            operators2=operators2,
            intent_weight=intent_weight,
            implementation_weight=impl_weight,
            cross_verification_weight=cross_weight,
            save_intermediate=save_intermediate,
            use_mock=use_mock
        )

        print(json.dumps(result, indent=2))

        sys.exit(0 if result.get('are_equivalent', False) else 1)

    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
