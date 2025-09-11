#!/usr/bin/env python3
"""
Dynamic checker for DocETL pipelines using LLM-based evaluation.
Implements a 3-step methodology with separation of concerns:
1. Pipeline -> Question (P -> Q')
2. Intent Alignment (Q vs Q' -> S_IntentAlign)
3. Constraint Adherence (Q vs P -> S_ConstraintAdherence)

Final score: S_Sem = w_intent * S_IntentAlign + w_constraint * S_ConstraintAdherence

Usage:
    from dynamic_checker import check_pipeline_dynamic
    
    result = check_pipeline_dynamic(
        question="Report the average number of reported identity thefts...",
        pipeline_path="pipeline.yaml"
    )
    
    # Or with YAML string
    result = check_pipeline_dynamic(
        question="...",
        pipeline_yaml="default_model: gpt-4o-mini\n..."
    )
"""

import json
import sys
import yaml
from typing import Dict, Any, Optional

# Prompt templates
PIPELINE_TO_QUESTION_PROMPT = """
# [Role]
You are an expert data scientist and system analyst. Your task is to reverse-engineer a data analysis pipeline written in a declarative format (e.g., YAML) to infer the high-level natural language question it was designed to answer.

# [Task]
Carefully analyze the provided pipeline definition. Synthesize its purpose into a concise and precise natural language question (we will call this Q'). Your focus should be on the overall strategic goal, not the low-level implementation details.

# [Analysis Instructions]
1.  **Data Source Analysis:** Identify the primary data sources and their conceptual meaning.
2.  **Operator Sequence Analysis:** Examine the sequence of operations to understand the main data flow and transformations. What is the core analytical objective (e.g., averaging, filtering, joining)?
3.  **Identify Intent-Defining Constraints:** Extract only the most critical constraints that define the core scope of the analysis, such as the primary entities being analyzed (e.g., "metropolitan areas") or fundamental filtering criteria (e.g., "with populations over a million").
4.  **Synthesize High-Level Question:** Combine your analysis into a coherent natural language question. **It is acceptable if minor implementation details, such as specific rounding rules, text normalization methods, or precise calculation formulas (e.g., linear interpolation), are summarized generally or omitted.**

# [Output Format]
Produce a JSON object with the following structure. Do not add any extra commentary outside of the JSON object.

{{
  "inferred_query": "A clear, natural language question representing the pipeline's main strategic goal.",
  "analysis_components": {{
    "main_goal": "What is the primary objective of the pipeline? (e.g., 'Calculate the average of a specific metric')",
    "data_sources_used": ["A list of dataset names or descriptions."],
    "key_processing_steps": [
      "A summary of the first major transformation or filtering step that defines the intent.",
      "A summary of the second major transformation or filtering step.",
      "..."
    ]
  }}
}}

# [Pipeline Definition to Analyze]
Here is the pipeline:
{pipeline_yaml_definition}
"""

INTENT_ALIGNMENT_CHECK_PROMPT = """
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

# [Output Format]
Produce a JSON object with the following structure. Do not add any extra commentary outside of the JSON object.
```json
{{
  "evaluation_summary": {{
    "core_task_domain": {{ "score": <float>, "rationale": "Brief justification." }},
    "filtering_selection": {{ "score": <float>, "rationale": "Brief justification." }},
    "transformation_calculation": {{ "score": <float>, "rationale": "Brief justification." }},
    "output_format_constraints": {{ "score": <float>, "rationale": "Brief justification." }}
  }},
  "intent_alignment_score": <float> // The weighted average of the dimensional scores.
}}
```
# [Queries to Compare]
## Original Query (Q):
{original_query}

## Inferred Query (Q'):
{inferred_query_from_prompt_1}
"""

CONSTRAINT_CHECK_PROMPT = """
# [Role]
You are a meticulous Quality Assurance (QA) analyst and code reviewer. Your sole task is to verify if a given data analysis pipeline correctly implements all the constraints specified in a natural language query.

# [Task]
1.  **Deconstruct the Query:** First, carefully read the Original Query (Q) and create a checklist of every explicit and implicit constraint. A constraint is any specific instruction that limits, directs, or formats the data, calculation, or output (e.g., numerical comparisons, rounding rules, specific formulas, text normalization steps).
2.  **Audit the Pipeline:** For each item on your checklist, meticulously scan the entire Pipeline Definition (P) to find the code or prompt snippet that is supposed to implement it.
3.  **Evaluate Implementation:** Judge whether the implementation is correct, partially correct, incorrect, or missing entirely.

# [Instructions]
- Be precise. Refer to specific operator names, code snippets, or parts of a prompt from the pipeline as evidence for your judgment.
- Do not evaluate the overall strategic logic of the pipeline. Your focus is strictly on its adherence to the specific constraints you have identified from the query.

# [Output Format]
Produce a JSON object with the following structure. Do not add any extra commentary outside of the JSON object.

{{
  "constraint_verification_list": [
    {{
      "constraint_description": "A description of the constraint extracted from the query (e.g., 'Metropolitan areas must be larger than one million in population').",
      "verification_status": "ENUM('VERIFIED', 'PARTIALLY_VERIFIED', 'INCORRECTLY_IMPLEMENTED', 'NOT_FOUND')",
      "evidence": "The specific part of the pipeline (e.g., 'code_filter: filter_large_metro_areas') that implements or fails to implement this constraint. Cite 'N/A' if not found.",
      "rationale": "A brief explanation of why the status was assigned. For example, 'The code correctly uses `doc['population_2023'] > 1_000_000`.' or 'The code incorrectly uses `>=`, which violates the strict `>` requirement.'",
      "score": <float> // 1.0 for VERIFIED, 0.5 for PARTIALLY_VERIFIED, 0.0 for INCORRECTLY_IMPLEMENTED or NOT_FOUND
    }}
  ],
  "constraint_adherence_score": <float> // The simple average of the scores from the list above.
}}

# [Inputs]
## Original Query (Q):
{original_query}

## Pipeline Definition (P):
{pipeline_yaml_definition}
"""

# Default weights for combining scores
DEFAULT_INTENT_WEIGHT = 0.6  # Strategic alignment weight
DEFAULT_CONSTRAINT_WEIGHT = 0.4  # Tactical constraint weight


class DocETLDynamicChecker:
    """Dynamic checker using LLM-based evaluation for DocETL pipelines."""
    
    def __init__(self, llm_client=None, intent_weight: float = DEFAULT_INTENT_WEIGHT, 
                 constraint_weight: float = DEFAULT_CONSTRAINT_WEIGHT):
        """
        Initialize the dynamic checker.
        
        Args:
            llm_client: LLM client for making API calls (if None, will use mock responses)
            intent_weight: Weight for intent alignment score (0.0-1.0)
            constraint_weight: Weight for constraint adherence score (0.0-1.0)
        """
        self.llm_client = llm_client
        self.intent_weight = intent_weight
        self.constraint_weight = constraint_weight
        
        # Validate weights
        if abs(intent_weight + constraint_weight - 1.0) > 1e-6:
            raise ValueError("Intent weight and constraint weight must sum to 1.0")
    
    def check(self, question: str, pipeline_yaml: str = None, pipeline_path: str = None) -> Dict[str, Any]:
        """
        Check pipeline against question using 3-step methodology.
        
        Args:
            question: Original natural language query
            pipeline_yaml: Pipeline YAML content (if provided)
            pipeline_path: Path to pipeline YAML file (if pipeline_yaml not provided)
        
        Returns:
            Dict containing semantic score and detailed step results
        """
        try:
            # Load pipeline content
            if pipeline_yaml is not None:
                pipeline_content = pipeline_yaml
            elif pipeline_path is not None:
                with open(pipeline_path, 'r', encoding='utf-8') as f:
                    pipeline_content = f.read()
            else:
                raise ValueError("Either pipeline_yaml or pipeline_path must be provided")
            
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
            
            return {
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
            
        except Exception as e:
            return {
                "semantic_score": 0.0,
                "error": f"Checker error: {str(e)}",
                "step_results": {}
            }
    
    def _pipeline_to_question(self, pipeline_content: str) -> Dict[str, Any]:
        """Step 1: Reverse-engineer pipeline to infer intended question."""
        try:
            prompt = PIPELINE_TO_QUESTION_PROMPT.format(pipeline_yaml_definition=pipeline_content)
            response = self._call_llm(prompt)
            
            # Parse JSON response
            result = json.loads(response)
            return result
            
        except json.JSONDecodeError as e:
            return {"error": f"Failed to parse LLM response as JSON: {str(e)}"}
        except Exception as e:
            return {"error": f"Step 1 error: {str(e)}"}
    
    def _intent_alignment_check(self, original_query: str, inferred_query: str) -> Dict[str, Any]:
        """Step 2: Check alignment between original and inferred queries."""
        try:
            prompt = INTENT_ALIGNMENT_CHECK_PROMPT.format(
                original_query=original_query,
                inferred_query_from_prompt_1=inferred_query
            )
            response = self._call_llm(prompt)
            
            # Parse JSON response
            result = json.loads(response)
            return result
            
        except json.JSONDecodeError as e:
            return {"error": f"Failed to parse LLM response as JSON: {str(e)}"}
        except Exception as e:
            return {"error": f"Step 2 error: {str(e)}"}
    
    def _constraint_check(self, original_query: str, pipeline_content: str) -> Dict[str, Any]:
        """Step 3: Check if pipeline correctly implements all constraints."""
        try:
            prompt = CONSTRAINT_CHECK_PROMPT.format(
                original_query=original_query,
                pipeline_yaml_definition=pipeline_content
            )
            response = self._call_llm(prompt)
            
            # Parse JSON response
            result = json.loads(response)
            return result
            
        except json.JSONDecodeError as e:
            return {"error": f"Failed to parse LLM response as JSON: {str(e)}"}
        except Exception as e:
            return {"error": f"Step 3 error: {str(e)}"}
    
    def _call_llm(self, prompt: str) -> str:
        """Call LLM with the given prompt."""
        if self.llm_client is None:
            # Return mock response for testing
            return "Mock inferred query from pipeline analysis"
        
        # Try different client methods
        if hasattr(self.llm_client, 'complete'):
            return self.llm_client.complete(prompt)
        elif hasattr(self.llm_client, 'chat'):
            messages = [{"role": "user", "content": prompt}]
            return self.llm_client.chat(messages)
        else:
            raise ValueError("LLM client must have either 'complete' or 'chat' method")


def check_pipeline_dynamic(question: str, 
                         pipeline_path: str = None,
                         pipeline_yaml: str = None,
                         llm_client=None,
                         intent_weight: float = DEFAULT_INTENT_WEIGHT,
                         constraint_weight: float = DEFAULT_CONSTRAINT_WEIGHT) -> Dict[str, Any]:
    """
    Convenience function to check pipeline semantics against a question.
    
    Args:
        question: Natural language query to evaluate against
        pipeline_path: Path to pipeline YAML file
        pipeline_yaml: Pipeline YAML content as string
        llm_client: LLM client for API calls (optional)
        intent_weight: Weight for strategic alignment (default: 0.6)
        constraint_weight: Weight for constraint adherence (default: 0.4)
    
    Returns:
        Dict with semantic_score and detailed results
    """
    checker = DocETLDynamicChecker(llm_client, intent_weight, constraint_weight)
    return checker.check(question, pipeline_yaml, pipeline_path)


def main():
    """Command-line interface for the dynamic checker."""
    if len(sys.argv) < 3:
        print("Usage: python3 dynamic_checker.py <question> <pipeline_yaml_path> [intent_weight]")
        print("Example: python3 dynamic_checker.py 'What is the average?' pipeline.yaml 0.8")
        sys.exit(1)
    
    question = sys.argv[1]
    pipeline_path = sys.argv[2]
    intent_weight = float(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_INTENT_WEIGHT
    constraint_weight = 1.0 - intent_weight
    
    result = check_pipeline_dynamic(
        question=question,
        pipeline_path=pipeline_path,
        intent_weight=intent_weight,
        constraint_weight=constraint_weight
    )
    
    print(json.dumps(result, indent=2))
    
    # Exit with appropriate code
    semantic_score = result.get("semantic_score", 0.0)
    sys.exit(0 if semantic_score > 0.5 else 1)


if __name__ == "__main__":
    main()