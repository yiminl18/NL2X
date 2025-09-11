#!/usr/bin/env python3
"""
Dynamic checker for LOTUS pipelines using LLM-based evaluation.

Implements the same 3-step methodology as DocETL but adapted for Python code:
1. Pipeline -> Question (P -> Q')
2. Intent Alignment (Q vs Q' -> S_IntentAlign)
3. Constraint Adherence (Q vs P -> S_ConstraintAdherence)

Usage:
    from dynamic_checker import check_lotus_pipeline_dynamic
    
    result = check_lotus_pipeline_dynamic(
        question="Report the average number of identity thefts...",
        pipeline_path="pipeline.py"
    )
"""

import json
import sys
import os
from typing import Dict, Any

# Import Azure OpenAI if available
try:
    from openai import AzureOpenAI
    AZURE_OPENAI_AVAILABLE = True
except ImportError:
    AZURE_OPENAI_AVAILABLE = False

# Prompt templates adapted for Python/LOTUS pipelines
PIPELINE_TO_QUESTION_PROMPT = """
# [Role]
You are an expert data scientist and system analyst. Your task is to reverse-engineer a LOTUS data analysis pipeline written in Python to infer the high-level natural language question it was designed to answer.

# [Task]
Carefully analyze the provided Python pipeline code using LOTUS framework. Synthesize its purpose into a concise and precise natural language question (we will call this Q'). Your focus should be on the overall strategic goal, not the low-level implementation details.

# [Analysis Instructions]
1. **Data Source Analysis:** Identify the primary data sources loaded via pandas or DirectoryReader.
2. **Operator Sequence Analysis:** Examine the sequence of LOTUS operators (sem_filter, sem_map, sem_agg, etc.) and pandas operations to understand the main data flow and transformations.
3. **Identify Intent-Defining Constraints:** Extract critical constraints from sem_filter predicates, sem_map instructions, and pandas operations.
4. **Synthesize High-Level Question:** Combine your analysis into a coherent natural language question.

# [Output Format]
Produce a JSON object with the following structure. Do not add any extra commentary outside of the JSON object.

{{
  "inferred_query": "A clear, natural language question representing the pipeline's main strategic goal.",
  "analysis_components": {{
    "main_goal": "What is the primary objective of the pipeline?",
    "data_sources_used": ["List of data sources/files used"],
    "key_processing_steps": [
      "Summary of first major transformation",
      "Summary of second major transformation",
      "..."
    ]
  }}
}}

# [Pipeline Code to Analyze]
Here is the pipeline:
{pipeline_code}
"""

INTENT_ALIGNMENT_CHECK_PROMPT = """
# [Role]
You are a meticulous and impartial evaluator. Your task is to assess the high-level semantic alignment between an Original Query (Q) and an Inferred Query (Q') that represents a LOTUS pipeline's strategic goal.

# [Task]
Compare Q and Q' across several strategic dimensions. For each dimension, provide a similarity score from 0.0 (complete mismatch) to 1.0 (perfect alignment) and a brief justification.

# [Evaluation Dimensions & Weights]
1. **Core Task & Domain Alignment (Weight: 50%):**
   - Does Q' correctly identify the primary analytical task and conceptual domain?
   - **Score (0.0 - 1.0):**
   - **Rationale:**

2. **Data Filtering & Selection Criteria Alignment (Weight: 30%):**
   - Does Q' accurately capture the fundamental filtering conditions?
   - **Score (0.0 - 1.0):**
   - **Rationale:**

3. **Data Transformation Alignment (Weight: 15%):**
   - Does Q' generally reflect the major data transformations mentioned in Q?
   - **Score (0.0 - 1.0):**
   - **Rationale:**

4. **Minor Constraints & Output Format Alignment (Weight: 5%):**
   - Does Q' capture any high-level instructions about the output?
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
You are a meticulous Quality Assurance (QA) analyst and code reviewer. Your sole task is to verify if a given LOTUS data analysis pipeline written in Python correctly implements all the constraints specified in a natural language query.

# [Task]
1. **Deconstruct the Query:** Carefully read the Original Query (Q) and create a checklist of every explicit and implicit constraint.
2. **Audit the Pipeline:** For each constraint, scan the Python code to find the implementation.
3. **Evaluate Implementation:** Judge whether the implementation is correct, partially correct, incorrect, or missing.

# [Instructions]
- Be precise. Refer to specific LOTUS operators (sem_filter, sem_map, etc.), pandas operations, or code snippets as evidence.
- Focus strictly on adherence to specific constraints from the query.

# [Output Format]
Produce a JSON object with the following structure. Do not add any extra commentary outside of the JSON object.

{{
  "constraint_verification_list": [
    {{
      "constraint_description": "A description of the constraint extracted from the query.",
      "verification_status": "ENUM('VERIFIED', 'PARTIALLY_VERIFIED', 'INCORRECTLY_IMPLEMENTED', 'NOT_FOUND')",
      "evidence": "The specific part of the pipeline that implements this constraint.",
      "rationale": "A brief explanation of why the status was assigned.",
      "score": <float> // 1.0 for VERIFIED, 0.5 for PARTIALLY_VERIFIED, 0.0 for others
    }}
  ],
  "constraint_adherence_score": <float> // The simple average of the scores from the list above.
}}

# [Inputs]
## Original Query (Q):
{original_query}

## Pipeline Code (P):
{pipeline_code}
"""

# Default weights for combining scores
DEFAULT_INTENT_WEIGHT = 0.6
DEFAULT_CONSTRAINT_WEIGHT = 0.4


class AzureGPT4Client:
    """Azure OpenAI GPT-4o client for LLM calls."""
    
    def __init__(self, api_key_path: str = '/Users/chiyuh/Workspace/NL2X/model/azuregpt4o.txt',
                 endpoint_url: str = 'https://text-db.openai.azure.com/',
                 api_version: str = '2025-01-01-preview',
                 model_name: str = 'gpt-4o',
                 max_tokens: int = 2000,
                 temperature: float = 0.0):
        """Initialize Azure GPT-4o client."""
        if not AZURE_OPENAI_AVAILABLE:
            raise ImportError("Azure OpenAI library not available. Install with: pip install openai")
            
        self.api_key_path = api_key_path
        self.endpoint_url = endpoint_url
        self.api_version = api_version
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        # Read API key
        try:
            with open(api_key_path, 'r') as f:
                self.api_key = f.read().strip()
        except FileNotFoundError:
            raise FileNotFoundError(f"API key file not found at {api_key_path}")
        
        # Initialize client
        self.client = AzureOpenAI(
            azure_endpoint=os.getenv("ENDPOINT_URL", self.endpoint_url),
            api_key=self.api_key,
            api_version=self.api_version,
        )
    
    def chat(self, messages):
        """Chat completion method."""
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                stream=False
            )
            return completion.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"Azure OpenAI API call failed: {str(e)}")
    
    def complete(self, prompt: str):
        """Completion method."""
        messages = [{"role": "user", "content": prompt}]
        return self.chat(messages)


class LotusDynamicChecker:
    """Dynamic checker using LLM-based evaluation for LOTUS pipelines."""
    
    def __init__(self, llm_client=None, intent_weight: float = DEFAULT_INTENT_WEIGHT, 
                 constraint_weight: float = DEFAULT_CONSTRAINT_WEIGHT):
        """Initialize the dynamic checker."""
        self.llm_client = llm_client
        self.intent_weight = intent_weight
        self.constraint_weight = constraint_weight
        
        # Validate weights
        if abs(intent_weight + constraint_weight - 1.0) > 1e-6:
            raise ValueError("Intent weight and constraint weight must sum to 1.0")
    
    def check(self, question: str, pipeline_code: str = None, pipeline_path: str = None) -> Dict[str, Any]:
        """
        Check pipeline against question using 3-step methodology.
        
        Args:
            question: Original natural language query
            pipeline_code: Pipeline Python code (if provided)
            pipeline_path: Path to pipeline Python file (if pipeline_code not provided)
        
        Returns:
            Dict containing semantic score and detailed step results
        """
        try:
            # Load pipeline content
            if pipeline_code is not None:
                code = pipeline_code
            elif pipeline_path is not None:
                with open(pipeline_path, 'r', encoding='utf-8') as f:
                    code = f.read()
            else:
                raise ValueError("Either pipeline_code or pipeline_path must be provided")
            
            # Step 1: Pipeline -> Question (P -> Q')
            step1_result = self._pipeline_to_question(code)
            
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
            step3_result = self._constraint_check(question, code)
            
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
    
    def _pipeline_to_question(self, pipeline_code: str) -> Dict[str, Any]:
        """Step 1: Reverse-engineer pipeline to infer intended question."""
        try:
            prompt = PIPELINE_TO_QUESTION_PROMPT.format(pipeline_code=pipeline_code)
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
    
    def _constraint_check(self, original_query: str, pipeline_code: str) -> Dict[str, Any]:
        """Step 3: Check if pipeline correctly implements all constraints."""
        try:
            prompt = CONSTRAINT_CHECK_PROMPT.format(
                original_query=original_query,
                pipeline_code=pipeline_code
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
            # Return mock JSON responses for testing
            if "Pipeline Code to Analyze" in prompt or "Here is the pipeline:" in prompt:
                # Mock response for Step 1 (Pipeline -> Question)
                return '''{
                    "inferred_query": "What is the average number of reported identity thefts for metropolitan areas with population over one million?",
                    "analysis_components": {
                        "main_goal": "Calculate average identity thefts for large metropolitan areas",
                        "data_sources_used": ["metropolitan_statistics.html.csv"],
                        "key_processing_steps": [
                            "Extract metropolitan area statistics using sem_extract",
                            "Filter areas with population over 1 million using sem_filter",
                            "Calculate average identity thefts using sem_agg"
                        ]
                    }
                }'''
            elif "Original Query (Q):" in prompt and "Inferred Query (Q'):" in prompt:
                # Mock response for Step 2 (Intent Alignment)
                return '''{
                    "evaluation_summary": {
                        "core_task_domain": {"score": 0.9, "rationale": "Both queries focus on identity theft statistics for metropolitan areas"},
                        "filtering_selection": {"score": 0.85, "rationale": "Both specify population threshold of one million"},
                        "transformation_calculation": {"score": 0.8, "rationale": "Both require averaging calculation"},
                        "output_format_constraints": {"score": 0.7, "rationale": "Output format generally aligned"}
                    },
                    "intent_alignment_score": 0.83
                }'''
            elif "constraint_verification_list" in prompt or "Pipeline Code (P):" in prompt:
                # Mock response for Step 3 (Constraint Check)
                return '''{
                    "constraint_verification_list": [
                        {
                            "constraint_description": "Metropolitan areas must be larger than one million in population",
                            "verification_status": "VERIFIED",
                            "evidence": "sem_filter with population > 1000000 condition",
                            "rationale": "Code correctly implements > 1 million population filter",
                            "score": 1.0
                        },
                        {
                            "constraint_description": "Calculate average identity thefts",
                            "verification_status": "VERIFIED", 
                            "evidence": "sem_agg operation with averaging instruction",
                            "rationale": "sem_agg correctly calculates average",
                            "score": 1.0
                        }
                    ],
                    "constraint_adherence_score": 1.0
                }'''
            else:
                return '{"mock_response": "Unable to determine prompt type"}'
        
        # Try different client methods
        if hasattr(self.llm_client, 'complete'):
            return self.llm_client.complete(prompt)
        elif hasattr(self.llm_client, 'chat'):
            messages = [{"role": "user", "content": prompt}]
            return self.llm_client.chat(messages)
        else:
            raise ValueError("LLM client must have either 'complete' or 'chat' method")


def create_azure_gpt4_client(api_key_path: str = '/Users/chiyuh/Workspace/NL2X/model/azuregpt4o.txt',
                            **kwargs):
    """Create an Azure GPT-4o client for use with the dynamic checker."""
    return AzureGPT4Client(api_key_path=api_key_path, **kwargs)


def check_lotus_pipeline_dynamic(question: str, 
                                pipeline_path: str = None,
                                pipeline_code: str = None,
                                llm_client=None,
                                use_azure_gpt4: bool = False,
                                api_key_path: str = '/Users/chiyuh/Workspace/NL2X/model/azuregpt4o.txt',
                                intent_weight: float = DEFAULT_INTENT_WEIGHT,
                                constraint_weight: float = DEFAULT_CONSTRAINT_WEIGHT) -> Dict[str, Any]:
    """
    Convenience function to check LOTUS pipeline semantics against a question.
    
    Args:
        question: Natural language query to evaluate against
        pipeline_path: Path to pipeline Python file
        pipeline_code: Pipeline Python code as string
        llm_client: LLM client for API calls (optional, overrides use_azure_gpt4)
        use_azure_gpt4: Whether to use Azure GPT-4o client (default: False)
        api_key_path: Path to Azure API key file (only used if use_azure_gpt4=True)
        intent_weight: Weight for strategic alignment (default: 0.6)
        constraint_weight: Weight for constraint adherence (default: 0.4)
    
    Returns:
        Dict with semantic_score and detailed results
    """
    # Create LLM client if needed
    if llm_client is None and use_azure_gpt4:
        llm_client = create_azure_gpt4_client(api_key_path)
    
    checker = LotusDynamicChecker(llm_client, intent_weight, constraint_weight)
    return checker.check(question, pipeline_code, pipeline_path)


def main():
    """Command-line interface for the dynamic checker."""
    if len(sys.argv) < 3:
        print("Usage: python3 dynamic_checker.py <question> <pipeline.py> [intent_weight] [--use-azure-gpt4]")
        print("Example: python3 dynamic_checker.py 'What is the average?' pipeline.py 0.8")
        print("Example with Azure: python3 dynamic_checker.py 'What is the average?' pipeline.py 0.8 --use-azure-gpt4")
        sys.exit(1)
    
    question = sys.argv[1]
    pipeline_path = sys.argv[2]
    intent_weight = DEFAULT_INTENT_WEIGHT
    use_azure_gpt4 = False
    api_key_path = '/Users/chiyuh/Workspace/NL2X/model/azuregpt4o.txt'
    
    # Parse arguments
    i = 3
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == '--use-azure-gpt4':
            use_azure_gpt4 = True
        elif arg == '--api-key-path':
            if i + 1 < len(sys.argv):
                api_key_path = sys.argv[i + 1]
                i += 1
            else:
                print("Error: --api-key-path requires a path argument")
                sys.exit(1)
        elif not arg.startswith('--'):
            # Assume it's the intent weight
            try:
                intent_weight = float(arg)
            except ValueError:
                print(f"Error: Invalid intent weight '{arg}'. Must be a float.")
                sys.exit(1)
        i += 1
    
    constraint_weight = 1.0 - intent_weight
    
    try:
        result = check_lotus_pipeline_dynamic(
            question=question,
            pipeline_path=pipeline_path,
            use_azure_gpt4=use_azure_gpt4,
            api_key_path=api_key_path,
            intent_weight=intent_weight,
            constraint_weight=constraint_weight
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