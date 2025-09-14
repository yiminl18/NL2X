#!/usr/bin/env python3
"""
Dynamic checker for DocETL pipelines using LLM-based evaluation.
Implements a 3-step methodology with separation of concerns:
1. Pipeline -> Question (P -> Q')
2. Intent Alignment (Q vs Q' -> S_IntentAlign)
3. Constraint Adherence (Q vs P -> S_ConstraintAdherence)

Final score: S_Sem = w_intent * S_IntentAlign + w_constraint * S_ConstraintAdherence

Usage:
    from dynamic_checker import check_pipeline_dynamic, create_azure_gpt4_client
    
    # With mock LLM (for testing)
    result = check_pipeline_dynamic(
        question="Report the average number of reported identity thefts...",
        pipeline_path="pipeline.yaml"
    )
    
    # With Azure GPT-4o
    result = check_pipeline_dynamic(
        question="Report the average number of reported identity thefts...",
        pipeline_path="pipeline.yaml",
        use_azure_gpt4=True
    )
    
    # Or with YAML string and custom Azure settings
    azure_client = create_azure_gpt4_client(max_tokens=2000, temperature=0.1)
    checker = DocETLDynamicChecker(azure_client, intent_weight=0.7)
    result = checker.check(question="...", pipeline_yaml="default_model: gpt-4o-mini\n...")
"""

import json
import sys
import os
from typing import Dict, Any, Optional
from string import Template
from pathlib import Path
from datetime import datetime

# Import Azure OpenAI if available
try:
    from openai import AzureOpenAI
    AZURE_OPENAI_AVAILABLE = True
except ImportError:
    AZURE_OPENAI_AVAILABLE = False

# Prompt templates using Template strings
PIPELINE_TO_QUESTION_PROMPT = Template("""
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
$pipeline_yaml_definition
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
$original_query

## Inferred Query (Q'):
$inferred_query_from_prompt_1
""")

CONSTRAINT_CHECK_PROMPT = Template("""
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
$original_query

## Pipeline Definition (P):
$pipeline_yaml_definition
""")

# Default weights for combining scores
DEFAULT_INTENT_WEIGHT = 0.6  # Strategic alignment weight
DEFAULT_CONSTRAINT_WEIGHT = 0.4  # Tactical constraint weight


class AzureGPT4Client:
    """Azure OpenAI GPT-4o client for LLM calls."""
    
    def __init__(self, api_key_path: str = '/Users/chiyuh/Workspace/NL2X/model/azuregpt4o.txt',
                 endpoint_url: str = 'https://text-db.openai.azure.com/',
                 api_version: str = '2025-01-01-preview',
                 model_name: str = 'gpt-4o',
                 max_tokens: int = 2000,
                 temperature: float = 0.0):
        """
        Initialize Azure GPT-4o client.
        
        Args:
            api_key_path: Path to API key file
            endpoint_url: Azure OpenAI endpoint URL
            api_version: API version
            model_name: Model deployment name
            max_tokens: Maximum tokens for response
            temperature: Response randomness (0-1)
        """
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
        """
        Chat completion method compatible with dynamic checker.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            
        Returns:
            str: Response content
        """
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
        """
        Completion method compatible with dynamic checker.
        
        Args:
            prompt: Text prompt string
            
        Returns:
            str: Response content
        """
        messages = [{"role": "user", "content": prompt}]
        return self.chat(messages)


class DocETLDynamicChecker:
    """Dynamic checker using LLM-based evaluation for DocETL pipelines."""
    
    def __init__(self, llm_client=None, intent_weight: float = DEFAULT_INTENT_WEIGHT, 
                 constraint_weight: float = DEFAULT_CONSTRAINT_WEIGHT, save_intermediate: bool = True):
        """
        Initialize the dynamic checker.
        
        Args:
            llm_client: LLM client for making API calls (if None, will use mock responses)
            intent_weight: Weight for intent alignment score (0.0-1.0)
            constraint_weight: Weight for constraint adherence score (0.0-1.0)
            save_intermediate: Whether to save intermediate responses to files
        """
        self.llm_client = llm_client
        self.intent_weight = intent_weight
        self.constraint_weight = constraint_weight
        self.save_intermediate = save_intermediate
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
    
    def _trim_json_response(self, response: str) -> str:
        """Trim response to extract only the JSON content between first { and last }."""
        if not response:
            return response
        
        # Find the first { and last }
        first_brace = response.find('{')
        last_brace = response.rfind('}')
        
        if first_brace != -1 and last_brace != -1 and first_brace < last_brace:
            return response[first_brace:last_brace + 1]
        
        return response
    
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
            # Create session-specific intermediate directory
            if self.save_intermediate:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                self.intermediate_dir = self.intermediate_base_dir / timestamp
                self.intermediate_dir.mkdir(exist_ok=True)
                
                # Save the question
                self._save_intermediate("question.txt", question)
            
            # Load pipeline content
            if pipeline_yaml is not None:
                pipeline_content = pipeline_yaml
            elif pipeline_path is not None:
                with open(pipeline_path, 'r', encoding='utf-8') as f:
                    pipeline_content = f.read()
            else:
                raise ValueError("Either pipeline_yaml or pipeline_path must be provided")
            
            # Save pipeline content
            if self.save_intermediate:
                self._save_intermediate("pipeline.yaml", pipeline_content)
            
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
            prompt = PIPELINE_TO_QUESTION_PROMPT.substitute(pipeline_yaml_definition=pipeline_content)
            
            # Save prompt
            if self.save_intermediate:
                self._save_intermediate("step1_prompt.txt", prompt)
            
            response = self._call_llm(prompt)
            
            # Save raw response
            if self.save_intermediate:
                self._save_intermediate("step1_response_raw.txt", response)
            
            # Trim response to JSON content
            trimmed_response = self._trim_json_response(response)
            
            # Save trimmed response
            if self.save_intermediate:
                self._save_intermediate("step1_response_trimmed.txt", trimmed_response)
            
            # Parse JSON response
            result = json.loads(trimmed_response)
            
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
            
            response = self._call_llm(prompt)
            
            # Save raw response
            if self.save_intermediate:
                self._save_intermediate("step2_response_raw.txt", response)
            
            # Trim response to JSON content
            trimmed_response = self._trim_json_response(response)
            
            # Save trimmed response
            if self.save_intermediate:
                self._save_intermediate("step2_response_trimmed.txt", trimmed_response)
            
            # Parse JSON response
            result = json.loads(trimmed_response)
            
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
                pipeline_yaml_definition=pipeline_content
            )
            
            # Save prompt
            if self.save_intermediate:
                self._save_intermediate("step3_prompt.txt", prompt)
            
            response = self._call_llm(prompt)
            
            # Save raw response
            if self.save_intermediate:
                self._save_intermediate("step3_response_raw.txt", response)
            
            # Trim response to JSON content
            trimmed_response = self._trim_json_response(response)
            
            # Save trimmed response
            if self.save_intermediate:
                self._save_intermediate("step3_response_trimmed.txt", trimmed_response)
            
            # Parse JSON response
            result = json.loads(trimmed_response)
            
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
    
    def _call_llm(self, prompt: str) -> str:
        """Call LLM with the given prompt."""
        if self.llm_client is None:
            # Return mock JSON responses for testing
            if "Pipeline Definition to Analyze" in prompt or "Here is the pipeline:" in prompt:
                # Mock response for Step 1 (Pipeline -> Question)
                return '''{
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
            elif "constraint_verification_list" in prompt or "Pipeline Definition (P):" in prompt:
                # Mock response for Step 3 (Constraint Check)
                return '''{
                    "constraint_verification_list": [
                        {
                            "constraint_description": "Metropolitan areas must be larger than one million in population",
                            "verification_status": "VERIFIED",
                            "evidence": "code_filter: return doc['population_2023'] > 1_000_000",
                            "rationale": "Code correctly implements > 1 million population filter",
                            "score": 1.0
                        },
                        {
                            "constraint_description": "Calculate average identity thefts",
                            "verification_status": "VERIFIED", 
                            "evidence": "reduce operation with averaging prompt",
                            "rationale": "Reduce operation correctly calculates average",
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
    """
    Create an Azure GPT-4o client for use with the dynamic checker.
    
    Args:
        api_key_path: Path to API key file
        **kwargs: Additional arguments for AzureGPT4Client
        
    Returns:
        AzureGPT4Client instance
    """
    return AzureGPT4Client(api_key_path=api_key_path, **kwargs)


def check_pipeline_dynamic(question: str, 
                         pipeline_path: str = None,
                         pipeline_yaml: str = None,
                         llm_client=None,
                         use_azure_gpt4: bool = False,
                         api_key_path: str = '/Users/chiyuh/Workspace/NL2X/model/azuregpt4o.txt',
                         intent_weight: float = DEFAULT_INTENT_WEIGHT,
                         constraint_weight: float = DEFAULT_CONSTRAINT_WEIGHT,
                         save_intermediate: bool = True) -> Dict[str, Any]:
    """
    Convenience function to check pipeline semantics against a question.
    
    Args:
        question: Natural language query to evaluate against
        pipeline_path: Path to pipeline YAML file
        pipeline_yaml: Pipeline YAML content as string
        llm_client: LLM client for API calls (optional, overrides use_azure_gpt4)
        use_azure_gpt4: Whether to use Azure GPT-4o client (default: False)
        api_key_path: Path to Azure API key file (only used if use_azure_gpt4=True)
        intent_weight: Weight for strategic alignment (default: 0.6)
        constraint_weight: Weight for constraint adherence (default: 0.4)
        save_intermediate: Whether to save intermediate responses to files
    
    Returns:
        Dict with semantic_score and detailed results
    """
    # Create LLM client if needed
    if llm_client is None and use_azure_gpt4:
        llm_client = create_azure_gpt4_client(api_key_path)
    
    checker = DocETLDynamicChecker(llm_client, intent_weight, constraint_weight, save_intermediate)
    return checker.check(question, pipeline_yaml, pipeline_path)


def main():
    """Command-line interface for the dynamic checker."""
    if len(sys.argv) < 3:
        print("Usage: python3 dynamic_checker.py <question> <pipeline_yaml_path> [intent_weight] [--use-azure-gpt4] [--api-key-path <path>] [--no-save-intermediate]")
        print("Example: python3 dynamic_checker.py 'What is the average?' pipeline.yaml 0.8")
        print("Example with Azure: python3 dynamic_checker.py 'What is the average?' pipeline.yaml 0.8 --use-azure-gpt4")
        sys.exit(1)
    
    question = sys.argv[1]
    pipeline_path = sys.argv[2]
    intent_weight = DEFAULT_INTENT_WEIGHT
    use_azure_gpt4 = True
    api_key_path = '/Users/chiyuh/Workspace/NL2X/model/azuregpt4o.txt'
    save_intermediate = True
    
    # Parse arguments
    i = 3
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == '--use-azure-gpt4':
            use_azure_gpt4 = True
        elif arg == '--no-save-intermediate':
            save_intermediate = False
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
        result = check_pipeline_dynamic(
            question=question,
            pipeline_path=pipeline_path,
            use_azure_gpt4=use_azure_gpt4,
            api_key_path=api_key_path,
            intent_weight=intent_weight,
            constraint_weight=constraint_weight,
            save_intermediate=save_intermediate
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