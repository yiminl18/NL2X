PIPELINE_TO_QUESTION_PROMPT = f"""
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

{
  "inferred_query": "A clear, natural language question representing the pipeline's main strategic goal.",
  "analysis_components": {
    "main_goal": "What is the primary objective of the pipeline? (e.g., 'Calculate the average of a specific metric')",
    "data_sources_used": ["A list of dataset names or descriptions."],
    "key_processing_steps": [
      "A summary of the first major transformation or filtering step that defines the intent.",
      "A summary of the second major transformation or filtering step.",
      "..."
    ]
  }
}

# [Pipeline Definition to Analyze]
Here is the pipeline:
{pipeline_yaml_definition}
"""

INTENT_ALIGNMENT_CHECK_PROMPT = f"""
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

CONSTRAINT_CHECK_PROMPT = f"""
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