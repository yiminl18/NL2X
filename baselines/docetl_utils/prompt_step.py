"""
Prompt templates for step-by-step DocETL pipeline generation.
"""

# Operator selection prompt - Step 1
OPERATOR_SELECTION_PROMPT = """
You are an expert at analyzing data processing tasks and selecting appropriate DocETL operators.

Given the following query and dataset, identify which operators are needed to accomplish the task.

QUERY:
{query}

DATASET SAMPLE:
{dataset_samples}

AVAILABLE OPERATORS:
{operator_definitions}

COMPLETE EXAMPLES FROM REAL PIPELINES:

Example 1: Presidential Debate Themes Analysis
Query: "Extract themes and viewpoints from debate transcripts and analyze how those themes evolve over time across multiple debates"
Dataset: Collection of presidential debate transcripts with fields: title, date, year, content

Selected operators and reasoning:
- map: extract themes and viewpoints from each debate transcript (transforms each debate into structured themes with viewpoints)
- unnest: expand the themes array into individual theme records (needed because map outputs a list of themes per debate)
- reduce: aggregate viewpoints by theme to analyze evolution over time (groups all instances of the same theme across debates)
- code_map: transform final output to result format (ensures output follows the required schema)

Example 2: Mining Product Reviews for Polarizing Themes
Query: "Identify polarizing themes in video game reviews that divide player opinions, resolve similar themes across reviews, and aggregate them to find common polarizing themes across different games"
Dataset: Video game reviews with fields: app_name, concatenated_reviews

Selected operators and reasoning:
- map: identify polarizing themes from concatenated reviews (analyzes each game's reviews to find divisive topics)
- unnest: expand polarizing_themes array into individual theme records (needed to process each theme separately)
- resolve: deduplicate and consolidate similar themes (merges themes that are essentially the same but worded differently)
- reduce: aggregate common themes across different games by theme (groups resolved themes to find patterns across games)
- code_map: transform final output to result format (formats the aggregated results properly)

Based on the query and dataset, list the operators needed in the order they should be applied.
For each operator, provide:
- The operator type
- A brief description of what this operator will do in the pipeline

Format your response as a list:
- [operator_type]: [what this operator will do]

IMPORTANT:
1. Only select operators that are actually needed
2. Consider the data flow between operators
3. Always include code_map at the end for final result transformation
4. For aggregation tasks, use reduce with appropriate reduce_key
5. For deduplication, use resolve
6. For splitting long text, use split followed by gather if context is needed
7. Use unnest when you need to expand arrays or nested structures
8. Chain operators logically - outputs of one operator should match inputs expected by the next

Your response:
"""

# Operator detail generation prompt - Step 3
OPERATOR_DETAIL_PROMPT = """
You are an expert at generating DocETL operator configurations.

Generate the detailed configuration for the following operator:

OPERATOR TYPE: {operator_type}
OPERATOR PURPOSE: {operator_purpose}

QUERY: {query}

DATASET SAMPLE:
{dataset_samples}

AVAILABLE FIELDS AT THIS STAGE:
{available_fields}

PREVIOUS OPERATORS IN PIPELINE:
{previous_operators}

OPERATOR EXAMPLES FOR {operator_type}:
{operator_examples}

CURRENT OPERATOR FRAMEWORK:
{operator_framework}

Fill in all the "TO_BE_GENERATED" placeholders with appropriate values.

IMPORTANT FIELD RULES:
- You can ONLY reference fields listed in "AVAILABLE FIELDS AT THIS STAGE"
- Fields from previous operators' outputs are included in available fields
- Do NOT reference fields that don't exist yet
- When creating output schemas, choose field names that don't conflict with existing fields

For prompts:
- Use Jinja2 templating: {{{{ input.field }}}} for single items, {{{{ inputs }}}} for reduce operations
- Be specific about what information to extract or process
- Match the output schema exactly
- ONLY use fields that are in the AVAILABLE FIELDS list

For schemas:
- Keep them simple and flat
- Use appropriate types: string, integer, number, boolean, list[...]
- For complex structures, use quoted strings: "list[{{field: type}}]"
- Output field names should be descriptive and unique

For operator-specific fields:
- reduce_key: must specify the field to group by for reduce operations (must exist in available fields)
- split_key: the field to split for split operations (must exist in available fields)
- unnest_key: the field containing arrays to unnest (must exist in available fields)
- document_keys: for extract operations, specify which field(s) to extract from (CANNOT be empty, defaults to ["src"] if not specified)

Return ONLY the filled operator configuration in YAML format.

```yaml
name: {operator_type}_operation
type: {operator_type}
# ... fill in all fields with actual values, no placeholders
```
"""

# Operator definitions with examples (extracted from original prompt.py)
OPERATOR_DEFINITIONS = """
1. **Map** — Per-document transformation using an LLM. Access fields via `input.<field>`.
   Example: Extract topics from news articles, transform each record

2. **Filter** — Keep or discard documents by returning a boolean.
   Example: Filter high-impact articles, remove invalid records

3. **Reduce** — Aggregate over groups using a `reduce_key`, combining multiple items into one.
   Example: Summarize by department, aggregate by category
   REQUIRES: reduce_key field

4. **Resolve** — Deduplicate or standardize entities by comparing and merging inputs.
   Example: Standardize patient names, merge duplicate records
   Use: optimize: true

5. **Rank** — Order items according to custom criteria with LLM-assisted scoring.
   Example: Rank by controversy, order by relevance

6. **Extract** — Pull verbatim sections of text matching patterns or prompts.
   Example: Extract key findings, pull specific quotes

7. **Cluster** — Group items into hierarchical categories using embeddings.
   Example: Group similar concepts, cluster related items

8. **Split** — Break long text fields into chunks by tokens, sentences, or rules.
   Example: Split long documents, chunk transcripts

9. **Gather** — Re-attach neighboring context after splitting.
   Example: Add surrounding context to chunks

10. **Unnest** — Expand array or nested fields into separate items.
    Example: Expand list fields, flatten nested structures

11. **Sample** — Subset the dataset uniformly or stratified.
    Example: Sample 10% of data, stratified sampling by category

12. **TopK** — Retrieve the top-k most relevant documents via embeddings.
    Example: Find most relevant tickets, top matching documents

13. **code_map** — Final result transformation using Python code.
    Example: Format final output, transform to result structure
    ALWAYS use at the end for result transformation
"""

# Validation prompt for checking operator consistency
OPERATOR_VALIDATION_PROMPT = """
Check if the following operator configuration is valid and consistent:

OPERATOR:
{operator_config}

PREVIOUS OPERATORS OUTPUT:
{previous_outputs}

DATASET FIELDS:
{dataset_fields}

Verify:
1. All input fields referenced exist (either from dataset or previous operators)
2. The output schema is appropriate for the operation
3. Required fields are present (e.g., reduce_key for reduce)
4. The prompt correctly uses Jinja2 templating

Return:
- VALID if the operator is correct
- INVALID with explanation if there are issues

Response format:
STATUS: [VALID/INVALID]
EXPLANATION: [if invalid, explain the issues]
"""
