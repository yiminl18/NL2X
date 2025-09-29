"""
Prompt templates for step-by-step DocETL pipeline generation.
"""

from string import Template

# Operator selection prompt - Step 1
OPERATOR_SELECTION_PROMPT = Template("""
You are an expert at analyzing data processing tasks and selecting appropriate DocETL operators.

Given the following query and dataset, identify which operators are needed to accomplish the task.

QUERY:
$query

DATASET SAMPLE:
$dataset_samples

AVAILABLE OPERATORS:
$operator_definitions

COMPLETE EXAMPLES FROM REAL PIPELINES:

Example 1: Presidential Debate Themes Analysis
Query: "Extract themes and viewpoints from debate transcripts and analyze how those themes evolve over time across multiple debates"
Dataset: Collection of presidential debate transcripts with fields: title, date, year, content

Selected operators and reasoning:
- map: extract themes and viewpoints from each debate transcript (transforms each debate into structured themes with viewpoints)
- unnest: expand the themes array into individual theme records (needed because map outputs a list of themes per debate)
- reduce: aggregate viewpoints by theme to analyze evolution over time (groups all instances of the same theme across debates)

Example 2: Mining Product Reviews for Polarizing Themes
Query: "Identify polarizing themes in video game reviews that divide player opinions, resolve similar themes across reviews, and aggregate them to find common polarizing themes across different games"
Dataset: Video game reviews with fields: app_name, concatenated_reviews

Selected operators and reasoning:
- map: identify polarizing themes from concatenated reviews (analyzes each game's reviews to find divisive topics)
- unnest: expand polarizing_themes array into individual theme records (needed to process each theme separately)
- resolve: deduplicate and consolidate similar themes (merges themes that are essentially the same but worded differently)
- reduce: aggregate common themes across different games by theme (groups resolved themes to find patterns across games)

Based on the query and dataset, list the operators needed in the order they should be applied.
For each operator, provide:
- The operator type
- A brief description of what this operator will do in the pipeline

Format your response as a list:
- [operator_type]: [what this operator will do]

IMPORTANT:
1. Only select operators that are actually needed
2. Consider the data flow between operators
3. For aggregation tasks, use reduce
4. For deduplication, use resolve
5. For splitting long text, use split followed by gather if context is needed
6. Use unnest when you need to expand arrays or nested structures
7. The map and extract operators has similar function - Using these two operators together is not recommended.
8. Chain operators logically - outputs of one operator should match inputs expected by the next

Your response:
""")

# Individual operator detail generation prompts - Step 3

MAP_OPERATOR_PROMPT = Template("""
You are an expert at generating DocETL map operator configurations.

Generate the detailed configuration for a MAP operator:

OPERATOR PURPOSE: $operator_purpose
QUERY: $query

DATASET SAMPLE:
$dataset_samples

AVAILABLE FIELDS AT THIS STAGE:
$available_fields

PREVIOUS OPERATORS IN PIPELINE:
$previous_operators

CURRENT OPERATOR FRAMEWORK:
$operator_framework

CRITICAL MAP OPERATOR RULES:
- Map transforms EACH document individually using {{ input.field_name }} syntax (Jinja2 template)
- You MUST use {{ input.field_name }} to reference fields in the prompt
- Available fields: $available_fields
- Create an output schema with new field names that don't conflict with existing fields
- The output schema MUST specify field types: "string", "int", "float", "bool", "list", "dict"
- Keep output schema simple and flat when possible

COMPLETE EXAMPLES (only showing fields you need to generate):

Example 1 - Extracting structured data:
{
  "prompt": "Analyze the text: {{ input.content }}. Extract the main themes discussed. Return a list of theme names.",
  "output_schema": {
    "themes": "list",
    "summary": "string"
  }
}

Example 2 - Sentiment analysis:
{
  "prompt": "Analyze sentiment of: {{ input.review_text }}. Return sentiment label and confidence score.",
  "output_schema": {
    "sentiment": "string",
    "confidence": "float"
  }
}

IMPORTANT: Return valid JSON (not YAML). Your response must be parseable JSON matching this structure.
The "output_schema" field must be an object with field names as keys and type strings as values.
Valid types: "string", "int", "float", "bool", "list", "dict"
""")

FILTER_OPERATOR_PROMPT = Template("""
You are an expert at generating DocETL filter operator configurations.

Generate the detailed configuration for a FILTER operator:

OPERATOR PURPOSE: $operator_purpose
QUERY: $query

DATASET SAMPLE:
$dataset_samples

AVAILABLE FIELDS AT THIS STAGE:
$available_fields

PREVIOUS OPERATORS IN PIPELINE:
$previous_operators

CURRENT OPERATOR FRAMEWORK:
$operator_framework

CRITICAL FILTER OPERATOR RULES:
- Filter keeps/discards documents based on {{ input.field_name }} syntax (Jinja2 template)
- You MUST use {{ input.field_name }} to reference fields in the prompt
- Available fields: $available_fields
- The output schema MUST have a boolean field (type: "bool")
- The prompt should ask for "true" or "false" as the response

COMPLETE EXAMPLES (only showing fields you need to generate):

Example 1 - Filter by relevance:
{
  "prompt": "Is this document relevant? Title: {{ input.title }}. Return true to keep, false to discard.",
  "output_schema": {
    "keep": "bool"
  }
}

Example 2 - Filter by score threshold:
{
  "prompt": "Quality score: {{ input.quality_score }}. Return true if score >= 7, else false.",
  "output_schema": {
    "passes_threshold": "bool"
  }
}

IMPORTANT: Return valid JSON (not YAML). Your response must be parseable JSON matching this structure.
The "output_schema" field must be an object with a boolean field.
""")

REDUCE_OPERATOR_PROMPT = Template("""
You are an expert at generating DocETL reduce operator configurations.

Generate the detailed configuration for a REDUCE operator:

OPERATOR PURPOSE: $operator_purpose
QUERY: $query

DATASET SAMPLE:
$dataset_samples

AVAILABLE FIELDS AT THIS STAGE:
$available_fields

PREVIOUS OPERATORS IN PIPELINE:
$previous_operators

CURRENT OPERATOR FRAMEWORK:
$operator_framework

CRITICAL REDUCE OPERATOR RULES:
- Reduce aggregates multiple documents grouped by reduce_key
- You MUST use {{ inputs }} (plural) to reference the group of documents (Jinja2 template)
- You MUST specify a reduce_key field that exists in available fields: $available_fields
- Access fields like: {{ inputs[0].field_name }} or {% for item in inputs %}{{ item.field_name }}{% endfor %}
- The output schema MUST specify field types: "string", "int", "float", "bool", "list", "dict"
- Create aggregated output schema with new field names

COMPLETE EXAMPLES (only showing fields you need to generate):

Example 1 - Aggregate themes by category:
{
  "reduce_key": "theme",
  "prompt": "Summarize all viewpoints for theme '{{ inputs[0].theme }}': {% for item in inputs %}{{ item.viewpoint }} {% endfor %}. Provide an aggregated summary.",
  "output_schema": {
    "theme": "string",
    "aggregated_summary": "string",
    "count": "int"
  }
}

Example 2 - Aggregate reviews by product:
{
  "reduce_key": "product_id",
  "prompt": "Product: {{ inputs[0].product_name }}. Reviews: {% for review in inputs %}{{ review.text }} {% endfor %}. Summarize common themes.",
  "output_schema": {
    "product_name": "string",
    "common_themes": "list",
    "average_sentiment": "string"
  }
}

IMPORTANT: Return valid JSON (not YAML). Your response must be parseable JSON matching this structure.
The "output_schema" field must be an object with field names as keys and type strings as values.
Valid types: "string", "int", "float", "bool", "list", "dict"
""")

RESOLVE_OPERATOR_PROMPT = Template("""
You are an expert at generating DocETL resolve operator configurations.

Generate the detailed configuration for a RESOLVE operator:

OPERATOR PURPOSE: $operator_purpose
QUERY: $query

DATASET SAMPLE:
$dataset_samples

AVAILABLE FIELDS AT THIS STAGE:
$available_fields

PREVIOUS OPERATORS IN PIPELINE:
$previous_operators

CURRENT OPERATOR FRAMEWORK:
$operator_framework

CRITICAL RESOLVE OPERATOR RULES:
- Resolve deduplicates/standardizes entities with comparison and resolution prompts (Jinja2 templates)
- comparison_prompt uses {{ input1.field }} and {{ input2.field }} to compare two items
- resolution_prompt uses {{ inputs }} to merge multiple similar items
- Available fields: $available_fields
- Set optimize: true for better performance
- The output schema MUST specify field types: "string", "int", "float", "bool", "list", "dict"
- Create output schema for the resolved/standardized entity

COMPLETE EXAMPLES (only showing fields you need to generate):

Example 1 - Resolve duplicate person names:
{
  "optimize": true,
  "comparison_prompt": "Are {{ input1.name }} and {{ input2.name }} the same person? Email1: {{ input1.email }} vs Email2: {{ input2.email }}. Return True or False.",
  "resolution_prompt": "Merge these person records: {% for item in inputs %}Name: {{ item.name }}, Email: {{ item.email }} {% endfor %}. Return canonical name and email.",
  "output_schema": {
    "canonical_name": "string",
    "canonical_email": "string"
  }
}

Example 2 - Resolve similar themes:
{
  "optimize": true,
  "comparison_prompt": "Are themes '{{ input1.theme_name }}' and '{{ input2.theme_name }}' similar? Return True or False.",
  "resolution_prompt": "Consolidate these similar themes: {% for t in inputs %}{{ t.theme_name }}, {{ t.description }} {% endfor %}. Return unified theme name.",
  "output_schema": {
    "unified_theme": "string",
    "consolidated_description": "string"
  }
}

IMPORTANT: Return valid JSON (not YAML). Your response must be parseable JSON matching this structure.
The "output_schema" field must be an object with field names as keys and type strings as values.
Valid types: "string", "int", "float", "bool", "list", "dict"
""")

RANK_OPERATOR_PROMPT = Template("""
You are an expert at generating DocETL rank operator configurations.

Generate the detailed configuration for a RANK operator:

OPERATOR PURPOSE: $operator_purpose
QUERY: $query

DATASET SAMPLE:
$dataset_samples

AVAILABLE FIELDS AT THIS STAGE:
$available_fields

PREVIOUS OPERATORS IN PIPELINE:
$previous_operators

CURRENT OPERATOR FRAMEWORK:
$operator_framework

CRITICAL RANK OPERATOR RULES:
- Rank orders documents by custom criteria using LLM scoring
- Specify input_keys with fields to consider for ranking: $available_fields
- The prompt should describe ranking criteria
- Set direction: "desc" for highest first, "asc" for lowest first
- Rank adds a "rank" field to output documents

Example configuration:
- input_keys: ["title", "content", "score"]
- prompt: "Rank by relevance considering title and content quality"

Fill in the "TO_BE_GENERATED" placeholders with appropriate values.

Return ONLY the filled operator configuration in JSON format matching the required schema.
""")

EXTRACT_OPERATOR_PROMPT = Template("""
You are an expert at generating DocETL extract operator configurations.

Generate the detailed configuration for an EXTRACT operator:

OPERATOR PURPOSE: $operator_purpose
QUERY: $query

DATASET SAMPLE:
$dataset_samples

AVAILABLE FIELDS AT THIS STAGE:
$available_fields

PREVIOUS OPERATORS IN PIPELINE:
$previous_operators

CURRENT OPERATOR FRAMEWORK:
$operator_framework

CRITICAL EXTRACT OPERATOR RULES:
- Extract pulls verbatim text sections from documents
- document_keys specifies which fields to extract from: $available_fields
- document_keys CANNOT be empty (defaults to ["src"] if not specified)
- The prompt should describe what text sections to extract
- Extracted content is added as new fields with suffix

Example configuration:
- document_keys: ["content", "article_text"]
- prompt: "Extract key findings, conclusions, and important quotes from the text"

Fill in the "TO_BE_GENERATED" placeholders with appropriate values.

Return ONLY the filled operator configuration in YAML format:

```yaml
name: $operator_type_operation
type: extract
prompt: |
  [Description of what text sections to extract]
document_keys: [list_of_fields_to_extract_from]
```
""")


CODE_FILTER_OPERATOR_PROMPT = Template("""
You are an expert at generating DocETL code_filter operator configurations.

Generate the detailed configuration for a CODE_FILTER operator:

OPERATOR PURPOSE: $operator_purpose
QUERY: $query

DATASET SAMPLE:
$dataset_samples

AVAILABLE FIELDS AT THIS STAGE:
$available_fields

PREVIOUS OPERATORS IN PIPELINE:
$previous_operators

CURRENT OPERATOR FRAMEWORK:
$operator_framework

CRITICAL CODE_FILTER OPERATOR RULES:
- Code_filter keeps/discards documents using Python code instead of LLM prompts
- Use doc['field_name'] syntax to access fields in the Python code
- Available fields: $available_fields
- The function must return a boolean (True to keep, False to discard)
- The function must be named 'filter' and take 'doc' parameter

Example code structure:
```python
def filter(doc) -> bool:
    return doc['field_name'] > threshold  # Use available fields
```

Fill in the "TO_BE_GENERATED" placeholders with appropriate values.

Return ONLY the filled operator configuration in YAML format:

```yaml
name: $operator_type_operation
type: code_filter
code: |
  def filter(doc) -> bool:
      [Your Python boolean logic using doc['field_name']]
      return [boolean_expression]
```
""")

SPLIT_OPERATOR_PROMPT = Template("""
You are an expert at generating DocETL split operator configurations.

Generate the detailed configuration for a SPLIT operator:

OPERATOR PURPOSE: $operator_purpose
QUERY: $query

DATASET SAMPLE:
$dataset_samples

AVAILABLE FIELDS AT THIS STAGE:
$available_fields

PREVIOUS OPERATORS IN PIPELINE:
$previous_operators

CURRENT OPERATOR FRAMEWORK:
$operator_framework

CRITICAL SPLIT OPERATOR RULES:
- Split breaks long text fields into smaller chunks
- split_key must specify which field to split: $available_fields
- method options: "token_count", "sentence", "delimiter"
- method_kwargs configures the splitting parameters
- Adds _split_id and _split_index fields to output

Example configuration:
- split_key: "content" (must be an available field)
- method: "token_count" with num_tokens parameter
- method: "sentence" for sentence-based splitting

Fill in the "TO_BE_GENERATED" placeholders with appropriate values.

Return ONLY the filled operator configuration in YAML format:

```yaml
name: $operator_type_operation
type: split
split_key: [field_to_split]
method: [token_count/sentence/delimiter]
method_kwargs:
  [method_parameters]
```
""")

GATHER_OPERATOR_PROMPT = Template("""
You are an expert at generating DocETL gather operator configurations.

Generate the detailed configuration for a GATHER operator:

OPERATOR PURPOSE: $operator_purpose
QUERY: $query

DATASET SAMPLE:
$dataset_samples

AVAILABLE FIELDS AT THIS STAGE:
$available_fields

PREVIOUS OPERATORS IN PIPELINE:
$previous_operators

CURRENT OPERATOR FRAMEWORK:
$operator_framework

CRITICAL GATHER OPERATOR RULES:
- Gather adds surrounding context to chunks after splitting
- content_key: field containing the chunk content
- doc_id_key: field identifying which document the chunk belongs to
- order_key: field indicating chunk order within document
- All keys must reference available fields: $available_fields
- Configure peripheral_chunks for context

Example configuration:
- content_key: "text_chunk"
- doc_id_key: "document_id"
- order_key: "chunk_number"

Fill in the "TO_BE_GENERATED" placeholders with appropriate values.

Return ONLY the filled operator configuration in YAML format:

```yaml
name: $operator_type_operation
type: gather
content_key: [chunk_content_field]
doc_id_key: [document_id_field]
order_key: [chunk_order_field]
```
""")

UNNEST_OPERATOR_PROMPT = Template("""
You are an expert at generating DocETL unnest operator configurations.

Generate the detailed configuration for an UNNEST operator:

OPERATOR PURPOSE: $operator_purpose
QUERY: $query

DATASET SAMPLE:
$dataset_samples

AVAILABLE FIELDS AT THIS STAGE:
$available_fields

PREVIOUS OPERATORS IN PIPELINE:
$previous_operators

CURRENT OPERATOR FRAMEWORK:
$operator_framework

CRITICAL UNNEST OPERATOR RULES:
- Unnest expands array or nested fields into separate documents
- unnest_key must specify which field contains the array/nested data: $available_fields
- For list unnesting: creates multiple documents, one per array element
- For dict unnesting: flattens nested fields into parent document
- Use recursive: true and depth: 2 to fully flatten nested structures

Example configuration:
- unnest_key: "themes" (to expand a themes array)
- unnest_key: "metadata" (to flatten nested metadata dict)

Fill in the "TO_BE_GENERATED" placeholders with appropriate values.

Return ONLY the filled operator configuration in YAML format:

```yaml
name: $operator_type_operation
type: unnest
unnest_key: [field_to_unnest]
```
""")

CLUSTER_OPERATOR_PROMPT = Template("""
You are an expert at generating DocETL cluster operator configurations.

Generate the detailed configuration for a CLUSTER operator:

OPERATOR PURPOSE: $operator_purpose
QUERY: $query

DATASET SAMPLE:
$dataset_samples

AVAILABLE FIELDS AT THIS STAGE:
$available_fields

PREVIOUS OPERATORS IN PIPELINE:
$previous_operators

CURRENT OPERATOR FRAMEWORK:
$operator_framework

CRITICAL CLUSTER OPERATOR RULES:
- Cluster groups similar documents using embeddings
- embedding_keys specifies which fields to use for similarity: $available_fields
- output_key names the field where cluster assignments are stored
- summary_prompt uses {{ inputs }} to describe cluster characteristics
- summary_schema defines the cluster summary structure

Example configuration:
- embedding_keys: ["title", "content"]
- summary_prompt: "Summarize this cluster: {% for item in inputs %}{{ item.title }}{% endfor %}"

Fill in the "TO_BE_GENERATED" placeholders with appropriate values.

Return ONLY the filled operator configuration in YAML format:

```yaml
name: $operator_type_operation
type: cluster
embedding_keys: [fields_for_similarity]
output_key: [cluster_field_name]
```
""")

SAMPLE_OPERATOR_PROMPT = Template("""
You are an expert at generating DocETL sample operator configurations.

Generate the detailed configuration for a SAMPLE operator:

OPERATOR PURPOSE: $operator_purpose
QUERY: $query

DATASET SAMPLE:
$dataset_samples

AVAILABLE FIELDS AT THIS STAGE:
$available_fields

PREVIOUS OPERATORS IN PIPELINE:
$previous_operators

CURRENT OPERATOR FRAMEWORK:
$operator_framework

CRITICAL SAMPLE OPERATOR RULES:
- Sample selects a subset of documents for processing
- method: "uniform" for random sampling, "stratified" for balanced sampling
- samples: fraction (0.1 = 10%) or integer count
- stratify_key: field to balance across when using stratified sampling: $available_fields
- random_state: seed for reproducible sampling

Example configuration:
- method: "uniform", samples: 0.1 (10% random sample)
- method: "stratified", samples: 100, stratify_key: "category"

Fill in the "TO_BE_GENERATED" placeholders with appropriate values.

Return ONLY the filled operator configuration in YAML format:

```yaml
name: $operator_type_operation
type: sample
method: [uniform/stratified]
samples: [fraction_or_count]
stratify_key: [field_for_balancing]
random_state: 42
```
""")

TOPK_OPERATOR_PROMPT = Template("""
You are an expert at generating DocETL topk operator configurations.

Generate the detailed configuration for a TOPK operator:

OPERATOR PURPOSE: $operator_purpose
QUERY: $query

DATASET SAMPLE:
$dataset_samples

AVAILABLE FIELDS AT THIS STAGE:
$available_fields

PREVIOUS OPERATORS IN PIPELINE:
$previous_operators

CURRENT OPERATOR FRAMEWORK:
$operator_framework

CRITICAL TOPK OPERATOR RULES:
- TopK retrieves the most relevant documents using embeddings or keywords
- method: "embedding" for semantic search, "keyword" for text matching
- k: number of documents to retrieve
- keys: fields to search within: $available_fields
- query: search query string
- embedding_model: model for embedding-based search

Example configuration:
- method: "embedding", k: 5, keys: ["title", "content"]
- query: "machine learning applications"

Fill in the "TO_BE_GENERATED" placeholders with appropriate values.

Return ONLY the filled operator configuration in YAML format:

```yaml
name: $operator_type_operation
type: topk
method: [embedding/keyword]
k: [number_of_documents]
keys: [fields_to_search]
query: [search_query_string]
```
""")

# Generic fallback prompt for any unlisted operators
GENERIC_OPERATOR_PROMPT = Template("""
You are an expert at generating DocETL operator configurations.

Generate the detailed configuration for the following operator:

OPERATOR TYPE: $operator_type
OPERATOR PURPOSE: $operator_purpose

QUERY: $query

DATASET SAMPLE:
$dataset_samples

AVAILABLE FIELDS AT THIS STAGE:
$available_fields

PREVIOUS OPERATORS IN PIPELINE:
$previous_operators

CURRENT OPERATOR FRAMEWORK:
$operator_framework

Fill in all the "TO_BE_GENERATED" placeholders with appropriate values.

IMPORTANT FIELD RULES:
- You can ONLY reference fields listed in "AVAILABLE FIELDS AT THIS STAGE"
- Fields from previous operators' outputs are included in available fields
- Do NOT reference fields that don't exist yet
- When creating output schemas, choose field names that don't conflict with existing fields

Return ONLY the filled operator configuration in YAML format.

```yaml
name: $operator_type_operation
type: $operator_type
# ... fill in all fields with actual values, no placeholders
```
""")

# Operator definitions with detailed examples organized by type
OPERATOR_DEFINITIONS = {
    "map": """
Map Operator — Per-document transformation using an LLM.

FIELD ACCESS: Use {{ input.field_name }} to access document fields.

Example:
```yaml
- name: analyze_articles
  type: map
  prompt: |
    Analyze the following article:
    Title: {{ input.title }}
    Content: {{ input.content }}

    Extract:
    1. Main topic (1-3 words)
    2. Summary (2-3 sentences)
    3. Sentiment (positive/negative/neutral)
  output:
    schema:
      main_topic: string
      summary: string
      sentiment: string
```
""",

    "filter": """
Filter Operator — Keep or discard documents by returning a boolean.

FIELD ACCESS: Use {{ input.field_name }} to access document fields.

Example:
```yaml
- name: filter_high_quality
  type: filter
  prompt: |
    Should we keep this article?
    Title: {{ input.title }}
    Quality Score: {{ input.quality_score }}

    Return "true" if quality_score > 7, else "false".
  output:
    schema:
      keep_article: boolean
```
""",

    "reduce": """
Reduce Operator — Aggregate over groups using a reduce_key.

Example:
```yaml
- name: summarize_by_category
  type: reduce
  reduce_key: category
  prompt: |
    Summarize articles for category {{ inputs[0].category }}:
    {% for article in inputs %}
    - Title: {{ article.title }}
    - Content: {{ article.content }}
    {% endfor %}

    Provide a comprehensive summary.
  output:
    schema:
      category_summary: string
```
""",

    "resolve": """
Resolve Operator — Deduplicate or standardize entities.

FIELD ACCESS:
- comparison_prompt: {{ input1.field }} and {{ input2.field }}
- resolution_prompt: {{ inputs }} for merging

Example:
```yaml
- name: standardize_names
  type: resolve
  optimize: true
  comparison_prompt: |
    Are these the same person?
    Person 1: {{ input1.name }} ({{ input1.email }})
    Person 2: {{ input2.name }} ({{ input2.email }})
    Return "True" or "False".
  resolution_prompt: |
    Standardize these names:
    {% for person in inputs %}
    - {{ person.name }}
    {% endfor %}
    Return the canonical name.
  output:
    schema:
      canonical_name: string
```
""",

    "rank": """
Rank Operator — Order items by custom criteria using LLM scoring.

FIELD ACCESS: Specify input_keys with fields to consider.

Example:
```yaml
- name: rank_by_relevance
  type: rank
  prompt: |
    Rank these documents by relevance and quality.
    Consider title clarity, content depth, and overall usefulness.
  input_keys: ["title", "content", "metadata"]
  direction: desc
```
""",

    "extract": """
Extract Operator — Pull verbatim text sections from documents. The difference between this operator and the map operator is that the extract operator fully copies text spans from the original texts.

Example:
```yaml
- name: extract_findings
  type: extract
  prompt: |
    Extract key findings, conclusions, and important quotes.
    Focus on:
    - Research results
    - Statistical data
    - Main conclusions
  document_keys: ["content", "abstract"]
```
""",

    "cluster": """
Cluster Operator — Group similar documents using embeddings.

FIELD ACCESS:
- embedding_keys: fields to use for similarity
- summary_prompt: use {{ inputs }} for cluster description

Example:
```yaml
- name: cluster_topics
  type: cluster
  embedding_keys: ["title", "content"]
  output_key: topic_cluster
  summary_prompt: |
    Describe this cluster of related articles:
    {% for article in inputs %}
    - {{ article.title }}
    {% endfor %}
    What topic connects these articles?
```
""",

    "split": """
Split Operator — Break long text into chunks.

FIELD ACCESS: Specify split_key with field to split.

Example:
```yaml
- name: split_documents
  type: split
  split_key: content
  method: token_count
  method_kwargs:
    num_tokens: 500
    model: azure/gpt-4o
```
""",

    "gather": """
Gather Operator — Add surrounding context to chunks.

FIELD ACCESS: Specify keys for content, document ID, and order.

Example:
```yaml
- name: add_context
  type: gather
  content_key: text_chunk
  doc_id_key: document_id
  order_key: chunk_number
  peripheral_chunks:
    previous:
      count: 1
    next:
      count: 1
```
""",

    "unnest": """
Unnest Operator — Expand arrays or nested fields.

FIELD ACCESS: Specify unnest_key with field containing array/nested data.

Example:
```yaml
- name: expand_topics
  type: unnest
  unnest_key: topics_array
  recursive: true
  depth: 2
```
""",

    "sample": """
Sample Operator — Select subset of documents.

FIELD ACCESS: Specify stratify_key for balanced sampling.

Example:
```yaml
- name: sample_data
  type: sample
  method: stratified
  samples: 0.1
  stratify_key: category
  random_state: 42
```
""",

    "topk": """
TopK Operator — Retrieve most relevant documents.

FIELD ACCESS: Specify keys with fields to search within.

Example:
```yaml
- name: find_relevant
  type: topk
  method: embedding
  k: 5
  keys: ["title", "content"]
  query: "machine learning applications"
  embedding_model: text-embedding-3-small
```
"""
}

# Validation prompt for checking operator consistency
OPERATOR_VALIDATION_PROMPT = Template("""
Check if the following operator configuration is valid and consistent:

OPERATOR:
$operator_config

PREVIOUS OPERATORS OUTPUT:
$previous_outputs

DATASET FIELDS:
$dataset_fields

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
""")
