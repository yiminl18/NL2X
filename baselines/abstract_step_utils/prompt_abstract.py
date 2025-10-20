"""
Abstract Layer Prompts
"""

from string import Template
from abstract.support import BaseSystem, format_operators_with_descriptions
from .dynamic_inst import get_operator_selection_rules


# =============================================================================
# Step 1: Operator Selection
# =============================================================================

ABSTRACT_OPERATOR_SELECTION_PROMPT = Template("""
You are an AI assistant that designs data processing pipelines using abstract operators.

Task: Analyze the query and dataset, then select appropriate abstract operators to build a pipeline.

Query: $query

Dataset Structure and Samples:
$dataset_samples

Available Abstract Operators:
$available_operators

COMPLETE EXAMPLES FROM REAL PIPELINES:

Example 1: Presidential Debate Themes Analysis
Query: "Extract themes and viewpoints from debate transcripts and analyze how those themes evolve over time across multiple debates"
Dataset: Collection of presidential debate transcripts with fields: title, date, year, content

Selected operators and reasoning:
- Map: extract themes and viewpoints from each debate transcript (transforms each debate into structured themes with viewpoints)
- Unnest: expand the themes array into individual theme records (needed because Map outputs a list of themes per debate)
- Reduce: aggregate viewpoints by theme to analyze evolution over time (groups all instances of the same theme across debates)

Example 2: Mining Product Reviews for Polarizing Themes
Query: "Identify polarizing themes in video game reviews that divide player opinions, resolve similar themes across reviews, and aggregate them to find common polarizing themes across different games"
Dataset: Video game reviews with fields: app_name, concatenated_reviews

Selected operators and reasoning:
- Map: identify polarizing themes from concatenated reviews (analyzes each game's reviews to find divisive topics)
- Unnest: expand polarizing_themes array into individual theme records (needed to process each theme separately)
- Resolve: deduplicate and consolidate similar themes (merges themes that are essentially the same but worded differently)
- Reduce: aggregate common themes across different games by theme (groups resolved themes to find patterns across games)

Based on the query and dataset, list the operators needed in the order they should be applied.
For each operator, provide:
- The operator type
- A brief description of what this operator will do in the pipeline

IMPORTANT:
$important_rules

Return a JSON object with this structure:
{
  "operators": [
    {"type": "Map", "purpose": "Extract key information from text"},
    {"type": "Filter", "purpose": "Keep only relevant records"},
    {"type": "Reduce", "purpose": "Aggregate results by category"}
  ]
}

Remember: Keep the pipeline simple and focused on answering the query.
""")


def get_operator_selection_prompt(
    query: str,
    dataset_samples: str,
    base_system: BaseSystem
) -> str:
    """
    Generate operator selection prompt based on base system support.

    Args:
        query: User query
        dataset_samples: Dataset samples (JSON formatted string)
        base_system: Base system type

    Returns:
        Complete prompt string
    """
    operators_info = format_operators_with_descriptions(base_system)
    rules_text = get_operator_selection_rules()
    return ABSTRACT_OPERATOR_SELECTION_PROMPT.substitute(
        query=query,
        dataset_samples=dataset_samples,
        available_operators=operators_info,
        important_rules=rules_text
    )


# =============================================================================
# Step 2: Detailed Operator Configuration Prompts
# =============================================================================

ABSTRACT_MAP_PROMPT = Template("""
Generate a Map operator configuration in abstract layer format.

A Map operator transforms EACH record independently using an LLM (1-to-1 mapping). It analyzes, summarizes, or transforms data to generate new fields. Unlike Extract, Map does NOT pull verbatim text - it performs transformations (e.g., sentiment analysis, summarization, structured extraction from analysis).

Context:
- Query: $query
- Operator Purpose: $operator_purpose
- Available Fields (with types): $available_fields
- Previous Operators: $previous_operators

Dataset Samples:
$dataset_samples

Generate a Map operator with:

1. **prompt**: A Jinja2 template string that describes the transformation
   - CRITICAL: MUST use {{ input.field_name }} to reference ALL input fields
   - Be specific about what to extract or transform
   - ✓ CORRECT: "Extract the person's name from: {{ input.text }}"
   - ✗ WRONG: "Extract the person's name from the field `text`"
   - ✗ WRONG: "Extract from the 'text' field"
   - ✗ WRONG: "Extract from field text"
   - For each field to be output, explain the meaning of it, for example, "medication: the medication contained in the doctor's prescription in the document" is better than "medication".

2. **input**: Specify input schema as {"fields": {"field1": "Type1", "field2": "Type2"}}
   - Only include fields you actually use in the prompt
   - Use the correct types from available fields
   - Example: {"fields": {"text": "String", "id": "Integer"}}

3. **output**: Specify output schema as {"field1": "Type1", "field2": "Type2", ...}
   - IMPORTANT: DO NOT include input fields in output (they are preserved automatically)
   - ONLY add NEW fields generated by this operator
   - Input fields are automatically carried forward, so only list the fields you CREATE
   - Use proper types with COMPLETE type specifications:
     * Basic types: String, Integer, Float, Boolean
     * List types: MUST specify element type
       - List[String] for string lists
       - List[Integer] for integer lists
       - List[Dict[{name: String, value: Integer}]] for lists of objects
     * Dict types: Can optionally specify fields
       - Dict for generic dictionary
       - Dict[{field1: String, field2: Integer}] for structured dictionary
   - Examples:
     * {"id": "Integer", "text": "String", "name": "String", "age": "Integer"}
     * {"names": "List[String]", "scores": "List[Integer]"}
     * {"metadata": "Dict[{created: String, count: Integer}]"}
     * {"entities": "List[Dict[{name: String, type: String}]]"}

4. **properties**: Additional configuration (can be empty {})
   - Only add if necessary (e.g., {"model": "gpt-4"})

COMPLETE EXAMPLES:

Example 1 - Extracting structured data:
{
  "prompt": "Analyze the text: {{ input.content }}. Extract the main themes discussed. Return a list of theme names.",
  "input": {"fields": {"content": "String"}},
  "output": {
    "themes": "List[String]",
    "summary": "String"
  },
  "properties": {}
}

Example 2 - Sentiment analysis:
{
  "prompt": "Analyze sentiment of: {{ input.review_text }}. Return sentiment label and confidence score.",
  "input": {"fields": {"review_text": "String"}},
  "output": {
    "sentiment": "String",
    "confidence": "Float"
  },
  "properties": {}
}

Return JSON:
{
  "prompt": "...",
  "input": {"fields": {...}},
  "output": {...},
  "properties": {...}
}

IMPORTANT:
- Use ONLY available fields in input
- DO NOT include input fields in output schema (they are automatically preserved)
- ONLY list NEW fields created by this operator in the output schema
- Make prompts specific to the task, not generic
- ALWAYS use Jinja2 format for field references: {{ input.field_name }}
- NEVER use bare field names like `field` or "field" - ALWAYS use {{ input.field }}
""")

ABSTRACT_FILTER_PROMPT = Template("""
Generate a Filter operator configuration in abstract layer format.

A Filter operator keeps or discards records based on a condition.

Context:
- Query: $query
- Operator Purpose: $operator_purpose
- Available Fields (with types): $available_fields
- Previous Operators: $previous_operators

Dataset Samples:
$dataset_samples

Generate a Filter operator with:

1. **prompt**: A Jinja2 template that describes the filtering condition
   - MUST use {{ input.field_name }} to reference fields (Jinja2 template)
   - The prompt should ask for "true" or "false" as the response
   - Clearly state what records should PASS the filter (return true to keep)
   - Example: "Is this relevant? {{ input.title }}. Return true to keep, false to discard."

2. **input**: Specify input schema
   - {"fields": {"field1": "Type1", ...}}
   - Only include fields used in the condition
   - Example: {"fields": {"score": "Float"}}

3. **output**: Specify output schema
   - MUST include a Boolean field for the filter decision
   - Include all input fields (filters preserve all fields)
   - Use complete type specifications including List[Type] and Dict[{fields}]
   - Example: {"id": "Integer", "text": "String", "score": "Float", "keep": "Boolean"}

4. **properties**: Additional configuration (can be empty {})

COMPLETE EXAMPLES:

Example 1 - Filter by relevance:
{
  "prompt": "Is this document relevant to the topic? Title: {{ input.title }}. Content: {{ input.content }}. Return true to keep this document, false to discard it.",
  "input": {"fields": {"title": "String", "content": "String"}},
  "output": {
    "title": "String",
    "content": "String",
    "keep": "Boolean"
  },
  "properties": {}
}

Example 2 - Filter by score threshold:
{
  "prompt": "Check the quality score: {{ input.quality_score }}. Return true if the score is greater than or equal to 7, otherwise return false.",
  "input": {"fields": {"quality_score": "Float"}},
  "output": {
    "quality_score": "Float",
    "passes_threshold": "Boolean"
  },
  "properties": {}
}

Return JSON:
{
  "prompt": "...",
  "input": {"fields": {...}},
  "output": {...},
  "properties": {...}
}

IMPORTANT:
- The output schema MUST have a Boolean field
- Filter output includes all input fields plus the Boolean decision field
- Make the filtering condition clear and specific
- Use Jinja2 format: {{ input.field_name }}
""")

ABSTRACT_REDUCE_PROMPT = Template("""
Generate a Reduce operator configuration in abstract layer format.

A Reduce operator aggregates/groups records by a key field.

Context:
- Query: $query
- Operator Purpose: $operator_purpose
- Available Fields (with types): $available_fields
- Previous Operators: $previous_operators

Dataset Samples:
$dataset_samples

Generate a Reduce operator with:

1. **reduce_key**: The field to group by (must exist in available fields)
   - Example: "category" (if grouping by category)

2. **prompt**: Jinja2 template describing how to aggregate
   - MUST use {{ inputs }} (plural) to reference the group of records (Jinja2 template)
   - Access fields like: {{ inputs[0].field_name }} or {% for item in inputs %}{{ item.field_name }}{% endfor %}
   - Example: "Summarize all viewpoints for theme '{{ inputs[0].theme }}': {% for item in inputs %}{{ item.viewpoint }} {% endfor %}"

3. **input**: Specify input schema
   - {"fields": {"field1": "Type1", ...}}
   - Include reduce_key and fields used in aggregation

4. **output**: Specify output schema
   - MUST include the reduce_key field
   - Add new aggregated fields
   - The output schema MUST specify field types: "String", "Integer", "Float", "Boolean", "List[...]", "Dict[{...}]"
   - Examples:
     * {"category": "String", "summary": "String", "count": "Integer"}
     * {"group_id": "Integer", "items": "List[String]", "stats": "Dict[{avg: Float, max: Float}]"}

5. **properties**: Additional configuration (can be empty {})

COMPLETE EXAMPLES:

Example 1 - Aggregate themes by category:
{
  "reduce_key": "theme",
  "prompt": "Summarize all viewpoints for the theme '{{ inputs[0].theme }}'. Here are all the viewpoints: {% for item in inputs %}{{ item.viewpoint }}. {% endfor %}Provide: aggregated_summary: a comprehensive summary combining all viewpoints; count: the total number of viewpoints.",
  "input": {"fields": {"theme": "String", "viewpoint": "String"}},
  "output": {
    "theme": "String",
    "aggregated_summary": "String",
    "count": "Integer"
  },
  "properties": {}
}

Example 2 - Aggregate reviews by product:
{
  "reduce_key": "product_id",
  "prompt": "Product: {{ inputs[0].product_name }}. Reviews: {% for review in inputs %}{{ review.text }}. {% endfor %}Provide: product_name: the name of the product; common_themes: a list of common themes mentioned across reviews; average_sentiment: the overall average sentiment (positive/negative/neutral).",
  "input": {"fields": {"product_id": "String", "product_name": "String", "text": "String"}},
  "output": {
    "product_id": "String",
    "product_name": "String",
    "common_themes": "List[String]",
    "average_sentiment": "String"
  },
  "properties": {}
}

Return JSON:
{
  "reduce_key": "...",
  "prompt": "...",
  "input": {"fields": {...}},
  "output": {...},
  "properties": {...}
}

IMPORTANT:
- You MUST specify a reduce_key field that exists in available fields
- You MUST use {{ inputs }} (plural) to reference the group of records
- Output must include the reduce_key field
- Create aggregated output schema with new field names
""")

ABSTRACT_RESOLVE_PROMPT = Template("""
Generate a Resolve operator configuration in abstract layer format.

A Resolve operator deduplicates or standardizes entities using comparison and resolution prompts.

Context:
- Query: $query
- Operator Purpose: $operator_purpose
- Available Fields (with types): $available_fields
- Previous Operators: $previous_operators

Dataset Samples:
$dataset_samples

Generate a Resolve operator with:

1. **comparison_prompt**: Jinja2 template for comparing two records
   - Uses {{ input1.field }} and {{ input2.field }} to compare two items
   - Should return "True" or "False"
   - Example: "Are {{ input1.name }} and {{ input2.name }} the same person? Return True or False."

2. **resolution_prompt**: Jinja2 template for merging records
   - Uses {{ inputs }} to merge multiple similar items
   - Describe how to create one standardized/merged record
   - Example: "Merge these person records: {% for item in inputs %}{{ item.name }}, {{ item.email }} {% endfor %}"

3. **input**: Specify input schema
   - {"fields": {"field1": "Type1", ...}}

4. **output**: Specify output schema (resolved/standardized entity schema)
   - The output schema MUST specify field types: "String", "Integer", "Float", "Boolean", "List[...]", "Dict[{...}]"
   - Create output schema for the resolved/standardized entity

5. **properties**: Additional configuration
   - Set "optimize": true for better performance (recommended)
   - Example: {"optimize": true}

COMPLETE EXAMPLES:

Example 1 - Resolve duplicate person names:
{
  "comparison_prompt": "Are {{ input1.name }} and {{ input2.name }} the same person? Compare their emails: {{ input1.email }} vs {{ input2.email }}. Return True if they are the same person, False otherwise.",
  "resolution_prompt": "Merge these person records: {% for item in inputs %}Name: {{ item.name }}, Email: {{ item.email }}. {% endfor %}Provide: canonical_name: the most complete/correct name; canonical_email: the primary email address.",
  "input": {"fields": {"name": "String", "email": "String"}},
  "output": {
    "canonical_name": "String",
    "canonical_email": "String"
  },
  "properties": {"optimize": true}
}

Example 2 - Resolve similar themes:
{
  "comparison_prompt": "Are the themes '{{ input1.theme_name }}' and '{{ input2.theme_name }}' similar or referring to the same concept? Consider their descriptions: {{ input1.description }} vs {{ input2.description }}. Return True if similar, False otherwise.",
  "resolution_prompt": "Consolidate these similar themes into one unified theme: {% for t in inputs %}Theme: {{ t.theme_name }}, Description: {{ t.description }}. {% endfor %}Provide: unified_theme: a single unified theme name; consolidated_description: a comprehensive description combining all themes.",
  "input": {"fields": {"theme_name": "String", "description": "String"}},
  "output": {
    "unified_theme": "String",
    "consolidated_description": "String"
  },
  "properties": {"optimize": true}
}

Return JSON:
{
  "comparison_prompt": "...",
  "resolution_prompt": "...",
  "input": {"fields": {...}},
  "output": {...},
  "properties": {...}
}
""")

ABSTRACT_EXTRACT_PROMPT = Template("""
Generate an Extract operator configuration in abstract layer format.

An Extract operator pulls verbatim text sections from documents. It extracts exact quotes, specific text spans, or passages without transformation - the output is the original text as-is.

CRITICAL EXTRACT OPERATOR RULES:
- Extract can ONLY output a SINGLE string field containing the extracted text
- Extract is NOT applicable for tasks needing analysis, transformation, or multiple output fields
- Extract pulls verbatim text sections from documents (exact text as-is)
- Use Extract when you need the original text exactly as written
- Use Map when you need analysis, summarization, structured extraction, or multiple output fields

Context:
- Query: $query
- Operator Purpose: $operator_purpose
- Available Fields (with types): $available_fields
- Previous Operators: $previous_operators

Dataset Samples:
$dataset_samples

Generate an Extract operator with:

1. **prompt**: Jinja2 template describing what text section to extract
   - Use {{ input.field }} to reference the document field
   - Be specific about what text sections to extract verbatim
   - Example: "Extract the key findings and conclusions section from: {{ input.content }}"

2. **document_keys**: List of field names containing the documents to extract from
   - Usually ["src"] or the text field name
   - Example: ["document"] or ["content"] or ["text"]

3. **input**: Specify input schema
   - {"fields": {"field1": "Type1", ...}}
   - Include the document field(s) specified in document_keys

4. **output**: DO NOT specify output schema for Extract operator
   - Extract automatically adds the extracted content as a new field
   - The extracted content is stored in a field with an appropriate suffix
   - Input fields are automatically preserved
   - IMPORTANT: Extract does NOT support custom output schemas with multiple fields

5. **properties**: Additional configuration (can be empty {})

COMPLETE EXAMPLES:

Example 1 - Extract research findings:
{
  "prompt": "Extract the key findings, conclusions, and important quotes from this research article: {{ input.content }}. Focus on: research results, statistical data, and main conclusions. Return the extracted text verbatim.",
  "document_keys": ["content"],
  "input": {"fields": {"content": "String", "article_id": "Integer"}},
  "properties": {}
}

Example 2 - Extract legal clauses:
{
  "prompt": "Extract the liability and indemnification clauses verbatim from this legal contract: {{ input.contract_text }}. Include the exact wording of these specific sections.",
  "document_keys": ["contract_text"],
  "input": {"fields": {"contract_text": "String", "contract_name": "String"}},
  "properties": {}
}

Return JSON:
{
  "prompt": "...",
  "document_keys": [...],
  "input": {"fields": {...}},
  "properties": {...}
}

IMPORTANT:
- Extract outputs ONLY a single string field with the extracted verbatim text
- DO NOT include an "output" key in the JSON response
- For tasks requiring multiple structured fields, use Map operator instead
- Extract is for pulling exact text sections, not for analysis or transformation
""")

# Map of operator types to their prompts
OPERATOR_TYPE_TO_PROMPT = {
    'Map': ABSTRACT_MAP_PROMPT,
    'Filter': ABSTRACT_FILTER_PROMPT,
    'Reduce': ABSTRACT_REDUCE_PROMPT,
    'Resolve': ABSTRACT_RESOLVE_PROMPT,
    'Extract': ABSTRACT_EXTRACT_PROMPT,
    # Additional operators can be added later
}


def get_abstract_operator_prompt(
    operator_type: str,
    operator_purpose: str,
    query: str,
    dataset_samples: str,
    previous_operators: str,
    available_fields: str
) -> str:
    """
    Get the appropriate prompt based on operator type.

    Args:
        operator_type: Operator type (Map, Filter, etc.)
        operator_purpose: Operator purpose
        query: User query
        dataset_samples: Dataset samples
        previous_operators: Previous operators (JSON string)
        available_fields: Available fields (formatted string)

    Returns:
        Complete prompt string

    Raises:
        ValueError: If operator type is not supported
    """
    prompt_template = OPERATOR_TYPE_TO_PROMPT.get(operator_type)

    if prompt_template is None:
        # For operators not yet implemented, return a generic prompt
        return f"""
Generate a {operator_type} operator configuration in abstract layer format.

Context:
- Query: {query}
- Operator Purpose: {operator_purpose}
- Available Fields: {available_fields}
- Previous Operators: {previous_operators}

Dataset Samples:
{dataset_samples}

Please generate appropriate configuration for this {operator_type} operator.
Include: input schema, output schema, and any operator-specific fields.
Return valid JSON.
"""

    return prompt_template.substitute(
        query=query,
        operator_purpose=operator_purpose,
        available_fields=available_fields,
        previous_operators=previous_operators,
        dataset_samples=dataset_samples
    )
