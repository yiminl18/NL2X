INSTRUCTION_PROMPT = """
**DocETL Pipeline Generation Instructions**

**Purpose & Scope**
DocETL provides a declarative, YAML-driven interface for defining pipelines that process unstructured datasets using LLM and custom code. Pipelines operate over files, perform map/filter/reduce/resolve tasks, enforce schema-validated outputs, and include optional code blocks and optimizations.

---

### Dataset Handling

1. **Supported Formats**

   * *JSON*: root must be a list of dict; fields accessible via `input.<field>`.
   * *CSV*: header row required; fields accessible via `input.<column>`.

2. **Dataset Config Example**

   ```yaml
   datasets:
     documents:
       type: file
       path: "data.json"
   ```

3. **Non-Standard Data**
   If the data includes audio, PDFs, or other formats, use parsing tools (e.g., Whisper or OCR) to convert them into structured text fields in an initial code operation.

4. **Consistency**
   Keep field names consistent across all dataset entries (e.g., always use `text` if multiple names exist, or generate a code operation to extract the primary field).

---

### Pipeline Structure

* **Top-level keys** must include:

  * `default_model` (model selection)
  * `datasets` (data sources)
  * `operations` (reusable steps)
  * `pipeline` (sequence of steps and final output)
  * Optional: `system_prompt` for dataset description and persona hints.

* **System Prompt (Optional)**

  ```yaml
  system_prompt:
    dataset_description: brief description of your data
    persona: role the LLM should assume
  ```

---

### Operation Types

DocETL supports the following operation types. Use them singly or in combination to form pipelines:

1. **Map** — Per-document transformation using an LLM or code. Access fields via `input.<field>`.
2. **Filter** — Keep or discard documents by returning a boolean.
3. **Reduce** — Aggregate over groups using a `reduce_key`, combining multiple items into one. Requires `reduce_key` to be specified.
4. **Resolve** — Deduplicate or standardize entities by comparing and merging inputs; set `optimize: true` for realistic workloads.
5. **Rank** — Order items according to custom criteria, often with LLM-assisted scoring.
6. **Extract** — Pull verbatim sections of text matching patterns or prompts (e.g., key findings).
7. **Cluster** — Group items into hierarchical categories using embeddings and summarization.
8. **Split** — Break long text fields into chunks by tokens, sentences, or rules.
9. **Gather** — Re-attach neighboring context (previous/next chunks, headers) after splitting.
10. **Unnest** — Expand array or nested fields into separate items.
11. **Sample** — Subset the dataset uniformly or stratified for debugging or prototyping.
12. **TopK** — Retrieve the top-k most relevant documents via embeddings, keywords, or LLM comparison. It's used to doc retrieval but not rank already loaded docs.
13. **Code Operations** — Deterministic transforms with Python:

    * `code_map`: per-document transformation
    * `code_filter`: keep/discard via boolean function
    * `code_reduce`: aggregate with deterministic logic

---

### Schemas & Types

* Use a schema to validate the structure of outputs:

  * `string`, `integer`, `number`, `boolean`
  * `list[...]` for lists of items
  * dict via `{ field: type }`
  * `enum[...]` only when the prompt explicitly lists all possible values
* Keep schemas **simple and flat**. Prefer strings or simple dict unless nested lists are required.
* Add validation rules judiciously and only when they materially improve reliability.

---

### Prompting Guidelines

* Use Jinja2 templating:

  * `input.<field>` for map/filter
  * `inputs` for reduce operations (plural)
  * `input1` and `input2` in resolve operations
* Be specific about fields you refer to and the exact format of the expected output.
* Match the requested output to the schema exactly (e.g., if your schema expects a list of dict, the prompt should ask for that list explicitly).

---

### Validation

* **Simple Validation**: Write boolean expressions that reference the output for minimal rule checks. Validation can be added as a filed of LLM-based operatiors. Example:

  ```yaml
  validate:
    - len(output["insights"]) >= 2
  ```

---

### Operator Examples

Below are concise examples for every DocETL operator. Each example includes a one‑sentence description and a YAML snippet demonstrating correct usage.

#### Map

Applies a transformation to each document independently, such as extracting topics and summarizing news articles.

```yaml
- name: analyze_news_article
  type: map
  prompt: |
    Analyze the following news article:
    "{{ input.article }}"

    Provide the following information:
    1. Main topic (1-3 words)
    2. Summary (2-3 sentences)
    3. Key entities mentioned (list up to 5, with brief descriptions)
    4. Sentiment towards the main topic (positive, negative, or neutral)
    5. Potential biases or slants in reporting (if any)
    6. Credibility score (1-10)
  output:
    schema:
      main_topic: string
      summary: string
      key_entities: list[{name: string, description: string}]
      sentiment: string
      biases: list[string]
      credibility_score: integer
```



#### Resolve

Identifies and canonicalizes duplicate or variant entities by comparing and merging entries.

```yaml
- name: standardize_patient_names
  type: resolve
  optimize: true
  comparison_prompt: |
    Compare:
    Patient 1: {{ input1.patient_name }}
    DOB 1: {{ input1.date_of_birth }}
    Patient 2: {{ input2.patient_name }}
    DOB 2: {{ input2.date_of_birth }}
    Are these entries likely the same patient? Respond "True" or "False".
  resolution_prompt: |
    Standardize these patient names:
    {% for e in inputs %}
    - {{ e.patient_name }}
    {% endfor %}
    Return "LastName, FirstName MiddleInitial".
  output:
    schema:
      patient_name: string
```



#### Reduce

Aggregates groups of documents into a single result, such as summarizing customer feedback by department. `reduce_key` must be specified.

```yaml
- name: summarize_feedback
  type: reduce
  reduce_key: department
  prompt: |
    Summarize the customer feedback for the {{ inputs[0].department }} department:
    {% for item in inputs %}
    Feedback {{ loop.index }}: {{ item.feedback }}
    {% endfor %}
    Provide a concise summary of the main points and overall sentiment.
  output:
    schema:
      summary: string
      sentiment: string
```



#### Filter

Keeps or discards documents based on a boolean condition, such as identifying high‑impact news articles.

```yaml
- name: filter_high_impact_articles
  type: filter
  prompt: |
    Analyze the following news article:
    Title: "{{ input.title }}"
    Content: "{{ input.content }}"
    Determine if this article is high-impact based on:
    1. Covers a significant global or national event
    2. Has potential long-term consequences
    3. Affects many people
    4. Is from a reputable source
    Respond "true" if at least 3 criteria are met, else "false".
  output:
    schema:
      is_high_impact: boolean
```



#### Rank

Sorts documents according to complex criteria using LLM‑assisted scoring, such as ranking political debates by controversy. Rank will leave a `rank` field in the output items.

```yaml
- name: rank_by_controversy
  type: rank
  prompt: |
    Order these debate transcripts based on how controversial the discussion is.
    Consider factors like disagreement, divisive topics, emotional language,
    conflicting viewpoints, and public reaction. Rank the most controversial highest.
  input_keys: ["content", "title", "date"]
  direction: desc
  rerank_call_budget: 10
  initial_ordering_method: "likert"
```



#### Extract

Pulls specific sections of text verbatim without summarization, ideal for isolating key findings in research reports.

```yaml
- name: findings
  type: extract
  prompt: |
    Extract all sections that discuss key findings, results, or conclusions:
    - Summarize experimental outcomes
    - Present statistical results
    - Describe discovered insights
    - State conclusions
    Only extract the most important and substantive findings.
  document_keys: ["report_text"]
  model: gpt-4.1-mini
```



#### Cluster

Groups items into hierarchical clusters based on embeddings, summarizing higher‑level categories for related concepts.

```yaml
- name: cluster_concepts
  type: cluster
  max_batch_size: 5
  embedding_keys:
    - concept
    - description
  output_key: categories
  summary_schema:
    concept: str
    description: str
  summary_prompt: |
    The following describes two related concepts. What concept encompasses both?
    If one concept subsumes the other, use that concept.
    {% for input in inputs %}
    {{ input.concept }}:
    {{ input.description }}
    {% endfor %}
    Provide the title and description of the super‑concept.
```



#### Split

Divides long text into manageable chunks for further processing, such as splitting customer support transcripts.

```yaml
- name: split_transcript
  type: split
  split_key: transcript
  method: token_count
  method_kwargs:
    num_tokens: 500
    model: gpt-4o-mini
```



#### Gather

Collects surrounding context for each chunk after splitting, ensuring downstream operations have sufficient information.

```yaml
- name: context_gatherer
  type: gather
  content_key: agreement_text_chunk
  doc_id_key: split_merger_agreement_id
  order_key: split_merger_agreement_chunk_num
  peripheral_chunks:
    previous:
      middle:
        content_key: agreement_text_chunk_summary
      tail:
        content_key: agreement_text_chunk
    next:
      head:
        count: 1
        content_key: agreement_text_chunk
  doc_header_key: headers
```



#### Unnest

Expands array or nested fields into individual items to enable granular analysis, such as analyzing salient quotes from product reviews.
For list-type unnesting: It generates multiple output itemes for each input item, replacing the original array in the `unnest_key` field with individual elements.
For dictionary-type unnesting: It expands the specified filds into the parent dictionary.  To fully flatten the nested lists, use two sequential `unnest` operations with the same `unnest_key`.

```yaml
- name: unnest_quotes
  type: unnest
  unnest_key: salient_quotes
```

#### Sample

Selects a subset of items from the input dataset for debugging or initial development, maintaining distribution via stratification if needed.

```yaml
- name: sample_concepts
  type: sample
  method: uniform
  samples: 0.1
  stratify_key: category
  random_state: 42
```

#### TopK

Retrieves the top‑k most relevant documents using embeddings, keyword search, or LLM comparisons; here we perform semantic search with embeddings.

```yaml
- name: find_relevant_tickets
  type: topk
  method: embedding
  k: 5
  keys:
    - subject
    - description
    - customer_feedback
  query: "payment processing errors with international transactions"
  embedding_model: text-embedding-3-small
```



#### Code

Executes deterministic Python transformations instead of LLM prompts; this example extracts keywords from text/average category volume/filter the documents.

```yaml
- name: extract_keywords
  type: code_map
  code: |
    def transform(doc) -> dict:
        keywords = doc['text'].lower().split()
        return {
            'keywords': keywords,
            'keyword_count': len(keywords)
        }
```

```yaml
- name: aggregate_stats
  type: code_reduce
  reduce_key: category
  code: |
    def transform(items) -> dict:
        total = sum(item['value'] for item in items)
        avg = total / len(items)
        return {
            'total': total,
            'average': avg,
            'count': len(items)
        }
```

```yaml
- name: filter_valid_entries
  type: code_filter  
  code: |
    def transform(doc) -> bool:
        # Return True to keep the document, False to filter it out
        return doc['score'] >= 0.5 and len(doc['text']) > 100
```

---


### Full Pipeline Examples

#### Presidential Debate Themes Analysis

This pipeline extracts themes and viewpoints from debate transcripts, then analyzes how those themes evolve over time.
**YAML:**

```yaml
datasets:
  debates:
    type: file
    path: "data.json"

system_prompt:
  dataset_description: a collection of transcripts of presidential debates
  persona: a political analyst

default_model: gpt-4o-mini

operations:
  - name: extract_themes_and_viewpoints
    type: map
    output:
      schema:
        themes: "list[{theme: str, viewpoints: str}]"
    prompt: |
      Analyze the following debate transcript for {{ input.title }} on {{ input.date }}:

      {{ input.content }}

      Extract the main themes discussed in this debate and the viewpoints of the candidates on these themes.
      Return a list of themes and corresponding viewpoints (including the specific quotes from the debate) in the following format:
      [
        {
          "theme": "Theme 1",
          "viewpoints": "Candidate A's viewpoint... Candidate B's viewpoint..."
        },
        {
          "theme": "Theme 2",
          "viewpoints": "Candidate A's viewpoint... Candidate B's viewpoint..."
        },
        ...
      ]

  - name: unnest_themes
    type: unnest
    unnest_key: themes
    recursive: true

  - name: summarize_theme_evolution
    type: reduce
    reduce_key: theme
    output:
      schema:
        theme: str
        report: str
    prompt: |
      Analyze the following viewpoints on the theme "{{ inputs[0].theme }}" from various debates over the years:

      {% for item in inputs %}
      Year: {{ item.year }}
      Date: {{ item.date }}
      Title: {{ item.title }}
      Viewpoints: {{ item.viewpoints }}

      {% endfor %}

      Generate a comprehensive summary of how Democratic and Republican viewpoints on this theme have evolved through the years. Include supporting quotes from the debates to illustrate key points or shifts in perspective.

      Your summary should:
      1. Identify *all* major trends or shifts in each party's stance over time
      2. Highlight any significant agreements or disagreements between the parties
      3. Note any external events or factors that may have influenced changes in viewpoints
      4. Use specific quotes to support your analysis
      5. The title should contain the start and end years of the analysis

      Format your response as a well-structured report.

pipeline:
  steps:
    - name: debate_analysis
      input: debates
      operations:
        - extract_themes_and_viewpoints
        - unnest_themes
        - summarize_theme_evolution

  output:
    type: file
    path: "theme_evolution_analysis.json"
    intermediate_dir: "checkpoints"
```

#### Mining Product Reviews: Identifying Polarizing Themes in Video Games

This example analyzes concatenated reviews of video games to find themes that sharply divide player opinions, resolves similar themes, and aggregates them across games.
**YAML:**

```yaml
default_model: gpt-4o-mini

system_prompt:
  dataset_description: a collection of reviews for video games
  persona: a marketing analyst analyzing player opinions and themes

datasets:
  steam_reviews:
    type: file
    path: "path/to/top_apps_steam_sample.json"

operations:
  - name: identify_polarizing_themes
    optimize: true
    type: map
    prompt: |
      Analyze the following concatenated reviews for a video game and identify polarizing themes that divide player opinions. A polarizing theme is one that some players love while others strongly dislike.

      Game: {{ input.app_name }}
      Reviews: {{ input.concatenated_reviews }}

      For each polarizing theme you identify:
      1. Provide a summary of the theme
      2. Explain why it's polarizing
      3. Include supporting quotes from both positive and negative perspectives

      Aim to identify ~10 polarizing themes, if present.

    output:
      schema:
        polarizing_themes: "list[{theme: str, summary: str, polarization_reason: str, positive_quotes: str, negative_quotes: str}]"

  - name: unnest_polarizing_themes
    type: unnest
    unnest_key: polarizing_themes
    recursive: true
    depth: 2

  - name: resolve_themes
    type: resolve
    optimize: true
    comparison_prompt: |
      Are the themes "{{ input1.theme }}" and "{{ input2.theme }}" the same?
      Here is some context to help you decide:

      Theme 1: {{ input1.theme }}
      Summary 1: {{ input1.summary }}

      Theme 2: {{ input2.theme }}
      Summary 2: {{ input2.summary }}
    resolution_prompt: |
      Given the following themes, please come up with a theme that best captures the essence of all the themes:

      {% for input in inputs %}
      Theme {{ loop.index }}: {{ input.theme }}
      {% if not loop.last %}
      ---
      {% endif %}
      {% endfor %}

      Based on these themes, provide a consolidated theme that captures the essence of all the above themes. Ensure that the consolidated theme is concise yet comprehensive.
    output:
      schema:
        theme: str

  - name: aggregate_common_themes
    type: reduce
    optimize: true
    reduce_key: theme
    prompt: |
      You are given a theme and summary that appears across multiple video games, along with various apps and review quotes related to this theme. Your task is to consolidate this information into a comprehensive report.

      For each input, you will receive:
      - theme: A specific polarizing theme
      - summary: A brief summary of the theme
      - app_name: The name of the game
      - positive_quotes: List of supporting quotes from positive perspectives
      - negative_quotes: List of supporting quotes from negative perspectives

      Create a report that includes:
      1. The name of the common theme
      2. A summary of the theme and why it's common across games
      3. Representative quotes from different games, both positive and negative

      Here's the information for the theme:
      Theme: {{ inputs[0].theme }}
      Summary: {{ inputs[0].summary }}

      {% for app in inputs %}
      Game: {{ app.app_name }}
      Positive Quotes: {{ app.positive_quotes }}
      Negative Quotes: {{ app.negative_quotes }}
      {% if not loop.last %}
      ----------------------------------------
      {% endif %}
      {% endfor %}
    output:
      schema:
        theme_summary: str
        representative_quotes: "list[{game: str, quote: str, sentiment: str}]"

pipeline:
  steps:
    - name: game_analysis
      input: steam_reviews
      operations:
        - identify_polarizing_themes
        - unnest_polarizing_themes
        - resolve_themes
        - aggregate_common_themes

  output:
    type: file
    path: "path/to/output_polarizing_themes.json"
    intermediate_dir: "path/to/intermediates"
```

#### Split & Gather Example: Analyzing a Legal Document

This pipeline processes a lengthy legal document (the Trump immunity case) by extracting metadata, splitting into chunks, gathering context, and identifying all people and their roles.
**YAML:**

```yaml
datasets:
  legal_doc:
    type: file
    path: /path/to/your/dataset.json
    parsing:
      - function: azure_di_read
        input_key: pdf_url
        output_key: extracted_text
        function_kwargs:
          use_url: true
          include_line_numbers: true

default_model: gpt-4o-mini

system_prompt:
  dataset_description: the Trump vs. United States case
  persona: a legal analyst

operations:
  - name: extract_metadata_find_people_and_involvements
    type: map
    model: gpt-4o-mini
    prompt: |
      Given the document excerpt: {{ input.extracted_text }}
      Extract all the people mentioned and summarize their involvements in the case described.
    output:
      schema:
        metadata: str

  - name: split_find_people_and_involvements
    type: split
    method: token_count
    method_kwargs:
      num_tokens: 3993
    split_key: extracted_text

  - name: header_extraction_extracted_text_find_people_and_involvements
    type: map
    model: gpt-4o-mini
    output:
      schema:
        headers: "list[{header: string, level: integer}]"
    prompt: |
      Analyze the following chunk of a document and extract any headers you see.

      { input.extracted_text_chunk }

      Examples of headers and their levels based on the document structure:
      - "GOVERNMENT'S MOTION FOR IMMUNITY DETERMINATIONS" (level 1)
      - "Legal Framework" (level 1)
      - "Section I" (level 2)
      - "Section II" (level 2)
      - "Section III" (level 2)
      - "A. Formation of the Conspiracies" (level 3)
      - "B. The Defendant Knew that His Claims of Outcome‑Determinative Fraud Were False" (level 3)
      - "1. Arizona" (level 4)
      - "2. Georgia" (level 4)

  - name: gather_extracted_text_find_people_and_involvements
    type: gather
    content_key: extracted_text_chunk
    doc_header_key: headers
    doc_id_key: split_find_people_and_involvements_id
    order_key: split_find_people_and_involvements_chunk_num
    peripheral_chunks:
      next:
        head:
          count: 1
      previous:
        tail:
          count: 1

  - name: submap_find_people_and_involvements
    type: map
    model: gpt-4o-mini
    output:
      schema:
        people_and_involvements: list[str]
    prompt: |
      Given the document excerpt: {{ input.extracted_text_chunk_rendered }}
      Extract all the people mentioned and summarize their involvements in the case described. Only process the main chunk.

  - name: subreduce_find_people_and_involvements
    type: reduce
    model: gpt-4o-mini
    associative: true
    pass_through: true
    synthesize_resolve: false
    output:
      schema:
        people_and_involvements: list[str]
    reduce_key:
      - split_find_people_and_involvements_id
    prompt: |
      Given the following extracted information about individuals involved in the case, compile a comprehensive list of people and their specific involvements in the case:

      {% for chunk in inputs %}
      {% for involvement in chunk.people_and_involvements %}
      - {{ involvement }}
      {% endfor %}
      {% endfor %}

      Make sure to include all the people and their involvements. If a person has multiple involvements, group them together.

pipeline:
  steps:
    - name: analyze_document
      input: legal_doc
      operations:
        - extract_metadata_find_people_and_involvements
        - split_find_people_and_involvements
        - header_extraction_extracted_text_find_people_and_involvements
        - gather_extracted_text_find_people_and_involvements
        - submap_find_people_and_involvements
        - subreduce_find_people_and_involvements

  output:
    type: file
    path: /path/to/your/output/people_and_involvements.json
    intermediate_dir: /path/to/your/intermediates
```
---

### Final Output Requirements

When generating a pipeline, the model must follow these rules:

1. **Do not ask clarifying questions**
   The user provides all necessary context in their first query. Make assumptions from the query and dataset samples.

2. **Simplicity first**
   Avoid unnecessary split/gather or multi-step indirection. Only use necessary operations to achieve the task.

3. **Top-level keys**
   Include only:

   * `default_model`
   * `datasets`
   * `operations`
   * `pipeline`

   *(Optional: `system_prompt` if dataset description or persona is useful.)*

4. **Schema discipline**

   * Keep schemas flat and simple.
   * Use lists or dict only when explicitly required.
   * Use `enum[...]` only if all possible values are enumerated in the prompt.
   * For vague tasks, default to:

     ```yaml
     output:
       schema:
         key_points: "list[string]"
         summary: string
     ```

5. **Field consistency**
   
   * Reference only fields present in the dataset or created earlier in the pipeline. If multiple text fields exist, normalize them into a single `text` field via `code_map`.
   * Common field unconsistency errors:
        - Wrong pluralization (e.g., `medications` vs. `medication`)
        - Synonyms (e.g., `review_text` vs. `text`)
        - Different casing (e.g., `Text` vs. `text`)
        - Nested fields (e.g., `input.medication.dosage` vs. `input.dosage`)

6. **Operator Correctness**

    * Use `reduce_key` for `reduce` operations. It **CANNOT** be null or omitted. `reduce`'s prompt must reference `inputs`.
    * All LLM-based operations must have a `prompt`.
    * For unnesting list of dicts, use `recursive: true` and `depth: 2` to fully flatten the structure.
7. **Optimization**

   * Set `optimize: true` for `resolve` steps.
   * Use it for map/reduce only when documents are long.
   * Otherwise, omit it.

8. **Validation (optional)**
   Add minimal validation rules only when they materially improve reliability (0-2). Avoid over-constraining outputs. 

9. **Output format**

   * Produce a single YAML string only.
   * No markdown fencing, comments, or prose.
   * YAML must be well-formed and valid.

10. **Chain of Thought (CoT)**
   Think step by step internally:

   1. Inspect dataset fields and formats
   2. Define the flat and minimal schema
   3. Apply safe defaults
   4. Assemble the final YAML

   Do not expose reasoning in the output.

11. **Best Practices**

1. Use `Rank` -> `code_filter` to select top-k items if possible, rather than `Filter` to improve efficiency. `Reduce` and `TopK` operators cannot be used in this way.

2. Use `resolve` to deduplicate or standardize entities before aggregation to improve quality.

3. Output several fields rather than a single complex dict to improve reliability.

4. Use several simple operatiors in sequence rather than one complex operatior, but still need to keep the pipeline correct.

5. Use `reduce` to aggregate information to summarize multiple items.
"""


PIPELINE_GENERATION_PROMPT = """
Task: Generate a minimal, correct DocETL pipeline YAML for the given query and datasets.
Query:
{query}
Datasets:
{profiles_str}
Output:
- Return only the pipeline YAML.
- You MUST follow "Final Output Requirements" strictly.
"""