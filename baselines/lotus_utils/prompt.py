INSTRUCTION_PROMPT = """
**Lotus Pipeline Generation Instructions**  
Guidelines for producing structured, reproducible pipelines in LOTUS by combining standard `pandas` methods with LOTUS’ semantic and utility operators.  

------

### Dataset Handling

- **Supported data sources**: LOTUS can load datasets from `pandas.DataFrame` or local directories/files (via `DirectoryReader`).  
- **Integration with pandas**: All datasets are exposed as `pandas.DataFrame`, so LOTUS operators can be freely mixed with native pandas functions.  
- **Consistency of fields**: Always maintain consistent column names and field references across operations.  
- **DirectoryReader** (Optional): Recursively reads files from a directory and prepares them for semantic operations. Example:  

```python
from lotus import DirectoryReader
df = DirectoryReader("data/").load()
````

---

### Quick Start of LOTUS Pipeline Structure

* LOTUS pipelines combine **pandas native methods** (filtering, merging, transformations) with **semantic operators** (e.g., `sem_map`, `sem_join`).
* Pipelines are expressed as ordered transformations, each returning a DataFrame.
* Orchestration resembles:

```python
import pandas as pd
import lotus
from lotus.models import LM, SentenceTransformersRM
from lotus.vector_store import FaissVS

# Configure models: language model, retrieval model and vector store
lm = LM(model="gpt-4o-mini")
rm = SentenceTransformersRM(model="intfloat/e5-base-v2")
vs = FaissVS()
lotus.settings.configure(lm=lm, rm=rm, vs=vs)

# Data: courses with descriptions and workloads:contentReference[oaicite:2]{index=2}
df = pd.DataFrame([
    ("Probability and Random Processes", "Focuses on markov chains and convergence of random processes. The workload is pretty high."),
    ("Deep Learning", "Focuses on theory and implementation of neural networks. Workload varies by professor but typically isn't terrible."),
    ("Digital Design and Integrated Circuits", "Focuses on building RISC‑V CPUs in Verilog. Students say the workload is VERY high."),
    ("Databases", "Focuses on implementation of a RDBMS with NoSQL topics at the end. Most students say the workload is not too high."),
], columns=["Course Name", "Description"])

# 1. Filter for machine‑learning courses:contentReference[oaicite:3]{index=3}
ml_df = df.sem_filter("{Description} indicates that the class is relevant for machine learning.")
# 2. Summarize how to succeed in those courses
tips = ml_df.sem_agg(
    "Given each {Course Name} and its {Description}, give me a study plan to succeed in my classes."
)._output[0]

# 3. Find the top two courses with the highest workload:contentReference[oaicite:4]{index=4}
top_2_hardest = df.sem_topk("What {Description} indicates the highest workload?", K=2)

# 4. Join a list of desired skills with the courses:contentReference[oaicite:5]{index=5}
skills_df = pd.DataFrame([("SQL"), ("Chip Design")], columns=["Skill"])
classes_for_skills = skills_df.sem_join(df, "Taking {Course Name} will make me better at {Skill}")

# 5. Create a semantic index and search for a course related to Convolutional Neural Networks:contentReference[oaicite:6]{index=6}
df_indexed = df.sem_index("Description", "index_dir")
top_conv_class = df_indexed.sem_search("Description", "Convolutional Neural Network", K=1)

# 6. Use sem_map to suggest next topics to explore:contentReference[oaicite:7]{index=7}
examples_df = pd.DataFrame(
    [("Computer Graphics", "Computer Vision"), ("Real Analysis", "Complex Analysis")],
    columns=["Course Name", "Answer"]
)
next_topics = df.sem_map(
    "Given {Course Name}, list a topic that will be good to explore next. Respond with just the topic name and nothing else.",
    examples=examples_df,
    suffix="Next Topics"
)

# print("Study plan:", tips)
# print("Top two hardest courses:\n", top_2_hardest)
# print("Courses to improve skills:\n", classes_for_skills)
# print("Class most relevant to CNNs:\n", top_conv_class)
# print("Suggested next topics:\n", next_topics)
```

---

### Operation Types

LOTUS has two broad operator categories:

* **Semantic Operators**: for reasoning with LLMs.

  * `sem_map`: Transform fields using LLM reasoning.
  * `sem_extract`: Extract structured values from text.
  * `sem_filter`: Select rows by semantic condition.
  * `sem_agg`: Aggregate with reasoning.
  * `sem_topk`: Rank/select top-k by semantic score.
  * `sem_join`: Semantic join between DataFrames.
  * `sem_search`: Query external knowledge.
  * `sem_sim_join`: Similarity-based join.
  * `sem_cluster_by`: Cluster rows by semantic similarity.

* **Utility Operators**: for dataset hygiene.

  * `sem_partition_by`: Split into partitions.
  * `sem_index`: Create stable row indices.
  * `sem_dedup`: Deduplicate by semantic similarity.

---

### Schemas & Types

* LOTUS outputs support **typed schemas** (via `pydantic`).
* Guidance:

  * Keep schemas **flat** (no deep nesting).
  * Use **simple fields** like strings, numbers, enums.

---

### Advanced Usage

#### Optimized Processing with Approximations (Cascades)

LOTUS can reduce latency and cost by using approximation cascades: a lightweight “helper” model performs cheaper but less accurate opertions, only sending uncertain cases to a more expensive “oracle” model.  

To use cascades, configure both models and pass a `CascadeArgs` object when calling the operator:  

```python
# Omit other imports for brevity
from lotus.types import CascadeArgs

# Configure both a full and a cheap LM
gpt_4o_mini = LM("gpt-4o-mini")
gpt_4o = LM("gpt-4o")
lotus.settings.configure(lm=gpt_4o, helper_lm=gpt_4o_mini)

# Sample data and predicate
df = pd.DataFrame({"Course Name": [
    "Probability and Random Processes",
    "Digital Design and Integrated Circuits",
    "Computer Security"
]})

# Set precision/recall targets and sampling fraction.
cascade_args = (
    recall_target=0.9,
    precision_target=0.9, # desired recall and precision for the cascade.
    sampling_percentage=0.5, # fraction of the data used to estimate decision thresholds.
    failure_probability=0.2 # probability that the accuracy targets will not be met.
)

# Apply the semantic filter with approximations; stats reveal thresholds etc.
filtered_df, stats = df.sem_filter(
    user_instruction="{Course Name} requires a lot of math",
    cascade_args=cascade_args,
    return_stats=True
)
```

Semantic operators that support cascades include `sem_filter`, `sem_join`, `sem_topk`.

#### Prompt Strategies

LOTUS supports advanced prompting techniques such as Chain‑of‑Thought (CoT), Zero‑Shot CoT (ZS\_COT) and few‑shot demonstrations.  You can supply an example DataFrame and set a reasoning strategy to guide the model:

```python
# Omit other imports for brevity
from lotus.types import ReasoningStrategy

# Configure LM
lm = LM(model="gpt-4o-mini")
lotus.settings.configure(lm=lm)

# Data and predicate
df = pd.DataFrame({"Course Name": [
    "Probability and Random Processes",
    "Optimization Methods in Engineering",
    "Digital Design and Integrated Circuits",
    "Computer Security"
]})
user_instruction = "{Course Name} requires a lot of math"

# Few‑shot examples with reasoning for CoT prompting
examples = pd.DataFrame({
    "Course Name": ["Machine Learning", "Reaction Mechanisms", "Nordic History"],
    "Answer": [True, True, False],
    "Reasoning": [
        "Machine Learning requires a solid understanding of linear algebra and calculus",
        "Reaction Engineering requires ordinary differential equations",
        "Nordic History has no math involved"
    ]
})

# Apply semantic filter with Chain‑of‑Thought strategy
df = df.sem_filter(
    user_instruction,
    examples=examples,
    strategy=ReasoningStrategy.COT
)
```

By providing examples and setting `ReasoningStrategy.COT`, the model will reason step‑by‑step, improving accuracy.  Other strategies include ZS\_COT (no examples, just “let’s think step by step”) and FEW\_SHOT (imitate examples without explicit reasoning).

#### Reasoning Models

LOTUS lets you swap in different reasoning‑optimized LLMs.  For instance, the DeepSeek‑R1 model is tuned for Chain‑of‑Thought prompts; recommended temperatures are around 0.5–0.7.  After configuring the model, you can apply semantic operators:

```python
# Omit other imports for brevity
from lotus.types import ReasoningStrategy

# Use a reasoning-optimized model with appropriate temperature
lotus.settings.configure(lm=LM(model="ollama/deepseek-r1:7b", temperature=0.5))

# Example: classify reviews as positive or negative with explanations
df = pd.DataFrame({"Reviews": [
    "I absolutely love this product. It exceeded all my expectations.",
    "Terrible experience. The product broke within a week.",
    "The quality is average, nothing special.",
    "Fantastic service and high quality!",
    "I would not recommend this to anyone.",
]})
df = df.sem_filter("{Reviews} are positive reviews", return_explanations=True, return_all=True)

# Example: map courses to similar courses using zero‑shot CoT reasoning
courses = pd.DataFrame({"Course Name": [
    "Probability and Random Processes",
    "Digital Design and Integrated Circuits",
    "Computer Security"
]})
courses = courses.sem_map(
    "What is a similar course to {Course Name}. Just give the course name.",
    return_explanations=True,
    strategy=ReasoningStrategy.ZS_COT
)
```

These examples show how to configure and use alternative reasoning models within LOTUS operations.

---

### Prompting Guidelines

* Always reference fields by their **DataFrame column names**.
* Use **templating** (e.g., `{column}` placeholders).
* Ensure generated outputs **exactly match the schema**.

---

### Operator Examples

**sem\_map – Row‑wise natural‑language mapping**

The `sem_map` operator applies an LLM-driven transformation to each row based on a prompt.  For example, to suggest a similar course for each class:

```python
# DataFrame of courses
df = pd.DataFrame({"Course Name": [
    "Probability and Random Processes",
    "Computer Security"
]})

# Map each course to a concise similar course description
df = df.sem_map("What is a similar course to {Course Name}. Be concise.")
```

**sem\_extract – Extract structured fields from unstructured text**

`sem_extract` produces multiple columns by projecting information out of one or more text columns.  You provide the input columns and a mapping of output names to optional descriptions:

```python
df = pd.DataFrame({"description": [
    "Yoshi is 25 years old",
    "Luigi is 15 years old"
]})
input_cols = ["description"]
output_cols = {
    "name": "The name of the person",
    "age": "The age of the person"
}
# Extract name and age into new columns
df = df.sem_extract(input_cols, output_cols)
```

**sem\_filter – Filter rows using a natural‑language predicate**

`sem_filter` keeps rows that satisfy a language‑model predicate.  Here, only courses requiring math are kept:

```python
df = pd.DataFrame({"Course Name": [
    "Probability and Random Processes",
    "Digital Design and Integrated Circuits"
]})
# Select courses that require a lot of math
df = df.sem_filter("{Course Name} requires a lot of math")
```

**sem\_agg – Semantic aggregation or summarization**

`sem_agg` combines multiple rows into a single semantic summary.  You can aggregate the entire DataFrame or group by a column:

```python
df = pd.DataFrame({"ArticleContent": [
    "Quantum computing harnesses the properties of quantum mechanics...",
    "Renewable energy helps mitigate climate change."
]})
# Summarize all articles in a single paragraph
df = df.sem_agg("Provide a concise summary of all {ArticleContent}.")
summary = df._output[0]
```

**sem\_topk – Rank and select the top‑K items**

`sem_topk` reorders records based on a natural‑language comparator and returns the top K rows.  You can also get statistics about the ranking:

```python
df = pd.DataFrame({"Course Name": [
    "Probability and Random Processes",
    "Computer Security",
    "Digital Design and Integrated Circuits"
]})
# Find the two courses that require the least math
sorted_df, stats = df.sem_topk(
    "Which {Course Name} requires the least math?",
    K=2, method="quick", return_stats=True
)
```

**sem\_join – Join two tables with a natural‑language predicate**

`sem_join` matches rows between two DataFrames according to a language‑model predicate.  For instance, join courses to skills they teach:

```python
df1 = pd.DataFrame({"Course Name": ["Operating Systems", "Compilers"]})
df2 = pd.DataFrame({"Skill": ["Computer Science", "Math"]})
join_instruction = "Taking {Course Name:left} will help me learn {Skill:right}"
# Perform semantic join
result = df1.sem_join(df2, join_instruction)
```

**sem\_search – Semantic search over an indexed column**

`sem_search` retrieves rows most relevant to a query by searching a semantic index.  You must build the index with `sem_index` first:

```python
df = pd.DataFrame({"Course Name": [
    "Computer Security",
    "Introduction to Robotics",
    "Introduction to Computer Networks"
]})
# Build a semantic index on the column
df = df.sem_index("Course Name", "index_dir")
# Search for courses related to computer security and rerank top matches
df = df.sem_search(
    "Course Name",
    "Which course name is most related to computer security?",
    K=3, n_rerank=1
)
```

**sem\_sim\_join – Similarity‑based join**

`sem_sim_join` joins two tables by nearest‑neighbour similarity instead of a custom predicate.  It returns the best matching row(s) and similarity scores:

```python
df1 = pd.DataFrame({"Course Name": ["Operating Systems", "Compilers"]})
df2 = pd.DataFrame({"Skill": ["Math", "Computer Science"]}).sem_index("Skill", "skill_index")
# Join on semantic similarity between course names and skills
result = df1.sem_sim_join(df2, left_on="Course Name", right_on="Skill", K=1)
```

**sem\_cluster\_by – Cluster rows by semantic similarity**

`sem_cluster_by` groups rows into clusters based on semantic closeness.  It can be used after building an index:

```python
df = pd.DataFrame({"Course Name": [
    "Digital Design and Integrated Circuits",
    "Computer Security",
    "Cooking"
]})
# Index and cluster into 2 groups
df = df.sem_index("Course Name", "course_index").sem_cluster_by("Course Name", 2)
# df now has a 'cluster_id' column indicating the cluster assignment
```

**sem\_partition\_by – Control aggregation partitions**

`sem_partition_by` assigns each row to a partition so that subsequent semantic aggregations respect a specific ordering.  By clustering first, you can influence summarization:

```python
df = pd.DataFrame({"Course Name": [
    "Digital Design and Integrated Circuits",
    "Computer Security",
    "Cooking"
]})
# Create a partitioning function (e.g., cluster into 2 groups)
partitions = lotus.utils.cluster("Course Name", 2)
df = df.sem_index("Course Name", "course_index").sem_partition_by(partitions)
# Aggregate while respecting the partition order
out = df.sem_agg("Summarize all {Course Name}")._output[0]
```

**sem\_index – Build a semantic index**

`sem_index` constructs a local semantic index for a column, enabling fast search and similarity joins.  Loading a DataFrame after indexing simply preserves the data:

```python
df = pd.DataFrame({"Course Name": [
    "Computer Security",
    "Introduction to Computer Science"
]})
# Build an index and save it to a directory
df = df.sem_index("Course Name", "index_dir")
```

**sem\_dedup – Remove semantically redundant rows**

`sem_dedup` uses semantic similarity to deduplicate entries.  After indexing, you specify a similarity threshold to remove paraphrased duplicates:

```python
df = pd.DataFrame({"Text": [
    "I don't know what day it is",
    "I don't know what time it is",
    "Harry Potter and the Sorcerer's Stone"
]})
# Index and deduplicate rows that are very similar
df = df.sem_index("Text", "index_dir").sem_dedup("Text", threshold=0.8)
```

---

### Full Pipeline Examples

#### Sample 1 – Analyzing leaked system prompts

This script loads a corpus of leaked system prompts, extracts model names, associates each with the vendor, and then analyzes prompt content (tasks, tone, unknown‑answer instructions) using semantic operators.  A semantic join with a cascade approximation links prompts to vendors efficiently.

```python
import pandas as pd
import lotus
from lotus.file_extractors import DirectoryReader
from lotus.models import LM, LiteLLMRM
from lotus.vector_store import FaissVS
from lotus.types import CascadeArgs

# Configure LOTUS with a main LM, helper LM and retrieval model
lotus.settings.configure(
    lm=LM("gpt-4o-mini"),
    helper_lm=LM("gpt-4.1-nano"),  # cheaper proxy model for cascades
    rm=LiteLLMRM(model="text-embedding-3-small"),
    vs=FaissVS()
)

# 1) Load all Markdown files in the leaked‑prompts repo
df = DirectoryReader().add(".").to_df()

# 2) Extract the model name from the file name (sem_extract projects new columns from text:contentReference[oaicite:1]{index=1})
input_cols = ["file_name"]
output_cols = {"model": "The model name in the file_name"}
df = df.sem_extract(input_cols, output_cols)

# 3) Map each model to its company via a semantic join
companies_df = pd.DataFrame({
    'company': ['Anthropic','OpenAI','Microsoft','Perplexity','Cursor','Google','Codeium','DeepSeek','GitHub','xAI'],
    'systems': ['Claude','GPT','Copilot','Perplexity','Cursor','Gemini','Codeium','DeepSeek','Copilot','Grok']
})
# CascadeArgs reduce LLM calls by using a helper model:contentReference[oaicite:2]{index=2}
cascade_args = CascadeArgs(recall_target=0.75, precision_target=0.75, sampling_percentage=0.1)
join_instruction = "The model {model:left} is associated with the company {company:right}."
df = df.sem_join(companies_df, join_instruction, cascade_args=cascade_args)

# 4) Extract lists of tasks and tone descriptors from the prompt content
df = df.sem_extract(
    ["content"],
    {
        "tasks": "List the specific tasks or domains (e.g., coding, math, science, writing) that the prompt says the model can help with",
        "tone": "What tone or personality is the model instructed to adopt in the prompt? (e.g., friendly, formal, concise)"
    }
)

# 5) Rank prompts by politeness using sem_topk:contentReference[oaicite:3]{index=3}
polite_instruction = "Rank the following prompts {content} from file {file_name} by how polite and friendly their tone is."
most_polite = df.sem_topk(polite_instruction, K=10)

# 6) Extract instructions for unknown answers
df = df.sem_extract(
    ["content"],
    {
        "dont_know_type": (
            "If the prompt mentions how the model should respond if it doesn’t know the answer, "
            "classify the instruction into one of the following categories: 'Apologize', 'Admit Uncertainty', "
            "'Suggest Alternatives', 'Refer to External Sources', or 'Other'. If not mentioned, return None."
        ),
        "dont_know_response": (
            "If the prompt mentions how the model should respond if it doesn’t know the answer, "
            "extract the exact instruction. If not mentioned, return None."
        ),
    }
)

# 7) Classify each prompt into supported tasks using sem_join and cascade args
tasks = ["coding","math","science","writing","language translation","summarization","reasoning","search","telling jokes","poetry"]
tasks_df = pd.DataFrame({"task": tasks})
join_instruction = (
    "Does the prompt {content:left} indicate that the model can help with the task {task:right}? "
    "Return True if yes, False otherwise."
)
df_task_map = df.sem_join(tasks_df, join_instruction, cascade_args=cascade_args)

# Now df contains extracted fields and join results for further analysis
```

---

#### Sample 2 – Analyzing multi‑agent system failures

This example loads agent traces, detects failure cases, summarises them, groups common failure modes, classifies each trace by failure type with a cascade‑optimized join, and produces per‑benchmark improvement suggestions.

```python
import json
import pandas as pd
import lotus
from lotus.models import LM, LiteLLMRM
from lotus.vector_store import FaissVS
from lotus.types import CascadeArgs

# Configure LOTUS with a single LM and retrieval model
lotus.settings.configure(
    lm=LM("gpt-4o-mini"),
    rm=LiteLLMRM(model="text-embedding-3-small"),
    vs=FaissVS()
)

# 1) Load the MAST dataset (here we pretend we've downloaded it and loaded as JSON)
with open("MAST_dataset.json") as f:
    data = json.load(f)
df = pd.DataFrame(data).head(100)
df["agent_trace"] = df.apply(lambda row: row['trace']['trajectory'], axis=1)

# 2) Identify traces that represent failures (sem_filter returns rows satisfying the predicate:contentReference[oaicite:4]{index=4})
fail_df = df.sem_filter("the {agent_trace} demonstrates a failure of the agent to complete the task")

# 3) Summarize each failure in one sentence (sem_map performs row‑wise transformation:contentReference[oaicite:5]{index=5})
fail_df = fail_df.sem_map(
    "Describe the cause of failure in the {agent_trace} in a sentence",
    suffix="failure_summary"
)

# 4) Aggregate all failure summaries into a list of common failure modes:contentReference[oaicite:6]{index=6}
failure_modes_df = fail_df.sem_agg(
    "given each agent's {failure_summary}, create a bullet point list of failure modes. "
    "each failure mode should be a few words"
)
# Parse the bullet list into a DataFrame of failure cases
failure_cases = [item.strip() for item in failure_modes_df.iloc[0]._output.split("- ") if item.strip()]
failure_cases_df = pd.DataFrame({"failure_case": failure_cases})

# 5) Classify each trace against each failure case using sem_join; build indices and use cascades
fail_df = fail_df.reset_index().sem_index("failure_summary", "failure_summ_idx")
failure_cases_df = failure_cases_df.reset_index().sem_index("failure_case", "failure_idx")
cascade_args = CascadeArgs(recall_target=0.8, precision_target=0.8, failure_probability=0.2)
join_instruction = (
    "given the agent's {failure_summary:left}, the agent is experiencing the {failure_case:right}"
)
classified_df, stats = fail_df.sem_join(
    failure_cases_df,
    join_instruction,
    cascade_args=cascade_args,
    return_stats=True
)

# 6) For each benchmark, recommend improvements using sem_agg with group_by:contentReference[oaicite:7]{index=7}
recs_df = classified_df.sem_agg(
    "given my {agent_trace}, recommend solutions to improve my agent so it can avoid the {failure_case} on the {benchmark_name}",
    group_by=["benchmark_name"]
)

# The recs_df DataFrame contains benchmark‑specific recommendations in its _output column
for i in range(len(recs_df)):
    benchmark = recs_df.iloc[i].benchmark_name
    # print(f"\nRecommendations for {benchmark}:\n{recs_df.iloc[i]._output}")
```

---

#### Sample 3 – Introducing semantic operators

This tutorial demonstrates core LOTUS operators over a small set of research papers and other datasets.  It also shows how to combine semantic operators with conventional filters and external data sources.

```python
import pandas as pd
import lotus
from lotus.models import LM, LiteLLMRM
from lotus.vector_store import FaissVS
from lotus.types import ReasoningStrategy

# Configure LOTUS with LM and retrieval model
lotus.settings.configure(
    lm=LM(model="gpt-4o-mini"),
    rm=LiteLLMRM(model="text-embedding-3-small"),
    vs=FaissVS()
)

# Load a sample of research papers (title, abstract, link, etc.)
papers_df = pd.read_csv("data/all_papers_df.csv").head(30)

# 1) Semantic filter – keep only papers whose abstracts discuss politics:contentReference[oaicite:8]{index=8}
politics_papers = papers_df.sem_filter("the paper {abstract} discusses politics")

# 2) Semantic filter with zero‑shot chain‑of‑thought reasoning to get explanations:contentReference[oaicite:9]{index=9}
explained = papers_df.sem_filter(
    "the paper {abstract} discusses politics",
    strategy=ReasoningStrategy.ZS_COT,
    return_explanations=True
)

# 3) Semantic top‑K – find the paper with the “best acronym”:contentReference[oaicite:10]{index=10}
best_acronym = papers_df.sem_topk("{abstract} has the best acronym", K=1)

# 4) Semantic join – match papers to companies whose models are mentioned in the abstract
companies_df = pd.DataFrame({"company": ["META", "OpenAI"]})
joined = papers_df.sem_join(
    companies_df,
    "{abstract} discusses one of the LLM models from {company}",
    strategy=ReasoningStrategy.ZS_COT,
    return_explanations=True
)

# 5) Index and search – build a semantic index on titles and find papers related to "language models"
indexed = papers_df.sem_index("title", "title_index_dir")
language_model_papers = indexed.sem_search("title", "language models", K=3)[["title"]]

# 6) Clustering and aggregation – cluster by title similarity and summarize each cluster:contentReference[oaicite:11]{index=11}
clustered = indexed.sem_cluster_by("title", 2)
cluster_summaries = clustered.sem_agg(
    "In one sentence, summarize the main themes from this paper group, given each paper {title}",
    group_by=["cluster_id"]
)

# 7) Compose semantic operators for an application:
#    - filter papers by date, domain and author h-index using standard pandas
#    - search abstracts for a research topic
#    - filter for additional relevance criteria
#    - summarize the results and refine the summary
filtered = papers_df.copy()
filtered["date_published"] = pd.to_datetime(filtered["date_published"], errors="coerce")
filtered = filtered[(filtered["max_author_hindex"] >= 2) & (filtered["categories"].str.contains("cs.AI|cs.ML"))]
filtered = filtered.sem_index("abstract", "abstract_index_dir")
filtered = filtered.sem_search("abstract", "retrieval augmented generation", K=10)
filtered = filtered.sem_filter("Based on the paper {abstract}, the paper meets the following criteria: provides an open-source repository of their code")

# Summarize and refine the papers for a digest
draft_summary = filtered.sem_agg(
    "You are writing a digest for a user who wants to catch up on recent papers. "
    "Given each paper {abstract}, connect the unifying themes among papers and relate them to retrieval augmented generation.",
    suffix="draft_summary"
)
refined_summary = draft_summary.sem_map(
    "Given the {draft_summary}, write a refined summary of the main topics discussed in the papers and how they relate to retrieval augmented generation.",
    suffix="final_summary"
)

# print(refined_summary.iloc[0].final_summary)
```

---

### Final Output Requirements

When generating a pipeline, the model must follow these rules:

1. **Do not ask clarifying questions**
   The user provides all necessary context in their first query. Make assumptions from the query and dataset samples.

2. **Simplicity first**
   
   * Only use necessary operations to achieve the task.
   * Prefer native pandas methods when possible.
   * Several short, clear steps are better than a few complex ones.

3. **Field consistency**
   
   * Reference only fields present in the dataset or created earlier in the pipeline.
   * Common field unconsistency errors:
        - Wrong pluralization (e.g., `medications` vs. `medication`)
        - Synonyms (e.g., `review_text` vs. `text`)
        - Different casing (e.g., `Text` vs. `text`)

4. **Data Passing**
    * Files needed to be loaded are stored in a dict called `data_dict`.
        * This dict is prepared by other components and passed to the pipeline code. Do not reload or redefine it.
        * The structure of `data_dict` is {<dataset_name>: <pandas.DataFrame>}. 

5. **Operator Correctness**

    * All semantic operators must have a `prompt`.
    * Combine pandas + LOTUS operators without breaking typing.

6. **Optimization**

   * Use `CascadeArgs` for semantic operators on too large datasets.
        * `CascadeArgs` are valid for `sem_filter`, `sem_join`, and `sem_topk`.
        * Helper model should be included in the configuration if `CascadeArgs` are used.
    * Use indexing (`sem_index`) before `sem_search` or `sem_sim_join`.

7. **Output format**

   * Produce a single Python code string only.
   * No markdown fencing, comments, or prose.
   * Python must be well-formed and valid.
   * Break down pipeline code into the following three stages: 
        1. Imports and configuration
        2. Pipeline body
        3. Finally, put the final result pd.DataFrame in a variable named `result`. Do not print or return it.
   * Only print necessary code snippets, the handling of `result` is owned by other components.
        
8. **Chain of Thought (CoT)**
   Think step by step internally:

   1. Inspect dataset fields and formats
   2. Define the flat and minimal schema
   3. Apply safe defaults
   4. Assemble the final Python code
   5. Generate comments for each step (not in the code output) to improve interpretability

   Do not expose reasoning in the output.


"""


PIPELINE_GENERATION_PROMPT = """
Task: Generate a minimal, correct Lotus pipeline Python code for the given query and datasets.
Query:
{query}
Datasets:
{profiles_str}
Output:
- Return only the pipeline Python code.
- You MUST follow "Final Output Requirements" strictly.
"""