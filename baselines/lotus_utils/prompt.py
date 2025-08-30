INSTRUCTION_PROMPT = """
**Lotus Pipeline Generation Instructions**  
Guidelines for producing structured, reproducible pipelines in LOTUS by combining standard `pandas` methods with LOTUS’ semantic and utility operators.  

------

### Dataset Handling

- **Supported data sources**: LOTUS can load datasets from `pandas.DataFrame` or local directories/files (via `DirectoryReader`).  
- **Integration with pandas**: All datasets are exposed as `pandas.DataFrame`, so LOTUS operators can be freely mixed with native pandas functions.  
- **Consistency of fields**: Always maintain consistent column names and field references across operations.  
- **DirectoryReader**: Recursively reads files from a directory and prepares them for semantic operations. Example:  

```python
from lotus import DirectoryReader
df = DirectoryReader("data/").load()
````

---

### Minimal LOTUS Pipeline Structure

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

print("Study plan:", tips)
print("Top two hardest courses:\n", top_2_hardest)
print("Courses to improve skills:\n", classes_for_skills)
print("Class most relevant to CNNs:\n", top_conv_class)
print("Suggested next topics:\n", next_topics)
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

LOTUS can reduce latency and cost by using approximation cascades: a lightweight “helper” model filters or joins most records, only sending uncertain cases to a more expensive “oracle” model.  To use cascades, configure both models and pass a `CascadeArgs` object when calling the operator:

```python
import pandas as pd
import lotus
from lotus.models import LM
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
    precision_target=0.9,
    sampling_percentage=0.5,
    failure_probability=0.2
)

# Apply the semantic filter with approximations; stats reveal thresholds etc.
filtered_df, stats = df.sem_filter(
    user_instruction="{Course Name} requires a lot of math",
    cascade_args=cascade_args,
    return_stats=True
)
```

#### Prompt Strategies

LOTUS supports advanced prompting techniques such as Chain‑of‑Thought (CoT), Zero‑Shot CoT (ZS\_COT) and few‑shot demonstrations.  You can supply an example DataFrame and set a reasoning strategy to guide the model:

```python
import pandas as pd
import lotus
from lotus.models import LM
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
import pandas as pd
import lotus
from lotus.models import LM
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

**Example 1: Resume Screening**
Task: Rank applicants by match to job description.

```python
df = DirectoryReader("resumes/").load()
df = df.pipe(sem_extract, field="text", schema=CandidateSchema)
df = df.pipe(sem_filter, prompt="Keep only candidates with 3+ years Python experience")
df = df.pipe(sem_topk, prompt="Rank by suitability for data scientist role", k=10)
```

**Example 2: Research Paper Analysis**
Task: Cluster abstracts and summarize trends.

```python
df = DirectoryReader("papers/").load()
df = df.pipe(sem_cluster_by, field="abstract")
df = df.pipe(sem_agg, groupby="cluster", prompt="Summarize cluster themes")
```

---

### Final Output Requirements

* Pipelines must:
  * Use **simple, reproducible steps**.
  * Respect **declared schemas** exactly.
  * Combine pandas + LOTUS operators without breaking typing.
  * Favor **clarity** over cleverness.
  * Prefer short, modular pipelines that are easy to debug.
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