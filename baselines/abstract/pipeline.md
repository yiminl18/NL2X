## Introduction of Procedure and Pipeline

In this system, a **Procedure** is a sequence of operators that are executed in a specific order with a base system (e.g. DocETL/Lotus...). A **Pipeline** is a set of Procedures that is orchestrated by the abstraction layer, and it might include non-base system connectors, opeartors, and other Procedures. The question is: how does the abstration layer scheduler put the Procedures together? The following sections will introduce 4 used patterns. 
### Pattern 1: Parallel Fan-Out (One-to-Many)

* **Map**
    * **Corresponds to:** "Map" phase of Map-Reduce.
    * **Description:** The scheduler **splits** a single large input (e.g., one file) into multiple smaller chunks. It then launches an **identical** underlying **procedure** instance for each chunk to process them in parallel.
    * **Example:** Split a 10GB log file into 100 x 100MB chunks; run 100 parallel instances of the log-processing **procedure**.
* **Conditional Routing** (implemented as `CONDITIONAL_ROUTING` node type)
    * **Corresponds to:** "Routing individual records to different downstream nodes."
    * **Description:** A procedure first processes all input records (possibly adding new fields), then routes each record to one of multiple output branches based on a routing field value. Each record is sent to exactly one branch.
    * **Implementation:** The procedure must produce a field (default: `_route_to`) containing the target branch name. The scheduler groups records by this field and sends each group to its designated downstream node.
    * **Example:** A procedure classifies documents by type, adding a `_route_to` field with values like "invoice", "receipt", or "other". Records are then routed to separate processing pipelines based on their classification.
* **Scatter**
    * **Description:** The scheduler sends the same input data to **multiple different procedures** simultaneously. Each **procedure** performs a different task on the same data.
    * **Example:** A product info file is sent to "**Procedure** A (update inventory)," "**Procedure** B (update website)," and "**Procedure** C (generate report)" all at the same time.

### Pattern 2: Aggregation Fan-In (Many-to-One)

* **Reduce**
    * **Corresponds to:** "Reduce" phase of Map-Reduce; "Aggregating outputs from multiple **procedures**."
    * **Description:** The scheduler not only collects outputs but also applies a specific logic (this can be a UDF or a dedicated "aggregator" **procedure**) to **combine** or **merge** the results into a single output.
    * **Example:** 100 **procedures** calculate word counts for their chunks. The scheduler gathers all 100 partial-count files and uses a "Reduce" UDF (or **procedure**) to sum them into one final word-count file.

### Pattern 3: Sequential Chaining (One-to-One)

* **Direct Chaining**
    * **Description:** `Procedure A` -> `Procedure B` -> `Procedure C`. The scheduler's job is to automatically trigger **Procedure** B as soon as **Procedure** A successfully completes, and feed **Procedure** B with **Procedure** A’s output.
* **Transformation Chaining**
    * **Corresponds to:** "Inserting a UDF operator between **procedures**."
    * **Description:** `Procedure A(s)` -> `UDF` -> `Procedure B(s)`. **Procedure** A's output format does not match **Procedure** B's required input format. The scheduler executes a lightweight "glue" UDF to transform, filter, or enrich the data before passing it to **Procedure** B.
    * **Example:**
        * **Procedure** A outputs CSV, but **Procedure** B requires JSON. The scheduler runs a UDF (CSV-to-JSON) on the output of **Procedure** A before starting **Procedure** B.
        * The scheduler samples the output of **Procedure** A before starting **Procedure** B.
* **Advanced Abstraction-Layer Chaining**
    * **Description:** `Procedure A` -> `Procedure C`. The scheduler's job is to add a `Procedure B` between the two **procedures** that:
        * Runs on the abstraction layer.
        * Contains operators not available in the underlying system.
    * **Examples: Direct Chaining** and **Transformation Chaining** are two corner cases respectively with 0 and 1 operator.
        * Sample
        * Web search
        * …

### Pattern 4: Loop

* **Fixed Loop**
    * **Description:** The scheduler use the output of the **procedure** as its input again, repeating fixed number of iterations.
    * **Examples:** Clean the data for 3 rounds using the same data cleaning **procedure**.
* **Conditional Loop**
    * **Description:** The scheduler check if the output of the **procedure** meets some conditions.
    * **Examples:** Clean the data to its quality satisfies the threshold.

## ConditionalRouting Node Type

### Overview

The `CONDITIONAL_ROUTING` node type enables record-level routing within a pipeline. Unlike file-level routing, this node processes all records through a procedure and then distributes them to different downstream nodes based on a field value in each record.

### Requirements

1. **Procedure**: Must produce a routing field (default: `_route_to`) containing the target branch name
2. **Multiple Outputs**: At least 2 output branches must be defined
3. **Single Routing**: Each record is sent to exactly one branch based on its routing field value
4. **Consistent Schema**: All output branches receive data with the same schema

### Usage Example

```python
from baselines.abstract.pipeline import Pipeline, DataReference

pipeline = Pipeline(name="document_classifier")

# Add a conditional routing node
pipeline.add_conditional_routing(
    procedure_path="classify_documents.json",
    node_id="classifier",
    data_sources=[DataReference("file", "documents.json")],
    output_branches={
        "invoice": "process_invoices",
        "receipt": "process_receipts",
        "other": "process_general"
    },
    routing_field="_route_to"  # Optional, defaults to "_route_to"
)
```

### Procedure Design

The procedure must add the routing field to each record. Example using a Map operator:

```json
{
    "type": "map",
    "prompt": "Classify the document type. Set _route_to to 'invoice', 'receipt', or 'other'",
    "output": {
        "schema": {
            "_route_to": "string"
        }
    }
}
```

### Validation Rules

- Must have at least 2 outputs
- All outputs must be node references (not files)
- Metadata must contain `routing_field`
- Records with unmatched routing values are sent to the 'default' branch if it exists

### Execution Behavior

1. Execute the procedure on all input records
2. Group records by routing field value
3. Send each group to its designated output branch
4. Downstream nodes access their branch using `DataReference` with `output_name`

### DataReference with Output Names

When connecting to a conditional routing node, specify which branch to read:

```python
# Child node reading from "invoice" branch
DataReference("node", "classifier", output_name="invoice")

# If output_name is omitted, defaults to "default"
DataReference("node", "classifier")  # Reads "default" branch
```