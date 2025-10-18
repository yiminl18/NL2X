from typing import List, Optional, Dict, Any
import json
from .pipeline import Pipeline
from .db import DataSource, DatasetManager
from model.litellm_client import llm_call

def _infer_field_type(value: Any, samples: List[Any] = None) -> str:
    """
    Infer abstraction layer type from sample value(s).

    Args:
        value: A sample value to infer type from
        samples: Optional list of additional samples for better inference

    Returns:
        Type string in abstraction layer format (e.g., "String", "List[Integer]", "Dict[{...}]")
    """
    if value is None:
        return "Null"

    if isinstance(value, bool):
        return "Boolean"
    elif isinstance(value, int):
        return "Integer"
    elif isinstance(value, float):
        return "Float"
    elif isinstance(value, str):
        return "String"
    elif isinstance(value, list):
        if len(value) == 0:
            return "List[Unknown]"

        # Try to infer element type from first non-null element
        element_type = None
        for item in value:
            if item is not None:
                element_type = _infer_field_type(item)
                break

        if element_type is None:
            element_type = "Unknown"

        return f"List[{element_type}]"
    elif isinstance(value, dict):
        if len(value) == 0:
            return "Dict"

        # Infer types for all fields in the dict
        field_types = {}
        for key, val in value.items():
            field_types[key] = _infer_field_type(val)

        # Format as Dict[{field1: Type1, field2: Type2}]
        fields_str = ", ".join([f"{k}: {v}" for k, v in field_types.items()])
        return f"Dict[{{{fields_str}}}]"
    else:
        return "Unknown"


def _extract_fields_with_types(data_samples: List[Dict], add_sample: bool = False) -> Dict[str, Dict[str, Any]]:
    """
    Extract all fields (including nested) with their types and optional samples.

    Args:
        data_samples: List of data dictionaries
        add_sample: Whether to include sample values

    Returns:
        Dictionary mapping field paths to their info:
        {
            "field_name": {"TYPE": "String", "SAMPLE": "value"},
            "contact": {"TYPE": "Dict[{...}]"},
            "contact.phone_numbers": {"TYPE": "List[Integer]", "SAMPLE": [1, 2, 3]}
        }
    """
    if not data_samples:
        return {}

    fields_info = {}

    def extract_from_dict(d: Dict, prefix: str = ""):
        """Recursively extract fields from a dictionary."""
        for key, value in d.items():
            field_path = f"{prefix}.{key}" if prefix else key

            # Infer type
            field_type = _infer_field_type(value)

            # Initialize field info
            field_info = {"TYPE": field_type}

            # Add sample if requested
            if add_sample:
                field_info["SAMPLE"] = value

            # Store field info
            fields_info[field_path] = field_info

            # Recursively process nested dicts
            if isinstance(value, dict) and value:
                extract_from_dict(value, field_path)
            elif isinstance(value, list) and value and isinstance(value[0], dict):
                # For lists of dicts, extract fields from first dict
                extract_from_dict(value[0], field_path)

    # Extract from first sample to get all fields
    if data_samples:
        extract_from_dict(data_samples[0])

    return fields_info


def _generate_field_descriptions(
    fields_info: Dict[str, Dict[str, Any]],
    data_samples: List[Dict],
    query: Optional[str] = None
) -> tuple[str, Dict[str, str]]:
    """
    Generate dataset summary and field descriptions using LLM.

    Args:
        fields_info: Dictionary with field types and samples
        data_samples: Original data samples for context
        query: Optional task description

    Returns:
        Tuple of (dataset_summary, field_descriptions_dict)
    """
    # Build field list for prompt
    fields_list = []
    for field_path, info in fields_info.items():
        field_entry = f"- {field_path}: {info['TYPE']}"
        fields_list.append(field_entry)

    fields_str = "\n".join(fields_list)

    # Build query section
    query_section = f"Task/Query: {query}\n\n" if query else ""

    # Build data samples section (limit to avoid token overflow)
    samples_to_show = data_samples[:3]  # Show max 3 samples
    samples_str = json.dumps(samples_to_show, indent=2, ensure_ascii=False)

    # Construct prompt
    prompt = f"""{query_section}Data Samples:
{samples_str}

Fields and Their Types:
{fields_str}

Please analyze the data samples and provide:
1. A concise overall summary of the entire dataset (what kind of data it contains, its purpose, and main characteristics)
2. Individual descriptions for each field

For nested fields (with dots), explain both the parent context and the specific sub-field meaning."""

    # Define schema for LLM response
    schema = {
        "type": "object",
        "properties": {
            "dataset_summary": {
                "type": "string",
                "description": "Overall summary of the entire dataset, describing what kind of data it contains and its purpose"
            },
            "field_descriptions": {
                "type": "object",
                "additionalProperties": {
                    "type": "string"
                },
                "description": "Map of field paths to their descriptions"
            }
        },
        "required": ["dataset_summary", "field_descriptions"]
    }

    system_prompt = "You are a data analyst that helps describe datasets and their fields. Provide clear, concise descriptions based on the data samples and their types."

    messages = [{"role": "user", "content": prompt}]

    try:
        response = llm_call(messages, schema=schema, system_prompt=system_prompt)
        result = json.loads(response)
        dataset_summary = result.get("dataset_summary", "Dataset summary not available")
        field_descriptions = result.get("field_descriptions", {})
        return dataset_summary, field_descriptions
    except Exception:
        # On error, return empty descriptions
        return "Dataset summary not available", {}


def summarize_dataset(
    data_samples: List[Dict],
    query: Optional[str] = None,
    add_sample: bool = False
) -> Dict[str, Any]:
    """
    Summarize dataset fields with types, descriptions, and optional samples.

    Args:
        data_samples: List of data dictionaries to analyze
        query: Optional query/task description for context
        add_sample: Whether to include sample values in output

    Returns:
        Dictionary with dataset summary and field information:
        {
            "DATASET_SUMMARY": "Each record contains personal information for job market.",
            "name": {
                "TYPE": "String",
                "FIELD_DESCRIPTION": "Person's name",
                "SAMPLE": "Tom"  # if add_sample=True
            },
            "age": {
                "TYPE": "Integer",
                "FIELD_DESCRIPTION": "Person's age in years",
                "SAMPLE": 18
            }
        }
    """
    if not data_samples:
        return {}

    # Step 1: Extract fields with types and optional samples
    fields_info = _extract_fields_with_types(data_samples, add_sample=add_sample)

    # Step 2: Generate dataset summary and field descriptions using LLM
    dataset_summary, descriptions = _generate_field_descriptions(fields_info, data_samples, query)

    # Step 3: Build result with DATASET_SUMMARY at the top
    result = {
        "DATASET_SUMMARY": dataset_summary
    }

    # Step 4: Add field information
    for field_path, info in fields_info.items():
        result[field_path] = {
            "TYPE": info["TYPE"],
            "FIELD_DESCRIPTION": descriptions.get(field_path, f"Field: {field_path}")
        }

        # Add sample if it exists
        if "SAMPLE" in info:
            result[field_path]["SAMPLE"] = info["SAMPLE"]

    return result


# ============================================================================
# Pipeline Subtasks Summarization Functions
# ============================================================================

def summarize_subtasks(
    pipeline: Pipeline,
    data_samples: List[Dict] = None,
    query: Optional[str] = None
) -> List[str]:
    """
    Infer operator subtasks for all operators in a pipeline using LLM.

    This function analyzes the pipeline operators and generates subtask descriptions
    for each operator in a single LLM call.

    Args:
        pipeline: Pipeline object containing operators
        data_samples: Optional list of data samples for context
        query: Optional task description for context

    Returns:
        List of subtask descriptions, one for each operator in execution order
    """
    if data_samples is None:
        data_samples = []

    # Get pipeline operators information
    operators_info = []
    execution_order = pipeline.get_execution_order()
    for node_id in execution_order:
        node = pipeline.nodes[node_id]
        op = node.operator
        operators_info.append({
            "type": op.type,
            "name": op.name,
            "input": op.input,
            "output": op.output,
            "properties": op.properties,
        })

    # Construct prompt for LLM
    # Build task description section
    task_section = f"Task Description: {query}\n\n" if query else ""

    # Build operators description
    operators_desc = ""
    for i, op_info in enumerate(operators_info, 1):
        properties_str = f"\n   Properties: {json.dumps(op_info['properties'], indent=2)}" if op_info['properties'] else ""
        operators_desc += f"""
{i}. Operator Type: {op_info['type']}
   Name: {op_info['name']}
   Input Schema: {json.dumps(op_info['input'], indent=2)}
   Output Schema: {json.dumps(op_info['output'], indent=2)}{properties_str}
"""

    # Build data samples section
    data_samples_section = f"\n\nData Samples ({len(data_samples)} examples):\n{json.dumps(data_samples, indent=2, ensure_ascii=False)}" if data_samples else ""

    # Construct final prompt
    prompt = f"""{task_section}Pipeline Operators:{operators_desc}{data_samples_section}

Based on the task description (if provided), pipeline operators, and data samples (if provided), infer the specific functionality/purpose of each operator in the pipeline. Provide a concise description for each operator that explains what it does in the context of solving the overall task."""

    # Define response schema
    schema = {
        "type": "object",
        "properties": {
            "subtasks": {
                "type": "array",
                "items": {
                    "type": "string"
                },
                "description": "List of descriptions for each operator's functionality, in execution order"
            }
        },
        "required": ["subtasks"]
    }

    system_prompt = "You are an AI assistant that analyzes data processing pipelines and infers the purpose of each operator. Always respond with valid JSON matching the required schema."

    messages = [{"role": "user", "content": prompt}]

    # Call LLM and parse response
    try:
        response = llm_call(messages, schema=schema, system_prompt=system_prompt)
        result = json.loads(response)
        subtasks = result.get("subtasks", [])

        # Validate that we have the right number of subtasks
        if len(subtasks) != len(operators_info):
            # If mismatch, use generic descriptions
            subtasks = [f"Process data with {op_info['type']} operator" for op_info in operators_info]

        return subtasks

    except Exception:
        # On error, return generic descriptions
        return [f"Process data with {op_info['type']} operator" for op_info in operators_info]


def summarize_subtasks_step(
    pipeline: Pipeline,
    data_samples: List[Dict] = None,
    query: Optional[str] = None
) -> List[str]:
    """
    Infer operator subtasks step-by-step, one operator at a time.

    This function generates subtasks incrementally. When generating the subtask for
    operator i, it includes the subtasks of all previous operators (0 to i-1) as
    context, allowing for better understanding of the pipeline flow.

    Args:
        pipeline: Pipeline object containing operators
        data_samples: Optional list of data samples for context
        query: Optional task description for context

    Returns:
        List of subtask descriptions, one for each operator in execution order
    """
    if data_samples is None:
        data_samples = []

    # Get pipeline operators information
    operators_info = []
    execution_order = pipeline.get_execution_order()
    for node_id in execution_order:
        node = pipeline.nodes[node_id]
        op = node.operator
        operators_info.append({
            "type": op.type,
            "name": op.name,
            "input": op.input,
            "output": op.output,
            "properties": op.properties,
        })

    # Generate subtasks incrementally, one operator at a time
    subtasks = []

    # Build data samples section (shared across all prompts)
    data_samples_section = f"\n\nData Samples ({len(data_samples)} examples):\n{json.dumps(data_samples, indent=2, ensure_ascii=False)}" if data_samples else ""

    for i, op_info in enumerate(operators_info):
        # Build task description section
        task_section = f"Task Description: {query}\n\n" if query else ""

        # Build previous operators' subtasks section
        previous_subtasks_section = ""
        if i > 0:
            previous_subtasks_section = "\n\nPrevious Operators' Subtask:\n"
            for j, prev_subtask in enumerate(subtasks, 1):
                previous_subtasks_section += f"Op. {j}: {prev_subtask}\n"

        # Build current operator description
        properties_str = f"\n   Properties: {json.dumps(op_info['properties'], indent=2)}" if op_info['properties'] else ""
        current_operator_desc = f"""
Current Operator (Op. {i + 1}):
   Type: {op_info['type']}
   Name: {op_info['name']}
   Input Schema: {json.dumps(op_info['input'], indent=2)}
   Output Schema: {json.dumps(op_info['output'], indent=2)}{properties_str}
"""

        # Construct prompt for current operator
        prompt = f"""{task_section}{previous_subtasks_section}{current_operator_desc}{data_samples_section}

Based on the task description (if provided), previous operators' subtasks (if any), current operator details, and data samples (if provided), infer the specific functionality/purpose of the current operator. Provide a concise description that explains what this operator does in the context of solving the overall task."""

        # Define response schema for single subtask
        schema = {
            "type": "object",
            "properties": {
                "subtask": {
                    "type": "string",
                    "description": "Description of the current operator's functionality"
                }
            },
            "required": ["subtask"]
        }

        system_prompt = "You are an AI assistant that analyzes data processing pipelines and infers the purpose of each operator. Always respond with valid JSON matching the required schema."

        messages = [{"role": "user", "content": prompt}]

        # Call LLM and parse response
        try:
            response = llm_call(messages, schema=schema, system_prompt=system_prompt)
            result = json.loads(response)
            subtask = result.get("subtask", f"Process data with {op_info['type']} operator")
            subtasks.append(subtask)
        except Exception:
            # On error, use generic description
            subtasks.append(f"Process data with {op_info['type']} operator")

    return subtasks


class PipelineOptimizer:
    """Optimizer for abstract pipelines with optional dataset support."""

    def __init__(
        self,
        pipeline: Pipeline,
        data_source: Optional[DataSource] = None,
        data_manager: Optional[DatasetManager] = None,
        query: Optional[str] = None,
        num_samples: int = 5
    ):
        """Initialize optimizer with pipeline and optional data source."""
        self.pipeline = pipeline
        self.data_manager = data_manager
        self.query = query
        self.num_samples = num_samples

        # If data_source not provided, try to load from pipeline
        if data_source is None and pipeline.input_path and data_manager:
            try:
                data_source = data_manager.load(pipeline.input_path)
            except Exception:
                # For optimizer, data is optional, so don't raise error
                pass

        self.data_source = data_source

        # Prepare data samples for subtask inference
        data_samples = []
        if self.data_source and self.data_source.data:
            sample_size = min(self.num_samples, len(self.data_source.data))
            data_samples = self.data_source.data[:sample_size]

        # Infer operator subtasks if not already set
        if self.pipeline.subtasks is None:
            self.pipeline.subtasks = summarize_subtasks(
                self.pipeline,
                data_samples,
                self.query
            )

    def should_optimize(self) -> bool:
        """Determine if pipeline should be optimized."""
        # TODO: Implement optimization decision logic
        return False

    def optimize(self) -> List[Pipeline]:
        """Optimize pipeline and return optimized variants."""
        # TODO: Implement pipeline optimization logic
        return [self.pipeline]
