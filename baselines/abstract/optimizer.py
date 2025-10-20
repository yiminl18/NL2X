from typing import List, Optional, Dict, Any
import json
import sys
from pathlib import Path
from .pipeline import Pipeline
from .db import DataSource, DatasetManager
from model.litellm_client import llm_call

sys.path.insert(0, str(Path(__file__).parent / "tools"))
from static_comparator import compare_pipelines

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

        field_types = {}
        for key, val in value.items():
            field_types[key] = _infer_field_type(val)

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
            field_type = _infer_field_type(value)
            field_info = {"TYPE": field_type}

            if add_sample:
                field_info["SAMPLE"] = value

            fields_info[field_path] = field_info

            if isinstance(value, dict) and value:
                extract_from_dict(value, field_path)
            elif isinstance(value, list) and value and isinstance(value[0], dict):
                extract_from_dict(value[0], field_path)

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
    fields_list = []
    for field_path, info in fields_info.items():
        field_entry = f"- {field_path}: {info['TYPE']}"
        fields_list.append(field_entry)

    fields_str = "\n".join(fields_list)
    query_section = f"Task/Query: {query}\n\n" if query else ""
    samples_to_show = data_samples[:3]
    samples_str = json.dumps(samples_to_show, indent=2, ensure_ascii=False)

    prompt = f"""{query_section}Data Samples:
{samples_str}

Fields and Their Types:
{fields_str}

Please analyze the data samples and provide:
1. A concise overall summary of the entire dataset (what kind of data it contains, its purpose, and main characteristics)
2. Individual descriptions for each field

For nested fields (with dots), explain both the parent context and the specific sub-field meaning."""

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

    fields_info = _extract_fields_with_types(data_samples, add_sample=add_sample)
    dataset_summary, descriptions = _generate_field_descriptions(fields_info, data_samples, query)

    result = {"DATASET_SUMMARY": dataset_summary}

    for field_path, info in fields_info.items():
        result[field_path] = {
            "TYPE": info["TYPE"],
            "FIELD_DESCRIPTION": descriptions.get(field_path, f"Field: {field_path}")
        }

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

    task_section = f"Task Description: {query}\n\n" if query else ""
    operators_desc = ""
    for i, op_info in enumerate(operators_info, 1):
        properties_str = f"\n   Properties: {json.dumps(op_info['properties'], indent=2)}" if op_info['properties'] else ""
        operators_desc += f"""
{i}. Operator Type: {op_info['type']}
   Name: {op_info['name']}
   Input Schema: {json.dumps(op_info['input'], indent=2)}
   Output Schema: {json.dumps(op_info['output'], indent=2)}{properties_str}
"""

    data_samples_section = f"\n\nData Samples ({len(data_samples)} examples):\n{json.dumps(data_samples, indent=2, ensure_ascii=False)}" if data_samples else ""

    prompt = f"""{task_section}Pipeline Operators:{operators_desc}{data_samples_section}

Based on the task description (if provided), pipeline operators, and data samples (if provided), infer the specific functionality/purpose of each operator in the pipeline. Provide a concise description for each operator that explains what it does in the context of solving the overall task."""

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

    try:
        response = llm_call(messages, schema=schema, system_prompt=system_prompt)
        result = json.loads(response)
        subtasks = result.get("subtasks", [])

        if len(subtasks) != len(operators_info):
            subtasks = [f"Process data with {op_info['type']} operator" for op_info in operators_info]

        return subtasks

    except Exception:
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

    subtasks = []
    data_samples_section = f"\n\nData Samples ({len(data_samples)} examples):\n{json.dumps(data_samples, indent=2, ensure_ascii=False)}" if data_samples else ""

    for i, op_info in enumerate(operators_info):
        task_section = f"Task Description: {query}\n\n" if query else ""

        previous_subtasks_section = ""
        if i > 0:
            previous_subtasks_section = "\n\nPrevious Operators' Subtask:\n"
            for j, prev_subtask in enumerate(subtasks, 1):
                previous_subtasks_section += f"Op. {j}: {prev_subtask}\n"

        properties_str = f"\n   Properties: {json.dumps(op_info['properties'], indent=2)}" if op_info['properties'] else ""
        current_operator_desc = f"""
Current Operator (Op. {i + 1}):
   Type: {op_info['type']}
   Name: {op_info['name']}
   Input Schema: {json.dumps(op_info['input'], indent=2)}
   Output Schema: {json.dumps(op_info['output'], indent=2)}{properties_str}
"""

        prompt = f"""{task_section}{previous_subtasks_section}{current_operator_desc}{data_samples_section}

Based on the task description (if provided), previous operators' subtasks (if any), current operator details, and data samples (if provided), infer the specific functionality/purpose of the current operator. Provide a concise description that explains what this operator does in the context of solving the overall task."""

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

        try:
            response = llm_call(messages, schema=schema, system_prompt=system_prompt)
            result = json.loads(response)
            subtask = result.get("subtask", f"Process data with {op_info['type']} operator")
            subtasks.append(subtask)
        except Exception:
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

        if data_source is None and pipeline.input_path and data_manager:
            try:
                data_source = data_manager.load(pipeline.input_path)
            except Exception:
                pass

        self.data_source = data_source

        data_samples = []
        if self.data_source and self.data_source.data:
            sample_size = min(self.num_samples, len(self.data_source.data))
            data_samples = self.data_source.data[:sample_size]

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


# ============================================================================
# Prompt Optimization Function
# ============================================================================

def optimize_prompt(
    operator,
    dataset_summary: Dict[str, Any],
    operator_subtask: str
) -> Optional:
    """
    Optimize the prompt of an operator for better task completion.
    Args:
        operator: Operator object with properties containing 'prompt' field
        dataset_summary: Dataset summary in the format returned by summarize_dataset,
            containing DATASET_SUMMARY and field information with TYPE and FIELD_DESCRIPTION
        operator_subtask: Description of the specific functionality this operator needs to implement
    Returns:
        Optimized Operator object with updated prompt, or None if optimization fails
    """
    from .ops.base import Operator

    if 'prompt' not in operator.properties:
        print(f"Error: Operator '{operator.name}' does not have a 'prompt' field in properties")
        return None

    original_prompt = operator.properties['prompt']

    dataset_info_lines = []
    dataset_info_lines.append(f"Dataset Summary: {dataset_summary.get('DATASET_SUMMARY', 'N/A')}")
    dataset_info_lines.append("\nField Descriptions:")

    for field_name, field_info in dataset_summary.items():
        if field_name == 'DATASET_SUMMARY':
            continue
        if isinstance(field_info, dict):
            field_type = field_info.get('TYPE', 'Unknown')
            field_desc = field_info.get('FIELD_DESCRIPTION', 'No description')
            dataset_info_lines.append(f"  - {field_name} ({field_type}): {field_desc}")

    dataset_info = "\n".join(dataset_info_lines)

    meta_prompt = f"""You are an expert prompt engineer tasked with optimizing a data processing prompt for a specific subtask.

Your goal is to create a TASK-SPECIFIC prompt that is tailored to the specific data and task requirements, rather than a generic prompt. The requirements and constraints should be directly related to the actual task and data characteristics.

## Subtask Description
{operator_subtask}

## Original Operator Information
Operator Type: {operator.type}
Operator Name: {operator.name}

Original Prompt:
{original_prompt}

Input Schema:
{json.dumps(operator.input, indent=2)}

Output Schema:
{json.dumps(operator.output, indent=2)}

## Dataset Information
{dataset_info}

## Task Instructions
Generate an optimized prompt that:
1. Is specifically tailored to the subtask: "{operator_subtask}"
2. Leverages the dataset characteristics and field descriptions
3. Provides clear, task-specific instructions (not generic ones)
4. Includes relevant constraints based on the data and task requirements

## Critical Constraints
1. **Variable Consistency**: You MUST use ONLY the variables that appear in the original prompt. Do NOT introduce any new variables that don't exist in the input schema.
2. **Schema Consistency**: The output must match the output schema exactly. Do not add or remove output fields.
3. **Jinja2 Format**: Maintain proper Jinja2 template syntax as demonstrated in the original prompt. Use the same variable reference style (e.g., {{{{ input.field_name }}}}).
4. **Task-Specific**: Make requirements and instructions specific to THIS task and THIS data, not generic best practices.

The original prompt already demonstrates the correct Jinja2 format and variable usage - maintain this style.

Provide your optimized prompt as a single string."""

    schema = {
        "type": "object",
        "properties": {
            "optimized_prompt": {
                "type": "string",
                "description": "The optimized task-specific prompt in Jinja2 format"
            }
        },
        "required": ["optimized_prompt"]
    }

    system_prompt = "You are an expert prompt engineer specializing in creating task-specific, data-aware prompts for data processing operations. You excel at tailoring prompts to specific tasks and datasets while maintaining technical constraints."

    try:
        messages = [{"role": "user", "content": meta_prompt}]
        response = llm_call(
            messages=messages,
            schema=schema,
            system_prompt=system_prompt,
            temperature=0.3
        )

        result = json.loads(response)
        optimized_prompt = result.get("optimized_prompt")

        if not optimized_prompt:
            print("Error: LLM did not return an optimized prompt")
            return None

    except Exception as e:
        print(f"Error calling LLM for prompt optimization: {e}")
        return None

    new_operator = Operator()
    new_operator.name = operator.name
    new_operator.type = operator.type
    new_operator.source = operator.source.copy() if isinstance(operator.source, dict) else operator.source
    new_operator.input = operator.input.copy() if isinstance(operator.input, dict) else operator.input
    new_operator.output = operator.output.copy() if isinstance(operator.output, dict) else operator.output
    new_operator.properties = operator.properties.copy() if isinstance(operator.properties, dict) else {}
    new_operator.properties['prompt'] = optimized_prompt

    def operator_to_dict(op):
        """Convert Operator object to dictionary format."""
        return {
            "name": op.name,
            "type": op.type,
            "source": op.source,
            "properties": op.properties,
            "input": op.input,
            "output": op.output
        }

    original_op_dict = operator_to_dict(operator)
    new_op_dict = operator_to_dict(new_operator)

    try:
        validation_result = compare_pipelines(
            operators1=[original_op_dict],
            operators2=[new_op_dict],
            verbose=False
        )

        if not validation_result['is_compatible']:
            print(f"Error: Optimized prompt failed static validation for operator '{operator.name}'")
            print(f"Validation errors:")
            for error in validation_result['errors']:
                print(f"  - {error}")
            return None

        print(f"Successfully optimized prompt for operator '{operator.name}'")
        return new_operator

    except Exception as e:
        print(f"Error during static validation: {e}")
        return None


# ============================================================================
# Filter Pushdown Optimization Functions
# ============================================================================

def should_pushdown_filter(pipeline: Pipeline) -> bool:
    """
    Check if any filter operator can be pushed down to an earlier position.

    Args:
        pipeline: Pipeline object to analyze

    Returns:
        True if at least one filter can be pushed earlier, False otherwise
    """
    try:
        execution_order = pipeline.get_execution_order()

        if len(execution_order) < 2:
            return False

        available_fields = set()
        if pipeline.dataset_schema and 'fields' in pipeline.dataset_schema:
            available_fields = set(pipeline.dataset_schema['fields'].keys())

        fields_at_position = [available_fields.copy()]

        for node_id in execution_order:
            operator = pipeline.nodes[node_id].operator

            if hasattr(operator, 'output') and isinstance(operator.output, dict):
                for field_name in operator.output.keys():
                    if field_name != 'type':
                        available_fields.add(field_name)

            fields_at_position.append(available_fields.copy())

        for current_pos, node_id in enumerate(execution_order):
            operator = pipeline.nodes[node_id].operator

            if operator.type == 'Filter':
                required_fields = set()
                if hasattr(operator, 'input') and isinstance(operator.input, dict):
                    input_fields = operator.input.get('fields', {})
                    if isinstance(input_fields, dict):
                        required_fields = set(input_fields.keys())

                if not required_fields:
                    continue

                earliest_position = None
                for pos in range(current_pos + 1):
                    if required_fields.issubset(fields_at_position[pos]):
                        earliest_position = pos
                        break

                if earliest_position is not None and earliest_position < current_pos:
                    return True

        return False

    except Exception as e:
        print(f"Error in should_pushdown_filter: {e}")
        return False


def _single_pushdown(pipeline: Pipeline) -> Pipeline:
    """
    Perform a single filter pushdown optimization.

    Args:
        pipeline: Pipeline object to optimize

    Returns:
        New Pipeline with one filter moved to an earlier position
    """
    import copy

    execution_order = pipeline.get_execution_order()
    available_fields = set()
    if pipeline.dataset_schema and 'fields' in pipeline.dataset_schema:
        available_fields = set(pipeline.dataset_schema['fields'].keys())

    fields_at_position = [available_fields.copy()]

    for node_id in execution_order:
        operator = pipeline.nodes[node_id].operator

        if hasattr(operator, 'output') and isinstance(operator.output, dict):
            for field_name in operator.output.keys():
                if field_name != 'type':
                    available_fields.add(field_name)

        fields_at_position.append(available_fields.copy())

    current_position = None
    target_position = None

    for current_pos, node_id in enumerate(execution_order):
        operator = pipeline.nodes[node_id].operator

        if operator.type == 'Filter':
            required_fields = set()
            if hasattr(operator, 'input') and isinstance(operator.input, dict):
                input_fields = operator.input.get('fields', {})
                if isinstance(input_fields, dict):
                    required_fields = set(input_fields.keys())

            if not required_fields:
                continue

            earliest_pos = None
            for pos in range(current_pos + 1):
                if required_fields.issubset(fields_at_position[pos]):
                    earliest_pos = pos
                    break

            if earliest_pos is not None and earliest_pos < current_pos:
                current_position = current_pos
                target_position = earliest_pos
                break

    new_pipeline = copy.deepcopy(pipeline)

    removed_filter = new_pipeline.remove_operator_at(current_position)
    new_pipeline.insert_operator(removed_filter, target_position)

    print(f"Filter '{removed_filter.name}' pushed down from position {current_position} to position {target_position}")

    return new_pipeline


def pushdown_filter(pipeline: Pipeline) -> Optional[Pipeline]:
    """
    Recursively push down all filter operators to their earliest possible positions.

    Args:
        pipeline: Pipeline object to optimize

    Returns:
        Optimized Pipeline if any filters were moved, None otherwise
    """
    if not should_pushdown_filter(pipeline):
        return None

    current_pipeline = pipeline
    optimization_count = 0

    while should_pushdown_filter(current_pipeline):
        current_pipeline = _single_pushdown(current_pipeline)
        optimization_count += 1

    print(f"Filter pushdown completed: {optimization_count} filter(s) optimized")

    return current_pipeline
