from typing import List, Optional
import json
from .pipeline import Pipeline
from .db import DataSource, DatasetManager
from model.litellm_client import llm_call


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

        # Prepare pipeline by inferring operator subtasks
        self.prepare_for_optimize()

    def prepare_for_optimize(self):
        """Prepare pipeline for optimization by inferring operator subtasks using LLM."""
        # If subtasks already exist, return early
        if self.pipeline.subtasks is not None:
            return

        # Gather information for LLM inference
        # 1. Get data samples if available
        data_samples = []
        if self.data_source and self.data_source.data:
            sample_size = min(self.num_samples, len(self.data_source.data))
            data_samples = self.data_source.data[:sample_size]

        # 2. Get pipeline operators information
        operators_info = []
        execution_order = self.pipeline.get_execution_order()
        for node_id in execution_order:
            node = self.pipeline.nodes[node_id]
            op = node.operator
            operators_info.append({
                "type": op.type,
                "name": op.name,
                "input": op.input,
                "output": op.output,
                "properties": op.properties,
            })

        # 3. Construct prompt for LLM
        # Build task description section
        task_section = f"Task Description: {self.query}\n\n" if self.query else ""

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

        # 4. Define response schema
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

        # 5. Call LLM and parse response
        try:
            response = llm_call(messages, schema=schema, system_prompt=system_prompt)
            result = json.loads(response)
            subtasks = result.get("subtasks", [])

            # Validate that we have the right number of subtasks
            if len(subtasks) != len(operators_info):
                # If mismatch, use generic descriptions
                subtasks = [f"Process data with {op_info['type']} operator" for op_info in operators_info]

            self.pipeline.subtasks = subtasks

        except Exception:
            return

    def prepare_for_optimize_step(self):
        """Prepare pipeline for optimization by inferring operator subtasks step-by-step using LLM.

        Unlike prepare_for_optimize(), this method generates subtasks incrementally, one operator at a time.
        When generating the subtask for operator i, it includes the subtasks of all previous operators
        (0 to i-1) as context, allowing for better understanding of the pipeline flow.
        """
        # If subtasks already exist, return early
        if self.pipeline.subtasks is not None:
            return

        # Gather information for LLM inference
        # 1. Get data samples if available
        data_samples = []
        if self.data_source and self.data_source.data:
            sample_size = min(self.num_samples, len(self.data_source.data))
            data_samples = self.data_source.data[:sample_size]

        # 2. Get pipeline operators information
        operators_info = []
        execution_order = self.pipeline.get_execution_order()
        for node_id in execution_order:
            node = self.pipeline.nodes[node_id]
            op = node.operator
            operators_info.append({
                "type": op.type,
                "name": op.name,
                "input": op.input,
                "output": op.output,
                "properties": op.properties,
            })

        # 3. Generate subtasks incrementally, one operator at a time
        subtasks = []

        # Build data samples section (shared across all prompts)
        data_samples_section = f"\n\nData Samples ({len(data_samples)} examples):\n{json.dumps(data_samples, indent=2, ensure_ascii=False)}" if data_samples else ""

        for i, op_info in enumerate(operators_info):
            # Build task description section
            task_section = f"Task Description: {self.query}\n\n" if self.query else ""

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

        # Store the generated subtasks
        self.pipeline.subtasks = subtasks

    def should_optimize(self) -> bool:
        """Determine if pipeline should be optimized."""
        # TODO: Implement optimization decision logic
        return False

    def optimize(self) -> List[Pipeline]:
        """Optimize pipeline and return optimized variants."""
        # TODO: Implement pipeline optimization logic
        return [self.pipeline]
