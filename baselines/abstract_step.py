"""
Abstract Layer Step-by-Step Pipeline Generation

This baseline generates abstract layer pipelines and converts them to base systems for execution.

The abstract layer is system-agnostic and can theoretically be converted to different
execution systems (currently only DocETL is supported).

Currently, we assume the dataset contains only one json.
"""

import time
from typing import Any, Dict, List, Optional, Tuple, Set
import os
import json
import traceback
from datetime import datetime
import logging

from .base import BaselineInterface, BaselineResult
from . import register_baseline
from .abstract_step_utils.ui import AbstractStepUserInterface
from .abstract_step_utils.validation import validate_pipeline_static
from .abstract_step_utils.file_manager import PipelineFileManager
from model.litellm_client import llm_call

# Schema tracking utilities using base system
from .abstract_step_utils.schema_tracker import (
    BaseSystemSchemaTracker,
    infer_schema_from_samples,
    format_fields_for_prompt,
)
from .abstract_step_utils.prompts import (
    get_next_operator_prompt,
    get_operator_repair_prompt,
    get_abstract_operator_prompt,
)
from .abstract.support import (
    BaseSystem,
    get_supported_operators,
    check_pipeline_compatibility,
)
from .abstract.ops.base import Operator
from .abstract.pipeline import Pipeline
from .abstract.convert.docetl import (
    abstract_to_docetl,
    operator_to_dict,
)
from .abstract.executor import DocETLExecutor
from .utils import DataTruncator


def load_sample_data(
    dataset_path: str,
    max_length: int = 2500,
    max_string_length: int = 1000,
    max_array_items: int = 3,
    max_dict_keys: int = 8
) -> List[Dict]:
    """
    Load sample data from a single dataset file with truncation.
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Initialize truncator
        truncator = DataTruncator(
            max_total_length=max_length,
            max_string_length=max_string_length,
            max_plain_text_length=5000,
            max_array_items=max_array_items,
            max_dict_keys=max_dict_keys
        )

        # Apply intelligent truncation
        truncated_data = truncator.truncate_data(data)

        return truncated_data
    except Exception as e:
        raise ValueError(f"Error loading dataset from {dataset_path}: {e}")


@register_baseline("abstract_step")
class AbstractStepBaseline(BaselineInterface):
    """
    Abstract Layer Step-by-Step Pipeline Generation
    """

    def _log(self, message: str, level: str = 'info', force: bool = False):
        """
        Unified logging interface with automatic verbose check.

        Args:
            message: Log message
            level: Log level ('info', 'warning', 'error', 'debug')
            force: If True, log regardless of verbose setting (for errors/warnings)
        """
        if force or self.config.verbose:
            getattr(self.logger, level)(message)

    def _setup(self):
        """Initialize baseline configuration and directories."""
        self.total_queries = 0
        self.total_time = 0

        self.max_attempts = self.config.max_attempts
        self.validate_answer = self.config.validate_answer

        # Set base system (currently only DocETL supported)
        self.base_system = BaseSystem.DOCETL

        if not self.config.verbose:
            logging.getLogger("LiteLLM").setLevel(logging.ERROR)
            logging.getLogger("httpcore").setLevel(logging.ERROR)
            logging.getLogger("openai").setLevel(logging.ERROR)
            logging.getLogger("asyncio").setLevel(logging.ERROR)
        else:
            logging.getLogger("LiteLLM").setLevel(logging.INFO)
            logging.getLogger("httpcore").setLevel(logging.INFO)
            logging.getLogger("openai").setLevel(logging.INFO)

        base_dir = os.getcwd()
        self.output_dirs = {
            'pipeline_output_dir': os.path.join(base_dir, "generated_pipelines", "abstract_step"),
            'converted_data_dir': os.path.join(base_dir, "converted_data", "abstract_step"),
            'abstract_pipeline_dir': os.path.join(base_dir, "abstract_pipelines", "abstract_step"),
        }

        for attr_name, dir_path in self.output_dirs.items():
            setattr(self, attr_name, dir_path)
            os.makedirs(dir_path, exist_ok=True)

        self.ui = AbstractStepUserInterface(self.config)

        # Initialize file manager for centralized file operations
        self.file_manager = PipelineFileManager(
            base_dirs={
                'pipeline_output': self.pipeline_output_dir,
                'abstract_pipeline': self.abstract_pipeline_dir,
                'converted_data': self.converted_data_dir
            },
            logger=self.logger,
            verbose=self.config.verbose
        )

        self.executor = self._create_executor()

        self._log(f"Initialized Abstract Step baseline with config: {self.config}")
        self._log(f"Base system: {self.base_system.value}")
        self._log(f"Executor: {self.executor.get_system_name()}")
        self._log(f"Supported operators: {get_supported_operators(self.base_system)}")
        self._log(f"Pipeline output directory: {self.pipeline_output_dir}")
        self._log(f"Abstract pipeline directory: {self.abstract_pipeline_dir}")

    def _extract_dataset_path(self, context: Optional[Dict[str, Any]]) -> str:
        """
        Extract dataset path from context dictionary.

        Args:
            context: Context dictionary containing dataset information
        """
        if not context:
            raise ValueError("No context provided")
        key, value = next(iter(context.items()))
        if value.path:
            return value.path

        if value.content:
            # Save to self.converted_data_dir
            dataset_path = os.path.join(self.converted_data_dir, f"{key}.json")
            with open(dataset_path, 'w', encoding='utf-8') as f:
                json.dump(value.content, f, ensure_ascii=False, indent=2)
            return dataset_path

        raise ValueError("No valid dataset path found in context")


    def _create_executor(self):
        """
        Create executor based on base system.

        Returns:
            BaseSystemExecutor instance for the configured base system
        """
        if self.base_system == BaseSystem.DOCETL:
            return DocETLExecutor(
                verbose=self.config.verbose,
                cache_enabled=False,  # Disable cache for baseline evaluation
                cache_dir=None,
                data_manager=None
            )
        else:
            raise ValueError(f"Unsupported base system: {self.base_system.value}")

    def process(self, query: str, context: Optional[Dict[str, Any]] = None) -> BaselineResult:
        """Process a single query with optional context using abstract layer pipeline generation."""
        from .base import BaselineResult as InterfaceBaselineResult

        start_time = time.time()
        self.total_queries += 1

        try:
            # Extract dataset path from context (simplified to support JSON files only)
            dataset_path = self._extract_dataset_path(context)

            dataset_samples = load_sample_data(dataset_path)
            print(dataset_samples)
            success = False
            result_output = None
            for attempt in range(self.max_attempts):
                try:
                    self._log(f"Attempt {attempt + 1}/{self.max_attempts} for query: {query[:100]}...")

                    abstract_pipeline = self._build_abstract_pipeline(query, dataset_samples, dataset_path, attempt)

                    success, result_output, _ = self._convert_and_execute(
                        abstract_pipeline, query, dataset_path, attempt
                    )

                    if success and result_output:
                        break

                except Exception as e:
                    self._log(f"Attempt {attempt + 1} failed: {e}", level='error', force=True)
                    self._log(traceback.format_exc(), level='error', force=True)

            execution_time = time.time() - start_time
            self.total_time += execution_time

            if success:
                return InterfaceBaselineResult(
                    query=query,
                    response={"answer": result_output},
                    metadata={},
                    execution_time=execution_time
                )
            else:
                raise ValueError(f"Pipeline generation failed after {self.max_attempts} attempts")

        except Exception as e:
            self._log(f"Error processing query with Abstract Step: {e}", level='error', force=True)
            self._log(f"Full traceback:\n{traceback.format_exc()}", level='error', force=True)

            execution_time = time.time() - start_time
            self.total_time += execution_time

            return InterfaceBaselineResult(
                query=query,
                response={"answer": "", "error": str(e)},
                metadata={},
                execution_time=execution_time,
                error=str(e)
            )

    def batch_process(self, queries: List[str], contexts: Optional[List[Dict[str, Any]]] = None) -> List[BaselineResult]:
        """Process multiple queries in batch."""
        if contexts is None:
            contexts = [None] * len(queries)

        results = []
        for query, context in zip(queries, contexts):
            results.append(self.process(query, context))

        return results

    def get_metrics(self) -> Dict[str, Any]:
        """Get baseline metrics."""
        return {
            "total_queries": self.total_queries,
            "total_time": self.total_time,
            "average_time": self.total_time / max(1, self.total_queries),
        }

    def _select_next_operator(
        self,
        query: str,
        dataset_samples: List[Dict],
        current_operators: List[Operator],
        schema_tracker: BaseSystemSchemaTracker,
        attempt: int = 0,
        bypass_cache: bool = False
    ) -> Tuple[bool, Optional[str], Optional[str], Optional[str]]:
        """
        Select the next operator type using JIT approach (selection only, not configuration).

        Args:
            query: User query
            dataset_samples: Sample data
            current_operators: List of operators generated so far
            schema_tracker: Current schema tracker
            attempt: Attempt number

        Returns:
            Tuple of (is_end, op_type, op_purpose, end_reason)
            - is_end: True if pipeline is complete
            - op_type: Operator type string (None if is_end)
            - op_purpose: Operator purpose string (None if is_end)
            - end_reason: Reason for ending (None if not is_end)
        """
        if self.config.verbose:
            self.logger.info(f"Generating next operator (current: {len(current_operators)} operators)...")

        # Format current pipeline state
        if current_operators:
            pipeline_state_parts = []
            for i, op in enumerate(current_operators):
                op_dict = operator_to_dict(op)
                pipeline_state_parts.append(f"{i+1}. {op.type} ({op.name}):\n{json.dumps(op_dict, indent=2)}")
            current_pipeline_state = "\n\n".join(pipeline_state_parts)
        else:
            current_pipeline_state = "No operators yet (this will be the first operator)"

        # Get current schema
        current_schema = schema_tracker.get_current_schema()
        available_fields_str = format_fields_for_prompt(current_schema)

        # Format dataset samples
        dataset_samples_str = json.dumps(dataset_samples, indent=2)

        # Get prompt
        prompt = get_next_operator_prompt(
            query=query,
            dataset_samples=dataset_samples_str,
            current_pipeline_state=current_pipeline_state,
            available_fields=available_fields_str,
            base_system=self.base_system
        )

        # Get supported operators for this base system to use as enum
        supported_ops = get_supported_operators(self.base_system)

        # Define schema for response
        parameters = {
            "type": "object",
            "properties": {
                "end": {"type": "boolean"},
                "reason": {"type": "string"},
                "operator": {
                    "type": "object",
                    "properties": {
                        "type": {"type": "string", "enum": supported_ops},
                        "purpose": {"type": "string"}
                    }
                }
            },
            "required": ["end"]
        }

        system_prompt = f"You are an AI assistant that builds data processing pipelines one operator at a time for {self.base_system.value}. Always respond with valid JSON matching the required schema."

        messages = [{"role": "user", "content": prompt}]

        # Confirm with user in debug/confirm mode
        user_choice = self.ui.confirm_step_before_llm(
            filled_operators=current_operators,
            step_name=f"Operator {len(current_operators) + 1}",
            iteration=len(current_operators) + 1,
            step_type="selection",
            step_prompt=prompt  # Always pass prompt, UI will handle display based on mode
        )

        if user_choice == 'abort':
            raise ValueError(f"User aborted at operator {len(current_operators) + 1}")

        # Call LLM (bypass cache if requested or after repair)
        should_bypass_cache = bypass_cache or (user_choice == 'regenerate')
        response = llm_call(messages, schema=parameters, system_prompt=system_prompt, bypass_cache=should_bypass_cache)

        try:
            result = json.loads(response)
        except json.JSONDecodeError as e:
            self._log(f"Failed to parse next operator response: {e}", level='error', force=True)
            raise

        # Check if pipeline is complete
        if result.get('end', False):
            end_reason = result.get('reason', 'Pipeline complete')
            self._log(f"Pipeline generation complete: {end_reason}", force=True)
            return True, None, None, end_reason

        # Extract operator type and purpose
        operator_data = result.get('operator')
        if not operator_data:
            raise ValueError("LLM response missing 'operator' field")

        op_type = operator_data.get('type')
        if not op_type:
            raise ValueError("Operator missing 'type' field")

        op_purpose = operator_data.get('purpose', '')

        if self.config.verbose:
            self.logger.info(f"Selected operator {len(current_operators) + 1}: {op_type} - {op_purpose}")

        return False, op_type, op_purpose, None

    def _get_abstract_operator_schema(self, op_type: str) -> Dict[str, Any]:
        """
        Get the JSON schema for a specific operator type.

        Args:
            op_type: Operator type (Map, Filter, etc.)

        Returns:
            JSON schema dictionary for the operator
        """
        from .abstract import ops

        operator_schema_map = {
            'Map': ops.Map,
            'Filter': ops.Filter,
            'Reduce': ops.Reduce,
            'Resolve': ops.Resolve,
            'Extract': ops.Extract,
            'Join': ops.Join,
            'Rank': ops.Rank,
            'TopK': ops.TopK,
            'Cluster': ops.Cluster,
            'Split': ops.Split,
            'Gather': ops.Gather,
            'Unnest': ops.Unnest,
            'Sample': ops.Sample,
        }

        # Get the operator class and call its get_json_schema() method
        operator_class = operator_schema_map.get(op_type, Operator)
        return operator_class.get_json_schema()

    def _fill_operator_details(
        self,
        op_type: str,
        op_purpose: str,
        query: str,
        dataset_samples: List[Dict],
        current_operators: List[Operator],
        schema_tracker: BaseSystemSchemaTracker,
        operator_index: int,
        attempt: int = 0,
        repair_context: Optional[Dict[str, Any]] = None,
        bypass_cache: bool = False
    ) -> Operator:
        """
        Fill in complete configuration for a selected operator.

        Args:
            op_type: Operator type (Map, Filter, etc.)
            op_purpose: Operator purpose/description
            query: User query
            dataset_samples: Sample data
            current_operators: List of operators generated so far
            schema_tracker: Current schema tracker
            operator_index: Index of this operator in pipeline
            attempt: Attempt number
            repair_context: Optional repair context from previous failed attempt
                           (contains failed_config, validation_errors, repair_rationale)

        Returns:
            Fully configured Operator object
        """
        if self.config.verbose:
            self.logger.info(f"Filling details for operator {operator_index + 1}: {op_type}")

        # Get current schema
        current_schema = schema_tracker.get_current_schema()
        available_fields_str = format_fields_for_prompt(current_schema)

        # Format dataset samples
        dataset_samples_str = json.dumps(dataset_samples, indent=2)

        # Format last operator
        last_operator_str = "None"
        if current_operators:
            last_op = current_operators[-1]
            last_operator_str = json.dumps(operator_to_dict(last_op), indent=2)

        # Get operator-specific prompt
        prompt = get_abstract_operator_prompt(
            operator_type=op_type,
            operator_purpose=op_purpose,
            query=query,
            dataset_samples=dataset_samples_str,
            last_operator=last_operator_str,
            available_fields=available_fields_str
        )

        # Add repair context if this is a regeneration after failed validation
        if repair_context:
            repair_section = "\n\n" + "="*20 + "\n"
            repair_section += "⚠️  PREVIOUS ATTEMPT FAILED - READ THIS CAREFULLY\n"

            if 'failed_config' in repair_context:
                repair_section += "FAILED OPERATOR CONFIGURATION:\n"
                repair_section += json.dumps(repair_context['failed_config'], indent=2)
                repair_section += "\n\n"

            if 'validation_errors' in repair_context:
                repair_section += "VALIDATION ERRORS:\n"
                for i, error in enumerate(repair_context['validation_errors'], 1):
                    repair_section += f"{i}. {error}\n"
                repair_section += "\n"

            if 'repair_rationale' in repair_context:
                repair_section += "REPAIR GUIDANCE:\n"
                repair_section += repair_context['repair_rationale']
                repair_section += "\n\n"

            repair_section += "Please regenerate the operator configuration addressing the above errors.\n"
            repair_section += "="*20

            prompt = prompt + repair_section

        # Get operator-specific schema
        parameters = self._get_abstract_operator_schema(op_type)
        system_prompt = f"You are an AI assistant that generates abstract layer {op_type} operator configurations. Always respond with valid JSON matching the required schema."

        messages = [{"role": "user", "content": prompt}]

        # Confirm with user in debug/confirm mode
        user_choice = self.ui.confirm_operator_before_llm(
            filled_operators=current_operators,
            operator_index=operator_index,
            total_operators=operator_index + 1,
            operator_type=op_type,
            operator_purpose=op_purpose,
            prompt=prompt,
            is_cached=False
        )

        if user_choice == 'abort':
            raise ValueError(f"User aborted at operator {operator_index + 1}")

        # Call LLM (bypass cache if requested or after repair)
        should_bypass_cache = bypass_cache or (user_choice == 'regenerate')
        response = llm_call(messages, schema=parameters, system_prompt=system_prompt, bypass_cache=should_bypass_cache)

        try:
            llm_response = json.loads(response)
        except json.JSONDecodeError as e:
            self._log(f"Failed to parse operator details: {e}", level='error', force=True)
            raise

        # Create Operator object from response
        abstract_operator = Operator()
        abstract_operator.name = f"op_{operator_index}_{op_type.lower()}"
        abstract_operator.type = op_type
        abstract_operator.source = {
            "system": "abstract",
            "name": abstract_operator.name
        }

        abstract_operator.input = llm_response.get('input', {})
        abstract_operator.output = llm_response.get('output', {})
        abstract_operator.properties = {}
        for key, value in llm_response.items():
            if key not in ['input', 'output']:
                abstract_operator.properties[key] = value

        if self.config.verbose:
            self.logger.info(f"Filled operator {operator_index + 1}: {op_type} with complete configuration")

        # Display generated operator
        user_choice = self.ui.display_generated_operator(
            filled_operators=current_operators,
            operator_index=operator_index,
            total_operators=operator_index + 1,
            operator_type=op_type,
            operator_config=llm_response
        )

        if user_choice == 'abort':
            raise ValueError(f"User aborted after filling operator {operator_index + 1}")
        elif user_choice == 'regenerate':
            # Recursively regenerate (bypass cache on regeneration)
            return self._fill_operator_details(
                op_type=op_type,
                op_purpose=op_purpose,
                query=query,
                dataset_samples=dataset_samples,
                current_operators=current_operators,
                schema_tracker=schema_tracker,
                operator_index=operator_index,
                attempt=attempt,
                repair_context=None,
                bypass_cache=True
            )

        return abstract_operator

    def _analyze_and_suggest_repair(
        self,
        query: str,
        current_operators: List[Operator],
        failed_operator: Operator,
        validation_errors: List[str],
        schema_tracker: BaseSystemSchemaTracker,
        operator_index: int
    ) -> Dict[str, Any]:
        """
        Analyze validation errors and get LLM's repair suggestion.

        Args:
            query: User query
            current_operators: Current operators in pipeline
            failed_operator: Operator that failed validation
            validation_errors: List of validation error messages
            schema_tracker: Current schema tracker
            operator_index: Index of failed operator

        Returns:
            Repair suggestion dictionary with action, analysis, rationale, and optional new_operator
        """
        if self.config.verbose:
            self.logger.info(f"Analyzing validation errors for operator {operator_index + 1}...")

        # Format current pipeline
        pipeline_parts = []
        for i, op in enumerate(current_operators):
            op_dict = operator_to_dict(op)
            pipeline_parts.append(f"{i+1}. {op.type} ({op.name}):\n{json.dumps(op_dict, indent=2)}")
        current_pipeline = "\n\n".join(pipeline_parts) if pipeline_parts else "Empty pipeline"

        # Format failed operator
        failed_op_dict = operator_to_dict(failed_operator)
        failed_operator_str = json.dumps(failed_op_dict, indent=2)

        # Get available fields
        current_schema = schema_tracker.get_current_schema()
        available_fields_str = format_fields_for_prompt(current_schema)

        # Get repair prompt
        prompt = get_operator_repair_prompt(
            query=query,
            current_pipeline=current_pipeline,
            failed_operator=failed_operator_str,
            validation_errors=validation_errors,
            available_fields=available_fields_str,
            operator_index=operator_index
        )

        # Define schema for repair response
        parameters = {
            "type": "object",
            "properties": {
                "analysis": {"type": "string"},
                "action": {
                    "type": "string",
                    "enum": ["DELETE", "INSERT_BEFORE", "REPLACE", "MODIFY"]
                },
                "rationale": {"type": "string"},
                "new_operator": {
                    "type": "object",
                    "properties": {
                        "type": {"type": "string"},
                        "purpose": {"type": "string"}
                    }
                }
            },
            "required": ["analysis", "action", "rationale"]
        }

        system_prompt = "You are an AI assistant that analyzes validation errors and suggests repairs for data processing pipelines. Always respond with valid JSON matching the required schema."

        messages = [{"role": "user", "content": prompt}]

        # Call LLM
        response = llm_call(messages, schema=parameters, system_prompt=system_prompt, bypass_cache=False)

        try:
            repair_suggestion = json.loads(response)
        except json.JSONDecodeError as e:
            self._log(f"Failed to parse repair suggestion: {e}", level='error', force=True)
            raise

        if self.config.verbose:
            self.logger.info(f"Repair suggestion: {repair_suggestion.get('action')}")

        return repair_suggestion

    def _apply_repair(
        self,
        repair_suggestion: Dict[str, Any],
        current_operators: List[Operator],
        schema_tracker: BaseSystemSchemaTracker,
        operator_index: int
    ) -> Tuple[int, bool]:
        """
        Apply repair action to operator list.

        Args:
            repair_suggestion: LLM's repair suggestion
            current_operators: Current list of operators (will be modified)
            schema_tracker: Schema tracker (will be modified)
            operator_index: Index of operator to repair

        Returns:
            Tuple of (new_position, should_regenerate)
            - new_position: Position to continue generation from
            - should_regenerate: Whether to regenerate from this position
        """
        action = repair_suggestion.get('action')

        if self.config.verbose:
            self.logger.info(f"Applying repair action: {action} at position {operator_index}")

        if action == 'DELETE':
            # Remove operator and rollback schema
            if operator_index < len(current_operators):
                removed_op = current_operators.pop(operator_index)
                schema_tracker.rollback(steps=1)
                self._log(f"Deleted operator {operator_index + 1}: {removed_op.type}", force=True)
            # Continue from same position (since we removed one)
            return operator_index, True

        elif action == 'INSERT_BEFORE':
            # Schema is already at correct state (before failed operator)
            # Just continue - next generation will insert before
            self._log(f"Will insert new operator before position {operator_index + 1}", force=True)
            return operator_index, True

        elif action == 'REPLACE':
            # Remove current operator and rollback schema
            if operator_index < len(current_operators):
                removed_op = current_operators.pop(operator_index)
                schema_tracker.rollback(steps=1)
                self._log(f"Removed operator {operator_index + 1} for replacement: {removed_op.type}", force=True)
            # Continue from same position with new operator
            return operator_index, True

        elif action == 'MODIFY':
            # Remove current operator and rollback schema, then regenerate
            if operator_index < len(current_operators):
                removed_op = current_operators.pop(operator_index)
                schema_tracker.rollback(steps=1)
                self._log(f"Removed operator {operator_index + 1} for modification: {removed_op.type}", force=True)
            # Continue from same position
            return operator_index, True

        else:
            raise ValueError(f"Unknown repair action: {action}")

    def _build_abstract_pipeline(
        self,
        query: str,
        dataset_samples: List[Dict],
        dataset_path: str,
        attempt: int = 0
    ) -> Pipeline:
        """
        Build complete abstract pipeline using JIT (Just-in-Time) step-by-step generation.

        Args:
            query: User query
            dataset_samples: Sample data (list of dicts)
            dataset_path: Path to dataset file
            attempt: Attempt number

        Returns:
            Abstract Pipeline object
        """
        if self.config.verbose:
            self.logger.info("Building pipeline using JIT step-by-step generation...")

        # Initialize schema tracker using base system - only use first sample
        samples_list = [dataset_samples[0]] if dataset_samples else [{}]

        dataset_schema = infer_schema_from_samples(samples_list)
        schema_tracker = BaseSystemSchemaTracker(
            base_system=self.base_system,
            initial_schema=dataset_schema,
            verbose=self.config.verbose
        )

        # JIT generation loop with position-based tracking
        filled_operators = []
        max_operators = 20  # Safety limit
        max_repair_attempts = 3  # Max repair attempts per operator

        # Track pending repair context for regeneration
        pending_repair_context = None
        pending_op_type = None
        pending_op_purpose = None
        bypass_cache_next = False  # Track if we should bypass cache after repair

        # Use position-based loop instead of iteration-based
        current_position = 0
        while current_position < max_operators:
            if self.config.verbose:
                self.logger.info(f"Position {current_position}: Generating operator {current_position + 1}...")

            # Step 1: Select next operator type (skip if we have a pending operator from repair)
            if pending_op_type is not None:
                # Using operator type from repair suggestion
                op_type = pending_op_type
                op_purpose = pending_op_purpose
                if self.config.verbose:
                    if pending_repair_context is not None:
                        self.logger.info(f"Regenerating operator at position {current_position}: {op_type} with repair context")
                    else:
                        self.logger.info(f"Inserting new operator at position {current_position}: {op_type} (from repair suggestion)")
            else:
                # Normal operator selection
                is_end, op_type, op_purpose, end_reason = self._select_next_operator(
                    query=query,
                    dataset_samples=dataset_samples,
                    current_operators=filled_operators,
                    schema_tracker=schema_tracker,
                    attempt=attempt,
                    bypass_cache=bypass_cache_next
                )

                # Check if pipeline is complete
                if is_end:
                    if self.config.verbose:
                        self.logger.info(f"Pipeline complete after {len(filled_operators)} operators: {end_reason}")
                    break

            # Step 2: Fill operator details (with repair context if available)
            operator = self._fill_operator_details(
                op_type=op_type,
                op_purpose=op_purpose,
                query=query,
                dataset_samples=dataset_samples,
                current_operators=filled_operators,
                schema_tracker=schema_tracker,
                operator_index=current_position,
                attempt=attempt,
                repair_context=pending_repair_context,
                bypass_cache=bypass_cache_next
            )

            # Clear repair context and cache bypass flag after use
            pending_repair_context = None
            pending_op_type = None
            pending_op_purpose = None
            bypass_cache_next = False

            # Step 3: Validate operator against current schema
            if self.base_system == BaseSystem.DOCETL:
                base_operator = abstract_to_docetl(operator)
            else:
                raise ValueError(f"Unsupported base system: {self.base_system.value}")

            validation_passed, validation_errors = schema_tracker.validate_operator(base_operator)

            # Handle validation errors with repair mechanism
            repair_attempts = 0
            while not validation_passed and repair_attempts < max_repair_attempts:
                self._log(f"⚠ Validation failed for operator at position {current_position}", level='warning', force=True)

                # Get repair suggestion from LLM
                repair_suggestion = self._analyze_and_suggest_repair(
                    query=query,
                    current_operators=filled_operators,
                    failed_operator=operator,
                    validation_errors=validation_errors,
                    schema_tracker=schema_tracker,
                    operator_index=current_position
                )

                # Ask user to confirm repair
                user_decision = self.ui.confirm_repair_suggestion(
                    filled_operators=filled_operators,
                    operator_index=current_position,
                    total_operators=current_position + 1,
                    repair_suggestion=repair_suggestion
                )

                if user_decision == 'abort':
                    raise ValueError(f"User aborted pipeline generation at position {current_position}")
                elif user_decision == 'skip':
                    # User chose to continue with errors
                    self._log(f"⚠ Continuing with validation errors at position {current_position}", level='warning', force=True)
                    validation_passed = True  # Force continue
                    break
                else:  # accept
                    # Store repair context before applying repair
                    repair_action = repair_suggestion.get('action')

                    if repair_action == 'MODIFY':
                        # For MODIFY: regenerate same operator type with repair context
                        pending_repair_context = {
                            'failed_config': operator_to_dict(operator),
                            'validation_errors': validation_errors,
                            'repair_rationale': repair_suggestion.get('rationale', ''),
                            'repair_action': repair_action
                        }
                        pending_op_type = op_type
                        pending_op_purpose = op_purpose
                    elif repair_action in ['INSERT_BEFORE', 'REPLACE']:
                        # For INSERT_BEFORE/REPLACE: use new operator suggested by LLM
                        new_operator = repair_suggestion.get('new_operator', {})
                        if new_operator:
                            pending_op_type = new_operator.get('type')
                            pending_op_purpose = new_operator.get('purpose', '')
                            # Don't set repair_context for new operators
                            pending_repair_context = None
                        else:
                            # Fallback: ask LLM to select new operator
                            pending_repair_context = None
                            pending_op_type = None
                            pending_op_purpose = None

                    # Apply repair and get new position
                    new_position, should_regenerate = self._apply_repair(
                        repair_suggestion=repair_suggestion,
                        current_operators=filled_operators,
                        schema_tracker=schema_tracker,
                        operator_index=current_position
                    )

                    # Handle repair regeneration
                    if should_regenerate:
                        repair_attempts += 1
                        # Set position for next iteration and bypass cache
                        current_position = new_position
                        bypass_cache_next = True
                        if self.config.verbose:
                            self.logger.info(f"Repair applied, continuing from position {current_position}")
                        # Use continue to restart loop from new position without validation
                        break

            # If we exhausted repair attempts, ask user
            if not validation_passed and repair_attempts >= max_repair_attempts:
                user_decision = self.ui.confirm_validation_error(
                    filled_operators=filled_operators,
                    operator_index=current_position,
                    total_operators=current_position + 1,
                    errors=validation_errors,
                    warnings=[]
                )

                if user_decision == 'abort':
                    raise ValueError(f"Pipeline generation aborted after {max_repair_attempts} repair attempts")
                else:
                    # Force continue
                    validation_passed = True

            # Update schema and add operator if validation passed without repair
            # OR if validation passed after user skipped errors
            if validation_passed and repair_attempts == 0:
                # Normal path: validation passed on first try
                try:
                    schema_tracker.update_schema(base_operator)
                except Exception as e:
                    error_msg = f"Failed to update schema for operator {operator.name} ({operator.type}): {str(e)}"
                    raise ValueError(error_msg)

                filled_operators.append(operator)

                if self.config.verbose:
                    self.logger.info(f"✓ Added operator {len(filled_operators)}: {operator.type}")

                # Move to next position
                current_position += 1

            elif validation_passed and repair_attempts > 0:
                # Validation passed but we applied repairs
                # This means user chose to skip errors
                try:
                    schema_tracker.update_schema(base_operator)
                except Exception as e:
                    error_msg = f"Failed to update schema for operator {operator.name} ({operator.type}): {str(e)}"
                    raise ValueError(error_msg)

                filled_operators.append(operator)

                if self.config.verbose:
                    self.logger.info(f"✓ Added operator {len(filled_operators)}: {operator.type} (with skipped errors)")

                # Move to next position
                current_position += 1
            # If validation_passed is False and repair_attempts > 0, the repair was applied
            # and we already set current_position in the repair handling above
            # So we don't increment position here - just continue to next iteration

        # Check if we have any operators
        if not filled_operators:
            raise ValueError("No operators generated - pipeline is empty")

        # Validate compatibility with base system
        operator_types = [op.type for op in filled_operators]
        is_compatible, unsupported = check_pipeline_compatibility(operator_types, self.base_system)
        if not is_compatible:
            raise ValueError(
                f"Pipeline contains unsupported operators for {self.base_system.value}: {unsupported}"
            )

        # Display pipeline summary
        self.ui.display_pipeline_summary(filled_operators, query)

        # Build Pipeline object
        pipeline_name = f"abstract_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        dataset_schema = {'fields': dataset_schema}

        abstract_pipeline = Pipeline.from_operators(
            operators=filled_operators,
            name=pipeline_name,
            input_path=None,  # Set later when executing
            output_path=None,  # Set later when executing
            properties={
                "query": query,
                "base_system": self.base_system.value,
                "default_model": "gpt-4o-mini",
            },
            dataset_schema=dataset_schema
        )

        if self.config.verbose:
            self.logger.info(f"Successfully built abstract pipeline with {len(filled_operators)} operators")

        return abstract_pipeline

    def _convert_and_execute(
        self,
        abstract_pipeline: Pipeline,
        query: str,
        dataset_path: str,
        attempt: int = 0
    ) -> Tuple[bool, Optional[str], Optional[Dict]]:
        """
        Convert abstract pipeline to base system format and execute.

        Args:
            abstract_pipeline: Abstract Pipeline object
            query: User query
            dataset_path: Path to dataset
            attempt: Current attempt number (for user confirmation)

        Returns:
            (success, output, pipeline_dict)
        """
        # Convert Pipeline to dict and save as JSON
        pipeline_dict = {
            "name": abstract_pipeline.name,
            "input_path": abstract_pipeline.input_path,
            "output_path": abstract_pipeline.output_path,
            "dataset_schema": abstract_pipeline.dataset_schema,
            "properties": abstract_pipeline.properties,
            "operators": [operator_to_dict(op) for op in abstract_pipeline.to_operators()]
        }

        # Convert abstract pipeline to base system format and execute using executor
        try:
            operators = abstract_pipeline.to_operators()

            # Build base system config using converter
            # For DocETL: convert operators to DocETL format
            if self.base_system == BaseSystem.DOCETL:
                docetl_operators = [abstract_to_docetl(op) for op in operators]
                output_path = self.file_manager.get_output_path(query, file_type='output')

                # Use executor's centralized pipeline config builder
                pipeline_dict = self.executor.build_pipeline_config(
                    operators=docetl_operators,
                    input_path=dataset_path,
                    output_path=output_path,
                    dataset_name='input',
                    step_name='generated_pipeline',
                    default_model=abstract_pipeline.properties.get('default_model', 'gpt-4o-mini')
                )
            else:
                raise ValueError(f"Unsupported base system: {self.base_system.value}")

            self._log(f"Successfully converted abstract pipeline to {self.base_system.value}")

        except Exception as e:
            self._log(f"Failed to convert abstract pipeline to {self.base_system.value}: {e}", level='error', force=True)
            self._log(traceback.format_exc(), level='error', force=True)
            return False, None, None

        try:
            # Save system config to file for validation and user confirmation
            yaml_path = self.file_manager.save_yaml(
                data=pipeline_dict,
                query=query,
                file_type='pipeline_temp',
                subdir_key='pipeline_output'
            )

            # Validate converted pipeline
            validation_result = validate_pipeline_static(
                system_name=self.base_system.value,
                pipeline_path=yaml_path,
                verbose=self.config.verbose,
                logger=self.logger
            )

            # Confirm execution with user, passing validation status
            if not self.ui.confirm_pipeline_execution(yaml_path, query, attempt, validation_passed=validation_result["passed"]):
                self._log("Pipeline execution skipped by user", force=True)
                return False, None, pipeline_dict

            # Execute pipeline using executor
            self._log(f"Executing pipeline using {self.executor.get_system_name()} executor...")
            execution_result = self.executor.execute_original_pipeline(yaml_path)

            if not execution_result.success:
                self._log(f"Pipeline execution failed: {execution_result.error}", level='error', force=True)
                return False, None, pipeline_dict

            output_data = execution_result.data
            if output_data is None:
                self._log("Pipeline executed but returned no output data", level='warning')
                output_data = []

            self._log(f"Pipeline execution completed with {len(output_data) if isinstance(output_data, list) else 'N/A'} results")

            return True, output_data, pipeline_dict

        except Exception as e:
            self._log(f"Pipeline execution failed: {e}", level='error', force=True)
            self._log(traceback.format_exc(), level='error', force=True)
            return False, None, pipeline_dict
