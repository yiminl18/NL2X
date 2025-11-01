"""
Abstract Layer Step-by-Step Pipeline Generation

This baseline generates abstract layer pipelines and converts them to base systems for execution.

The abstract layer is system-agnostic and can theoretically be converted to different
execution systems (currently only DocETL is supported).

Currently, we assume the dataset contains only one json.
"""

import time
from typing import Any, Dict, List, Optional, Tuple
import os
import json
import traceback
from datetime import datetime
import logging
from dataclasses import dataclass, field

from .base import BaselineInterface, BaselineResult
from . import register_baseline
from .abstract_step_utils.ui import AbstractStepUserInterface
from .abstract_step_utils.validation import validate_pipeline_static
from .abstract_step_utils.file_manager import PipelineFileManager
from .abstract_step_utils.context import UserContext
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
    format_pipeline_compact,
    format_repair_context,
)
from .abstract.support import (
    BaseSystem,
    get_supported_operators,
    check_pipeline_compatibility,
)
from .abstract.ops.base import Operator
from .abstract.procedure import Procedure
from .abstract.convert.docetl import (
    abstract_to_docetl,
    operator_to_dict,
)
from .abstract.executor import DocETLExecutor
from .utils import DataTruncator


@dataclass
class RepairContext:
    """Context for operator repair during validation failures."""
    failed_config: Dict[str, Any]
    validation_errors: List[str]
    repair_rationale: str


@dataclass
class PipelineGenerationState:
    """State management for pipeline generation process."""
    pending_repair_context: Optional[RepairContext] = None
    pending_op_type: Optional[str] = None
    pending_op_purpose: Optional[str] = None
    bypass_cache_next: bool = False
    repair_attempts: int = 0
    current_position: int = 0

    def clear_pending_state(self) -> None:
        """Clear pending operation state after use."""
        self.pending_repair_context = None
        self.pending_op_type = None
        self.pending_op_purpose = None
        self.bypass_cache_next = False

    def set_repair_state(self, op_type: str, op_purpose: str,
                        repair_context: Optional[RepairContext] = None) -> None:
        """Set state for repair operation."""
        self.pending_op_type = op_type
        self.pending_op_purpose = op_purpose
        self.pending_repair_context = repair_context
        self.bypass_cache_next = True


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

    def _convert_to_base_system(self, operator: Operator) -> Dict[str, Any]:
        # Convert abstract operator to base system format.
        if self.base_system == BaseSystem.DOCETL:
            return abstract_to_docetl(operator)
        else:
            raise ValueError(f"Unsupported base system: {self.base_system.value}")


    def _update_schema(self, schema_tracker: BaseSystemSchemaTracker,
                           base_operator: Dict[str, Any],
                           operator: Operator) -> None:
        try:
            schema_tracker.update_schema(base_operator)
        except Exception as e:
            error_msg = f"Failed to update schema for operator {operator.name} ({operator.type}): {str(e)}"
            raise ValueError(error_msg)

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
        initial_schema: Dict[str, str],
        attempt: int = 0,
        bypass_cache: bool = False,
        user_context_obj: Optional[UserContext] = None
    ) -> Tuple[bool, Optional[str], Optional[str], Optional[str]]:
        """
        Select the next operator type using JIT approach (selection only, not configuration).

        Args:
            query: User query
            dataset_samples: Sample data
            current_operators: List of operators generated so far
            schema_tracker: Current schema tracker
            initial_schema: Initial dataset schema
            attempt: Attempt number
            bypass_cache: Whether to bypass cache
            user_context_obj: Optional UserContext object for managing user constraints

        Returns:
            Tuple of (is_end, op_type, op_purpose, end_reason)
            - is_end: True if pipeline is complete
            - op_type: Operator type string (None if is_end)
            - op_purpose: Operator purpose string (None if is_end)
            - end_reason: Reason for ending (None if not is_end)
        """
        if self.config.verbose:
            self.logger.info(f"Generating next operator (current: {len(current_operators)} operators)...")

        # Format current pipeline in compact format
        current_pipeline = format_pipeline_compact(
            operators=current_operators,
            initial_schema=initial_schema,
        )

        # Get current schema
        current_schema = schema_tracker.get_current_schema()
        available_fields_str = format_fields_for_prompt(current_schema)

        # Format dataset samples
        dataset_samples_str = json.dumps(dataset_samples, indent=2)

        # Get prompt
        prompt = get_next_operator_prompt(
            query=query,
            dataset_samples=dataset_samples_str,
            current_pipeline=current_pipeline,
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

        # Confirm with user in debug/confirm mode
        user_choice, user_constraints = self.ui.confirm_step_before_llm(
            filled_operators=current_operators,
            step_name=f"Operator {len(current_operators) + 1}",
            iteration=len(current_operators) + 1,
            step_type="selection",
            step_prompt=prompt  # Always pass prompt, UI will handle display based on mode
        )

        if user_choice == 'abort':
            raise ValueError(f"User aborted at operator {len(current_operators) + 1}")

        # Add user constraints to context if provided
        operator_position = len(current_operators)
        if user_context_obj and user_constraints:
            for constraint in user_constraints:
                user_context_obj.add_for_operator(operator_position, constraint)

        # Build system prompt with constraints (if any)
        base_system_prompt = f"You are an AI assistant that builds data processing pipelines one operator at a time for {self.base_system.value}."

        if user_context_obj and user_context_obj.has_constraints(operator_position):
            # Get constraints for this operator selection
            constraints = user_context_obj.get_for_operator(operator_position)
            constraint_text = "\n".join(f"   - {c}" for c in constraints)

            system_prompt = f"""{base_system_prompt}

CRITICAL: The user has specified MANDATORY constraints for operator selection:

{constraint_text}

You MUST consider these constraints when selecting the operator type and purpose.
These requirements override any default behavior or standard practices.

Always respond with valid JSON matching the required schema."""
        else:
            system_prompt = f"{base_system_prompt} Always respond with valid JSON matching the required schema."

        # Append user context to prompt as reminder
        final_prompt = prompt
        if user_context_obj:
            context_section = user_context_obj.format_for_prompt(operator_position)
            if context_section:
                final_prompt = prompt + context_section

        messages = [{"role": "user", "content": final_prompt}]

        # Call LLM (bypass cache if requested or after repair)
        should_bypass_cache = bypass_cache or (user_choice == 'regenerate')
        regenerate_temperature = 0.5 if (user_choice == 'regenerate') else 0.3
        response = llm_call(messages, schema=parameters, system_prompt=system_prompt, bypass_cache=should_bypass_cache, temperature=regenerate_temperature)

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

    def _prepare_operator_prompt(
        self,
        op_type: str,
        op_purpose: str,
        query: str,
        dataset_samples: List[Dict],
        current_operators: List[Operator],
        schema_tracker: BaseSystemSchemaTracker,
        initial_schema: Dict[str, str],
        repair_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Prepare prompt for operator detail generation.

        Args:
            op_type: Operator type
            op_purpose: Operator purpose
            query: User query
            dataset_samples: Sample data
            current_operators: List of current operators
            schema_tracker: Schema tracker
            initial_schema: Initial dataset schema
            repair_context: Optional repair context

        Returns:
            Formatted prompt string
        """
        # Get current schema
        current_schema = schema_tracker.get_current_schema()
        available_fields_str = format_fields_for_prompt(current_schema)

        # Format dataset samples
        dataset_samples_str = json.dumps(dataset_samples, indent=2)

        # Format current pipeline in compact format
        current_pipeline = format_pipeline_compact(
            operators=current_operators,
            initial_schema=initial_schema,
        )

        # Get operator-specific prompt
        prompt = get_abstract_operator_prompt(
            operator_type=op_type,
            operator_purpose=op_purpose,
            query=query,
            dataset_samples=dataset_samples_str,
            current_pipeline=current_pipeline,
            available_fields=available_fields_str
        )

        # Add repair context if this is a regeneration after failed validation
        if repair_context:
            repair_section = format_repair_context(repair_context)
            prompt = prompt + repair_section

        return prompt

    def _get_operator_from_llm(
        self,
        op_type: str,
        prompt: str,
        operator_index: int,
        current_operators: List[Operator],
        op_purpose: str,
        bypass_cache: bool = False,
        user_context_obj: Optional['UserContext'] = None
    ) -> Dict[str, Any]:
        """Call LLM to get operator configuration.

        Args:
            op_type: Operator type
            prompt: Prepared prompt
            operator_index: Index of this operator
            current_operators: Current operators list
            op_purpose: Operator purpose
            bypass_cache: Whether to bypass cache
            user_context_obj: Optional UserContext object for managing constraints

        Returns:
            LLM response as dictionary
        """
        # Get operator-specific schema
        parameters = self._get_abstract_operator_schema(op_type)

        # Confirm with user in debug/confirm mode
        user_choice, user_constraints = self.ui.confirm_operator_before_llm(
            filled_operators=current_operators,
            operator_index=operator_index,
            total_operators=operator_index + 1,
            operator_type=op_type,
            operator_purpose=op_purpose,
            prompt=prompt,
            is_cached=False
        )

        # Add user constraints to context if provided
        if user_context_obj and user_constraints:
            for constraint in user_constraints:
                user_context_obj.add_for_operator(operator_index, constraint)

        # Build system prompt with constraints (if any)
        base_system_prompt = f"You are an AI assistant that generates abstract layer {op_type} operator configurations."

        if user_context_obj and user_context_obj.has_constraints(operator_index):
            # Get constraints for this operator
            constraints = user_context_obj.get_for_operator(operator_index)
            constraint_text = "\n".join(f"   - {c}" for c in constraints)

            system_prompt = f"""{base_system_prompt}

CRITICAL: The user has specified MANDATORY constraints for this operator:

{constraint_text}

You MUST ensure the generated configuration strictly adheres to ALL these constraints.
These requirements override any default behavior or standard practices.

Always respond with valid JSON matching the required schema."""
        else:
            system_prompt = f"{base_system_prompt} Always respond with valid JSON matching the required schema."

        # Append user context to prompt as reminder
        final_prompt = prompt
        if user_context_obj:
            context_section = user_context_obj.format_for_prompt(operator_index)
            if context_section:
                final_prompt = prompt + context_section

        messages = [{"role": "user", "content": final_prompt}]

        if user_choice == 'abort':
            raise ValueError(f"User aborted at operator {operator_index + 1}")

        # Call LLM (bypass cache if requested or user chose regenerate)
        should_bypass_cache = bypass_cache or (user_choice == 'regenerate')
        regenerate_temperature = 0.5 if (user_choice == 'regenerate') else 0.3
        response = llm_call(messages, schema=parameters, system_prompt=system_prompt, bypass_cache=should_bypass_cache, temperature=regenerate_temperature)

        try:
            llm_response = json.loads(response)
        except json.JSONDecodeError as e:
            self._log(f"Failed to parse operator details: {e}", level='error', force=True)
            raise

        return llm_response

    def _clean_escaped_chars(self, obj: Any) -> Any:
        """Recursively clean escaped characters from strings in data structures.

        Args:
            obj: Object to clean (can be dict, list, str, or other types)

        Returns:
            Cleaned object with escaped characters removed from strings
        """
        if isinstance(obj, dict):
            return {key: self._clean_escaped_chars(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._clean_escaped_chars(item) for item in obj]
        elif isinstance(obj, str):
            # Replace escaped characters with their actual equivalents
            return obj.replace('\\"', '"').replace('\\n', '\n').replace('\\t', '\t')
        else:
            return obj

    def _create_operator_instance(
        self,
        llm_response: Dict[str, Any],
        op_type: str,
        operator_index: int,
        op_purpose: str = ""
    ) -> Operator:
        """Create Operator object from LLM response.

        Args:
            llm_response: LLM response dictionary (already cleaned)
            op_type: Operator type
            operator_index: Index of this operator
            op_purpose: Operator purpose (stored in properties)

        Returns:
            Operator instance
        """
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

        # Store operator purpose in properties
        if op_purpose:
            abstract_operator.properties['purpose'] = op_purpose

        if self.config.verbose:
            self.logger.info(f"Created operator {operator_index + 1}: {op_type} with complete configuration")

        return abstract_operator

    def _fill_operator_details(
        self,
        op_type: str,
        op_purpose: str,
        query: str,
        dataset_samples: List[Dict],
        current_operators: List[Operator],
        schema_tracker: BaseSystemSchemaTracker,
        initial_schema: Dict[str, str],
        operator_index: int,
        attempt: int = 0,
        repair_context: Optional[Dict[str, Any]] = None,
        bypass_cache: bool = False,
        user_context_obj: Optional[UserContext] = None
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
            initial_schema: Initial dataset schema
            operator_index: Index of this operator in pipeline
            attempt: Attempt number
            repair_context: Optional repair context from previous failed attempt
                           (contains failed_config, validation_errors, repair_rationale)
            bypass_cache: Whether to bypass cache
            user_context_obj: Optional UserContext object for managing user constraints

        Returns:
            Fully configured Operator object
        """
        if self.config.verbose:
            self.logger.info(f"Filling details for operator {operator_index + 1}: {op_type}")

        # Step 1: Prepare prompt
        prompt = self._prepare_operator_prompt(
            op_type=op_type,
            op_purpose=op_purpose,
            query=query,
            dataset_samples=dataset_samples,
            current_operators=current_operators,
            schema_tracker=schema_tracker,
            initial_schema=initial_schema,
            repair_context=repair_context
        )

        # Step 2: Get operator configuration from LLM
        llm_response = self._get_operator_from_llm(
            op_type=op_type,
            prompt=prompt,
            operator_index=operator_index,
            current_operators=current_operators,
            op_purpose=op_purpose,
            bypass_cache=bypass_cache,
            user_context_obj=user_context_obj
        )

        # Clean escaped characters from LLM response
        # llm_response = self._clean_escaped_chars(llm_response)

        # Step 3: Create Operator instance
        abstract_operator = self._create_operator_instance(
            llm_response=llm_response,
            op_type=op_type,
            operator_index=operator_index,
            op_purpose=op_purpose
        )

        # Step 4: Display generated operator and handle user choice
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
                initial_schema=initial_schema,
                operator_index=operator_index,
                attempt=attempt,
                repair_context=None,
                bypass_cache=True,
                user_context_obj=user_context_obj
            )
        elif user_choice == 'edit':
            # Allow user to edit configuration
            edited_config = self.ui.edit_operator_config(
                operator_config=llm_response,
                operator_type=op_type,
                operator_index=operator_index
            )

            # Recreate operator with edited config
            abstract_operator = self._create_operator_instance(
                llm_response=edited_config,
                op_type=op_type,
                operator_index=operator_index,
                op_purpose=op_purpose
            )

            # Ask user if they want to continue or edit again
            print(f"\n{self.ui.CYAN}Proceed with edited configuration? (Y/e/n):{self.ui.RESET} ", end="")
            confirm = input().strip().lower()

            if confirm == 'n':
                raise ValueError(f"User aborted after editing operator {operator_index + 1}")
            elif confirm == 'e':
                # Edit again (recursive call with edited config as starting point)
                return self._fill_operator_details(
                    op_type=op_type,
                    op_purpose=op_purpose,
                    query=query,
                    dataset_samples=dataset_samples,
                    current_operators=current_operators,
                    schema_tracker=schema_tracker,
                    initial_schema=initial_schema,
                    operator_index=operator_index,
                    attempt=attempt,
                    repair_context=None,
                    bypass_cache=True,
                    user_context_obj=user_context_obj
                )

        return abstract_operator

    def _analyze_and_suggest_repair(
        self,
        query: str,
        current_operators: List[Operator],
        failed_operator: Operator,
        validation_errors: List[str],
        schema_tracker: BaseSystemSchemaTracker,
        initial_schema: Dict[str, str],
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
            initial_schema: Initial dataset schema
            operator_index: Index of failed operator

        Returns:
            Repair suggestion dictionary with action, analysis, rationale, and optional new_operator
        """
        if self.config.verbose:
            self.logger.info(f"Analyzing validation errors for operator {operator_index + 1}...")

        # Format current pipeline in compact format
        current_pipeline = format_pipeline_compact(
            operators=current_operators,
            initial_schema=initial_schema,
        ) if current_operators else "No operators yet (this will be the first operator)"

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
                "rationale": {"type": "string"},
                "new_operator": {
                    "type": "object",
                    "properties": {
                        "type": {"type": "string"},
                        "purpose": {"type": "string"}
                    },
                    "required": ["type", "purpose"]
                }
            },
            "required": ["analysis", "rationale", "new_operator"]
        }

        system_prompt = "You are an AI assistant that analyzes validation errors and suggests repairs for data processing pipelines. Always respond with valid JSON matching the required schema."

        messages = [{"role": "user", "content": prompt}]

        # Confirm with user before LLM call
        user_choice = self.ui.confirm_repair_analysis_before_llm(
            filled_operators=current_operators,
            operator_index=operator_index,
            prompt=prompt
        )

        if user_choice == 'abort':
            raise ValueError(f"User aborted repair analysis at operator {operator_index + 1}")

        # Call LLM (bypass cache if user chose regenerate)
        should_bypass_cache = (user_choice == 'regenerate')
        regenerate_temperature = 0.5 if (user_choice == 'regenerate') else 0.3
        response = llm_call(messages, schema=parameters, system_prompt=system_prompt, bypass_cache=should_bypass_cache, temperature=regenerate_temperature)

        try:
            repair_suggestion = json.loads(response)
        except json.JSONDecodeError as e:
            self._log(f"Failed to parse repair suggestion: {e}", level='error', force=True)
            raise

        if self.config.verbose:
            self.logger.info(f"Repair suggestion: {repair_suggestion.get('action')}")

        return repair_suggestion

    def _generate_next_operator(
        self,
        state: PipelineGenerationState,
        query: str,
        dataset_samples: List[Dict],
        filled_operators: List[Operator],
        schema_tracker: BaseSystemSchemaTracker,
        initial_schema: Dict[str, str],
        attempt: int = 0,
        user_context_obj: Optional[UserContext] = None
    ) -> Tuple[bool, Optional[Operator], Optional[str]]:
        """Generate the next operator, either from pending state or by selection.

        Args:
            state: Pipeline generation state
            query: User query
            dataset_samples: Sample data
            filled_operators: Already filled operators
            schema_tracker: Schema tracker
            initial_schema: Initial dataset schema
            attempt: Attempt number
            user_context_obj: Optional UserContext object for managing user constraints

        Returns:
            Tuple of (is_end, operator, end_reason)
        """
        # Check if we have a pending operator from repair
        if state.pending_op_type is not None:
            # Using operator from repair suggestion
            op_type = state.pending_op_type
            op_purpose = state.pending_op_purpose

            if self.config.verbose:
                if state.pending_repair_context is not None:
                    self.logger.info(f"Regenerating operator at position {state.current_position}: {op_type} with repair context")
                else:
                    self.logger.info(f"Inserting new operator at position {state.current_position}: {op_type} (from repair suggestion)")

            # Convert RepairContext to dict if present
            repair_context_dict = None
            if state.pending_repair_context:
                repair_context_dict = {
                    'failed_config': state.pending_repair_context.failed_config,
                    'validation_errors': state.pending_repair_context.validation_errors,
                    'repair_rationale': state.pending_repair_context.repair_rationale
                }

            # Fill operator details with repair context
            operator = self._fill_operator_details(
                op_type=op_type,
                op_purpose=op_purpose,
                query=query,
                dataset_samples=dataset_samples,
                current_operators=filled_operators,
                schema_tracker=schema_tracker,
                initial_schema=initial_schema,
                operator_index=state.current_position,
                attempt=attempt,
                repair_context=repair_context_dict,
                bypass_cache=state.bypass_cache_next,
                user_context_obj=user_context_obj
            )

            return False, operator, None

        # Normal operator selection
        is_end, op_type, op_purpose, end_reason = self._select_next_operator(
            query=query,
            dataset_samples=dataset_samples,
            current_operators=filled_operators,
            schema_tracker=schema_tracker,
            initial_schema=initial_schema,
            attempt=attempt,
            bypass_cache=state.bypass_cache_next,
            user_context_obj=user_context_obj
        )

        if is_end:
            if self.config.verbose:
                self.logger.info(f"Pipeline complete after {len(filled_operators)} operators: {end_reason}")
            return True, None, end_reason

        # Fill operator details
        operator = self._fill_operator_details(
            op_type=op_type,
            op_purpose=op_purpose,
            query=query,
            dataset_samples=dataset_samples,
            current_operators=filled_operators,
            schema_tracker=schema_tracker,
            initial_schema=initial_schema,
            operator_index=state.current_position,
            attempt=attempt,
            repair_context=None,
            bypass_cache=state.bypass_cache_next,
            user_context_obj=user_context_obj
        )
        return False, operator, None

    def _validate_and_repair_operator(
        self,
        state: PipelineGenerationState,
        operator: Operator,
        query: str,
        filled_operators: List[Operator],
        schema_tracker: BaseSystemSchemaTracker,
        initial_schema: Dict[str, str],
        max_repair_attempts: int = 3
    ) -> Tuple[bool, Optional[str]]:
        """Validate operator and handle repair if needed.

        Args:
            state: Pipeline generation state
            operator: Operator to validate
            query: User query
            filled_operators: Current operators list
            schema_tracker: Schema tracker
            initial_schema: Initial dataset schema
            max_repair_attempts: Max repair attempts

        Returns:
            Tuple of (should_continue_to_next, unused)
        """
        # Convert to base system for validation
        base_operator = self._convert_to_base_system(operator)
        validation_passed, validation_errors = schema_tracker.validate_operator(base_operator)

        if validation_passed:
            # Update schema and add operator
            self._update_schema(schema_tracker, base_operator, operator)
            filled_operators.append(operator)

            if self.config.verbose:
                self.logger.info(f"✓ Added operator {len(filled_operators)}: {operator.type}")

            return True, None

        # Handle validation errors with repair
        state.repair_attempts = 0
        while not validation_passed and state.repair_attempts < max_repair_attempts:
            self._log(f"⚠ Validation failed for operator at position {state.current_position}", level='warning', force=True)

            # Get repair suggestion from LLM
            repair_suggestion = self._analyze_and_suggest_repair(
                query=query,
                current_operators=filled_operators,
                failed_operator=operator,
                validation_errors=validation_errors,
                schema_tracker=schema_tracker,
                initial_schema=initial_schema,
                operator_index=state.current_position
            )

            # Ask user to confirm repair
            user_decision = self.ui.confirm_repair_suggestion(
                filled_operators=filled_operators,
                operator_index=state.current_position,
                total_operators=state.current_position + 1,
                repair_suggestion=repair_suggestion
            )

            if user_decision == 'abort':
                raise ValueError(f"User aborted pipeline generation at position {state.current_position}")
            elif user_decision == 'skip':
                # User chose to continue with errors
                self._log(f"⚠ Continuing with validation errors at position {state.current_position}", level='warning', force=True)
                self._update_schema(schema_tracker, base_operator, operator)
                filled_operators.append(operator)
                return True, None
            else:  # accept
                # Store repair context before applying repair
                new_operator = repair_suggestion.get('new_operator', {})

                # Create repair context with complete failure information
                repair_ctx = RepairContext(
                    failed_config=operator_to_dict(operator),
                    validation_errors=validation_errors,
                    repair_rationale=repair_suggestion.get('rationale', '')
                )

                # Set repair state with complete context
                state.set_repair_state(
                    new_operator.get('type'),
                    new_operator.get('purpose', ''),
                    repair_ctx
                )

                # Apply repair and get new position
                new_position, should_regenerate = self._apply_repair(
                    repair_suggestion=repair_suggestion,
                    current_operators=filled_operators,
                    schema_tracker=schema_tracker,
                    operator_index=state.current_position
                )

                if should_regenerate:
                    state.repair_attempts += 1
                    state.current_position = new_position
                    if self.config.verbose:
                        self.logger.info(f"Repair applied, continuing from position {state.current_position}")
                    return False, None

        # If we exhausted repair attempts, ask user
        if not validation_passed:
            user_decision = self.ui.confirm_validation_error(
                filled_operators=filled_operators,
                operator_index=state.current_position,
                total_operators=state.current_position + 1,
                errors=validation_errors,
                warnings=[]
            )

            if user_decision == 'abort':
                raise ValueError(f"Pipeline generation aborted after {max_repair_attempts} repair attempts")
            else:
                # Force continue
                self._update_schema(schema_tracker, base_operator, operator)
                filled_operators.append(operator)
                if self.config.verbose:
                    self.logger.info(f"✓ Added operator {len(filled_operators)}: {operator.type} (with skipped errors)")
                return True, None

        return True, None

    def _apply_repair(
        self,
        repair_suggestion: Dict[str, Any],
        current_operators: List[Operator],
        schema_tracker: BaseSystemSchemaTracker,
        operator_index: int
    ) -> Tuple[int, bool]:
        """
        Apply repair: remove failed operator and rollback schema for regeneration.

        Args:
            repair_suggestion: LLM's repair suggestion (unused, kept for compatibility)
            current_operators: Current list of operators (will be modified)
            schema_tracker: Schema tracker (will be modified)
            operator_index: Index of operator to repair

        Returns:
            Tuple of (new_position, should_regenerate)
            - new_position: Position to continue generation from
            - should_regenerate: Always True (will regenerate from this position)
        """
        if self.config.verbose:
            self.logger.info(f"Applying repair at position {operator_index}")

        # Remove failed operator and rollback schema
        if operator_index < len(current_operators):
            removed_op = current_operators.pop(operator_index)
            schema_tracker.rollback(steps=1)
            self._log(f"Removed operator {operator_index + 1} for repair: {removed_op.type}", force=True)

        return operator_index, True

    def _build_abstract_pipeline(
        self,
        query: str,
        dataset_samples: List[Dict],
        dataset_path: str,  # Used for pipeline metadata
        attempt: int = 0
    ) -> Procedure:
        """
        Build complete abstract procedure using JIT (Just-in-Time) step-by-step generation.

        Args:
            query: User query
            dataset_samples: Sample data (list of dicts)
            dataset_path: Path to dataset file (for pipeline metadata)
            attempt: Attempt number

        Returns:
            Abstract Procedure object
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

        # Initialize state and operators list
        state = PipelineGenerationState()
        filled_operators = []
        max_operators = 20  # Safety limit
        max_repair_attempts = 3  # Max repair attempts per operator

        # Initialize user context for interactive constraints
        user_context = UserContext()

        while state.current_position < max_operators:
            if self.config.verbose:
                self.logger.info(f"Position {state.current_position}: Generating operator {state.current_position + 1}...")

            # Step 1: Generate next operator
            is_end, operator, end_reason = self._generate_next_operator(
                state=state,
                query=query,
                dataset_samples=dataset_samples,
                filled_operators=filled_operators,
                schema_tracker=schema_tracker,
                initial_schema=dataset_schema,
                attempt=attempt,
                user_context_obj=user_context
            )

            # Check if pipeline is complete
            if is_end:
                if self.config.verbose:
                    self.logger.info(f"Pipeline complete: {end_reason}")
                break

            # Clear pending state after generation
            state.clear_pending_state()

            # Step 2: Validate and repair if needed
            should_continue, _ = self._validate_and_repair_operator(
                state=state,
                operator=operator,
                query=query,
                filled_operators=filled_operators,
                schema_tracker=schema_tracker,
                initial_schema=dataset_schema,
                max_repair_attempts=max_repair_attempts
            )

            if should_continue:
                # Move to next position
                state.current_position += 1
            # else: repair was applied and position already updated

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

        # Build Procedure object
        pipeline_name = f"abstract_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        dataset_schema = {'fields': dataset_schema}

        abstract_pipeline = Procedure.from_operators(
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
        abstract_pipeline: Procedure,
        query: str,
        dataset_path: str,
        attempt: int = 0
    ) -> Tuple[bool, Optional[str], Optional[Dict]]:
        """
        Convert abstract procedure to base system format and execute.

        Args:
            abstract_pipeline: Abstract Procedure object
            query: User query
            dataset_path: Path to dataset
            attempt: Current attempt number (for user confirmation)

        Returns:
            (success, output, pipeline_dict)
        """
        # Convert Procedure to dict and save as JSON
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
                docetl_operators = [self._convert_to_base_system(op) for op in operators]
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
