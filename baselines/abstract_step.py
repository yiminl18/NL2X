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
from .abstract_step_utils.checker_utils import validate_pipeline_static
from .abstract_step_utils.log_utils import PipelineFileManager
from model.litellm_client import llm_call, is_cached

# Schema tracking utilities using base system
from .abstract_step_utils.base_schema_tracker import (
    BaseSystemSchemaTracker,
    infer_schema_from_samples,
    format_fields_for_prompt,
)
from .abstract_step_utils.prompt_abstract import (
    get_operator_selection_prompt,
    get_abstract_operator_prompt,
)
from .abstract.support import (
    BaseSystem,
    get_supported_operators,
    validate_pipeline_compatibility,
)
from .abstract import ops
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

                    abstract_pipeline = self._build_abstract_pipeline(query, dataset_samples, attempt)

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

    def _build_abstract_pipeline(
        self,
        query: str,
        dataset_samples: List[Dict],
        attempt: int = 0
    ) -> Pipeline:
        """
        Build complete abstract pipeline.

        Args:
            query: User query
            dataset_samples: Sample data (list of dicts)
            attempt: Attempt number

        Returns:
            Abstract Pipeline object
        """
        if self.config.verbose:
            self.logger.info("Step 1: Selecting operators...")

        operators = self._select_operators(query, dataset_samples, attempt)

        if not operators:
            raise ValueError("No operators selected")

        if self.config.verbose:
            self.logger.info(f"Step 2: Generating details for {len(operators)} operators...")

        # Initialize schema tracker using base system - only use first sample
        samples_list = [dataset_samples[0]] if dataset_samples else [{}]

        dataset_schema = infer_schema_from_samples(samples_list)
        schema_tracker = BaseSystemSchemaTracker(
            base_system=self.base_system,
            initial_schema=dataset_schema,
            verbose=self.config.verbose
        )

        filled_operators = []
        for i, op in enumerate(operators):
            filled_op = self._generate_operator_details(
                operator=op,
                query=query,
                dataset_samples=dataset_samples,
                previous_operators=filled_operators,
                schema_tracker=schema_tracker,
                operator_index=i,
                attempt=attempt,
                total_operators=len(operators)
            )
            filled_operators.append(filled_op)

        # Validate compatibility with base system
        operator_types = [op.type for op in filled_operators]
        is_compatible, unsupported = validate_pipeline_compatibility(operator_types, self.base_system)
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

    def _select_operators(
        self,
        query: str,
        dataset_samples: List[Dict],
        attempt: int = 0,
        collect_messages: bool = False
    ) -> List[Dict[str, str]]:
        """
        Step 1: Select abstract operators needed for the pipeline.

        Args:
            query: User query
            dataset_samples: Sample data (list of dicts)
            attempt: Attempt number (for retry logic)
            collect_messages: Whether to collect messages for debugging

        Returns:
            List of selected operators with type and purpose
        """
        dataset_samples_str = json.dumps(dataset_samples, indent=2)

        # Get operator selection prompt (filtered by base system support)
        prompt = get_operator_selection_prompt(
            query=query,
            dataset_samples=dataset_samples_str,
            base_system=self.base_system
        )

        # Confirm before calling LLM in confirm/debug mode
        user_choice = self.ui.confirm_step_before_llm(
            step_name="Operator Selection (Abstract Layer)",
            step_number="1",
            step_prompt=prompt
        )
        if user_choice == 'abort':
            raise ValueError("User aborted pipeline generation at Step 1")

        # Define schema for structured operator selection
        # Only include operators supported by the base system
        supported_ops = get_supported_operators(self.base_system)
        operator_enum = sorted([op for op in supported_ops])

        parameters = {
            "type": "object",
            "properties": {
                "operators": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "type": {
                                "type": "string",
                                "enum": operator_enum  # Only supported operators
                            },
                            "purpose": {"type": "string"}
                        },
                        "required": ["type", "purpose"]
                    }
                }
            },
            "required": ["operators"]
        }

        system_prompt = f"You are an AI assistant that selects appropriate abstract operators for data processing pipelines. Only use operators supported by {self.base_system.value}. Always respond with valid JSON matching the required schema."

        messages = [{"role": "user", "content": prompt}]

        try:
            bypass_cache = (user_choice == 'regenerate')
            response = llm_call(messages, schema=parameters, system_prompt=system_prompt, bypass_cache=bypass_cache)

            result = json.loads(response)
            operators = result.get("operators", [])

            if collect_messages:
                messages.append({"role": "assistant", "content": response})

        except (json.JSONDecodeError, KeyError) as e:
            if self.config.verbose:
                self.logger.warning(f"Failed to parse structured operator selection: {e}")
            operators = []

        if self.config.verbose:
            self.logger.info(f"Selected {len(operators)} operators: {[op['type'] for op in operators]}")

        # Confirm this step in confirm/debug mode
        if not self.ui.confirm_step_execution("Step 1: Operator Selection (Abstract)", operators, query, attempt):
            raise ValueError("User aborted pipeline generation at operator selection step")

        if collect_messages:
            return operators, messages
        return operators

    def _create_operator_from_response(
        self,
        llm_response: Dict[str, Any],
        operator_index: int,
        op_type: str
    ) -> Operator:
        """
        Create an Operator object from LLM response.

        Args:
            llm_response: Parsed JSON response from LLM
            operator_index: Index of this operator in the pipeline
            op_type: Type of the operator (e.g., 'Map', 'Filter')

        Returns:
            Operator object with populated fields
        """
        abstract_operator = Operator()
        abstract_operator.name = f"op_{operator_index}_{op_type.lower()}"
        abstract_operator.type = op_type
        abstract_operator.source = {
            "system": "abstract",
            "name": abstract_operator.name
        }

        abstract_operator.input = llm_response.get('input', {})
        abstract_operator.output = llm_response.get('output', {})

        # Store other properties
        abstract_operator.properties = {}
        for key, value in llm_response.items():
            if key not in ['input', 'output']:
                abstract_operator.properties[key] = value

        return abstract_operator

    def _get_abstract_operator_schema(self, op_type: str) -> Dict[str, Any]:
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

    def _generate_operator_details(
        self,
        operator: Dict[str, Any],
        query: str,
        dataset_samples: List[Dict],
        previous_operators: List[Operator],
        schema_tracker: BaseSystemSchemaTracker = None,
        collect_messages: bool = False,
        operator_index: int = 0,
        attempt: int = 0,
        total_operators: int = 1
    ) -> Dict[str, Any]:
        """
        Step 2: Generate details for a single abstract operator.

        Args:
            operator: Operator framework (type and purpose)
            query: User query
            dataset_samples: Sample data (list of dicts)
            previous_operators: Previously generated operators
            schema_tracker: Schema tracker using base system
            collect_messages: Whether to collect messages
            operator_index: Index of this operator
            attempt: Attempt number
            total_operators: Total number of operators

        Returns:
            Filled operator configuration
        """
        op_type = operator['type']
        operator_purpose = operator['purpose']

        # Get current schema in abstract format for LLM prompt
        current_schema = schema_tracker.get_current_schema()
        available_fields_str = format_fields_for_prompt(current_schema)

        dataset_samples_str = json.dumps(dataset_samples, indent=2)

        # Format previous operators (convert Operator objects to dicts)
        if previous_operators:
            previous_operators_dicts = [operator_to_dict(op) for op in previous_operators]
            previous_operators_str = json.dumps(previous_operators_dicts, indent=2)
        else:
            previous_operators_str = "None"

        # Get operator-specific prompt
        prompt = get_abstract_operator_prompt(
            operator_type=op_type,
            operator_purpose=operator_purpose,
            query=query,
            dataset_samples=dataset_samples_str,
            previous_operators=previous_operators_str,
            available_fields=available_fields_str
        )

        # Prepare LLM call parameters
        parameters = self._get_abstract_operator_schema(op_type)
        system_prompt = f"You are an AI assistant that generates abstract layer {op_type} operator configurations. Always respond with valid JSON matching the required schema."
        messages = [{"role": "user", "content": prompt}]

        # Check if this prompt is already cached
        prompt_is_cached = is_cached(
            messages=messages,
            schema=parameters,
            system_prompt=system_prompt
        )

        # Confirm this operator generation with user
        user_choice = self.ui.confirm_operator_before_llm(
            operator_index=operator_index,
            total_operators=total_operators,
            operator_type=op_type,
            operator_purpose=operator_purpose,
            prompt=prompt,
            is_cached=prompt_is_cached,
        )

        if user_choice == 'abort':
            raise ValueError(f"User aborted operator generation at operator {operator_index + 1}")

        try:
            bypass_cache = (user_choice == 'regenerate')
            response = llm_call(messages, schema=parameters, system_prompt=system_prompt, bypass_cache=bypass_cache)

            llm_response = json.loads(response)

            # Create operator from LLM response
            abstract_operator = self._create_operator_from_response(llm_response, operator_index, op_type)

            # Collect response for debugging
            if collect_messages:
                messages.append({"role": "assistant", "content": response})

        except (json.JSONDecodeError, KeyError) as e:
            if self.config.verbose:
                self.logger.warning(f"Failed to parse operator {operator_index}: {e}")
            raise

        # Apply operator to schema tracker (validates and updates schema)
        success, errors = schema_tracker.apply_operator(abstract_operator)
        if not success:
            error_msg = f"Failed to apply operator to schema: {', '.join(errors)}"
            if self.config.verbose:
                self.logger.warning(error_msg)
            # Continue anyway - let base system handle the error during execution

        if self.config.verbose:
            self.logger.info(f"Generated operator {operator_index + 1}/{total_operators}: {op_type}")

        # Display generated operator and ask for confirmation
        user_choice = self.ui.display_generated_operator(
            operator_index=operator_index,
            total_operators=total_operators,
            operator_type=op_type,
            operator_config=llm_response
        )

        if user_choice == 'abort':
            raise ValueError(f"User aborted after generating operator {operator_index + 1}")
        elif user_choice == 'regenerate':
            # Recursively regenerate this operator with force_regenerate flag
            if self.config.verbose:
                self.logger.info(f"Regenerating operator {operator_index + 1}...")

            response = llm_call(messages, schema=parameters, system_prompt=system_prompt, bypass_cache=True)

            llm_response = json.loads(response)

            # Create operator from regenerated LLM response
            abstract_operator = self._create_operator_from_response(llm_response, operator_index, op_type)

            # Apply operator to schema tracker (validates and updates schema)
            success, errors = schema_tracker.apply_operator(abstract_operator)
            if not success:
                error_msg = f"Failed to apply regenerated operator to schema: {', '.join(errors)}"
                if self.config.verbose:
                    self.logger.warning(error_msg)

            # Show regenerated operator
            regenerate_choice = self.ui.display_generated_operator(
                operator_index=operator_index,
                total_operators=total_operators,
                operator_type=op_type,
                operator_config=llm_response
            )

            if regenerate_choice == 'abort':
                raise ValueError(f"User aborted after regenerating operator {operator_index + 1}")
            elif regenerate_choice == 'regenerate':
                # User wants to regenerate again - recursive call
                return self._generate_operator_details(
                    operator=operator,
                    query=query,
                    dataset_samples=dataset_samples,
                    previous_operators=previous_operators,
                    schema_tracker=schema_tracker,
                    collect_messages=collect_messages,
                    operator_index=operator_index,
                    attempt=attempt,
                    total_operators=total_operators
                )

        if collect_messages:
            return abstract_operator, messages
        return abstract_operator


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
            validate_pipeline_static(
                system_name=self.base_system.value,
                pipeline_path=yaml_path,
                verbose=self.config.verbose,
                logger=self.logger
            )

            # Confirm execution with user
            if not self.ui.confirm_pipeline_execution(yaml_path, query, attempt):
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
