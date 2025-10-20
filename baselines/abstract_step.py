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
from .docetl_utils.data_utils import DocETLDataProcessor
from .abstract_step_utils.ui import AbstractStepUserInterface
from .abstract_step_utils.checker_utils import validate_pipeline_static
from .abstract_step_utils.log_utils import PipelineFileManager
from .docetl_utils.pipeline_utils import (
    load_sample_data,
    execute_single_pipeline,
    FailedPipeline,
    find_output_path,
)
from model.litellm_client import llm_call

# Abstract layer utilities
from .abstract_step_utils.type_utils_abstract import (
    AbstractTypeSystem,
    infer_schema_from_samples,
    format_fields_for_prompt,
)
# Import unified type system for validation
from .abstract.type import validate_type_syntax
from .abstract_step_utils.prompt_abstract import (
    get_operator_selection_prompt,
    get_abstract_operator_prompt,
)
from .abstract.support import (
    BaseSystem,
    get_supported_operators,
    validate_pipeline_compatibility,
    filter_unsupported_operators,
)

# Use existing converter from abstract layer
from .abstract.ops.base import Operator
from .abstract.ops import Map, Filter, Reduce, Resolve, Extract
from .abstract.pipeline import Pipeline
from .abstract.convert.docetl import (
    abstract_to_docetl,
    operator_to_dict,
)


@register_baseline("abstract_step")
class AbstractStepBaseline(BaselineInterface):
    """
    Abstract Layer Step-by-Step Pipeline Generation

    Steps:
    1. Select abstract operators needed
    2. Generate details for each operator (abstract format)
    3. Build abstract Pipeline object
    4. Convert to DocETL YAML
    5. Execute and validate
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
        self.total_generation_attempts = 0
        self.successful_pipelines = 0
        self.failed_pipelines = []

        self.max_attempts = self.config.max_attempts
        self.validate_answer = self.config.validate_answer

        # Set base system (currently only DocETL supported)
        self.base_system = BaseSystem.DOCETL

        # Configure logging
        if not self.config.verbose:
            logging.getLogger("LiteLLM").setLevel(logging.ERROR)
            logging.getLogger("httpcore").setLevel(logging.ERROR)
            logging.getLogger("openai").setLevel(logging.ERROR)
            logging.getLogger("asyncio").setLevel(logging.ERROR)
        else:
            logging.getLogger("LiteLLM").setLevel(logging.INFO)
            logging.getLogger("httpcore").setLevel(logging.INFO)
            logging.getLogger("openai").setLevel(logging.INFO)

        # Setup output directories
        base_dir = os.getcwd()
        self.output_dirs = {
            'pipeline_output_dir': os.path.join(base_dir, "generated_pipelines", "abstract_step"),
            'converted_data_dir': os.path.join(base_dir, "converted_data", "abstract_step"),
            'abstract_pipeline_dir': os.path.join(base_dir, "abstract_pipelines", "abstract_step"),
        }

        for attr_name, dir_path in self.output_dirs.items():
            setattr(self, attr_name, dir_path)
            os.makedirs(dir_path, exist_ok=True)

        # Initialize data processor
        self.data_processor = DocETLDataProcessor(
            converted_data_dir=self.converted_data_dir,
            verbose=self.config.verbose,
            logger=self.logger
        )

        # Initialize user interface for confirmations
        self.ui = AbstractStepUserInterface(self.config)

        # Initialize file manager for centralized file operations
        self.file_manager = PipelineFileManager(
            base_dirs={
                'pipeline_output': self.pipeline_output_dir,
                'abstract_pipeline': self.abstract_pipeline_dir,
                'converted_data': self.converted_data_dir
            },
            data_processor=self.data_processor,
            logger=self.logger,
            verbose=self.config.verbose
        )

        self._log(f"Initialized Abstract Step baseline with config: {self.config}")
        self._log(f"Base system: {self.base_system.value}")
        self._log(f"Supported operators: {get_supported_operators(self.base_system)}")
        self._log(f"Pipeline output directory: {self.pipeline_output_dir}")
        self._log(f"Abstract pipeline directory: {self.abstract_pipeline_dir}")

    def _select_operators(
        self,
        query: str,
        dataset_samples: Dict[str, Any],
        attempt: int = 0,
        collect_messages: bool = False
    ) -> List[Dict[str, str]]:
        """
        Step 1: Select abstract operators needed for the pipeline.

        Args:
            query: User query
            dataset_samples: Sample data
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
        user_choice = self.ui.confirm_step_before_llm("Operator Selection (Abstract Layer)", "1")
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

            operators = filter_unsupported_operators(operators, self.base_system)

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

    def _get_abstract_operator_schema(self, op_type: str) -> Dict[str, Any]:
        # Map operator types to their corresponding classes
        operator_schema_map = {
            'Map': Map,
            'Filter': Filter,
            'Reduce': Reduce,
            'Resolve': Resolve,
            'Extract': Extract,
        }

        # Get the operator class and call its get_json_schema() method
        operator_class = operator_schema_map.get(op_type, Operator)
        return operator_class.get_json_schema()

    def _generate_operator_details(
        self,
        operator: Dict[str, Any],
        query: str,
        dataset_samples: Dict[str, Any],
        previous_operators: List[Operator],
        type_system: AbstractTypeSystem = None,
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
            dataset_samples: Sample data
            previous_operators: Previously generated operators
            type_system: Current type system state
            collect_messages: Whether to collect messages
            operator_index: Index of this operator
            attempt: Attempt number
            total_operators: Total number of operators

        Returns:
            Filled operator configuration
        """
        op_type = operator['type']
        operator_purpose = operator['purpose']

        current_schema = type_system.get_current_schema()
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

        # Confirm this operator generation with user
        user_choice = self.ui.confirm_operator_before_llm(
            operator_index=operator_index,
            total_operators=total_operators,
            operator_type=op_type,
            operator_purpose=operator_purpose,
            prompt=prompt,
        )

        if user_choice == 'abort':
            raise ValueError(f"User aborted operator generation at operator {operator_index + 1}")

        parameters = self._get_abstract_operator_schema(op_type)

        system_prompt = f"You are an AI assistant that generates abstract layer {op_type} operator configurations. Always respond with valid JSON matching the required schema."

        messages = [{"role": "user", "content": prompt}]

        try:
            bypass_cache = (user_choice == 'regenerate')
            response = llm_call(messages, schema=parameters, system_prompt=system_prompt, bypass_cache=bypass_cache)

            llm_response = json.loads(response)

            abstract_operator = Operator()
            abstract_operator.name = f"op_{operator_index}_{op_type.lower()}"
            abstract_operator.type = op_type
            abstract_operator.source = {
                "system": "llm_generated",
                "name": abstract_operator.name
            }

            abstract_operator.input = llm_response.get('input', {})
            abstract_operator.output = llm_response.get('output', {})

            # Validate and fix output schema types
            if 'fields' in abstract_operator.input and isinstance(abstract_operator.input['fields'], dict):
                abstract_operator.input['fields'] = self._validate_and_fix_types(
                    abstract_operator.input['fields'],
                    f"operator_{operator_index}_input"
                )

            if isinstance(abstract_operator.output, dict):
                abstract_operator.output = self._validate_and_fix_types(
                    abstract_operator.output,
                    f"operator_{operator_index}_output"
                )

            abstract_operator.properties = {}
            for key, value in llm_response.items():
                if key not in ['input', 'output']:
                    abstract_operator.properties[key] = value

            # Validate and fix prompt if it exists
            if 'prompt' in abstract_operator.properties:
                input_field_names = set(abstract_operator.input.get('fields', {}).keys())
                fixed_prompt, was_modified = self._validate_and_fix_prompt(
                    abstract_operator.properties['prompt'],
                    input_field_names,
                    abstract_operator.name
                )
                if was_modified:
                    abstract_operator.properties['prompt'] = fixed_prompt

            # Remove duplicate input fields from output schema
            if isinstance(abstract_operator.output, dict) and 'fields' in abstract_operator.input:
                input_fields = abstract_operator.input['fields']
                output_fields_to_remove = []

                for field_name in list(abstract_operator.output.keys()):
                    if field_name == 'type':  # Skip special keys
                        continue

                    # Check if this field exists in input with same type
                    if field_name in input_fields:
                        input_type = input_fields[field_name]
                        output_type = abstract_operator.output[field_name]

                        # If types match, this is a duplicate (input fields are preserved automatically)
                        if input_type == output_type:
                            output_fields_to_remove.append(field_name)
                            if self.config.verbose:
                                self.logger.info(
                                    f"{abstract_operator.name}: Removed duplicate field '{field_name}' "
                                    f"from output schema (already in input with same type '{input_type}')"
                                )

                # Remove duplicate fields
                for field_name in output_fields_to_remove:
                    del abstract_operator.output[field_name]

            # Collect response for debugging
            if collect_messages:
                messages.append({"role": "assistant", "content": response})

        except (json.JSONDecodeError, KeyError) as e:
            if self.config.verbose:
                self.logger.warning(f"Failed to parse operator {operator_index}: {e}")
            raise

        # Update type system with this operator's transformation
        self._apply_operator_to_type_system(type_system, abstract_operator)

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

            abstract_operator = Operator()
            abstract_operator.name = f"op_{operator_index}_{op_type.lower()}"
            abstract_operator.type = op_type
            abstract_operator.source = {
                "system": self.base_system.value,
                "name": abstract_operator.name
            }

            abstract_operator.input = llm_response.get('input', {})
            abstract_operator.output = llm_response.get('output', {})

            # Validate and fix types for regenerated operator
            if 'fields' in abstract_operator.input and isinstance(abstract_operator.input['fields'], dict):
                abstract_operator.input['fields'] = self._validate_and_fix_types(
                    abstract_operator.input['fields'],
                    f"operator_{operator_index}_input_regenerated"
                )

            if isinstance(abstract_operator.output, dict):
                abstract_operator.output = self._validate_and_fix_types(
                    abstract_operator.output,
                    f"operator_{operator_index}_output_regenerated"
                )

            abstract_operator.properties = {}
            for key, value in llm_response.items():
                if key not in ['input', 'output']:
                    abstract_operator.properties[key] = value

            # Update type system
            self._apply_operator_to_type_system(type_system, abstract_operator)

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
                    type_system=type_system,
                    collect_messages=collect_messages,
                    operator_index=operator_index,
                    attempt=attempt,
                    total_operators=total_operators
                )

        if collect_messages:
            return abstract_operator, messages
        return abstract_operator

    def _validate_and_fix_types(self, schema: Dict[str, Any], schema_name: str = "schema") -> Dict[str, Any]:
        """
        Validate and potentially fix type strings in a schema.

        Args:
            schema: Schema dictionary with type strings
            schema_name: Name for logging purposes

        Returns:
            Validated (and potentially fixed) schema
        """
        if not isinstance(schema, dict):
            return schema

        fixed_schema = {}
        warnings = []

        for field, type_str in schema.items():
            if not isinstance(type_str, str):
                fixed_schema[field] = type_str
                continue

            is_valid, error = validate_type_syntax(type_str)

            if is_valid:
                fixed_schema[field] = type_str
            else:
                # Type is invalid, try to fix common issues
                original_type = type_str

                # Fix: bare "List" -> "List[String]" (default to String)
                if type_str == "List":
                    type_str = "List[String]"
                    warnings.append(f"Fixed bare 'List' to 'List[String]' for field '{field}'")

                # Fix: bare "Dict" is actually OK
                elif type_str == "Dict":
                    pass  # This is valid

                is_valid_after_fix, _ = validate_type_syntax(type_str)

                if is_valid_after_fix:
                    fixed_schema[field] = type_str
                    if type_str != original_type:
                        if self.config.verbose:
                            self.logger.warning(f"Auto-fixed type for {schema_name}.{field}: {original_type} -> {type_str}")
                else:
                    # Can't fix, keep original but warn
                    fixed_schema[field] = original_type
                    warnings.append(f"Invalid type for field '{field}': {original_type} (Error: {error})")
                    if self.config.verbose:
                        self.logger.warning(f"Invalid type in {schema_name}.{field}: {original_type} - {error}")

        return fixed_schema

    def _validate_and_fix_prompt(self, prompt: str, input_fields: Set[str], operator_name: str = "") -> tuple:
        """
        Validate prompt and attempt to auto-fix bare field references.

        Args:
            prompt: The prompt string to validate
            input_fields: Set of available input field names
            operator_name: Name of the operator (for logging)

        Returns:
            Tuple of (fixed_prompt, was_modified)
        """
        import re

        modified = False
        fixed_prompt = prompt

        # Check for bare field references and fix them
        for field in input_fields:
            # Pattern 1: backtick references like `fieldname`
            bare_pattern = f'`{field}`'
            jinja_replacement = f'{{{{ input.{field} }}}}'

            if bare_pattern in fixed_prompt:
                fixed_prompt = fixed_prompt.replace(bare_pattern, jinja_replacement)
                modified = True
                if self.config.verbose:
                    self.logger.warning(
                        f"{operator_name}: Auto-fixed bare field reference: `{field}` -> {{{{ input.{field} }}}}"
                    )

            # Pattern 2: "field 'fieldname'" or "field \"fieldname\""
            for quote in ["'", '"']:
                pattern = f"field {quote}{field}{quote}"
                if pattern in fixed_prompt.lower():
                    # Find the exact match with correct case
                    match = re.search(re.escape(pattern), fixed_prompt, re.IGNORECASE)
                    if match:
                        fixed_prompt = fixed_prompt[:match.start()] + jinja_replacement + fixed_prompt[match.end():]
                        modified = True
                        if self.config.verbose:
                            self.logger.warning(
                                f"{operator_name}: Auto-fixed bare field reference: {match.group()} -> {{{{ input.{field} }}}}"
                            )

        return fixed_prompt, modified

    def _apply_operator_to_type_system(
        self,
        type_system: AbstractTypeSystem,
        operator: Operator
    ) -> None:
        """
        Apply operator transformation to the type system.

        Args:
            type_system: Type system to update
            operator: Operator object
        """
        op_type = operator.type
        output_schema = operator.output

        if op_type == 'Map':
            type_system.apply_map_operator(output_schema)
        elif op_type == 'Filter':
            type_system.apply_filter_operator()
        elif op_type == 'Reduce':
            reduce_key = operator.properties.get('reduce_key')
            type_system.apply_reduce_operator(output_schema, reduce_key)
        elif op_type == 'Resolve':
            type_system.apply_resolve_operator(output_schema)
        elif op_type == 'Extract':
            type_system.apply_extract_operator(output_schema)
        else:
            # Generic operator
            type_system.apply_generic_operator(output_schema)

    def _build_abstract_pipeline(
        self,
        query: str,
        dataset_samples: Dict[str, Any],
        attempt: int = 0
    ) -> Pipeline:
        """
        Build complete abstract pipeline.

        Args:
            query: User query
            dataset_samples: Sample data
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

        # Initialize type system from dataset (assume single json dataset)
        samples_list = dataset_samples[:10] if dataset_samples else []
        dataset_schema = infer_schema_from_samples(samples_list if samples_list else [{}])
        type_system = AbstractTypeSystem(dataset_schema)

        filled_operators = []
        for i, op in enumerate(operators):
            filled_op = self._generate_operator_details(
                operator=op,
                query=query,
                dataset_samples=dataset_samples,
                previous_operators=filled_operators,
                type_system=type_system,
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

    def _convert_and_execute(
        self,
        abstract_pipeline: Pipeline,
        query: str,
        dataset_path: str
    ) -> Tuple[bool, Optional[str], Optional[Dict]]:
        """
        Convert abstract pipeline to DocETL and execute.

        Args:
            abstract_pipeline: Abstract Pipeline object
            query: User query
            dataset_path: Path to dataset

        Returns:
            (success, output, yaml_config)
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

        abstract_pipeline_path = self.file_manager.save_json(
            data=pipeline_dict,
            query=query,
            file_type='pipeline',
            subdir_key='abstract_pipeline'
        )

        # Validate abstract pipeline before conversion to base system
        validate_pipeline_static(
            system_name="abstract",
            pipeline_path=abstract_pipeline_path,
            verbose=self.config.verbose,
            logger=self.logger,
            dataset_schema=abstract_pipeline.dataset_schema
        )

        # Convert to DocETL YAML using existing converter
        try:
            operators = abstract_pipeline.to_operators()
            docetl_operators = [abstract_to_docetl(op) for op in operators]

            # Build DocETL config with output path
            output_path = self.file_manager.get_output_path(query, file_type='output')

            docetl_config = {
                'datasets': {
                    'input': {
                        'type': 'file',
                        'path': dataset_path
                    }
                },
                'operations': docetl_operators,
                'pipeline': {
                    'steps': [{
                        'name': 'generated_pipeline',
                        'input': 'input',
                        'operations': [op['name'] for op in docetl_operators]
                    }],
                    'output': {
                        'type': 'file',
                        'path': output_path
                    }
                },
                'default_model': abstract_pipeline.properties.get('default_model', 'gpt-4o-mini')
            }

            self._log("Successfully converted abstract pipeline to DocETL")

        except Exception as e:
            self._log(f"Failed to convert abstract pipeline to DocETL: {e}", level='error', force=True)
            self._log(traceback.format_exc(), level='error', force=True)
            return False, None, None

        try:
            # Save DocETL config to YAML file for execution
            yaml_path = self.file_manager.save_yaml(
                data=docetl_config,
                query=query,
                file_type='pipeline_temp',
                subdir_key='pipeline_output'
            )

            validate_pipeline_static(
                system_name=self.base_system.value,
                pipeline_path=yaml_path,
                verbose=self.config.verbose,
                logger=self.logger
            )

            if not self.ui.confirm_pipeline_execution(yaml_path, query, attempt):
                self._log("Pipeline execution skipped by user", force=True)
                return False, None, docetl_config

            success, error_msg = execute_single_pipeline(yaml_path)

            if not success:
                self._log(f"Pipeline execution failed: {error_msg}", level='error', force=True)
                return False, None, docetl_config

            output_path = find_output_path(yaml_path)
            if output_path and os.path.exists(output_path):
                with open(output_path, 'r', encoding='utf-8') as f:
                    output = json.load(f)
            else:
                self._log("Could not find output file, using empty output", level='warning')
                output = []

            self._log(f"Pipeline execution completed with {len(output) if isinstance(output, list) else 'N/A'} results")

            return True, output, docetl_config

        except Exception as e:
            self._log(f"Pipeline execution failed: {e}", level='error', force=True)
            self._log(traceback.format_exc(), level='error', force=True)
            return False, None, docetl_config

    def process(self, query: str, context: Optional[Dict[str, Any]] = None) -> BaselineResult:
        """Process a single query with optional context using abstract layer pipeline generation."""
        from .base import BaselineResult as InterfaceBaselineResult

        start_time = time.time()
        self.total_queries += 1

        try:
            # Prepare dataset file from context (assume single json dataset)
            dataset_paths = self.data_processor.prepare_dataset_files(context)

            if not dataset_paths:
                raise ValueError("No dataset path available")

            dataset_path = dataset_paths[0]

            dataset_samples = load_sample_data([dataset_path])

            # Attempt pipeline generation with retries
            success = False
            result_output = None

            for attempt in range(self.max_attempts):
                self.total_generation_attempts += 1

                try:
                    self._log(f"Attempt {attempt + 1}/{self.max_attempts} for query: {query[:100]}...")

                    abstract_pipeline = self._build_abstract_pipeline(query, dataset_samples, attempt)

                    success, result_output, _ = self._convert_and_execute(
                        abstract_pipeline, query, dataset_path
                    )

                    if success and result_output:
                        self.successful_pipelines += 1
                        break

                except Exception as e:
                    self._log(f"Attempt {attempt + 1} failed: {e}", level='error', force=True)
                    self._log(traceback.format_exc(), level='error', force=True)

                    if attempt == self.max_attempts - 1:
                        # Final attempt failed
                        self.failed_pipelines.append(FailedPipeline(
                            pipeline_yaml=f"# Query: {query}\n# Failed after {self.max_attempts} attempts",
                            error_type='execution',
                            error_message=str(e)
                        ))

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
