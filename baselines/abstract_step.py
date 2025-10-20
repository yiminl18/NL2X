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
import yaml
import traceback
from datetime import datetime
import logging

from .base import BaselineInterface, BaselineResult
from . import register_baseline
from .docetl_utils.data_utils import DocETLDataProcessor
from .abstract_step_utils.ui import AbstractStepUserInterface
from .docetl_utils.log_utils import get_filename_base
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

        if self.config.verbose:
            self.logger.info(f"Initialized Abstract Step baseline with config: {self.config}")
            self.logger.info(f"Base system: {self.base_system.value}")
            self.logger.info(f"Supported operators: {get_supported_operators(self.base_system)}")
            self.logger.info(f"Pipeline output directory: {self.pipeline_output_dir}")
            self.logger.info(f"Abstract pipeline directory: {self.abstract_pipeline_dir}")

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
        # Infer schema from samples (assume single json dataset)
        samples_list = dataset_samples[:10] if dataset_samples else []
        schema = infer_schema_from_samples(samples_list if samples_list else [{}])

        # Format dataset samples for prompt
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
            # Call LLM (with bypass_cache if regenerating)
            bypass_cache = (user_choice == 'regenerate')
            response = llm_call(messages, schema=parameters, system_prompt=system_prompt, bypass_cache=bypass_cache)

            result = json.loads(response)
            operators = result.get("operators", [])

            # Filter out unsupported operators (safety check)
            operators = filter_unsupported_operators(operators, self.base_system)

            # Collect response for debugging
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
        """
        Generate JSON schema for abstract operator type.

        Args:
            op_type: Operator type (Map, Filter, etc.)

        Returns:
            JSON schema dictionary
        """
        if op_type == 'Map':
            return {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Jinja2 template for the transformation. Use {{ input.field_name }} to reference fields."
                    },
                    "input": {
                        "type": "object",
                        "properties": {
                            "fields": {
                                "type": "object",
                                "description": "Input field types, e.g., {\"text\": \"String\", \"id\": \"Integer\"}"
                            }
                        },
                        "required": ["fields"]
                    },
                    "output": {
                        "type": "object",
                        "description": "Output schema with ALL fields (preserved + new). IMPORTANT: For List types, MUST specify element type as 'List[ElementType]' (e.g., 'List[String]', 'List[Integer]', 'List[Dict[{name: String}]]'). For Dict types, can use 'Dict' or 'Dict[{field1: Type1, field2: Type2}]'. Examples: {\"text\": \"String\", \"names\": \"List[String]\", \"data\": \"Dict[{count: Integer}]\"}"
                    },
                    "properties": {
                        "type": "object",
                        "description": "Additional configuration (optional)"
                    }
                },
                "required": ["prompt", "input", "output"]
            }

        elif op_type == 'Filter':
            return {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Jinja2 template for the filter condition. Use {{ input.field_name }} to reference fields."
                    },
                    "input": {
                        "type": "object",
                        "properties": {
                            "fields": {
                                "type": "object",
                                "description": "Input field types used in the condition"
                            }
                        },
                        "required": ["fields"]
                    },
                    "output": {
                        "type": "object",
                        "description": "Output schema (same as all available fields). Use complete type specifications: 'List[String]' not 'List', 'Dict[{field: Type}]' for structured dicts."
                    },
                    "properties": {
                        "type": "object",
                        "description": "Additional configuration (optional)"
                    }
                },
                "required": ["prompt", "input", "output"]
            }

        elif op_type == 'Reduce':
            return {
                "type": "object",
                "properties": {
                    "reduce_key": {
                        "type": "string",
                        "description": "Field name to group by (must exist in available fields)"
                    },
                    "prompt": {
                        "type": "string",
                        "description": "Jinja2 template for aggregation. Use {{ inputs }} to reference grouped records."
                    },
                    "input": {
                        "type": "object",
                        "properties": {
                            "fields": {
                                "type": "object",
                                "description": "Input field types including reduce_key"
                            }
                        },
                        "required": ["fields"]
                    },
                    "output": {
                        "type": "object",
                        "description": "Output schema including reduce_key and aggregated fields. Use complete types: 'List[String]', 'Dict[{avg: Float, max: Float}]' etc."
                    },
                    "properties": {
                        "type": "object",
                        "description": "Additional configuration (optional)"
                    }
                },
                "required": ["reduce_key", "prompt", "input", "output"]
            }

        elif op_type == 'Resolve':
            return {
                "type": "object",
                "properties": {
                    "comparison_prompt": {"type": "string"},
                    "resolution_prompt": {"type": "string"},
                    "input": {
                        "type": "object",
                        "properties": {
                            "fields": {"type": "object"}
                        },
                        "required": ["fields"]
                    },
                    "output": {"type": "object"},
                    "properties": {"type": "object"}
                },
                "required": ["comparison_prompt", "resolution_prompt", "input", "output"]
            }

        elif op_type == 'Extract':
            return {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Jinja2 template describing what to extract. Use {{ input.field }} for document fields."
                    },
                    "document_keys": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of field names containing the documents to extract from"
                    },
                    "input": {
                        "type": "object",
                        "properties": {
                            "fields": {
                                "type": "object",
                                "description": "Input field types including document fields"
                            }
                        },
                        "required": ["fields"]
                    },
                    "output": {
                        "type": "object",
                        "description": "Output schema with extracted fields. CRITICAL: Specify complete types. For lists extracted from documents, use 'List[String]' not just 'List'. Examples: {\"document_name\": \"List[String]\", \"parties\": \"List[String]\", \"agreement_date\": \"List[String]\"}"
                    },
                    "properties": {
                        "type": "object",
                        "description": "Additional configuration (optional)"
                    }
                },
                "required": ["prompt", "document_keys", "input", "output"]
            }

        # For other operators, use a generic schema
        return {
            "type": "object",
            "properties": {
                "prompt": {"type": "string"},
                "input": {
                    "type": "object",
                    "properties": {
                        "fields": {"type": "object"}
                    }
                },
                "output": {"type": "object"},
                "properties": {"type": "object"}
            },
            "required": []
        }

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

        # Format available fields
        current_schema = type_system.get_current_schema()
        available_fields_str = format_fields_for_prompt(current_schema)

        # Format dataset samples
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
            prompt_file=None
        )

        if user_choice == 'abort':
            raise ValueError(f"User aborted operator generation at operator {operator_index + 1}")

        # Get operator-specific schema
        parameters = self._get_abstract_operator_schema(op_type)

        system_prompt = f"You are an AI assistant that generates abstract layer {op_type} operator configurations. Always respond with valid JSON matching the required schema."

        messages = [{"role": "user", "content": prompt}]

        try:
            # Call LLM (with bypass_cache if regenerating)
            bypass_cache = (user_choice == 'regenerate')
            response = llm_call(messages, schema=parameters, system_prompt=system_prompt, bypass_cache=bypass_cache)

            llm_response = json.loads(response)

            # Create Operator object
            abstract_operator = Operator()
            abstract_operator.name = f"op_{operator_index}_{op_type.lower()}"
            abstract_operator.type = op_type
            abstract_operator.source = {
                "system": "llm_generated",
                "name": abstract_operator.name
            }

            # Extract input and output schemas
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

            # Store all other fields as properties
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

            # Clean up output schema: remove duplicate input fields
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

            # Call LLM again with bypass_cache to force regeneration
            response = llm_call(messages, schema=parameters, system_prompt=system_prompt, bypass_cache=True)

            # Parse and create new operator
            llm_response = json.loads(response)

            # Create new Operator object
            abstract_operator = Operator()
            abstract_operator.name = f"op_{operator_index}_{op_type.lower()}"
            abstract_operator.type = op_type
            abstract_operator.source = {
                "system": "llm_generated",
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
                # Type is not a string, keep as-is
                fixed_schema[field] = type_str
                continue

            # Validate type syntax
            is_valid, error = validate_type_syntax(type_str)

            if is_valid:
                # Type is valid, keep it
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

                # Re-validate after fix attempt
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
        # Step 1: Select operators
        if self.config.verbose:
            self.logger.info("Step 1: Selecting operators...")

        operators = self._select_operators(query, dataset_samples, attempt)

        if not operators:
            raise ValueError("No operators selected")

        # Step 2: Generate operator details
        if self.config.verbose:
            self.logger.info(f"Step 2: Generating details for {len(operators)} operators...")

        # Initialize type system from dataset (assume single json dataset)
        samples_list = dataset_samples[:10] if dataset_samples else []
        initial_schema = infer_schema_from_samples(samples_list if samples_list else [{}])
        type_system = AbstractTypeSystem(initial_schema)

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
        dataset_schema = {'fields': initial_schema}

        abstract_pipeline = Pipeline.from_operators(
            operators=filled_operators,
            name=pipeline_name,
            input_path=None,  # Set later when executing
            output_path=None,  # Set later when executing
            properties={
                "query": query,
                "base_system": self.base_system.value,
                "default_model": "gpt-4o-mini",
                "created_at": datetime.now().isoformat()
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
        # Save abstract pipeline as JSON
        abstract_pipeline_path = os.path.join(
            self.abstract_pipeline_dir,
            f"{get_filename_base(self.data_processor, query, 'pipeline')}.json"
        )

        # Convert Pipeline to dict for JSON serialization
        pipeline_dict = {
            "name": abstract_pipeline.name,
            "input_path": abstract_pipeline.input_path,
            "output_path": abstract_pipeline.output_path,
            "dataset_schema": abstract_pipeline.dataset_schema,
            "properties": abstract_pipeline.properties,
            "operators": [operator_to_dict(op) for op in abstract_pipeline.to_operators()]
        }

        with open(abstract_pipeline_path, 'w', encoding='utf-8') as f:
            json.dump(pipeline_dict, f, indent=2, ensure_ascii=False)

        if self.config.verbose:
            self.logger.info(f"Saved abstract pipeline to {abstract_pipeline_path}")

        # Convert to DocETL YAML using existing converter
        try:
            operators = abstract_pipeline.to_operators()
            docetl_operators = [abstract_to_docetl(op) for op in operators]

            # Build DocETL config with output path
            output_filename = f"{get_filename_base(self.data_processor, query, 'output')}.json"
            output_path = os.path.join(self.pipeline_output_dir, output_filename)

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
                        'name': 'main',
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

            if self.config.verbose:
                self.logger.info("Successfully converted abstract pipeline to DocETL")

        except Exception as e:
            if self.config.verbose:
                self.logger.error(f"Failed to convert abstract pipeline to DocETL: {e}")
                self.logger.error(traceback.format_exc())
            return False, None, None

        # Execute pipeline
        try:
            # Save DocETL config to temporary YAML file for execution
            # execute_single_pipeline expects a file path, not a config dict
            yaml_path = os.path.join(
                self.pipeline_output_dir,
                f"{get_filename_base(self.data_processor, query, 'pipeline_temp')}.yaml"
            )
            with open(yaml_path, 'w') as f:
                yaml.dump(docetl_config, f, default_flow_style=False)

            # Validate the generated DocETL YAML using the static checker
            if self.config.verbose:
                self.logger.info("Validating generated DocETL YAML with static checker...")

            try:
                # Import the DocETL static checker
                import sys
                import os as os_mod

                # Add the path to sys.path if needed
                grader_dir = os_mod.path.join(os_mod.path.dirname(__file__), 'grader', 'docetl')
                if grader_dir not in sys.path:
                    sys.path.insert(0, grader_dir)

                from baselines.grader.docetl.static_checker import check_pipeline_file

                # Run static checker
                validation_result = check_pipeline_file(yaml_path)

                if self.config.verbose:
                    if validation_result.get('score') == 1:
                        self.logger.info("✓ Static validation passed")
                    else:
                        self.logger.warning("⚠ Static validation found issues:")
                        for error in validation_result.get('errors', []):
                            self.logger.warning(f"  - {error.get('message', str(error))}")

                        if validation_result.get('warnings'):
                            self.logger.info("Warnings:")
                            for warning in validation_result.get('warnings', []):
                                self.logger.info(f"  - {warning.get('message', str(warning))}")

                # Continue execution even if validation fails (may be overly strict)
                # but log the issues for debugging

            except Exception as validation_error:
                if self.config.verbose:
                    self.logger.warning(f"Static validation skipped (error: {validation_error})")

            # Confirm pipeline execution in debug/confirm mode
            if not self.ui.confirm_pipeline_execution(yaml_path, query, attempt):
                # User chose not to execute, treat as skipped (not a failure)
                if self.config.verbose:
                    self.logger.info("Pipeline execution skipped by user")
                return False, None, docetl_config

            # Execute the pipeline using the YAML file
            success, error_msg = execute_single_pipeline(yaml_path)

            if not success:
                if self.config.verbose:
                    self.logger.error(f"Pipeline execution failed: {error_msg}")
                return False, None, docetl_config

            # Load the output data
            output_path = find_output_path(yaml_path)
            if output_path and os.path.exists(output_path):
                with open(output_path, 'r', encoding='utf-8') as f:
                    output = json.load(f)
            else:
                if self.config.verbose:
                    self.logger.warning("Could not find output file, using empty output")
                output = []

            if self.config.verbose:
                self.logger.info(f"Pipeline execution completed with {len(output) if isinstance(output, list) else 'N/A'} results")

            return True, output, docetl_config

        except Exception as e:
            if self.config.verbose:
                self.logger.error(f"Pipeline execution failed: {e}")
                self.logger.error(traceback.format_exc())
            return False, None, docetl_config

    # Required abstract methods from BaselineInterface

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

            # Load dataset samples
            dataset_samples = load_sample_data([dataset_path])

            # Attempt pipeline generation with retries
            success = False
            result_output = None

            for attempt in range(self.max_attempts):
                self.total_generation_attempts += 1

                try:
                    if self.config.verbose:
                        self.logger.info(f"Attempt {attempt + 1}/{self.max_attempts} for query: {query[:100]}...")

                    # Build abstract pipeline
                    abstract_pipeline = self._build_abstract_pipeline(query, dataset_samples, attempt)

                    # Convert and execute
                    success, result_output, _ = self._convert_and_execute(
                        abstract_pipeline, query, dataset_path
                    )

                    if success and result_output:
                        self.successful_pipelines += 1
                        break

                except Exception as e:
                    if self.config.verbose:
                        self.logger.error(f"Attempt {attempt + 1} failed: {e}")
                        self.logger.error(traceback.format_exc())

                    if attempt == self.max_attempts - 1:
                        # Final attempt failed
                        # FailedPipeline expects (pipeline_yaml, error_type, error_message)
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
                    response={"answer": result_output, "pipeline_generated": True, "method": "abstract_step"},
                    metadata={
                        "baseline": "abstract_step",
                        "base_system": self.base_system.value,
                        "context_provided": context is not None,
                        "num_datasets": 1,  # Single json dataset
                        "generation_attempts": self.total_generation_attempts,
                        "success": True
                    },
                    execution_time=execution_time
                )
            else:
                error_msg = "Pipeline generation failed"
                return InterfaceBaselineResult(
                    query=query,
                    response={"answer": "", "error": error_msg},
                    metadata={
                        "baseline": "abstract_step",
                        "base_system": self.base_system.value,
                        "context_provided": context is not None,
                        "num_datasets": 1,  # Single json dataset
                        "generation_attempts": self.total_generation_attempts,
                        "success": False
                    },
                    execution_time=execution_time,
                    error=error_msg
                )

        except Exception as e:
            self.logger.error(f"Error processing query with Abstract Step: {e}")
            self.logger.error(f"Full traceback:\n{traceback.format_exc()}")

            execution_time = time.time() - start_time
            self.total_time += execution_time

            return InterfaceBaselineResult(
                query=query,
                response={"answer": "", "error": str(e)},
                metadata={
                    "baseline": "abstract_step",
                    "base_system": self.base_system.value,
                    "context_provided": context is not None,
                    "error": True
                },
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
            "total_generation_attempts": self.total_generation_attempts,
            "successful_pipelines": self.successful_pipelines,
            "failed_pipelines_count": len(self.failed_pipelines),
            "success_rate": self.successful_pipelines / max(1, self.total_queries),
            "average_attempts_per_query": self.total_generation_attempts / max(1, self.total_queries),
            "baseline_type": "abstract_step",
            "base_system": self.base_system.value
        }
