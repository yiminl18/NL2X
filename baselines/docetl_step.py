import time
from typing import Any, Dict, List, Optional, Tuple
import os
import json
import yaml
import traceback
from datetime import datetime
import logging

from .base import BaselineInterface, BaselineResult
from . import register_baseline
from .docetl_data_utils import DocETLDataProcessor
from .docetl_ui import DocETLUserInterface
from .docetl_cache import LLMCache
from .docetl_log_utils import save_prompt, save_messages, save_validation, save_step_output, get_filename_base
from .docetl_utils.llm2pipeline_docetl import (
    load_sample_data,
    execute_single_pipeline,
    validate_pipeline_output,
    FailedPipeline,
)
from .docetl_litellm_client import llm_call
from .docetl_type_utils import (
    TypeSystem,
    extract_type_system,
    apply_operator_transformation,
    format_available_fields_with_types,
    validate_operator_inputs,
)
from .docetl_utils.prompt_step import (
    OPERATOR_SELECTION_PROMPT,
    OPERATOR_DEFINITIONS,
    get_op_prompt,
)

@register_baseline("docetl_step")
class DocETLStepBaseline(BaselineInterface):
    """
    1. Select operators needed
    2. Create draft framework
    3. Generate details for each operator
    4. Fill operator drafts
    5. Connect and finalize pipeline
    """

    def _setup(self):
        self.total_queries = 0
        self.total_time = 0
        self.total_generation_attempts = 0
        self.successful_pipelines = 0
        self.failed_pipelines = []

        self.max_attempts = self.config.max_attempts
        self.validate_answer = self.config.validate_answer

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
            'pipeline_output_dir': os.path.join(base_dir, "generated_pipelines", "docetl_step"),
            'prompts_output_dir': os.path.join(base_dir, "generated_prompts", "docetl_step"),
            'validations_output_dir': os.path.join(base_dir, "validations", "docetl_step"),
            'messages_output_dir': os.path.join(base_dir, "messages", "docetl_step"),
            'converted_data_dir': os.path.join(base_dir, "converted_data", "docetl_step"),
            'steps_output_dir': os.path.join(base_dir, "pipeline_steps", "docetl_step")
        }

        for attr_name, dir_path in self.output_dirs.items():
            setattr(self, attr_name, dir_path)
            os.makedirs(dir_path, exist_ok=True)

        self.data_processor = DocETLDataProcessor(
            converted_data_dir=self.converted_data_dir,
            verbose=self.config.verbose,
            logger=self.logger
        )

        # Initialize user interface for confirmations
        self.ui = DocETLUserInterface(self.config)

        # Initialize LLM cache
        cache_dir = os.path.join(base_dir, "llm_cache", "docetl_step")
        self.llm_cache = LLMCache(cache_dir)

        if self.config.verbose:
            self.logger.info(f"Initialized DocETL Step-by-Step baseline with config: {self.config}")
            self.logger.info(f"Pipeline output directory: {self.pipeline_output_dir}")
            self.logger.info(f"Cache directory: {cache_dir}")





    def _select_operators(self, query: str, dataset_samples: Dict[str, Any], attempt: int = 0, collect_messages: bool = False):
        """Step 1: Select operators needed for the pipeline."""
        # Convert OPERATOR_DEFINITIONS dict to string for operator selection
        operator_definitions_str = ""
        for op_type, definition in OPERATOR_DEFINITIONS.items():
            # Include the full operator definition for better context
            operator_definitions_str += f"\n{definition.strip()}\n"

        # Extract type system for better understanding of dataset structure
        type_system = extract_type_system(dataset_samples)
        dataset_structure = type_system.format_for_prompt()

        prompt = OPERATOR_SELECTION_PROMPT.substitute(
            query=query,
            dataset_samples=f"{json.dumps(dataset_samples, indent=2)}\n\nDataset Structure:\n{dataset_structure}",
            operator_definitions=operator_definitions_str
        )

        # Save prompt for debugging
        filename_base = get_filename_base(self.data_processor, query, f'step1_attempt{attempt}')
        save_prompt(self.prompts_output_dir, filename_base, prompt, query, 0)

        # Confirm before calling LLM in confirm/debug mode
        user_choice = self.ui.confirm_step_before_llm("Operator Selection", "1")
        if user_choice == 'abort':
            raise ValueError("User aborted pipeline generation at Step 1")

        # Define schema for structured operator selection
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
                                "enum": ["map", "filter", "reduce", "resolve", "rank", "extract",
                                        "cluster", "split", "gather", "unnest", "sample", "topk",
                                        "code_filter"]
                            },
                            "purpose": {"type": "string"}
                        },
                        "required": ["type", "purpose"]
                    }
                }
            },
            "required": ["operators"]
        }

        system_prompt = "You are an AI assistant that selects appropriate DocETL operators for data processing pipelines. Always respond with valid JSON matching the required schema."

        messages = [{"role": "user", "content": prompt}]

        try:
            # Check cache first (with regeneration flag if requested)
            force_regenerate = (user_choice == 'regenerate')
            response = self.llm_cache.get(prompt, force_regenerate=force_regenerate)
            if response is None:
                # No cache or regeneration requested, call LLM
                response = llm_call(messages, schema=parameters, system_prompt=system_prompt)
                # Save to cache (will overwrite if regenerating)
                self.llm_cache.set(prompt, response)
            else:
                if self.config.verbose:
                    self.logger.info("Using cached response for Step 1: Operator Selection")

            result = json.loads(response)
            operators = result.get("operators", [])

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
        if not self.ui.confirm_step_execution("Step 1: Operator Selection", operators, query, attempt):
            raise ValueError("User aborted pipeline generation at operator selection step")

        if collect_messages:
            return operators, messages
        return operators

    def _create_operator_framework(self, operators: List[Dict[str, str]], query: str, attempt: int = 0) -> List[Dict[str, Any]]:
        """Step 2: Create draft framework for each operator."""
        frameworks = []

        for i, op in enumerate(operators):
            framework = {
                'name': f"op_{i}_{op['type']}",
                'type': op['type'],
                'purpose': op['purpose']
            }

            # Some ops might need default settings.
            # if op['type'] == 'resolve':
            #     framework['optimize'] = True

            if op['type'] == 'rank':
                framework['direction'] = "desc"

            elif op['type'] == 'split':
                framework['method'] = "token_count"
                framework['method_kwargs'] = {}

            elif op['type'] == 'topk':
                framework['method'] = "embedding"
                framework['k'] = 5

            elif op['type'] == 'sample':
                framework['method'] = "uniform"
                framework['samples'] = 0.1
                framework['random_state'] = 42

            frameworks.append(framework)

        if self.config.verbose:
            self.logger.info(f"Created framework for {len(frameworks)} operators")

        # Confirm this step in confirm/debug mode
        if not self.ui.confirm_step_execution("Step 2: Operator Framework", frameworks, query, attempt):
            raise ValueError("User aborted pipeline generation at framework creation step")

        return frameworks

    def _extract_dataset_fields(self, dataset_samples: Dict[str, Any]) -> List[str]:
        """Extract field names from dataset samples."""
        type_system = extract_type_system(dataset_samples)
        return type_system.get_all_fields()

    def _track_available_fields(self, dataset_samples: Dict[str, Any],
                               filled_operators: List[Dict[str, Any]]) -> List[str]:
        """Track fields available at current pipeline stage."""
        # Initialize type system from dataset
        type_system = extract_type_system(dataset_samples)

        # Apply transformations from each operator
        for operator in filled_operators:
            type_system = apply_operator_transformation(type_system, operator)

        return sorted(type_system.get_all_fields())

    def _get_type_system_at_stage(self, dataset_samples: Dict[str, Any],
                                 filled_operators: List[Dict[str, Any]]) -> TypeSystem:
        """Get type system at current pipeline stage."""
        # Initialize type system from dataset
        type_system = extract_type_system(dataset_samples)

        # Apply transformations from each operator
        for operator in filled_operators:
            type_system = apply_operator_transformation(type_system, operator)

        return type_system

    def _validate_field_references(self, operator: Dict[str, Any], type_system: TypeSystem) -> tuple:
        """Validate operator field references."""
        # Use the new validate_operator_inputs function from type_utils
        return validate_operator_inputs(type_system, operator)

    def _get_operator_schema(self, op_type: str) -> Dict[str, Any]:
        """Generate JSON schema for operator type."""
        if op_type in ['map', 'filter']:
            return {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string"},
                    "output_schema": {"type": "object"}
                },
                "required": ["prompt", "output_schema"]
            }

        elif op_type == 'extract':
            return {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string"},
                    "document_keys": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                },
                "required": ["prompt", "document_keys"]
            }

        elif op_type == 'reduce':
            return {
                "type": "object",
                "properties": {
                    "reduce_key": {"type": "string"},
                    "prompt": {"type": "string"},
                    "output_schema": {"type": "object"}
                },
                "required": ["reduce_key", "prompt", "output_schema"]
            }

        elif op_type == 'resolve':
            return {
                "type": "object",
                "properties": {
                    "comparison_prompt": {"type": "string"},
                    "resolution_prompt": {"type": "string"},
                    "output_schema": {"type": "object"}
                },
                "required": ["comparison_prompt", "resolution_prompt", "output_schema"]
            }

        elif op_type == 'rank':
            return {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string"},
                    "input_keys": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "direction": {"type": "string", "enum": ["asc", "desc"]}
                },
                "required": ["prompt", "input_keys"]
            }

        elif op_type == 'split':
            return {
                "type": "object",
                "properties": {
                    "split_key": {"type": "string"},
                    "method": {"type": "string"},
                    "method_kwargs": {"type": "object"}
                },
                "required": ["split_key", "method"]
            }

        elif op_type == 'gather':
            return {
                "type": "object",
                "properties": {
                    "content_key": {"type": "string"},
                    "doc_id_key": {"type": "string"},
                    "order_key": {"type": "string"}
                },
                "required": ["content_key", "doc_id_key"]
            }

        elif op_type == 'unnest':
            return {
                "type": "object",
                "properties": {
                    "unnest_key": {"type": "string"}
                },
                "required": ["unnest_key"]
            }

        elif op_type in ['code_map', 'code_filter']:
            return {
                "type": "object",
                "properties": {
                    "code": {"type": "string"}
                },
                "required": ["code"]
            }

        elif op_type == 'cluster':
            return {
                "type": "object",
                "properties": {
                    "embedding_keys": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "output_key": {"type": "string"}
                },
                "required": ["embedding_keys", "output_key"]
            }

        elif op_type == 'topk':
            return {
                "type": "object",
                "properties": {
                    "method": {"type": "string"},
                    "k": {"type": "integer"},
                    "keys": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "query": {"type": "string"}
                },
                "required": ["k", "keys"]
            }

        elif op_type == 'sample':
            return {
                "type": "object",
                "properties": {
                    "method": {"type": "string", "enum": ["uniform", "stratified"]},
                    "samples": {"type": ["number", "integer"]},
                    "stratify_key": {"type": "string"},
                },
                "required": ["method", "samples"]
            }

        # Default empty schema for unknown operator types
        return {
            "type": "object",
            "properties": {},
            "required": []
        }

    def _generate_operator_details(self, operator: Dict[str, Any], query: str,
                                  dataset_samples: Dict[str, Any],
                                  previous_operators: List[Dict[str, Any]],
                                  available_fields: List[str],
                                  type_system: TypeSystem = None,
                                  collect_messages: bool = False,
                                  operator_index: int = 0,
                                  attempt: int = 0,
                                  total_operators: int = 1):
        """Step 3: Generate details for a single operator."""
        # Get operator-specific prompt template
        op_type = operator['type']

        # Select the appropriate prompt template
        prompt_template = get_op_prompt(op_type)

        fields_with_types = format_available_fields_with_types(type_system)

        prompt = prompt_template.substitute(
            operator_type=op_type,
            operator_purpose=operator['purpose'],
            query=query,
            dataset_samples=json.dumps(dataset_samples, indent=2),
            previous_operators=json.dumps(previous_operators, indent=2) if previous_operators else "None",
            available_fields=fields_with_types,
            operator_framework=json.dumps(operator, indent=2)
        )

        # Save prompt for debugging
        filename_base = get_filename_base(self.data_processor, query, f'step3_op{operator_index}_{op_type}_attempt{attempt}')
        prompt_file_path = save_prompt(self.prompts_output_dir, filename_base, prompt, query, 0)

        # Confirm this operator generation with user in Step 4
        user_choice = self.ui.confirm_operator_before_llm(
            operator_index=operator_index,
            total_operators=total_operators,
            operator_type=op_type,
            operator_purpose=operator.get('purpose', ''),
            prompt=prompt,
            prompt_file=prompt_file_path
        )

        if user_choice == 'abort':
            raise ValueError(f"User aborted operator generation at operator {operator_index + 1}")

        # Get operator-specific schema
        parameters = self._get_operator_schema(op_type)

        system_prompt = f"You are an AI assistant that generates detailed configurations for DocETL {op_type} operators. Always respond with valid JSON matching the required schema. Use the available fields and previous operators context to create appropriate configurations."

        messages = [{"role": "user", "content": prompt}]

        try:
            # Check cache first (with regeneration flag if requested)
            force_regenerate = (user_choice == 'regenerate')
            response = self.llm_cache.get(prompt, force_regenerate=force_regenerate)
            if response is None:
                # No cache or regeneration requested, call LLM
                response = llm_call(messages, schema=parameters, system_prompt=system_prompt)
                # Save to cache (will overwrite if regenerating)
                self.llm_cache.set(prompt, response)
            else:
                if self.config.verbose:
                    self.logger.info(f"Using cached response for Step 3: operator {operator_index} ({op_type})")

            filled_operator = json.loads(response)

            # Collect response for debugging
            if collect_messages:
                messages.append({"role": "assistant", "content": response})

            # Merge with original operator to preserve framework structure
            for key, value in filled_operator.items():
                operator[key] = value

            # Transform output_schema to nested output.schema format for DocETL
            # This applies to operators that use output schemas: map, filter, reduce, resolve
            if 'output_schema' in operator and operator['type'] in ['map', 'filter', 'reduce', 'resolve']:
                if operator['output_schema']:  # Only transform if not empty
                    operator['output'] = {'schema': operator['output_schema']}
                    del operator['output_schema']
                else:
                    # If empty, still create the nested structure with empty schema
                    operator['output'] = {'schema': {}}
                    del operator['output_schema']

            # Special validation for extract operators
            if operator['type'] == 'extract':
                if 'document_keys' in operator and (not operator['document_keys'] or operator['document_keys'] == []):
                    if self.config.verbose:
                        self.logger.warning(f"Extract operator {operator.get('name', 'unknown')} has empty document_keys, defaulting to ['src']")
                    operator['document_keys'] = ["src"]

            if collect_messages:
                return operator, messages
            return operator

        except (json.JSONDecodeError, KeyError) as e:
            if self.config.verbose:
                self.logger.error(f"Failed to parse structured operator details: {e}")
            raise e

    def _identify_answer_fields(self, query: str, available_fields: List[str],
                               filled_operators: List[Dict[str, Any]],
                               collect_messages: bool = False,
                               force_regenerate: bool = False) -> tuple:
        """Identify fields containing the answer."""
        prompt = f"""
Given this query and the available fields after all processing steps,
identify which specific fields contain the information needed to answer the query.

Query: {query}

Available fields after processing:
{json.dumps(available_fields, indent=2)}

Processing steps completed:
{json.dumps([{'name': op.get('name'), 'type': op['type'],
              'purpose': op.get('purpose', '')}
             for op in filled_operators], indent=2)}

Return ONLY the field names that can directly answer the query without any further processing.
For example: 
If the query is "Extract the medication and dosage from the document", and the available fields are ["src", "medication", "dosage"],
then the answer fields are ["medication", "dosage"]. Even if the "src" field is also available, it need to be processed further to extract the medication and dosage.
"""

        # Use structured output to get field list
        schema = {
            "type": "object",
            "properties": {
                "answer_fields": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            },
            "required": ["answer_fields"]
        }

        messages = [{"role": "user", "content": prompt}]

        try:
            # Check cache first (with regeneration flag if requested)
            response = self.llm_cache.get(prompt, force_regenerate=force_regenerate)
            if response is None:
                # No cache or regeneration requested, call LLM
                response = llm_call(messages, schema=schema,
                                   system_prompt="You are an AI that identifies which fields answer a query")
                # Save to cache
                self.llm_cache.set(prompt, response)
            else:
                if self.config.verbose:
                    self.logger.info("Using cached response for answer field identification")

            result = json.loads(response)
            answer_fields = result.get("answer_fields", [])

            # Collect response for debugging
            if collect_messages:
                messages.append({"role": "assistant", "content": response})

            if not answer_fields:
                raise ValueError("No answer fields identified from the LLM response.")

            if self.config.verbose:
                self.logger.info(f"Identified answer fields: {answer_fields}")

            if collect_messages:
                return answer_fields, messages
            return answer_fields, []

        except Exception as e:
            self.logger.error(f"Failed to identify answer fields: {e}")
            raise e

    def _synthesize_final_code_map(self, answer_fields: List[str]) -> Dict[str, Any]:
        """Generate final code_map operator."""
        # Generate safe field extraction code
        field_extractions = []
        for field in answer_fields:
            safe_field = field.replace("'", "\\'")
            field_extractions.append(f"        '{field}': doc.get('{safe_field}', '')")

        join_str = ',\n'
        code = f"""
        def transform(doc) -> dict:
            return {{
                {join_str.join(field_extractions)}
            }}
        """

        if self.config.verbose:
            self.logger.info(f"Synthesized final code_map for fields: {answer_fields}")

        return {
            'name': 'final_extract_result',
            'type': 'code_map',
            'code': code
        }

    def _fill_operator_draft(self, frameworks: List[Dict[str, Any]], query: str,
                            dataset_samples: Dict[str, Any], attempt: int = 0, collect_messages: bool = False):
        """Step 4: Fill operator drafts with details."""
        filled_operators = []
        all_messages = []

        for i, framework in enumerate(frameworks):
            if self.config.verbose:
                self.logger.info(f"Generating details for operator {i+1}/{len(frameworks)}: {framework['type']}")

            # Track available fields at this stage
            available_fields = self._track_available_fields(dataset_samples, filled_operators)
            # Get complete type system at this stage
            current_type_system = self._get_type_system_at_stage(dataset_samples, filled_operators)

            if self.config.verbose:
                self.logger.info(f"Available fields for operator {framework['type']}: {available_fields}")

            # Generate details for this operator
            filled_op, operator_messages = self._generate_operator_details(
                framework,
                query,
                dataset_samples,
                filled_operators,  # Pass previously filled operators for context
                available_fields,  # Pass available fields
                current_type_system,  # Pass type system for better prompting
                collect_messages=collect_messages,
                operator_index=i,
                attempt=attempt,
                total_operators=len(frameworks)  # Pass total number of operators
            )

            # Collect messages if requested
            if collect_messages and operator_messages:
                all_messages.extend(operator_messages)

            # Validate field references using the current type system
            is_valid, validation_errors = self._validate_field_references(filled_op, current_type_system)

            if not is_valid and self.config.verbose:
                self.logger.warning(f"Field validation issues in operator {framework['type']}: {validation_errors}")
                # Continue anyway, but log the issues

            # Clean up the operator name
            filled_op['name'] = f"{framework['type']}_{i}"

            filled_operators.append(filled_op)

        # Confirm this step in confirm/debug mode
        if not self.ui.confirm_step_execution("Step 3-4: Operator Details", filled_operators, query, attempt):
            raise ValueError("User aborted pipeline generation at operator detail generation step")

        # After all operators are filled, identify answer fields and synthesize final code_map
        if filled_operators:
            available_fields = self._track_available_fields(dataset_samples, filled_operators)
            answer_fields, answer_messages = self._identify_answer_fields(
                query, available_fields, filled_operators, collect_messages=collect_messages
            )

            # Collect messages from answer field identification
            if collect_messages and answer_messages:
                all_messages.extend(answer_messages)

            if answer_fields:
                final_code_map = self._synthesize_final_code_map(answer_fields)
                filled_operators.append(final_code_map)

                if self.config.verbose:
                    self.logger.info(f"Auto-synthesized final code_map with fields: {answer_fields}")

        if collect_messages:
            return filled_operators, all_messages
        return filled_operators

    def _connect_pipeline(self, operators: List[Dict[str, Any]], dataset_paths: List[str],
                         query: str, attempt: int = 0) -> str:
        """Step 5: Connect operators and create pipeline YAML."""
        # Determine if we have merged datasets
        is_merged = len(dataset_paths) > 1 or 'merged_datasets' in dataset_paths[0] if dataset_paths else False

        # Create the pipeline structure
        pipeline = {
            'default_model': 'azure/gpt-4o',
            'datasets': {},
            'operations': operators,
            'pipeline': {}
        }

        # Set up datasets section
        if dataset_paths:
            if is_merged or len(dataset_paths) > 1:
                # Use the merged dataset path
                dataset_path = dataset_paths[0] if len(dataset_paths) == 1 else self.data_processor.merge_datasets_to_json(dataset_paths)
                pipeline['datasets']['merged_data'] = {
                    'type': 'file',
                    'path': dataset_path
                }
                input_name = 'merged_data'
            else:
                # Single dataset
                pipeline['datasets']['input_data'] = {
                    'type': 'file',
                    'path': dataset_paths[0]
                }
                input_name = 'input_data'
        else:
            # No dataset paths provided, panic
            raise ValueError("No dataset paths provided. Please provide at least one dataset path.")

        # Set up pipeline steps
        pipeline['pipeline']['steps'] = [{
            'name': 'process_data',
            'input': input_name,
            'operations': [op['name'] for op in operators]
        }]

        # Set up output
        timestamp = self.data_processor._get_timestamp()
        output_filename = f"output_{timestamp}.json"
        pipeline['pipeline']['output'] = {
            'type': 'file',
            'path': os.path.join(self.pipeline_output_dir, output_filename),
            'intermediate_dir': os.path.join(self.pipeline_output_dir, 'intermediates')
        }

        # Convert to YAML
        pipeline_yaml = yaml.dump(pipeline, default_flow_style=False, sort_keys=False)

        # Confirm this step in confirm/debug mode
        if not self.ui.confirm_step_execution("Step 5: Pipeline Connection", pipeline_yaml, query, attempt):
            raise ValueError("User aborted pipeline generation at pipeline connection step")

        return pipeline_yaml

    def _generate_pipeline_step_by_step(self, query: str, dataset_paths: List[str],
                                       attempt: int = 0) -> Tuple[bool, str, str]:
        """Orchestrate step-by-step pipeline generation."""
        # Initialize message collection for this pipeline generation attempt
        all_messages = []

        try:
            # Load dataset samples
            dataset_samples = load_sample_data(dataset_paths, max_length=3000, max_string_length=2000)

            if self.config.verbose:
                self.logger.info(f"Step-by-step pipeline generation - Attempt {attempt + 1}")

            # Step 1: Select operators
            if self.config.verbose:
                self.logger.info("Step 1: Selecting operators...")
            operators, messages_step1 = self._select_operators(query, dataset_samples, attempt, collect_messages=True)
            all_messages.extend(messages_step1)
            filename_base = get_filename_base(self.data_processor, query)
            save_step_output(self.steps_output_dir, filename_base, 'operators_selected', operators, query, attempt)

            if not operators:
                raise ValueError("No operators selected for the pipeline")

            # Step 2: Create operator framework
            if self.config.verbose:
                self.logger.info("Step 2: Creating operator framework...")
            frameworks = self._create_operator_framework(operators, query, attempt)
            save_step_output(self.steps_output_dir, filename_base, 'operator_frameworks', frameworks, query, attempt)

            # Step 3 & 4: Generate and fill operator details
            if self.config.verbose:
                self.logger.info("Step 3-4: Generating operator details...")
            filled_operators, messages_step34 = self._fill_operator_draft(frameworks, query, dataset_samples, attempt, collect_messages=True)
            all_messages.extend(messages_step34)
            save_step_output(self.steps_output_dir, filename_base, 'filled_operators', filled_operators, query, attempt)

            # Step 5: Connect pipeline
            if self.config.verbose:
                self.logger.info("Step 5: Connecting pipeline...")
            pipeline_yaml = self._connect_pipeline(filled_operators, dataset_paths, query, attempt)
            save_step_output(self.steps_output_dir, filename_base, 'final_pipeline', pipeline_yaml, query, attempt)

            if self.config.verbose:
                self.logger.info("Pipeline generation completed successfully")

            # Save all collected messages for successful pipeline generation
            filename_base = get_filename_base(self.data_processor, query, f'attempt{attempt}')
            save_messages(self.messages_output_dir, filename_base, all_messages, query)

            return True, pipeline_yaml, None

        except Exception as e:
            error_msg = f"Step-by-step pipeline generation failed: {str(e)}\n{traceback.format_exc()}"
            # Always log critical pipeline generation errors
            self.logger.error(error_msg)

            # Save messages even for failed attempts
            if all_messages:
                filename_base = get_filename_base(self.data_processor, query, f'attempt{attempt}')
                save_messages(self.messages_output_dir, filename_base, all_messages, query)

            return False, None, error_msg

    def _generate_and_execute_pipeline(self, query: str, dataset_paths: List[str]) -> tuple:
        """Generate and execute pipeline."""
        start_time = time.time()

        if self.config.verbose:
            self.logger.info(f"Processing {len(dataset_paths)} dataset files:")
            for path in dataset_paths:
                self.logger.info(f"  - {path}")

        # Handle multiple datasets: merge them into a single JSON file
        if len(dataset_paths) > 1:
            merged_file_path = self.data_processor.merge_datasets_to_json(dataset_paths)
            final_dataset_paths = [merged_file_path]
            if self.config.verbose:
                self.logger.info(f"Merged {len(dataset_paths)} files into single dataset: {merged_file_path}")
        else:
            final_dataset_paths = dataset_paths

        # Track pipeline generation history
        pipeline_history = []

        for attempt in range(self.max_attempts):
            self.total_generation_attempts += 1

            # Create pipeline filename with attempt number
            pipeline_filename = f"{self.data_processor._create_base_filename(query, f'attempt{attempt}')}.yaml"
            pipeline_file = os.path.join(self.pipeline_output_dir, pipeline_filename)

            if self.config.verbose:
                self.logger.info(f"Pipeline generation attempt {attempt + 1}/{self.max_attempts}")

            try:
                # Generate pipeline using step-by-step approach
                success, pipeline_yaml, error_msg = self._generate_pipeline_step_by_step(
                    query, final_dataset_paths, attempt
                )

                if not success:
                    raise ValueError(error_msg)

                # Save pipeline to file
                with open(pipeline_file, "w") as f:
                    f.write(f"{self.ui.format_query_as_comments(query)}\n")
                    f.write(f"# Attempt: {attempt}\n")
                    f.write(f"# Generated at: {datetime.now().isoformat()}\n")
                    f.write(f"# Method: Step-by-Step Generation\n")
                    f.write("#" + "="*50 + "\n\n")
                    f.write(pipeline_yaml)

                # Confirm pipeline execution in debug/confirm mode
                if not self.ui.confirm_pipeline_execution(pipeline_file, query, attempt):
                    continue

                # Execute pipeline
                exec_success, exec_error_msg = execute_single_pipeline(pipeline_file)

                if not exec_success:
                    if self.config.verbose:
                        self.logger.warning(f"Pipeline execution failed: {exec_error_msg}")

                    pipeline_history.append(FailedPipeline(
                        pipeline_yaml=pipeline_yaml,
                        error_type='execution',
                        error_message=exec_error_msg
                    ))
                    continue

                # Validate answer if enabled
                if self.validate_answer:
                    is_valid, validation_msg, sample_output = validate_pipeline_output(
                        pipeline_file, query, final_dataset_paths, llm_call
                    )

                    if not is_valid:
                        if self.config.verbose:
                            self.logger.warning(f"Pipeline validation failed: {validation_msg}")
                            if sample_output:
                                self.logger.warning(f"Sample output: {sample_output[:500]}...")

                        pipeline_history.append(FailedPipeline(
                            pipeline_yaml=pipeline_yaml,
                            error_type='validation',
                            error_message=f"{validation_msg}\nSample output: {sample_output[:500] if sample_output else 'None'}"
                        ))
                        continue

                # Success! Load the output
                output_path = self._find_output_path(pipeline_file)
                result = self._load_pipeline_output(output_path)

                execution_time = time.time() - start_time
                self.successful_pipelines += 1

                return True, result, execution_time

            except Exception as e:
                error_msg = f"Pipeline generation error: {str(e)}"
                # Always log pipeline generation errors with traceback
                self.logger.error(f"{error_msg}\nTraceback:\n{traceback.format_exc()}")

                pipeline_history.append(FailedPipeline(
                    pipeline_yaml="",
                    error_type='generation',
                    error_message=error_msg
                ))

        # All attempts failed
        execution_time = time.time() - start_time
        self.failed_pipelines.append({
            'query': query,
            'attempts': len(pipeline_history),
            'history': pipeline_history
        })

        return False, {"error": "Failed to generate working pipeline after all attempts"}, execution_time

    def _find_output_path(self, pipeline_file: str) -> Optional[str]:
        """Find the output path from pipeline configuration."""
        try:
            with open(pipeline_file, 'r') as f:
                pipeline_config = yaml.safe_load(f)

            pipeline_output = pipeline_config.get('pipeline', {}).get('output', {})
            if isinstance(pipeline_output, dict) and 'path' in pipeline_output:
                output_path = pipeline_output['path']
                if not os.path.isabs(output_path):
                    output_path = os.path.join(os.path.dirname(pipeline_file), output_path)
                return output_path

            for dataset in pipeline_config.get('datasets', []):
                output_name = pipeline_config.get('pipeline', {}).get('output', {}).get('path')
                if dataset.get('name') == output_name:
                    return dataset.get('path')

            return None
        except Exception as e:
            if self.config.verbose:
                self.logger.warning(f"Failed to find output path: {e}")
            return None

    def _load_pipeline_output(self, output_path: str) -> Any:
        """Load and return pipeline output data."""
        if not output_path or not os.path.exists(output_path):
            return None

        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                if output_path.endswith('.json'):
                    data = json.load(f)
                elif output_path.endswith('.csv'):
                    import pandas as pd
                    df = pd.read_csv(output_path)
                    data = df.to_dict('records')
                else:
                    data = f.read()

            # Extract 'result' field if present
            if isinstance(data, dict) and 'result' in data:
                return data['result']
            elif isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict) and 'result' in data[0]:
                results = [item.get('result', item) for item in data]
                if len(results) == 1:
                    return results[0]
                return results
            else:
                return data

        except Exception as e:
            if self.config.verbose:
                self.logger.warning(f"Failed to load output: {e}")
            return None

    def process(self, query: str, context: Optional[Dict[str, Any]] = None) -> BaselineResult:
        """Process a single query with optional context using step-by-step DocETL pipeline generation."""
        start_time = time.time()
        self.total_queries += 1

        try:
            # Prepare dataset files from context
            dataset_paths = self.data_processor.prepare_dataset_files(context)

            if not dataset_paths and context:
                raise ValueError("No valid datasets could be prepared from context")

            # Generate and execute pipeline
            success, result, pipeline_time = self._generate_and_execute_pipeline(query, dataset_paths)

            execution_time = time.time() - start_time
            self.total_time += execution_time

            if success:
                return BaselineResult(
                    query=query,
                    response={"answer": result, "pipeline_generated": True, "method": "step-by-step"},
                    metadata={
                        "baseline": "docetl_step",
                        "context_provided": context is not None,
                        "num_datasets": len(dataset_paths),
                        "generation_attempts": self.total_generation_attempts,
                        "pipeline_time": pipeline_time,
                        "success": True
                    },
                    execution_time=execution_time
                )
            else:
                return BaselineResult(
                    query=query,
                    response={"answer": "", "error": result.get("error", "Pipeline generation failed")},
                    metadata={
                        "baseline": "docetl_step",
                        "context_provided": context is not None,
                        "num_datasets": len(dataset_paths),
                        "generation_attempts": self.total_generation_attempts,
                        "success": False
                    },
                    execution_time=execution_time,
                    error=result.get("error", "Pipeline generation failed")
                )

        except Exception as e:
            self.logger.error(f"Error processing query with DocETL Step-by-Step: {e}")
            # Always show full traceback for critical errors in main process method
            self.logger.error(f"Full traceback:\n{traceback.format_exc()}")

            execution_time = time.time() - start_time
            self.total_time += execution_time

            return BaselineResult(
                query=query,
                response={"answer": "", "error": str(e)},
                metadata={
                    "baseline": "docetl_step",
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
            "baseline_type": "docetl_step"
        }