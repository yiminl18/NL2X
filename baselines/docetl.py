import time
from typing import Any, Dict, List, Optional
import os
import json
import yaml
import traceback
from datetime import datetime
import logging

from .base import BaselineInterface, BaselineResult
from . import register_baseline
from .docetl_utils.data_utils import DocETLDataProcessor
from .docetl_utils.log_utils import save_prompt, save_messages, save_validation, get_filename_base
from .docetl_utils.llm2pipeline_docetl import (
    create_initial_messages,
    add_error_message,
    llm_call_with_messages,
    load_sample_data,
    execute_single_pipeline,
    validate_pipeline_output,
    extract_yaml_from_response,
    validate_generated_pipeline,
    FailedPipeline
)
from .docetl_utils.prompt import INSTRUCTION_PROMPT

@register_baseline("docetl")
class DocETLBaseline(BaselineInterface):
    """
    DocETL baseline implementation that generates and executes data processing pipelines
    from natural language queries using LLM-driven pipeline generation.
    """
    
    def _setup(self):
        self.total_queries = 0
        self.total_time = 0
        self.total_generation_attempts = 0
        self.successful_pipelines = 0
        self.failed_pipelines = []

        # Use configuration from BaselineConfig
        self.max_attempts = self.config.max_attempts
        self.validate_answer = self.config.validate_answer

        # Control LiteLLM logging based on mode
        # In confirm-only mode (not verbose), suppress most LiteLLM output
        if not self.config.verbose:
            logging.getLogger("LiteLLM").setLevel(logging.ERROR)
            # Also disable other noisy loggers
            logging.getLogger("httpcore").setLevel(logging.ERROR)
            logging.getLogger("openai").setLevel(logging.ERROR)
            logging.getLogger("asyncio").setLevel(logging.ERROR)
        else:
            # In verbose mode, show INFO level but not DEBUG
            logging.getLogger("LiteLLM").setLevel(logging.INFO)
            logging.getLogger("httpcore").setLevel(logging.INFO)
            logging.getLogger("openai").setLevel(logging.INFO)

        # Create persistent directories for saving outputs
        base_dir = os.getcwd()
        self.output_dirs = {
            'pipeline_output_dir': os.path.join(base_dir, "generated_pipelines", "docetl"),
            'prompts_output_dir': os.path.join(base_dir, "generated_prompts", "docetl"),
            'validations_output_dir': os.path.join(base_dir, "validations", "docetl"),
            'messages_output_dir': os.path.join(base_dir, "messages", "docetl"),
            'converted_data_dir': os.path.join(base_dir, "converted_data", "docetl")
        }

        # Set instance attributes and create directories
        for attr_name, dir_path in self.output_dirs.items():
            setattr(self, attr_name, dir_path)
            os.makedirs(dir_path, exist_ok=True)

        # Initialize data processor
        self.data_processor = DocETLDataProcessor(
            converted_data_dir=self.converted_data_dir,
            verbose=self.config.verbose,
            logger=self.logger
        )

        if self.config.verbose:
            self.logger.info(f"Initialized DocETL baseline with config: {self.config}")
            self.logger.info(f"Pipeline output directory: {self.pipeline_output_dir}")


    def _confirm_pipeline_execution(self, pipeline_file: str, query: str, attempt: int) -> bool:
        """
        Ask user for confirmation before executing pipeline in confirm/debug mode.

        Args:
            pipeline_file: Path to the generated pipeline file
            query: The original query
            attempt: Current attempt number

        Returns:
            True if user confirms execution, False otherwise
        """
        if not (self.config.confirm or self.config.debug):
            return True

        print("\n" + "="*80)
        mode_text = "[DEBUG MODE]" if self.config.debug else "[CONFIRM MODE]"
        print(f"{mode_text} Pipeline Generated - Attempt {attempt + 1}")
        print("="*80)

        print(f"\n📋 Query:")
        print("-"*40)
        print(query)
        print("-"*40)

        print(f"\n📄 Generated Pipeline File:")
        print(f"  {pipeline_file}")

        print(f"\n📝 Pipeline Preview (first 50 lines):")
        print("-"*40)
        try:
            with open(pipeline_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for i, line in enumerate(lines[:50]):
                    print(f"{i+1:3d}: {line.rstrip()}")
                if len(lines) > 50:
                    print(f"... ({len(lines) - 50} more lines)")
        except Exception as e:
            print(f"Error reading pipeline file: {e}")
        print("-"*40)

        print("\n⚠️  Execute this pipeline? (Y/n): ", end="")
        user_input = input().strip().lower()

        if user_input and user_input != 'y':
            print("❌ Pipeline execution skipped by user")
            return False

        print("✅ Proceeding with pipeline execution...")
        return True

    
    


    def _generate_and_execute_pipeline(self, query: str, dataset_paths: List[str]) -> tuple:
        """
        Generate and execute DocETL pipeline for the query.

        Args:
            query: Natural language query
            dataset_paths: List of dataset file paths

        Returns:
            Tuple of (success: bool, result: dict, execution_time: float)
        """
        start_time = time.time()

        if self.config.verbose:
            self.logger.info(f"Processing {len(dataset_paths)} dataset files:")
            for path in dataset_paths:
                self.logger.info(f"  - {path}")

        # Handle multiple datasets: merge them into a single JSON file
        if len(dataset_paths) > 1:
            # Merge multiple files into a single JSON dataset
            merged_file_path = self.data_processor.merge_datasets_to_json(dataset_paths)
            final_dataset_paths = [merged_file_path]
            if self.config.verbose:
                self.logger.info(f"Merged {len(dataset_paths)} files into single dataset: {merged_file_path}")
        else:
            # Single file, use as-is
            final_dataset_paths = dataset_paths

        # Load dataset samples
        dataset_samples = load_sample_data(final_dataset_paths, max_length=1500, max_string_length=200)

        if self.config.verbose:
            self.logger.info(f"Loaded samples from {len(dataset_samples)} files")
        
        # Initialize conversation with initial messages
        initial_prompt, messages = create_initial_messages(INSTRUCTION_PROMPT, query, dataset_samples)
        
        # Track pipeline generation history
        pipeline_history = []
        
        for attempt in range(self.max_attempts):
            self.total_generation_attempts += 1

            # Create pipeline filename with attempt number
            pipeline_filename = f"{self.data_processor._create_base_filename(query, f'attempt{attempt}')}.yaml"
            pipeline_file = os.path.join(self.pipeline_output_dir, pipeline_filename)

            if self.config.verbose:
                self.logger.info(f"Pipeline generation attempt {attempt + 1}/{self.max_attempts}")

            # Save initial prompt for each attempt
            self._save_prompt(initial_prompt if attempt == 0 else f"Retry attempt {attempt}\n\n{initial_prompt}", query, attempt)

            try:
                # Generate pipeline
                response = llm_call_with_messages(messages)
                messages.append({"role": "assistant", "content": response})

                # Extract YAML from response
                pipeline_yaml = extract_yaml_from_response(response)
                if not pipeline_yaml:
                    raise ValueError("No valid YAML found in LLM response")

                # Validate pipeline structure
                validate_generated_pipeline(pipeline_yaml)

                # Save pipeline to file with metadata
                with open(pipeline_file, "w") as f:
                    # Handle multi-line queries by commenting each line
                    query_lines = query.split('\n')
                    f.write(f"# Query: {query_lines[0]}\n")
                    for line in query_lines[1:]:
                        f.write(f"# {line}\n")
                    f.write(f"# Attempt: {attempt}\n")
                    f.write(f"# Generated at: {datetime.now().isoformat()}\n")
                    f.write("#" + "="*50 + "\n\n")
                    f.write(pipeline_yaml)

                # Confirm pipeline execution in debug/confirm mode
                if not self._confirm_pipeline_execution(pipeline_file, query, attempt):
                    # User declined to execute - skip this attempt
                    continue

                # Execute pipeline
                success, error_msg = execute_single_pipeline(pipeline_file)
                
                if not success:
                    # Execution failed - add error feedback for retry
                    if self.config.verbose:
                        self.logger.warning(f"Pipeline execution failed: {error_msg}")

                    messages = add_error_message(messages, "execution", error_msg)
                    pipeline_history.append(FailedPipeline(
                        pipeline_yaml=pipeline_yaml,
                        error_type='execution',
                        error_message=error_msg
                    ))
                    continue
                
                # Validate answer if enabled
                if self.validate_answer:
                    is_valid, validation_msg, sample_output = validate_pipeline_output(
                        pipeline_file, query, dataset_paths, llm_call_with_messages
                    )
                    
                    if not is_valid:
                        # Validation failed - add feedback for retry
                        if self.config.verbose:
                            self.logger.warning(f"Pipeline validation failed: {validation_msg}")
                        
                        messages = add_error_message(
                            messages, 
                            "validation", 
                            validation_msg,
                            previous_pipeline=pipeline_yaml,
                            sample_output=sample_output
                        )
                        pipeline_history.append(FailedPipeline(
                            pipeline_yaml=pipeline_yaml,
                            error_type='validation',
                            error_message=validation_msg
                        ))
                        continue
                
                # Success! Load the output
                output_path = self._find_output_path(pipeline_file)
                result = self._load_pipeline_output(output_path)

                # Save complete message history for successful run
                filename_base = get_filename_base(self.data_processor, query)
                save_messages(self.messages_output_dir, filename_base, messages, query, pipeline_history)

                # Keep pipeline file without renaming

                execution_time = time.time() - start_time
                self.successful_pipelines += 1

                return True, result, execution_time
                
            except Exception as e:
                # Generation or parsing error
                error_msg = f"Pipeline generation error: {str(e)}"
                if self.config.verbose:
                    self.logger.warning(error_msg)
                
                messages = add_error_message(messages, "execution", error_msg)
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

        # Save message history for failed run
        filename_base = get_filename_base(self.data_processor, query)
        save_messages(self.messages_output_dir, filename_base, messages, query, pipeline_history)

        # Keep pipeline file without renaming

        return False, {"error": "Failed to generate working pipeline after all attempts"}, execution_time
    
    def _find_output_path(self, pipeline_file: str) -> Optional[str]:
        """Find the output path from pipeline configuration."""
        try:
            with open(pipeline_file, 'r') as f:
                pipeline_config = yaml.safe_load(f)
            
            # Check pipeline output configuration
            pipeline_output = pipeline_config.get('pipeline', {}).get('output', {})
            if isinstance(pipeline_output, dict) and 'path' in pipeline_output:
                # Make path absolute if relative
                output_path = pipeline_output['path']
                if not os.path.isabs(output_path):
                    output_path = os.path.join(os.path.dirname(pipeline_file), output_path)
                return output_path
            
            # Check datasets for output
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
        """Load and return pipeline output data, extracting 'result' field if present."""
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

            # Extract 'result' field if present (as required by pipeline prompt)
            if isinstance(data, dict) and 'result' in data:
                return data['result']
            elif isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict) and 'result' in data[0]:
                # Handle array format where each item might have 'result' field
                results = [item.get('result', item) for item in data]
                # For single-item lists, return just the result value to avoid list wrapping
                if len(results) == 1:
                    return results[0]
                return results
            else:
                # Backward compatibility: return original data if no 'result' field
                return data

        except Exception as e:
            if self.config.verbose:
                self.logger.warning(f"Failed to load output: {e}")
            return None
    
    def process(self, query: str, context: Optional[Dict[str, Any]] = None) -> BaselineResult:
        """Process a single query with optional context using DocETL pipeline generation."""
        start_time = time.time()
        self.total_queries += 1
        
        try:
            # Prepare dataset files from context
            dataset_paths = self.data_processor.prepare_dataset_files(context)

            if not dataset_paths and context:
                # No valid datasets could be prepared
                raise ValueError("No valid datasets could be prepared from context")
            
            # Generate and execute pipeline
            success, result, pipeline_time = self._generate_and_execute_pipeline(query, dataset_paths)
            
            execution_time = time.time() - start_time
            self.total_time += execution_time
            
            if success:
                return BaselineResult(
                    query=query,
                    response={"answer": result, "pipeline_generated": True},
                    metadata={
                        "baseline": "docetl",
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
                        "baseline": "docetl",
                        "context_provided": context is not None,
                        "num_datasets": len(dataset_paths),
                        "generation_attempts": self.total_generation_attempts,
                        "success": False
                    },
                    execution_time=execution_time,
                    error=result.get("error", "Pipeline generation failed")
                )
            
        except Exception as e:
            self.logger.error(f"Error processing query with DocETL: {e}")
            if self.config.verbose:
                self.logger.error(traceback.format_exc())
            
            execution_time = time.time() - start_time
            self.total_time += execution_time
            
            return BaselineResult(
                query=query,
                response={"answer": "", "error": str(e)},
                metadata={
                    "baseline": "docetl",
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
            "baseline_type": "docetl"
        }
    
