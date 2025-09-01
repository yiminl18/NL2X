import time
from typing import Any, Dict, List, Optional
import os
import json
import yaml
import tempfile
import traceback
from pathlib import Path

from .base import BaselineInterface, BaselineResult
from . import register_baseline
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
from .docetl_utils.prompt import INSTRUCTION_PROMPT, PIPELINE_GENERATION_PROMPT

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
        self.temp_dir = tempfile.mkdtemp(prefix="docetl_baseline_")
        
        if self.config.verbose:
            self.logger.info(f"Initialized DocETL baseline with config: {self.config}")
            self.logger.info(f"Temporary directory: {self.temp_dir}")
    
    def _prepare_dataset_files(self, context: Dict[str, Any]) -> List[str]:
        """
        Prepare dataset files from context for DocETL pipeline.
        
        Args:
            context: Context dictionary with data
            
        Returns:
            List of dataset file paths
        """
        dataset_paths = []
        
        if not context:
            return dataset_paths
        
        for key, value in context.items():
            try:
                # Determine the file type and content
                if hasattr(value, 'type'):
                    file_type = value.type
                else:
                    file_type = 'json'  # Default to JSON
                
                # Get content or path
                if hasattr(value, 'content'):
                    content = value.content
                    
                    # Create temporary file for the content
                    if file_type in ['json', 'text']:
                        # Save as JSON if it's structured data
                        temp_file = os.path.join(self.temp_dir, f"{key}.json")
                        
                        # If content is already structured, save it
                        if isinstance(content, (list, dict)):
                            with open(temp_file, 'w', encoding='utf-8') as f:
                                json.dump(content, f, indent=2, ensure_ascii=False)
                        else:
                            # Try to parse as JSON first
                            try:
                                parsed_content = json.loads(content)
                                with open(temp_file, 'w', encoding='utf-8') as f:
                                    json.dump(parsed_content, f, indent=2, ensure_ascii=False)
                            except:
                                # Save as list with single text item
                                with open(temp_file, 'w', encoding='utf-8') as f:
                                    json.dump([{"text": content}], f, indent=2, ensure_ascii=False)
                        
                        dataset_paths.append(temp_file)
                    
                    elif file_type == 'csv':
                        # Save CSV content
                        temp_file = os.path.join(self.temp_dir, f"{key}.csv")
                        with open(temp_file, 'w', encoding='utf-8') as f:
                            f.write(content)
                        dataset_paths.append(temp_file)
                
                elif hasattr(value, 'path') and value.path:
                    # Use existing file path
                    if os.path.exists(value.path):
                        dataset_paths.append(value.path)
                    else:
                        if self.config.verbose:
                            self.logger.warning(f"File not found: {value.path}")
                
            except Exception as e:
                if self.config.verbose:
                    self.logger.warning(f"Failed to prepare dataset {key}: {e}")
        
        return dataset_paths
    
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
        pipeline_file = os.path.join(self.temp_dir, "pipeline.yaml")
        
        # Load dataset samples
        dataset_samples = load_sample_data(dataset_paths, max_length=1500, max_string_length=200)
        
        # Initialize conversation with initial messages
        messages = create_initial_messages(INSTRUCTION_PROMPT, query, dataset_samples)
        
        # Track pipeline generation history
        pipeline_history = []
        
        for attempt in range(self.max_attempts):
            self.total_generation_attempts += 1
            
            if self.config.verbose:
                self.logger.info(f"Pipeline generation attempt {attempt + 1}/{self.max_attempts}")
            
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
                
                # Save pipeline to file
                with open(pipeline_file, "w") as f:
                    f.write(pipeline_yaml)
                
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
        """Load and return pipeline output data."""
        if not output_path or not os.path.exists(output_path):
            return None
        
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                if output_path.endswith('.json'):
                    return json.load(f)
                elif output_path.endswith('.csv'):
                    import pandas as pd
                    df = pd.read_csv(output_path)
                    return df.to_dict('records')
                else:
                    return f.read()
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
            dataset_paths = self._prepare_dataset_files(context)
            
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
    
    def __del__(self):
        """Cleanup temporary directory on deletion."""
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            try:
                import shutil
                shutil.rmtree(self.temp_dir)
            except Exception as e:
                if hasattr(self, 'logger'):
                    self.logger.warning(f"Failed to cleanup temp directory: {e}")