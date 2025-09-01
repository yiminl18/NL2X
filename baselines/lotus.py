import time
from typing import Any, Dict, List, Optional
import os
import json
import tempfile
import traceback

from .base import BaselineInterface, BaselineResult
from . import register_baseline
from .lotus_utils.llm2pipeline_lotus import (
    create_initial_messages,
    add_error_message,
    llm_call_with_messages,
    load_sample_data,
    execute_single_pipeline,
    validate_pipeline_output,
    extract_python_from_response,
    validate_generated_pipeline,
    convert_json_to_csv,
    create_wrapped_pipeline_code,
    FailedPipeline
)
from .lotus_utils.prompt import INSTRUCTION_PROMPT

@register_baseline("lotus")
class LOTUSBaseline(BaselineInterface):
    """
    LOTUS baseline implementation that generates and executes semantic data processing pipelines
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
        self.temp_dir = tempfile.mkdtemp(prefix="lotus_baseline_")
        
        if self.config.verbose:
            self.logger.info(f"Initialized LOTUS baseline with config: {self.config}")
            self.logger.info(f"Temporary directory: {self.temp_dir}")
    
    def _prepare_dataset_files(self, context: Dict[str, Any]) -> List[str]:
        """
        Prepare dataset files from context for LOTUS pipeline.
        Converts all data to CSV format for consistent DataFrame handling.
        
        Args:
            context: Context dictionary with data
            
        Returns:
            List of dataset file paths (CSV files)
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
                if hasattr(value, 'content') and value.content is not None:
                    content = value.content
                    
                    # Create temporary JSON file first, then convert to CSV
                    temp_json_file = os.path.join(self.temp_dir, f"{key}.json")
                    
                    if file_type in ['json', 'text']:
                        # Save as JSON if it's structured data
                        if isinstance(content, (list, dict)):
                            with open(temp_json_file, 'w', encoding='utf-8') as f:
                                json.dump(content, f, indent=2, ensure_ascii=False)
                        else:
                            # Try to parse as JSON first
                            try:
                                parsed_content = json.loads(content)
                                with open(temp_json_file, 'w', encoding='utf-8') as f:
                                    json.dump(parsed_content, f, indent=2, ensure_ascii=False)
                            except:
                                # Save as list with single text item
                                with open(temp_json_file, 'w', encoding='utf-8') as f:
                                    json.dump([{"text": content}], f, indent=2, ensure_ascii=False)
                        
                        # Convert JSON to CSV for LOTUS processing
                        csv_file = convert_json_to_csv(temp_json_file)
                        dataset_paths.append(csv_file)
                    
                    elif file_type == 'csv':
                        # Save CSV content directly
                        temp_csv_file = os.path.join(self.temp_dir, f"{key}.csv")
                        with open(temp_csv_file, 'w', encoding='utf-8') as f:
                            f.write(content)
                        dataset_paths.append(temp_csv_file)
                
                elif hasattr(value, 'path') and value.path:
                    # Use existing file path, convert to CSV if needed
                    if os.path.exists(value.path):
                        if value.path.endswith('.json'):
                            # Convert JSON file to CSV
                            csv_path = convert_json_to_csv(value.path)
                            dataset_paths.append(csv_path)
                        elif value.path.endswith('.csv'):
                            dataset_paths.append(value.path)
                        else:
                            # For text files and other types, create a temporary CSV
                            temp_json_file = os.path.join(self.temp_dir, f"{key}.json")
                            with open(value.path, 'r', encoding='utf-8') as src:
                                content = src.read()
                            
                            # Handle text files - keep entire file as single entry
                            if hasattr(value, 'type') and value.type == 'text':
                                # Keep entire text file as single document
                                json_data = [{"text": content}]
                            else:
                                # For other types, keep as single document
                                json_data = [{"text": content}]
                            
                            with open(temp_json_file, 'w', encoding='utf-8') as dst:
                                json.dump(json_data, dst, indent=2, ensure_ascii=False)
                            csv_path = convert_json_to_csv(temp_json_file)
                            dataset_paths.append(csv_path)
                    else:
                        if self.config.verbose:
                            self.logger.warning(f"File not found: {value.path}")
                
            except Exception as e:
                if self.config.verbose:
                    self.logger.warning(f"Failed to prepare dataset {key}: {e}")
        
        return dataset_paths
    
    def _generate_and_execute_pipeline(self, query: str, dataset_paths: List[str]) -> tuple:
        """
        Generate and execute LOTUS pipeline for the query.
        
        Args:
            query: Natural language query
            dataset_paths: List of dataset file paths (CSV files)
            
        Returns:
            Tuple of (success: bool, result: dict, execution_time: float)
        """
        start_time = time.time()
        pipeline_file = os.path.join(self.temp_dir, "pipeline.py")
        
        # Load dataset samples (this handles CSV files)
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
                
                # Extract Python code from response
                pipeline_code = extract_python_from_response(response)
                if not pipeline_code:
                    raise ValueError("No valid Python code found in LLM response")
                
                # Validate pipeline structure
                validate_generated_pipeline(pipeline_code)
                
                # Add wrapper code if needed
                if "__name__" not in pipeline_code:
                    output_file = os.path.join(self.temp_dir, "pipeline_output.json")
                    pipeline_code = create_wrapped_pipeline_code(pipeline_code, dataset_paths, output_file)
                
                # Save pipeline to file
                with open(pipeline_file, "w") as f:
                    f.write(pipeline_code)
                os.chmod(pipeline_file, 0o755)  # Make executable
                
                # Execute pipeline
                success, error_msg, output_data = execute_single_pipeline(pipeline_file, dataset_paths)
                
                if not success:
                    # Execution failed - add error feedback for retry
                    if self.config.verbose:
                        self.logger.warning(f"Pipeline execution failed: {error_msg}")
                    
                    messages = add_error_message(messages, "execution", error_msg)
                    pipeline_history.append(FailedPipeline(
                        pipeline_code=pipeline_code,
                        error_type='execution',
                        error_message=error_msg
                    ))
                    continue
                
                # Validate answer if enabled
                if self.validate_answer:
                    is_valid, validation_msg, sample_output = validate_pipeline_output(
                        query, dataset_paths, output_data, llm_call_with_messages
                    )
                    
                    if not is_valid:
                        # Validation failed - add feedback for retry
                        if self.config.verbose:
                            self.logger.warning(f"Pipeline validation failed: {validation_msg}")
                        
                        messages = add_error_message(
                            messages, 
                            "validation", 
                            validation_msg,
                            previous_pipeline=pipeline_code,
                            sample_output=sample_output
                        )
                        pipeline_history.append(FailedPipeline(
                            pipeline_code=pipeline_code,
                            error_type='validation',
                            error_message=validation_msg
                        ))
                        continue
                
                # Success!
                execution_time = time.time() - start_time
                self.successful_pipelines += 1
                
                return True, output_data, execution_time
                
            except Exception as e:
                # Generation or parsing error
                error_msg = f"Pipeline generation error: {str(e)}"
                if self.config.verbose:
                    self.logger.warning(error_msg)
                
                messages = add_error_message(messages, "execution", error_msg)
                pipeline_history.append(FailedPipeline(
                    pipeline_code="",
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
    
    def process(self, query: str, context: Optional[Dict[str, Any]] = None) -> BaselineResult:
        """Process a single query with optional context using LOTUS pipeline generation."""
        start_time = time.time()
        self.total_queries += 1
        
        try:
            # Prepare dataset files from context (converted to CSV)
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
                        "baseline": "lotus",
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
                        "baseline": "lotus",
                        "context_provided": context is not None,
                        "num_datasets": len(dataset_paths),
                        "generation_attempts": self.total_generation_attempts,
                        "success": False
                    },
                    execution_time=execution_time,
                    error=result.get("error", "Pipeline generation failed")
                )
            
        except Exception as e:
            self.logger.error(f"Error processing query with LOTUS: {e}")
            if self.config.verbose:
                self.logger.error(traceback.format_exc())
            
            execution_time = time.time() - start_time
            self.total_time += execution_time
            
            return BaselineResult(
                query=query,
                response={"answer": "", "error": str(e)},
                metadata={
                    "baseline": "lotus",
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
            "baseline_type": "lotus"
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