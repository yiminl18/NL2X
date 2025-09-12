import time
from typing import Any, Dict, List, Optional
import os
import json
import yaml
import tempfile
import traceback
import shutil
from pathlib import Path
from datetime import datetime
from bs4 import BeautifulSoup

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
        
        # Create persistent directories for saving outputs
        self.pipeline_output_dir = os.path.join(os.getcwd(), "generated_pipelines", "docetl")
        self.prompts_output_dir = os.path.join(os.getcwd(), "generated_prompts", "docetl")
        self.validations_output_dir = os.path.join(os.getcwd(), "validations", "docetl")
        self.messages_output_dir = os.path.join(os.getcwd(), "messages", "docetl")
        self.converted_data_dir = os.path.join(os.getcwd(), "converted_data", "docetl")
        
        os.makedirs(self.pipeline_output_dir, exist_ok=True)
        os.makedirs(self.prompts_output_dir, exist_ok=True)
        os.makedirs(self.validations_output_dir, exist_ok=True)
        os.makedirs(self.messages_output_dir, exist_ok=True)
        os.makedirs(self.converted_data_dir, exist_ok=True)
        
        if self.config.verbose:
            self.logger.info(f"Initialized DocETL baseline with config: {self.config}")
            self.logger.info(f"Temporary directory: {self.temp_dir}")
            self.logger.info(f"Pipeline output directory: {self.pipeline_output_dir}")
    
    def _parse_html_to_dict(self, html_content: str) -> Dict[str, Any]:
        """
        Parse HTML content into a structured dictionary.
        
        Args:
            html_content: HTML string content
            
        Returns:
            Dictionary with parsed HTML structure
        """
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Extract metadata
        metadata = {}
        if soup.title:
            metadata['title'] = soup.title.string
        
        # Extract meta tags
        meta_tags = {}
        for meta in soup.find_all('meta'):
            if meta.get('name'):
                meta_tags[meta.get('name')] = meta.get('content', '')
            elif meta.get('property'):
                meta_tags[meta.get('property')] = meta.get('content', '')
        if meta_tags:
            metadata['meta'] = meta_tags
        
        # Extract main content
        result = {
            'metadata': metadata,
            'text': soup.get_text(separator=' ', strip=True),
            'structure': {}
        }
        
        # Extract structured content
        # Headers
        headers = []
        for i in range(1, 7):
            for header in soup.find_all(f'h{i}'):
                headers.append({
                    'level': i,
                    'text': header.get_text(strip=True)
                })
        if headers:
            result['structure']['headers'] = headers
        
        # Links
        links = []
        for link in soup.find_all('a', href=True):
            links.append({
                'text': link.get_text(strip=True),
                'href': link['href']
            })
        if links:
            result['structure']['links'] = links
        
        # Images
        images = []
        for img in soup.find_all('img'):
            img_info = {
                'src': img.get('src', ''),
                'alt': img.get('alt', '')
            }
            if img.get('title'):
                img_info['title'] = img['title']
            images.append(img_info)
        if images:
            result['structure']['images'] = images
        
        # Tables
        tables = []
        for table in soup.find_all('table'):
            table_data = []
            for row in table.find_all('tr'):
                row_data = []
                for cell in row.find_all(['td', 'th']):
                    row_data.append(cell.get_text(strip=True))
                if row_data:
                    table_data.append(row_data)
            if table_data:
                tables.append(table_data)
        if tables:
            result['structure']['tables'] = tables
        
        # Lists
        lists = []
        for list_elem in soup.find_all(['ul', 'ol']):
            list_items = []
            for li in list_elem.find_all('li'):
                list_items.append(li.get_text(strip=True))
            if list_items:
                lists.append({
                    'type': list_elem.name,
                    'items': list_items
                })
        if lists:
            result['structure']['lists'] = lists
        
        return result
    
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
                if hasattr(value, 'content') and value.content is not None:
                    content = value.content
                    
                    # Create persistent file for HTML content
                    if file_type == 'html':
                        # Parse HTML content into structured dictionary
                        # Save in both temp_dir (for processing) and converted_data_dir (for persistence)
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        persistent_file = os.path.join(self.converted_data_dir, f"{timestamp}_{key}_html_to_json.json")
                        temp_file = os.path.join(self.temp_dir, f"{key}.json")
                        
                        parsed_html = self._parse_html_to_dict(content)
                        
                        # Save to persistent directory
                        with open(persistent_file, 'w', encoding='utf-8') as f:
                            json.dump([parsed_html], f, indent=2, ensure_ascii=False)
                        
                        # Also save to temp directory for processing
                        with open(temp_file, 'w', encoding='utf-8') as f:
                            json.dump([parsed_html], f, indent=2, ensure_ascii=False)
                        
                        # Use persistent file path instead of temp file
                        dataset_paths.append(persistent_file)
                        
                        if self.config.verbose:
                            self.logger.info(f"Saved HTML-to-JSON conversion to: {persistent_file}")
                    
                    elif file_type in ['json', 'text']:
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
                    # Handle file based on type
                    if os.path.exists(value.path):
                        # Check if it's an HTML file
                        if hasattr(value, 'type') and value.type == 'html':
                            # Read HTML file and parse to structured format
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            persistent_file = os.path.join(self.converted_data_dir, f"{timestamp}_{key}_html_to_json.json")
                            temp_file = os.path.join(self.temp_dir, f"{key}.json")
                            
                            with open(value.path, 'r', encoding='utf-8') as f:
                                html_content = f.read()
                            
                            parsed_html = self._parse_html_to_dict(html_content)
                            
                            # Save to persistent directory
                            with open(persistent_file, 'w', encoding='utf-8') as f:
                                json.dump([parsed_html], f, indent=2, ensure_ascii=False)
                            
                            # Also save to temp directory for processing
                            with open(temp_file, 'w', encoding='utf-8') as f:
                                json.dump([parsed_html], f, indent=2, ensure_ascii=False)
                            
                            # Use persistent file path instead of temp file
                            dataset_paths.append(persistent_file)
                            
                            if self.config.verbose:
                                self.logger.info(f"Saved HTML-to-JSON conversion to: {persistent_file}")
                                
                        elif value.path.endswith('.html') or value.path.endswith('.htm'):
                            # Auto-detect HTML files by extension
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            persistent_file = os.path.join(self.converted_data_dir, f"{timestamp}_{key}_html_to_json.json")
                            temp_file = os.path.join(self.temp_dir, f"{key}.json")
                            
                            with open(value.path, 'r', encoding='utf-8') as f:
                                html_content = f.read()
                            
                            parsed_html = self._parse_html_to_dict(html_content)
                            
                            # Save to persistent directory
                            with open(persistent_file, 'w', encoding='utf-8') as f:
                                json.dump([parsed_html], f, indent=2, ensure_ascii=False)
                            
                            # Also save to temp directory for processing
                            with open(temp_file, 'w', encoding='utf-8') as f:
                                json.dump([parsed_html], f, indent=2, ensure_ascii=False)
                            
                            # Use persistent file path instead of temp file
                            dataset_paths.append(persistent_file)
                            
                            if self.config.verbose:
                                self.logger.info(f"Saved HTML-to-JSON conversion to: {persistent_file}")
                        elif hasattr(value, 'type') and value.type == 'text':
                            # Read text file and convert to JSON format
                            temp_file = os.path.join(self.temp_dir, f"{key}.json")
                            with open(value.path, 'r', encoding='utf-8') as f:
                                text_content = f.read()
                            
                            # Save as JSON with text content as single entry
                            with open(temp_file, 'w', encoding='utf-8') as f:
                                # Keep entire text file as single document
                                json.dump([{"text": text_content}], f, indent=2, ensure_ascii=False)
                            
                            dataset_paths.append(temp_file)
                        else:
                            # For other file types (JSON, CSV, etc.), use the file directly
                            dataset_paths.append(value.path)
                    else:
                        if self.config.verbose:
                            self.logger.warning(f"File not found: {value.path}")
                
            except Exception as e:
                if self.config.verbose:
                    self.logger.warning(f"Failed to prepare dataset {key}: {e}")
        
        return dataset_paths
    
    def _create_base_filename(self, query: str, suffix: str = "") -> str:
        """Create a base filename with timestamp and query snippet."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        query_snippet = query[:50].replace(" ", "_").replace("/", "_").replace("\\", "_")
        query_snippet = "".join(c for c in query_snippet if c.isalnum() or c in "_-")
        if suffix:
            return f"{timestamp}_{suffix}_{query_snippet}"
        return f"{timestamp}_{query_snippet}"
    
    def _save_prompt(self, prompt: str, query: str, attempt: int = 0) -> str:
        """Save prompt to persistent directory."""
        filename = f"{self._create_base_filename(query, f'attempt{attempt}')}.txt"
        filepath = os.path.join(self.prompts_output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"Query: {query}\n")
            f.write(f"Attempt: {attempt}\n")
            f.write(f"Generated at: {datetime.now().isoformat()}\n")
            f.write("="*50 + "\n\n")
            f.write(prompt)
        
        return filepath
    
    def _save_validation(self, validation_prompt: str, validation_result: dict, query: str) -> str:
        """Save validation prompt and result."""
        filename = f"{self._create_base_filename(query, 'validation')}.json"
        filepath = os.path.join(self.validations_output_dir, filename)
        
        validation_data = {
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "validation_prompt": validation_prompt,
            "validation_result": validation_result
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(validation_data, f, indent=2, ensure_ascii=False)
        
        return filepath
    
    def _save_messages(self, messages: list, query: str, pipeline_history: list = None) -> str:
        """Save complete message history."""
        filename = f"{self._create_base_filename(query, 'messages')}.json"
        filepath = os.path.join(self.messages_output_dir, filename)
        
        messages_data = {
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "total_messages": len(messages),
            "messages": messages,
            "pipeline_history": []
        }
        
        if pipeline_history:
            for failed in pipeline_history:
                messages_data["pipeline_history"].append({
                    "error_type": failed.error_type,
                    "error_message": failed.error_message,
                    "pipeline_yaml": failed.pipeline_yaml
                })
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(messages_data, f, indent=2, ensure_ascii=False)
        
        return filepath
    
    def _save_pipeline_to_persistent_dir(self, pipeline_content: str, query: str, success: bool = True) -> str:
        """
        Save pipeline to persistent directory for manual inspection.
        
        Args:
            pipeline_content: The pipeline YAML content
            query: The original query
            success: Whether the pipeline was successful
            
        Returns:
            Path to saved pipeline file
        """
        # Create filename with timestamp and query snippet
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        query_snippet = query[:50].replace(" ", "_").replace("/", "_").replace("\\", "_")
        status = "success" if success else "failed"
        filename = f"{timestamp}_{status}_{query_snippet}.yaml"
        
        filepath = os.path.join(self.pipeline_output_dir, filename)
        
        # Save pipeline with metadata
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"# Query: {query}\n")
            f.write(f"# Generated at: {datetime.now().isoformat()}\n")
            f.write(f"# Status: {status}\n")
            f.write("#" + "="*50 + "\n\n")
            f.write(pipeline_content)
        
        if self.config.verbose:
            self.logger.info(f"Saved pipeline to: {filepath}")
        
        return filepath
    
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
        initial_prompt, messages = create_initial_messages(INSTRUCTION_PROMPT, query, dataset_samples)
        
        # Track pipeline generation history
        pipeline_history = []
        
        for attempt in range(self.max_attempts):
            self.total_generation_attempts += 1
            
            if self.config.verbose:
                self.logger.info(f"Pipeline generation attempt {attempt + 1}/{self.max_attempts}")
            
            # Save initial prompt on first attempt
            if attempt == 0:
                self._save_prompt(initial_prompt, query, attempt)
            
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
                    
                    # Save failed pipeline for inspection (only on last attempt)
                    if attempt == self.max_attempts - 1:
                        self._save_pipeline_to_persistent_dir(pipeline_yaml, query, success=False)
                    
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
                
                # Save successful pipeline to persistent directory
                self._save_pipeline_to_persistent_dir(pipeline_yaml, query, success=True)
                
                # Save complete message history for successful run
                self._save_messages(messages, query, pipeline_history)
                
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
        self._save_messages(messages, query, pipeline_history)
        
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