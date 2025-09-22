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
from .utils import parse_html_to_dict, convert_txt_to_json, convert_xlsx_to_csv, convert_csv_to_json
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
        
        if self.config.verbose:
            self.logger.info(f"Initialized DocETL baseline with config: {self.config}")
            self.logger.info(f"Pipeline output directory: {self.pipeline_output_dir}")

    def _get_timestamp(self) -> str:
        """Get current timestamp in standard format."""
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def _get_char_limits(self, value: Any) -> tuple:
        """Get head and tail character limits from value object."""
        head_chars = getattr(value, 'head_chars', 2500) if hasattr(value, 'head_chars') else 2500
        tail_chars = getattr(value, 'tail_chars', 2500) if hasattr(value, 'tail_chars') else 2500
        return head_chars, tail_chars

    def _write_json(self, data: Any, filepath: str) -> None:
        """Write JSON data to file with standard formatting."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


    def _rename_pipeline_file(self, pipeline_file: str, status: str) -> None:
        """Rename pipeline file based on status (success or failed)."""
        if not os.path.exists(pipeline_file):
            return

        dir_name = os.path.dirname(pipeline_file)
        base_name = os.path.basename(pipeline_file)
        new_filename = base_name.replace("attempt", f"{status}_attempt")
        new_file = os.path.join(dir_name, new_filename)

        try:
            os.rename(pipeline_file, new_file)
            if self.config.verbose:
                self.logger.info(f"{status.capitalize()} pipeline renamed to: {new_file}")
        except Exception as e:
            if self.config.verbose:
                self.logger.warning(f"Could not rename {status} pipeline: {e}")
                self.logger.info(f"{status.capitalize()} pipeline kept at: {pipeline_file}")

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

    def _process_html_content(self, key: str, content: str) -> str:
        """Process HTML content and save as JSON."""
        timestamp = self._get_timestamp()
        # Remove .html/.htm extension from key if present to avoid duplication in filename
        key_base = key.replace('.html', '').replace('.HTML', '').replace('.htm', '').replace('.HTM', '')
        persistent_file = os.path.join(self.converted_data_dir, f"{timestamp}_{key_base}_html_to_json.json")

        parsed_html = parse_html_to_dict(content)
        self._write_json([parsed_html], persistent_file)

        if self.config.verbose:
            self.logger.info(f"Saved HTML-to-JSON conversion to: {persistent_file}")

        return persistent_file

    def _process_text_content(self, key: str, content: str, value: Any, filename: str = None) -> str:
        """Process text content and save as JSON."""
        timestamp = self._get_timestamp()
        # Remove .txt extension from key if present to avoid duplication in filename
        key_base = key.replace('.txt', '').replace('.TXT', '')
        persistent_file = os.path.join(self.converted_data_dir, f"{timestamp}_{key_base}_txt_to_json.json")

        head_chars, tail_chars = self._get_char_limits(value)
        filename = filename or f"{key}.txt"

        json_data = convert_txt_to_json(content, filename, head_chars, tail_chars)
        self._write_json([json_data], persistent_file)

        if self.config.verbose:
            self.logger.info(f"Saved TXT-to-JSON conversion to: {persistent_file}")

        return persistent_file

    def _process_json_content(self, key: str, content: Any) -> str:
        """Process JSON content and save to file."""
        timestamp = self._get_timestamp()
        persistent_file = os.path.join(self.converted_data_dir, f"{timestamp}_{key}.json")

        # Determine the data to save
        if isinstance(content, (list, dict)):
            data = content
        else:
            try:
                data = json.loads(content)
            except:
                data = [{"text": content}]

        self._write_json(data, persistent_file)
        return persistent_file

    def _process_xlsx_content(self, key: str, content: Any) -> List[str]:
        """Process XLSX content and convert to CSV."""
        # Write XLSX content to converted_data_dir first
        timestamp = self._get_timestamp()
        temp_xlsx = os.path.join(self.converted_data_dir, f"{timestamp}_{key}_temp.xlsx")
        if isinstance(content, bytes):
            with open(temp_xlsx, 'wb') as f:
                f.write(content)
        else:
            # If content is text, write as is (though XLSX should be binary)
            with open(temp_xlsx, 'w', encoding='utf-8') as f:
                f.write(content)

        # Convert to CSV
        csv_files = convert_xlsx_to_csv(temp_xlsx, self.converted_data_dir)

        if self.config.verbose:
            self.logger.info(f"Converted XLSX to {len(csv_files)} CSV file(s)")

        # Clean up temp file
        os.unlink(temp_xlsx)

        return csv_files

    def _process_csv_content(self, key: str, content: str) -> str:
        """Process CSV content and save to file."""
        timestamp = self._get_timestamp()
        persistent_file = os.path.join(self.converted_data_dir, f"{timestamp}_{key}.csv")
        with open(persistent_file, 'w', encoding='utf-8') as f:
            f.write(content)
        return persistent_file

    def _process_html_file(self, key: str, file_path: str) -> str:
        """Process HTML file and convert to JSON."""
        with open(file_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        return self._process_html_content(key, html_content)

    def _process_text_file(self, key: str, file_path: str, value: Any) -> str:
        """Process text file and convert to JSON."""
        with open(file_path, 'r', encoding='utf-8') as f:
            text_content = f.read()

        filename = os.path.basename(file_path)
        return self._process_text_content(key, text_content, value, filename)

    def _process_xlsx_file(self, file_path: str) -> List[str]:
        """Process XLSX file and convert to CSV."""
        csv_files = convert_xlsx_to_csv(file_path, self.converted_data_dir)

        if self.config.verbose:
            self.logger.info(f"Converted {file_path} to {len(csv_files)} CSV file(s)")

        return csv_files

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
                # Determine the file type
                file_type = getattr(value, 'type', 'json')  # Default to JSON

                # Process based on content or path
                if hasattr(value, 'content') and value.content is not None:
                    # Handle content directly
                    dataset_paths.extend(self._process_content(key, value.content, file_type, value))

                elif hasattr(value, 'path') and value.path:
                    # Handle file path
                    if os.path.exists(value.path):
                        dataset_paths.extend(self._process_file_path(key, value.path, file_type, value))
                    else:
                        if self.config.verbose:
                            self.logger.warning(f"File not found: {value.path}")

            except Exception as e:
                if self.config.verbose:
                    self.logger.warning(f"Failed to prepare dataset {key}: {e}")

        return dataset_paths

    def _process_content(self, key: str, content: Any, file_type: str, value: Any) -> List[str]:
        """Process content based on file type."""
        paths = []

        if file_type == 'html':
            paths.append(self._process_html_content(key, content))
        elif file_type in ['text', 'txt']:
            paths.append(self._process_text_content(key, content, value))
        elif file_type == 'json':
            paths.append(self._process_json_content(key, content))
        elif file_type in ['xlsx', 'excel']:
            paths.extend(self._process_xlsx_content(key, content))
        elif file_type == 'csv':
            paths.append(self._process_csv_content(key, content))

        return paths

    def _process_file_path(self, key: str, file_path: str, file_type: str, value: Any) -> List[str]:
        """Process file path based on file type or extension."""
        paths = []

        # Check explicit type first
        if file_type == 'html' or file_path.endswith(('.html', '.htm')):
            paths.append(self._process_html_file(key, file_path))
        elif file_type == 'text' or file_path.endswith('.txt'):
            paths.append(self._process_text_file(key, file_path, value))
        elif file_type in ['xlsx', 'excel'] or file_path.endswith(('.xlsx', '.xls', '.xlsm')):
            paths.extend(self._process_xlsx_file(file_path))
        else:
            # For other file types (JSON, CSV, etc.), use the file directly
            paths.append(file_path)

        return paths
    
    def _create_base_filename(self, query: str, suffix: str = "") -> str:
        """Create a base filename with timestamp and query snippet."""
        timestamp = self._get_timestamp()
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

        self._write_json(validation_data, filepath)
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

        self._write_json(messages_data, filepath)
        return filepath

    def _merge_datasets_to_json(self, dataset_paths: List[str]) -> str:
        """
        Merge multiple dataset files into a single JSON file as a list of dictionaries.

        Args:
            dataset_paths: List of dataset file paths

        Returns:
            Path to the merged JSON file
        """
        timestamp = self._get_timestamp()
        merged_filename = f"{timestamp}_merged_datasets.json"
        merged_filepath = os.path.join(self.converted_data_dir, merged_filename)

        merged_data = []  # Changed from dict to list

        for file_path in dataset_paths:
            try:
                # Get a clean key name from the file path (replace . with _ in extension)
                filename = os.path.basename(file_path)

                # Handle converted files: extract original filename
                if '_txt_to_json.json' in filename:
                    # Extract original name from converted filename like "20250915_213120_introduction_txt_to_json.json"
                    # Remove timestamp and conversion suffix
                    parts = filename.split('_')
                    if len(parts) >= 3:
                        # Find the part that's not timestamp (doesn't start with digit pattern)
                        original_name = None
                        for part in parts:
                            if not (part.isdigit() or (len(part) == 6 and part.isdigit())):
                                # This should be the original filename part
                                original_name = part
                                break
                        if original_name:
                            key_name = f"{original_name}_txt"
                        else:
                            # Fallback: use the part before txt_to_json
                            key_name = f"{parts[-3]}_txt" if len(parts) >= 3 else "text_txt"
                    else:
                        key_name = "text_txt"
                elif '_html_to_json.json' in filename:
                    # Similar handling for HTML files
                    parts = filename.split('_')
                    if len(parts) >= 3:
                        original_name = None
                        for part in parts:
                            if not (part.isdigit() or (len(part) == 6 and part.isdigit())):
                                original_name = part
                                break
                        if original_name:
                            key_name = f"{original_name}_html"
                        else:
                            key_name = f"{parts[-3]}_html" if len(parts) >= 3 else "html_html"
                    else:
                        key_name = "html_html"
                else:
                    # Regular file handling
                    name, ext = os.path.splitext(filename)
                    if ext:
                        key_name = f"{name}_{ext[1:]}"  # Remove the dot and add underscore
                    else:
                        key_name = name

                # Load pure content based on file type
                if file_path.endswith('.json'):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        json_content = json.load(f)

                    # Check if this is a text-to-json conversion with metadata
                    if (isinstance(json_content, list) and len(json_content) == 1 and
                        isinstance(json_content[0], dict) and 'content' in json_content[0] and
                        'file_name' in json_content[0]):
                        # Extract pure text content from converted text file
                        content = json_content[0]['content']
                    else:
                        # Regular JSON content
                        content = json_content
                elif file_path.endswith('.csv'):
                    # Convert CSV to pure list of records
                    import pandas as pd
                    df = pd.read_csv(file_path)
                    content = df.to_dict('records')
                else:
                    # For text files, just read the content
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                # Add as dictionary with filename and content fields
                merged_data.append({
                    "filename": key_name,
                    "content": content
                })

                if self.config.verbose:
                    if isinstance(content, list):
                        self.logger.info(f"Merged {filename} with {len(content)} records")
                    elif isinstance(content, str):
                        self.logger.info(f"Merged {filename} with text content ({len(content)} chars)")
                    else:
                        self.logger.info(f"Merged {filename} with data")

            except Exception as e:
                if self.config.verbose:
                    self.logger.warning(f"Failed to merge {file_path}: {e}")
                # Skip problematic files rather than failing completely
                continue

        # Save merged data
        with open(merged_filepath, 'w', encoding='utf-8') as f:
            json.dump(merged_data, f, indent=2, ensure_ascii=False)

        if self.config.verbose:
            self.logger.info(f"Created merged dataset with {len(merged_data)} file sources: {merged_filepath}")

        return merged_filepath

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
            merged_file_path = self._merge_datasets_to_json(dataset_paths)
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
            pipeline_filename = f"{self._create_base_filename(query, f'attempt{attempt}')}.yaml"
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
                self._save_messages(messages, query, pipeline_history)

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
        self._save_messages(messages, query, pipeline_history)

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
    
