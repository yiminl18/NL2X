import time
from typing import Any, Dict, List, Optional
import os
import json
import tempfile
import traceback
import shutil
import re
from datetime import datetime
from bs4 import BeautifulSoup
import pandas as pd

from .base import BaselineInterface, BaselineResult
from . import register_baseline
from .utils import convert_xlsx_to_csv
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

        # Create persistent directories for saving outputs (matching DocETL structure)
        base_dir = os.getcwd()
        self.output_dirs = {
            'pipeline_output_dir': os.path.join(base_dir, "generated_pipelines", "lotus"),
            'prompts_output_dir': os.path.join(base_dir, "generated_prompts", "lotus"),
            'validations_output_dir': os.path.join(base_dir, "validations", "lotus"),
            'messages_output_dir': os.path.join(base_dir, "messages", "lotus"),
            'converted_data_dir': os.path.join(base_dir, "converted_data", "lotus")
        }

        # Set instance attributes and create directories
        for attr_name, dir_path in self.output_dirs.items():
            setattr(self, attr_name, dir_path)
            os.makedirs(dir_path, exist_ok=True)

        if self.config.verbose:
            self.logger.info(f"Initialized LOTUS baseline with config: {self.config}")
            self.logger.info(f"Temporary directory: {self.temp_dir}")
            self.logger.info(f"Pipeline output directory: {self.pipeline_output_dir}")

    def _get_timestamp(self) -> str:
        """Get current timestamp in standard format."""
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def _get_char_limits(self, value: Any) -> tuple:
        """Get head and tail character limits from value object."""
        head_chars = getattr(value, 'head_chars', 2000) if hasattr(value, 'head_chars') else 2000
        tail_chars = getattr(value, 'tail_chars', 2000) if hasattr(value, 'tail_chars') else 2000
        return head_chars, tail_chars

    def _write_json(self, data: Any, filepath: str) -> None:
        """Write JSON data to file with standard formatting."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _save_json_to_both_dirs(self, data: Any, persistent_file: str, temp_file: str) -> None:
        """Save JSON data to both persistent and temp directories."""
        self._write_json(data, persistent_file)
        self._write_json(data, temp_file)

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

    def _extract_text_from_html(self, html_content: str) -> str:
        """
        Extract plain text from HTML content.
        
        Args:
            html_content: HTML string content
            
        Returns:
            Plain text extracted from HTML
        """
        soup = BeautifulSoup(html_content, 'html.parser')
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get text with proper spacing
        text = soup.get_text(separator=' ', strip=True)
        
        # Clean up extra whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return text

    def _write_single_row_csv(self, file_path: str, file_name: str, content: str) -> str:
        """Write a single row CSV with (file_name, content) format."""
        df = pd.DataFrame([{
            'file_name': file_name,
            'content': content
        }])
        df.to_csv(file_path, index=False, encoding='utf-8')
        return file_path

    def _write_multi_row_csv(self, file_path: str, data: list) -> str:
        """Write a multi-row CSV from a list of dictionaries."""
        df = pd.DataFrame(data)
        df.to_csv(file_path, index=False, encoding='utf-8')
        return file_path

    def _clean_csv_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean CSV DataFrame by removing unnamed columns and fixing column names."""
        # Get original columns
        original_columns = list(df.columns)

        # Find columns that are unnamed or empty
        columns_to_drop = []
        new_column_names = []

        for i, col in enumerate(original_columns):
            if (isinstance(col, str) and col.startswith('Unnamed:')) or pd.isna(col):
                # Check if this column has any data
                if df.iloc[:, i].dropna().empty or df.iloc[:, i].dropna().astype(str).str.strip().eq('').all():
                    # Column is completely empty, mark for removal
                    columns_to_drop.append(col)
                else:
                    # Column has data but unnamed, give it a meaningful name
                    new_column_names.append(f'column_{i}')
            else:
                new_column_names.append(col)

        # Drop empty columns
        if columns_to_drop:
            df = df.drop(columns=columns_to_drop)
            if self.config.verbose:
                self.logger.info(f"Removed {len(columns_to_drop)} empty unnamed columns")

        # Rename remaining columns
        if len(new_column_names) == len(df.columns):
            df.columns = new_column_names

        return df

    def _clean_csv_file(self, csv_path: str) -> str:
        """Clean a CSV file by removing unnamed columns and save the cleaned version."""
        try:
            # Read the CSV
            df = pd.read_csv(csv_path)

            # Clean columns
            df_cleaned = self._clean_csv_columns(df)

            # Save cleaned version back to the same file
            df_cleaned.to_csv(csv_path, index=False, encoding='utf-8')

            if self.config.verbose:
                self.logger.info(f"Cleaned CSV: {csv_path} ({df.shape[1]} -> {df_cleaned.shape[1]} columns)")

            return csv_path

        except Exception as e:
            if self.config.verbose:
                self.logger.warning(f"Failed to clean CSV {csv_path}: {e}")
            return csv_path

    def _process_html_content(self, key: str, content: str) -> str:
        """Process HTML content and save as single-row CSV with (file_name, content)."""
        # Extract plain text from HTML
        text_content = self._extract_text_from_html(content)

        # Create CSV file path in persistent directory
        timestamp = self._get_timestamp()
        csv_filename = f"{timestamp}_{key}.csv"
        csv_file = os.path.join(self.converted_data_dir, csv_filename)

        # Write single row CSV
        self._write_single_row_csv(csv_file, key, text_content)

        if self.config.verbose:
            self.logger.info(f"Saved HTML as single-row CSV: {csv_file}")

        return csv_file

    def _process_text_content(self, key: str, content: str, value: Any, filename: str = None) -> str:
        """Process text content and save as single-row CSV with (file_name, content)."""
        # Apply character limits if specified
        head_chars, tail_chars = self._get_char_limits(value)

        # Truncate content if needed
        original_length = len(content)
        if head_chars is not None and tail_chars is not None:
            total_chars = head_chars + tail_chars
            if original_length > total_chars:
                content = content[:head_chars] + "..." + content[-tail_chars:]
        elif head_chars is not None:
            if original_length > head_chars:
                content = content[:head_chars] + "..."
        elif tail_chars is not None:
            if original_length > tail_chars:
                content = "..." + content[-tail_chars:]

        # Create CSV file path in persistent directory
        timestamp = self._get_timestamp()
        csv_filename = f"{timestamp}_{key}.csv"
        csv_file = os.path.join(self.converted_data_dir, csv_filename)

        # Write single row CSV
        filename = filename or key
        self._write_single_row_csv(csv_file, filename, content)

        if self.config.verbose:
            self.logger.info(f"Saved TXT as single-row CSV: {csv_file}")

        return csv_file

    def _process_json_content(self, key: str, content: Any) -> str:
        """Process JSON content and save as CSV (multi-row for lists, single-row otherwise)."""
        timestamp = self._get_timestamp()
        csv_filename = f"{timestamp}_{key}.csv"
        csv_file = os.path.join(self.converted_data_dir, csv_filename)

        # Parse content if it's a string
        if isinstance(content, str):
            try:
                data = json.loads(content)
            except:
                # If parsing fails, treat as plain text
                self._write_single_row_csv(csv_file, key, content)
                return csv_file
        else:
            data = content

        # Check if data is a list
        if isinstance(data, list):
            # Multi-row CSV for list
            if len(data) > 0 and isinstance(data[0], dict):
                # List of dictionaries - write directly as multi-row CSV
                self._write_multi_row_csv(csv_file, data)
            else:
                # List of non-dict items - convert to list of dicts
                converted_data = [{"value": item} for item in data]
                self._write_multi_row_csv(csv_file, converted_data)
        else:
            # Single-row CSV for non-list (dict or other)
            if isinstance(data, dict):
                # Convert dict to JSON string for content field
                json_str = json.dumps(data, ensure_ascii=False)
            else:
                # Other types, convert to string
                json_str = str(data)
            self._write_single_row_csv(csv_file, key, json_str)

        if self.config.verbose:
            self.logger.info(f"Saved JSON as CSV: {csv_file}")

        return csv_file

    def _split_xlsx_by_sheets(self, xlsx_path: str, output_dir: str) -> List[str]:
        """
        Split XLSX file by sheets and convert each to cleaned CSV.
        Following the user's requested flow: XLSX → single sheet XLSX → CSV → clean
        """
        csv_paths = []

        try:
            # Read Excel file to get sheet names
            xls = pd.ExcelFile(xlsx_path)
            base_name = os.path.splitext(os.path.basename(xlsx_path))[0]

            for sheet_name in xls.sheet_names:
                # Read specific sheet
                df = pd.read_excel(xlsx_path, sheet_name=sheet_name)

                # Clean the DataFrame
                df_cleaned = self._clean_csv_columns(df)

                # Create CSV filename
                safe_sheet_name = re.sub(r'[^\w\s-]', '_', sheet_name)
                csv_filename = f"{base_name}_{safe_sheet_name}.csv"
                csv_path = os.path.join(output_dir, csv_filename)

                # Save cleaned CSV
                df_cleaned.to_csv(csv_path, index=False, encoding='utf-8')
                csv_paths.append(csv_path)

                if self.config.verbose:
                    self.logger.info(f"Created cleaned CSV: {csv_filename} ({df.shape[1]} -> {df_cleaned.shape[1]} columns)")

        except Exception as e:
            if self.config.verbose:
                self.logger.warning(f"Failed to split XLSX by sheets: {e}")

        return csv_paths

    def _process_xlsx_content(self, key: str, content: Any) -> List[str]:
        """Process XLSX content and convert to cleaned CSV files by sheets."""
        # Write XLSX content to temp file first
        temp_xlsx = os.path.join(self.temp_dir, f"{key}_temp.xlsx")
        if isinstance(content, bytes):
            with open(temp_xlsx, 'wb') as f:
                f.write(content)
        else:
            # If content is text, write as is (though XLSX should be binary)
            with open(temp_xlsx, 'w', encoding='utf-8') as f:
                f.write(content)

        # Use new split and clean method
        csv_files = self._split_xlsx_by_sheets(temp_xlsx, self.converted_data_dir)

        if self.config.verbose:
            self.logger.info(f"Converted XLSX to {len(csv_files)} cleaned CSV file(s)")

        # Clean up temp file
        os.unlink(temp_xlsx)

        return csv_files

    def _process_csv_content(self, key: str, content: str) -> str:
        """Process CSV content and save to file."""
        timestamp = self._get_timestamp()
        csv_filename = f"{timestamp}_{key}.csv"
        csv_file = os.path.join(self.converted_data_dir, csv_filename)
        with open(csv_file, 'w', encoding='utf-8') as f:
            f.write(content)
        return csv_file

    def _process_html_file(self, key: str, file_path: str) -> str:
        """Process HTML file and convert to JSON, then CSV."""
        with open(file_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        return self._process_html_content(key, html_content)

    def _process_text_file(self, key: str, file_path: str, value: Any) -> str:
        """Process text file and convert to JSON, then CSV."""
        with open(file_path, 'r', encoding='utf-8') as f:
            text_content = f.read()

        filename = os.path.basename(file_path)
        return self._process_text_content(key, text_content, value, filename)

    def _process_xlsx_file(self, file_path: str) -> List[str]:
        """Process XLSX file and convert to cleaned CSV files by sheets."""
        csv_files = self._split_xlsx_by_sheets(file_path, self.converted_data_dir)

        if self.config.verbose:
            self.logger.info(f"Converted {file_path} to {len(csv_files)} cleaned CSV file(s)")

        return csv_files

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
        elif file_path.endswith('.json'):
            # Convert JSON file to CSV for LOTUS compatibility
            csv_path = convert_json_to_csv(file_path)
            paths.append(csv_path)
        elif file_path.endswith('.csv'):
            paths.append(file_path)
        else:
            # For other file types, treat as text
            paths.append(self._process_text_file(key, file_path, value))

        return paths

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
    
    def _create_base_filename(self, query: str, suffix: str = "") -> str:
        """Create a base filename with timestamp and query snippet."""
        timestamp = self._get_timestamp()
        # Clean query: remove newlines, replace special chars, keep only safe characters
        query_snippet = query.strip().replace("\n", " ").replace("\r", " ")[:50]
        query_snippet = query_snippet.replace(" ", "_").replace("/", "_").replace("\\", "_")
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
                    "pipeline_code": failed.pipeline_code
                })
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(messages_data, f, indent=2, ensure_ascii=False)
        
        return filepath
    

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

        if self.config.verbose:
            self.logger.info(f"Processing {len(dataset_paths)} dataset files:")
            for path in dataset_paths:
                self.logger.info(f"  - {path}")

        # Load dataset samples (this handles CSV files)
        # No merging - keep files separate for LOTUS to process independently
        dataset_samples = load_sample_data(dataset_paths, max_length=1500, max_string_length=200)

        if self.config.verbose:
            self.logger.info(f"Loaded samples from {len(dataset_samples)} files")
        
        # Initialize conversation with initial messages
        initial_prompt, messages = create_initial_messages(INSTRUCTION_PROMPT, query, dataset_samples)
        
        # Track pipeline generation history
        pipeline_history = []
        
        for attempt in range(self.max_attempts):
            self.total_generation_attempts += 1

            # Create pipeline filename with attempt number (like DocETL)
            pipeline_filename = f"{self._create_base_filename(query, f'attempt{attempt}')}.py"
            pipeline_file = os.path.join(self.pipeline_output_dir, pipeline_filename)

            if self.config.verbose:
                self.logger.info(f"Pipeline generation attempt {attempt + 1}/{self.max_attempts}")

            # Save initial prompt on first attempt
            if attempt == 0:
                self._save_prompt(initial_prompt, query, attempt)
            
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
                    # Create output file in project directory with timestamp
                    timestamp = self._get_timestamp()
                    output_filename = f"{timestamp}_attempt{attempt}_output.json"
                    output_file = os.path.join(self.pipeline_output_dir, output_filename)
                    pipeline_code = create_wrapped_pipeline_code(pipeline_code, dataset_paths, output_file)
                
                # Save pipeline to file with metadata (like DocETL)
                with open(pipeline_file, "w") as f:
                    # Handle multi-line queries by commenting each line properly
                    query_lines = query.strip().split('\n')
                    f.write(f"# Query: {query_lines[0]}\n")
                    for line in query_lines[1:]:
                        # Ensure each line is properly commented, handle empty lines
                        if line.strip():
                            f.write(f"# {line}\n")
                        else:
                            f.write("#\n")
                    f.write(f"# Attempt: {attempt}\n")
                    f.write(f"# Generated at: {datetime.now().isoformat()}\n")
                    f.write("#" + "="*50 + "\n\n")
                    f.write(pipeline_code)
                os.chmod(pipeline_file, 0o755)  # Make executable

                # Confirm pipeline execution in debug/confirm mode
                if not self._confirm_pipeline_execution(pipeline_file, query, attempt):
                    # User declined to execute - skip this attempt
                    continue

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
                # Save complete message history for successful run
                self._save_messages(messages, query, pipeline_history)
                
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
        
        # Save message history for failed run
        self._save_messages(messages, query, pipeline_history)
        
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