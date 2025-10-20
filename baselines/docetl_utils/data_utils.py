"""
Shared dataset preprocessing utilities for DocETL baselines.

This module contains common data processing functionality used by both
docetl.py and docetl_step.py baselines to avoid code duplication.
"""

import os
import json
import pandas as pd
from typing import Any, Dict, List
from datetime import datetime
from ..utils import DataTruncator, parse_html_to_dict, convert_txt_to_json, convert_xlsx_to_csv


class DocETLDataProcessor:

    def __init__(self, converted_data_dir: str, verbose: bool = False, logger = None):
        """
        Initialize the data processor.

        Args:
            converted_data_dir: Directory to store converted data files
            verbose: Enable verbose logging
            logger: Logger instance for output
        """
        self.converted_data_dir = converted_data_dir
        self.verbose = verbose
        self.logger = logger

        # Ensure the output directory exists
        os.makedirs(converted_data_dir, exist_ok=True)

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

    def _create_base_filename(self, query: str, suffix: str = "") -> str:
        """Create a base filename with timestamp and query snippet."""
        timestamp = self._get_timestamp()
        query_snippet = query[:50].replace(" ", "_").replace("/", "_").replace("\\", "_")
        query_snippet = "".join(c for c in query_snippet if c.isalnum() or c in "_-")
        if suffix:
            return f"{timestamp}_{suffix}_{query_snippet}"
        return f"{timestamp}_{query_snippet}"

    def _process_html_content(self, key: str, content: str) -> str:
        """Process HTML content and save as JSON."""
        timestamp = self._get_timestamp()
        key_base = key.replace('.html', '').replace('.HTML', '').replace('.htm', '').replace('.HTM', '')
        persistent_file = os.path.join(self.converted_data_dir, f"{timestamp}_{key_base}_html_to_json.json")

        parsed_html = parse_html_to_dict(content)
        self._write_json([parsed_html], persistent_file)

        if self.verbose and self.logger:
            self.logger.info(f"Saved HTML-to-JSON conversion to: {persistent_file}")

        return persistent_file

    def _process_text_content(self, key: str, content: str, value: Any, filename: str = None) -> str:
        """Process text content and save as JSON."""
        timestamp = self._get_timestamp()
        key_base = key.replace('.txt', '').replace('.TXT', '')
        persistent_file = os.path.join(self.converted_data_dir, f"{timestamp}_{key_base}_txt_to_json.json")

        head_chars, tail_chars = self._get_char_limits(value)
        filename = filename or f"{key}.txt"

        json_data = convert_txt_to_json(content, filename, head_chars, tail_chars)
        self._write_json([json_data], persistent_file)

        if self.verbose and self.logger:
            self.logger.info(f"Saved TXT-to-JSON conversion to: {persistent_file}")

        return persistent_file

    def _process_json_content(self, key: str, content: Any) -> str:
        """Process JSON content and save to file."""
        timestamp = self._get_timestamp()
        persistent_file = os.path.join(self.converted_data_dir, f"{timestamp}_{key}.json")

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
        timestamp = self._get_timestamp()
        temp_xlsx = os.path.join(self.converted_data_dir, f"{timestamp}_{key}_temp.xlsx")

        if isinstance(content, bytes):
            with open(temp_xlsx, 'wb') as f:
                f.write(content)
        else:
            with open(temp_xlsx, 'w', encoding='utf-8') as f:
                f.write(content)

        csv_files = convert_xlsx_to_csv(temp_xlsx, self.converted_data_dir)

        if self.verbose and self.logger:
            self.logger.info(f"Converted XLSX to {len(csv_files)} CSV file(s)")

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

        if self.verbose and self.logger:
            self.logger.info(f"Converted {file_path} to {len(csv_files)} CSV file(s)")

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

    def prepare_dataset_files(self, context: Dict[str, Any]) -> List[str]:
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
                        if self.verbose and self.logger:
                            self.logger.warning(f"File not found: {value.path}")

            except Exception as e:
                if self.verbose and self.logger:
                    self.logger.warning(f"Failed to prepare dataset {key}: {e}")

        return dataset_paths

    def merge_datasets_to_json(self, dataset_paths: List[str]) -> str:
        """
        Merge multiple dataset files into a single JSON file.

        Args:
            dataset_paths: List of dataset file paths

        Returns:
            Path to the merged JSON file
        """
        timestamp = self._get_timestamp()
        merged_filename = f"{timestamp}_merged_datasets.json"
        merged_filepath = os.path.join(self.converted_data_dir, merged_filename)

        merged_data = []

        for file_path in dataset_paths:
            try:
                filename = os.path.basename(file_path)

                # Handle converted files: extract original filename
                if '_txt_to_json.json' in filename:
                    parts = filename.split('_')
                    if len(parts) >= 3:
                        original_name = None
                        for part in parts:
                            if not (part.isdigit() or (len(part) == 6 and part.isdigit())):
                                original_name = part
                                break
                        if original_name:
                            key_name = f"{original_name}_txt"
                        else:
                            key_name = f"{parts[-3]}_txt" if len(parts) >= 3 else "text_txt"
                    else:
                        key_name = "text_txt"
                elif '_html_to_json.json' in filename:
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
                    name, ext = os.path.splitext(filename)
                    if ext:
                        key_name = f"{name}_{ext[1:]}"
                    else:
                        key_name = name

                # Load content based on file type
                if file_path.endswith('.json'):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        json_content = json.load(f)

                    # Check if this is a text-to-json conversion with metadata
                    if (isinstance(json_content, list) and len(json_content) == 1 and
                        isinstance(json_content[0], dict) and 'content' in json_content[0] and
                        'file_name' in json_content[0]):
                        content = json_content[0]['content']
                    else:
                        content = json_content
                elif file_path.endswith('.csv'):
                    df = pd.read_csv(file_path)
                    content = df.to_dict('records')
                else:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                merged_data.append({
                    "filename": key_name,
                    "content": content
                })

                if self.verbose and self.logger:
                    if isinstance(content, list):
                        self.logger.info(f"Merged {filename} with {len(content)} records")
                    elif isinstance(content, str):
                        self.logger.info(f"Merged {filename} with text content ({len(content)} chars)")
                    else:
                        self.logger.info(f"Merged {filename} with data")

            except Exception as e:
                if self.verbose and self.logger:
                    self.logger.warning(f"Failed to merge {file_path}: {e}")
                continue

        with open(merged_filepath, 'w', encoding='utf-8') as f:
            json.dump(merged_data, f, indent=2, ensure_ascii=False)

        if self.verbose and self.logger:
            self.logger.info(f"Created merged dataset with {len(merged_data)} file sources: {merged_filepath}")

        return merged_filepath


def load_sample_data(dataset_paths: List[str], max_length: int = 1500, max_string_length: int = 200, max_plain_text_length: int = 5000, csv_sample_rows: int = 5) -> Dict[str, Any]:
    """
    Load samples from all dataset files with clean formatting.

    Args:
        dataset_paths: List of dataset file paths
        max_length: Maximum total length for each file's sample
        max_string_length: Maximum length for individual string fields
        max_plain_text_length: Maximum length for plain text content
        csv_sample_rows: Number of rows to sample from CSV files (default 5)

    Returns:
        Dictionary with {file_path: sample_data} format
    """
    # Type validation - fail fast if not a list
    if not isinstance(dataset_paths, list):
        raise TypeError(
            f"dataset_paths must be a list of strings, got {type(dataset_paths).__name__}. "
            f"Value: {dataset_paths!r}. "
            f"If you have a single path, wrap it in a list: [path]"
        )

    # Path existence validation
    for path in dataset_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset path does not exist: {path}")

    dataset_samples = {}

    # Initialize intelligent truncator
    truncator = DataTruncator(
        max_total_length=max_length,
        max_string_length=max_string_length,
        max_plain_text_length=max_plain_text_length,
        max_array_items=3,
        max_dict_keys=8
    )

    for file_path in dataset_paths:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.endswith('.json'):
                    data = json.load(f)

                    # Check if this is a merged dataset (list with filename/content dicts)
                    if (isinstance(data, list) and
                        'merged_datasets' in os.path.basename(file_path) and
                        len(data) > 0 and isinstance(data[0], dict) and
                        'filename' in data[0] and 'content' in data[0]):

                        # This is a merged dataset, sample from each source
                        sampled_data = []

                        for item in data:
                            sampled_item = {"filename": item["filename"]}
                            content = item["content"]

                            if isinstance(content, list):
                                # For list data (like CSV records), sample first N items
                                sample_size = min(csv_sample_rows, len(content))
                                sampled_item["content"] = content[:sample_size]
                            else:
                                # For other data types (like text), keep as is but truncate if too long
                                if isinstance(content, str) and len(content) > 2000:
                                    sampled_item["content"] = content[:2000] + "..."
                                else:
                                    sampled_item["content"] = content

                            sampled_data.append(sampled_item)

                        data = sampled_data
                elif file_path.endswith('.csv'):
                    import pandas as pd
                    df = pd.read_csv(file_path)

                    # For CSV files, sample more rows to show data structure
                    # Include schema (column names and types) and sample rows
                    sample_data = {
                        "schema": {col: str(df[col].dtype) for col in df.columns},
                        "shape": {"rows": len(df), "columns": len(df.columns)},
                        "sample_rows": df.head(csv_sample_rows).to_dict('records')
                    }

                    # If the DataFrame has more rows than sample_rows, add indication
                    if len(df) > csv_sample_rows:
                        sample_data["note"] = f"Showing first {csv_sample_rows} rows of {len(df)} total rows"

                    data = sample_data
                else:
                    # For other file types, try to read as text
                    content = f.read()
                    # Clean up the content and treat as plain text
                    content = ' '.join(content.split())
                    # Use plain text truncation for text files
                    truncated_content = truncator.truncate_text(content, is_plain_text=True)
                    data = [{"text": truncated_content}]

                # Apply intelligent truncation
                if file_path.endswith('.csv'):
                    # CSV files already have controlled sampling, just apply truncation to values
                    if isinstance(data, dict) and 'sample_rows' in data:
                        data['sample_rows'] = truncator.truncate_data(data['sample_rows'])
                    dataset_samples[file_path] = data
                elif isinstance(data, dict) and 'merged_datasets' in os.path.basename(file_path):
                    # Handle merged datasets - apply truncation to each file's content
                    for filename, content in data.items():
                        if isinstance(content, list):
                            data[filename] = truncator.truncate_data(content)
                        elif isinstance(content, str):
                            data[filename] = truncator.truncate_text(content, is_plain_text=True)
                    dataset_samples[file_path] = data
                elif not file_path.endswith(('.txt', '.md', '.html')):
                    truncated_data = truncator.truncate_data(data)
                    dataset_samples[file_path] = truncated_data
                else:
                    dataset_samples[file_path] = data

        except Exception as e:
            # print(f"  Warning: Could not load data from {file_path}: {e}")
            dataset_samples[file_path] = f"Error loading file: {str(e)}"

    # If only one dataset, return content directly without the file path key
    if len(dataset_samples) == 1:
        return list(dataset_samples.values())[0]

    return dataset_samples