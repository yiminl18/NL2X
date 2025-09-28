import time
from typing import Any, Dict, List, Optional, Tuple
import os
import json
import yaml
import traceback
from datetime import datetime
import logging
import re

from .base import BaselineInterface, BaselineResult
from . import register_baseline
from .utils import parse_html_to_dict, convert_txt_to_json, convert_xlsx_to_csv
from .docetl_utils.llm2pipeline_docetl import (
    load_sample_data,
    execute_single_pipeline,
    validate_pipeline_output,
    extract_yaml_from_response,
    FailedPipeline,
    llm_call_with_messages,
)
from .docetl_utils.prompt_step import (
    OPERATOR_SELECTION_PROMPT,
    OPERATOR_DETAIL_PROMPT,
    OPERATOR_DEFINITIONS
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

        # Use configuration from BaselineConfig
        self.max_attempts = self.config.max_attempts
        self.validate_answer = self.config.validate_answer

        # Control LiteLLM logging based on mode
        if not self.config.verbose:
            logging.getLogger("LiteLLM").setLevel(logging.ERROR)
            logging.getLogger("httpcore").setLevel(logging.ERROR)
            logging.getLogger("openai").setLevel(logging.ERROR)
            logging.getLogger("asyncio").setLevel(logging.ERROR)
        else:
            logging.getLogger("LiteLLM").setLevel(logging.INFO)
            logging.getLogger("httpcore").setLevel(logging.INFO)
            logging.getLogger("openai").setLevel(logging.INFO)

        # Create persistent directories for saving outputs
        base_dir = os.getcwd()
        self.output_dirs = {
            'pipeline_output_dir': os.path.join(base_dir, "generated_pipelines", "docetl_step"),
            'prompts_output_dir': os.path.join(base_dir, "generated_prompts", "docetl_step"),
            'validations_output_dir': os.path.join(base_dir, "validations", "docetl_step"),
            'messages_output_dir': os.path.join(base_dir, "messages", "docetl_step"),
            'converted_data_dir': os.path.join(base_dir, "converted_data", "docetl_step"),
            'steps_output_dir': os.path.join(base_dir, "pipeline_steps", "docetl_step")
        }

        # Set instance attributes and create directories
        for attr_name, dir_path in self.output_dirs.items():
            setattr(self, attr_name, dir_path)
            os.makedirs(dir_path, exist_ok=True)

        if self.config.verbose:
            self.logger.info(f"Initialized DocETL Step-by-Step baseline with config: {self.config}")
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

    def _confirm_pipeline_execution(self, pipeline_file: str, query: str, attempt: int) -> bool:
        """Ask user for confirmation before executing pipeline in confirm/debug mode."""
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

    # Data processing methods (reuse from original docetl.py)
    def _process_html_content(self, key: str, content: str) -> str:
        """Process HTML content and save as JSON."""
        timestamp = self._get_timestamp()
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

        if self.config.verbose:
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

    def _prepare_dataset_files(self, context: Dict[str, Any]) -> List[str]:
        """Prepare dataset files from context for DocETL pipeline."""
        dataset_paths = []

        if not context:
            return dataset_paths

        for key, value in context.items():
            try:
                file_type = getattr(value, 'type', 'json')

                if hasattr(value, 'content') and value.content is not None:
                    dataset_paths.extend(self._process_content(key, value.content, file_type, value))
                elif hasattr(value, 'path') and value.path:
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

        if file_type == 'html' or file_path.endswith(('.html', '.htm')):
            paths.append(self._process_html_file(key, file_path))
        elif file_type == 'text' or file_path.endswith('.txt'):
            paths.append(self._process_text_file(key, file_path, value))
        elif file_type in ['xlsx', 'excel'] or file_path.endswith(('.xlsx', '.xls', '.xlsm')):
            paths.extend(self._process_xlsx_file(file_path))
        else:
            paths.append(file_path)

        return paths

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

    def _merge_datasets_to_json(self, dataset_paths: List[str]) -> str:
        """Merge multiple dataset files into a single JSON file."""
        timestamp = self._get_timestamp()
        merged_filename = f"{timestamp}_merged_datasets.json"
        merged_filepath = os.path.join(self.converted_data_dir, merged_filename)

        merged_data = []

        for file_path in dataset_paths:
            try:
                filename = os.path.basename(file_path)

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

                if file_path.endswith('.json'):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        json_content = json.load(f)

                    if (isinstance(json_content, list) and len(json_content) == 1 and
                        isinstance(json_content[0], dict) and 'content' in json_content[0] and
                        'file_name' in json_content[0]):
                        content = json_content[0]['content']
                    else:
                        content = json_content
                elif file_path.endswith('.csv'):
                    import pandas as pd
                    df = pd.read_csv(file_path)
                    content = df.to_dict('records')
                else:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()

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
                continue

        with open(merged_filepath, 'w', encoding='utf-8') as f:
            json.dump(merged_data, f, indent=2, ensure_ascii=False)

        if self.config.verbose:
            self.logger.info(f"Created merged dataset with {len(merged_data)} file sources: {merged_filepath}")

        return merged_filepath

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

    def _save_step_output(self, step_name: str, content: Any, query: str, attempt: int = 0) -> str:
        """Save intermediate step output."""
        filename = f"{self._create_base_filename(query, f'attempt{attempt}_{step_name}')}.json"
        filepath = os.path.join(self.steps_output_dir, filename)

        step_data = {
            "query": query,
            "attempt": attempt,
            "step": step_name,
            "timestamp": datetime.now().isoformat(),
            "content": content
        }

        self._write_json(step_data, filepath)
        return filepath

    # Step-by-step pipeline generation methods
    def _confirm_step_execution(self, step_name: str, step_data: Any, query: str, attempt: int) -> bool:
        """
        Ask user for confirmation after each step in confirm/debug mode.

        Args:
            step_name: Name of the step just completed
            step_data: Data generated in this step
            query: The original query
            attempt: Current attempt number

        Returns:
            True if user wants to continue, False to abort
        """
        if not (self.config.confirm or self.config.debug):
            return True

        print("\n" + "="*80)
        mode_text = "[DEBUG MODE]" if self.config.debug else "[CONFIRM MODE]"
        print(f"{mode_text} Step Completed - Attempt {attempt + 1}")
        print("="*80)

        print(f"\n📋 Query:")
        print("-"*40)
        print(query)
        print("-"*40)

        print(f"\n✅ Completed Step: {step_name}")
        print("-"*40)

        # Display step-specific data
        if step_name == "Step 1: Operator Selection":
            print("Selected Operators:")
            for i, op in enumerate(step_data, 1):
                print(f"  {i}. {op['type']}: {op['purpose']}")

        elif step_name == "Step 2: Operator Framework":
            print("Created Frameworks:")
            for framework in step_data:
                print(f"  - {framework['name']} ({framework['type']})")
                print(f"    Purpose: {framework['purpose']}")

        elif step_name.startswith("Step 3-4: Operator Details"):
            print("Filled Operators:")
            for op in step_data:
                print(f"  - {op['name']} ({op['type']})")
                if 'prompt' in op and op['prompt'] != "TO_BE_GENERATED":
                    preview = op['prompt'][:100] + "..." if len(op['prompt']) > 100 else op['prompt']
                    print(f"    Prompt preview: {preview}")
                if 'output' in op and 'schema' in op['output']:
                    print(f"    Output schema: {list(op['output']['schema'].keys()) if op['output']['schema'] else 'empty'}")

        elif step_name == "Step 5: Pipeline Connection":
            print("Final Pipeline Preview (first 30 lines):")
            lines = step_data.split('\n')
            for i, line in enumerate(lines[:30]):
                print(f"  {line}")
            if len(lines) > 30:
                print(f"  ... ({len(lines) - 30} more lines)")

        print("-"*40)

        print(f"\n⚠️  Continue to next step? (Y/n): ", end="")
        user_input = input().strip().lower()

        if user_input and user_input != 'y':
            print("❌ Pipeline generation aborted by user")
            return False

        print("✅ Proceeding to next step...")
        return True

    def _select_operators(self, query: str, dataset_samples: Dict[str, Any], attempt: int = 0) -> List[Dict[str, str]]:
        """
        Step 1: Select operators needed for the pipeline.

        Returns a list of dictionaries with 'type' and 'purpose' for each operator.
        """
        prompt = OPERATOR_SELECTION_PROMPT.format(
            query=query,
            dataset_samples=json.dumps(dataset_samples, indent=2)[:2000],
            operator_definitions=OPERATOR_DEFINITIONS
        )

        messages = [{"role": "user", "content": prompt}]
        response = llm_call_with_messages(messages)

        # Parse the response to extract operators
        operators = []
        lines = response.strip().split('\n')

        for line in lines:
            # Look for patterns like "- map: extract information from each document"
            match = re.match(r'^[-*]\s*(\w+):\s*(.+)$', line.strip())
            if match:
                op_type = match.group(1).strip()
                purpose = match.group(2).strip()
                operators.append({
                    'type': op_type,
                    'purpose': purpose
                })

        # If no operators found in expected format, try to extract from text
        if not operators:
            # Look for operator names mentioned in the text
            operator_types = ['map', 'filter', 'reduce', 'resolve', 'rank', 'extract',
                            'cluster', 'split', 'gather', 'unnest', 'sample', 'topk',
                            'code_map', 'code_filter']

            for op_type in operator_types:
                if op_type in response.lower():
                    # Find the context around the operator mention
                    pattern = rf'\b{op_type}\b[^.]*'
                    matches = re.findall(pattern, response, re.IGNORECASE)
                    for match in matches:
                        operators.append({
                            'type': op_type,
                            'purpose': match.strip()
                        })
                        break  # Only take the first mention of each operator

        # Always add code_map at the end for final transformation
        if not any(op['type'] == 'code_map' for op in operators):
            operators.append({
                'type': 'code_map',
                'purpose': 'Transform final output to result format'
            })

        if self.config.verbose:
            self.logger.info(f"Selected {len(operators)} operators: {[op['type'] for op in operators]}")

        # Confirm this step in confirm/debug mode
        if not self._confirm_step_execution("Step 1: Operator Selection", operators, query, attempt):
            raise ValueError("User aborted pipeline generation at operator selection step")

        return operators

    def _create_operator_framework(self, operators: List[Dict[str, str]], query: str, attempt: int = 0) -> List[Dict[str, Any]]:
        """
        Step 2: Create draft framework for each operator.

        Returns a list of operator skeletons with placeholders.
        """
        frameworks = []

        for i, op in enumerate(operators):
            framework = {
                'name': f"op_{i}_{op['type']}",
                'type': op['type'],
                'purpose': op['purpose']
            }

            # Add type-specific placeholders
            if op['type'] in ['map', 'filter', 'extract']:
                framework['prompt'] = "TO_BE_GENERATED"
                framework['output'] = {'schema': {}}

            elif op['type'] == 'reduce':
                framework['reduce_key'] = "TO_BE_GENERATED"
                framework['prompt'] = "TO_BE_GENERATED"
                framework['output'] = {'schema': {}}

            elif op['type'] == 'resolve':
                framework['optimize'] = True
                framework['comparison_prompt'] = "TO_BE_GENERATED"
                framework['resolution_prompt'] = "TO_BE_GENERATED"
                framework['output'] = {'schema': {}}

            elif op['type'] == 'rank':
                framework['prompt'] = "TO_BE_GENERATED"
                framework['input_keys'] = []
                framework['direction'] = "desc"

            elif op['type'] == 'split':
                framework['split_key'] = "TO_BE_GENERATED"
                framework['method'] = "token_count"
                framework['method_kwargs'] = {}

            elif op['type'] == 'gather':
                framework['content_key'] = "TO_BE_GENERATED"
                framework['doc_id_key'] = "TO_BE_GENERATED"
                framework['order_key'] = "TO_BE_GENERATED"

            elif op['type'] == 'unnest':
                framework['unnest_key'] = "TO_BE_GENERATED"

            elif op['type'] == 'code_map':
                framework['code'] = "TO_BE_GENERATED"

            elif op['type'] == 'code_filter':
                framework['code'] = "TO_BE_GENERATED"

            elif op['type'] == 'cluster':
                framework['embedding_keys'] = []
                framework['output_key'] = "TO_BE_GENERATED"

            elif op['type'] == 'topk':
                framework['method'] = "embedding"
                framework['k'] = 5
                framework['keys'] = []
                framework['query'] = "TO_BE_GENERATED"

            frameworks.append(framework)

        if self.config.verbose:
            self.logger.info(f"Created framework for {len(frameworks)} operators")

        # Confirm this step in confirm/debug mode
        if not self._confirm_step_execution("Step 2: Operator Framework", frameworks, query, attempt):
            raise ValueError("User aborted pipeline generation at framework creation step")

        return frameworks

    def _generate_operator_details(self, operator: Dict[str, Any], query: str,
                                  dataset_samples: Dict[str, Any],
                                  previous_operators: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Step 3: Generate details for a single operator.

        Returns the operator with filled-in details.
        """
        # Get operator-specific example from OPERATOR_DEFINITIONS
        op_type = operator['type']
        op_examples = ""

        # Extract example for this operator type from the definitions
        if op_type in OPERATOR_DEFINITIONS:
            op_examples = OPERATOR_DEFINITIONS

        prompt = OPERATOR_DETAIL_PROMPT.format(
            operator_type=op_type,
            operator_purpose=operator['purpose'],
            query=query,
            dataset_samples=json.dumps(dataset_samples, indent=2)[:1500],
            previous_operators=json.dumps(previous_operators, indent=2) if previous_operators else "None",
            operator_examples=op_examples,
            operator_framework=json.dumps(operator, indent=2)
        )

        messages = [{"role": "user", "content": prompt}]
        response = llm_call_with_messages(messages)

        # Extract YAML from response
        yaml_content = extract_yaml_from_response(response)
        if not yaml_content:
            # If no YAML found, return the original operator
            return operator

        try:
            # Parse the YAML to get the filled operator
            filled_operator = yaml.safe_load(yaml_content)

            # Merge with original operator to preserve structure
            for key, value in filled_operator.items():
                if value != "TO_BE_GENERATED":
                    operator[key] = value

            return operator

        except Exception as e:
            if self.config.verbose:
                self.logger.warning(f"Failed to parse operator details: {e}")
            return operator

    def _fill_operator_draft(self, frameworks: List[Dict[str, Any]], query: str,
                            dataset_samples: Dict[str, Any], attempt: int = 0) -> List[Dict[str, Any]]:
        """
        Step 4: Fill all operator drafts with generated details.

        Returns list of fully specified operators.
        """
        filled_operators = []

        for i, framework in enumerate(frameworks):
            if self.config.verbose:
                self.logger.info(f"Generating details for operator {i+1}/{len(frameworks)}: {framework['type']}")

            # Generate details for this operator
            filled_op = self._generate_operator_details(
                framework,
                query,
                dataset_samples,
                filled_operators  # Pass previously filled operators for context
            )

            # Clean up the operator name
            filled_op['name'] = f"{framework['type']}_{i}"

            filled_operators.append(filled_op)

        # Confirm this step in confirm/debug mode
        if not self._confirm_step_execution("Step 3-4: Operator Details", filled_operators, query, attempt):
            raise ValueError("User aborted pipeline generation at operator detail generation step")

        return filled_operators

    def _connect_pipeline(self, operators: List[Dict[str, Any]], dataset_paths: List[str],
                         query: str, attempt: int = 0) -> str:
        """
        Step 5: Connect operators and create final pipeline YAML.

        Returns the complete pipeline YAML string.
        """
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
                dataset_path = dataset_paths[0] if len(dataset_paths) == 1 else self._merge_datasets_to_json(dataset_paths)
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
            # No dataset paths provided
            pipeline['datasets']['input_data'] = {
                'type': 'file',
                'path': 'data.json'
            }
            input_name = 'input_data'

        # Set up pipeline steps
        pipeline['pipeline']['steps'] = [{
            'name': 'process_data',
            'input': input_name,
            'operations': [op['name'] for op in operators]
        }]

        # Set up output
        timestamp = self._get_timestamp()
        output_filename = f"output_{timestamp}.json"
        pipeline['pipeline']['output'] = {
            'type': 'file',
            'path': os.path.join(self.pipeline_output_dir, output_filename),
            'intermediate_dir': os.path.join(self.pipeline_output_dir, 'intermediates')
        }

        # Convert to YAML
        pipeline_yaml = yaml.dump(pipeline, default_flow_style=False, sort_keys=False)

        # Confirm this step in confirm/debug mode
        if not self._confirm_step_execution("Step 5: Pipeline Connection", pipeline_yaml, query, attempt):
            raise ValueError("User aborted pipeline generation at pipeline connection step")

        return pipeline_yaml

    def _generate_pipeline_step_by_step(self, query: str, dataset_paths: List[str],
                                       attempt: int = 0) -> Tuple[bool, str, str]:
        """
        Main orchestration function for step-by-step pipeline generation.

        Returns (success, pipeline_yaml, error_message)
        """
        try:
            # Load dataset samples
            dataset_samples = load_sample_data(dataset_paths, max_length=1500, max_string_length=200)

            if self.config.verbose:
                self.logger.info(f"Step-by-step pipeline generation - Attempt {attempt + 1}")

            # Step 1: Select operators
            if self.config.verbose:
                self.logger.info("Step 1: Selecting operators...")
            operators = self._select_operators(query, dataset_samples, attempt)
            self._save_step_output('operators_selected', operators, query, attempt)

            if not operators:
                raise ValueError("No operators selected for the pipeline")

            # Step 2: Create operator framework
            if self.config.verbose:
                self.logger.info("Step 2: Creating operator framework...")
            frameworks = self._create_operator_framework(operators, query, attempt)
            self._save_step_output('operator_frameworks', frameworks, query, attempt)

            # Step 3 & 4: Generate and fill operator details
            if self.config.verbose:
                self.logger.info("Step 3-4: Generating operator details...")
            filled_operators = self._fill_operator_draft(frameworks, query, dataset_samples, attempt)
            self._save_step_output('filled_operators', filled_operators, query, attempt)

            # Step 5: Connect pipeline
            if self.config.verbose:
                self.logger.info("Step 5: Connecting pipeline...")
            pipeline_yaml = self._connect_pipeline(filled_operators, dataset_paths, query, attempt)
            self._save_step_output('final_pipeline', pipeline_yaml, query, attempt)

            if self.config.verbose:
                self.logger.info("Pipeline generation completed successfully")

            return True, pipeline_yaml, None

        except Exception as e:
            error_msg = f"Step-by-step pipeline generation failed: {str(e)}\n{traceback.format_exc()}"
            if self.config.verbose:
                self.logger.error(error_msg)
            return False, None, error_msg

    def _generate_and_execute_pipeline(self, query: str, dataset_paths: List[str]) -> tuple:
        """
        Generate and execute DocETL pipeline using step-by-step approach.
        """
        start_time = time.time()

        if self.config.verbose:
            self.logger.info(f"Processing {len(dataset_paths)} dataset files:")
            for path in dataset_paths:
                self.logger.info(f"  - {path}")

        # Handle multiple datasets: merge them into a single JSON file
        if len(dataset_paths) > 1:
            merged_file_path = self._merge_datasets_to_json(dataset_paths)
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
            pipeline_filename = f"{self._create_base_filename(query, f'attempt{attempt}')}.yaml"
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
                    f.write(f"# Query: {query}\n")
                    f.write(f"# Attempt: {attempt}\n")
                    f.write(f"# Generated at: {datetime.now().isoformat()}\n")
                    f.write(f"# Method: Step-by-Step Generation\n")
                    f.write("#" + "="*50 + "\n\n")
                    f.write(pipeline_yaml)

                # Confirm pipeline execution in debug/confirm mode
                if not self._confirm_pipeline_execution(pipeline_file, query, attempt):
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
                        pipeline_file, query, final_dataset_paths, llm_call_with_messages
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
                if self.config.verbose:
                    self.logger.warning(error_msg)

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
            dataset_paths = self._prepare_dataset_files(context)

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
            if self.config.verbose:
                self.logger.error(traceback.format_exc())

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