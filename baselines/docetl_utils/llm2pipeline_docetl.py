#!/usr/bin/env python3
"""
LLM2Pipeline4DocETL Test Script
Based on medical_analysis.py, uses generate_pipeline.py to create pipeline from natural language query
"""

import os
import sys
import json
from dotenv import load_dotenv
import yaml
import subprocess
from typing import List, Dict, Any, Optional, Tuple, NamedTuple
import traceback
import time

from docetl.runner import DSLRunner

# Data structure for pipeline-error pairs
class FailedPipeline(NamedTuple):
    """Store a failed pipeline with its error information"""
    pipeline_yaml: str
    error_type: str  # 'execution' or 'validation'
    error_message: str

# Add parent directory to path to import azuregpt4o
sys.path.append('/Users/chiyuh/Workspace/NL2X/model')
from azuregpt4o import gpt_4o_azure
from .prompt import (
    INSTRUCTION_PROMPT,
    PIPELINE_GENERATION_PROMPT_TEMPLATE,
    VALIDATION_PROMPT_TEMPLATE,
    OUTPUT_VALIDATION_PROMPT_TEMPLATE
)
from docetl.api import Pipeline

# Import smart truncation utility
from ..utils import DataTruncator

def create_sample_medical_transcripts() -> str:
    """Create sample medical transcripts dataset for testing"""
    transcripts = [
        {
            "id": 1,
            "patient_id": "P001",
            "date": "2024-01-15",
            "doctor": "Dr. Smith",
            "src": """
Doctor: Good morning! How are you feeling today?
Patient: I've been having trouble sleeping and feeling anxious lately.
Doctor: I see. Have you been taking the Zoloft I prescribed last month?
Patient: Yes, 50mg daily as you said. But I'm experiencing some nausea.
Doctor: That's a common side effect of Zoloft initially. It usually subsides after 2-3 weeks. 
Let's also add Trazodone 50mg at bedtime to help with your sleep issues.
Patient: Will there be any interactions?
Doctor: No significant interactions between Zoloft and Trazodone. Trazodone will help with both sleep and has some anxiolytic effects.
Patient: Thank you, doctor.
"""
        },
        {
            "id": 2,
            "patient_id": "P002", 
            "date": "2024-01-20",
            "doctor": "Dr. Johnson",
            "src": """
Doctor: Hello! What brings you in today?
Patient: I've been having severe back pain for weeks now.
Doctor: On a scale of 1-10, how would you rate your pain?
Patient: About 7 or 8, especially in the mornings.
Doctor: I'm prescribing Naproxen 500mg twice daily with food. This is an anti-inflammatory that should help.
Patient: I've heard that can upset your stomach?
Doctor: Yes, that's why you should take it with food. I'm also prescribing Omeprazole 20mg daily to protect your stomach.
Patient: How long should I take these?
Doctor: Let's start with 2 weeks and reassess. The Naproxen is for pain and inflammation, while Omeprazole prevents stomach irritation.
"""
        },
        {
            "id": 3,
            "patient_id": "P003",
            "date": "2024-01-25", 
            "doctor": "Dr. Chen",
            "src": """
Doctor: Your blood pressure is still elevated. How are you tolerating the Lisinopril?
Patient: I've had a persistent dry cough since starting it.
Doctor: That's a known side effect of ACE inhibitors like Lisinopril. Let's switch you to Losartan 50mg daily.
Patient: Will that work the same way?
Doctor: Losartan is an ARB, similar effectiveness but without the cough side effect. 
I'm also adding Hydrochlorothiazide 12.5mg for better blood pressure control.
Patient: Any side effects I should watch for?
Doctor: Hydrochlorothiazide is a diuretic, so you may urinate more frequently. Stay hydrated and watch your potassium levels.
We'll check your labs in a month.
"""
        },
        {
            "id": 4,
            "patient_id": "P004",
            "date": "2024-02-01",
            "doctor": "Dr. Martinez",
            "src": """
Doctor: How's your diabetes management going?
Patient: My blood sugar has been running high, around 180-200.
Doctor: Let's increase your Metformin to 1000mg twice daily. Continue taking it with meals.
Patient: I sometimes get stomach upset with it.
Doctor: That's common with Metformin. Taking it with food helps. The extended-release version might be better tolerated.
I'm also starting you on Glimepiride 2mg once daily to help lower your blood sugar further.
Patient: Will I need insulin?
Doctor: Not yet. Glimepiride helps your pancreas produce more insulin. Monitor for hypoglycemia symptoms like shakiness or sweating.
"""
        },
        {
            "id": 5,
            "patient_id": "P005",
            "date": "2024-02-05",
            "doctor": "Dr. Wilson",
            "src": """
Doctor: Your cholesterol levels are concerning. Total cholesterol is 280.
Patient: Is that very high?
Doctor: Yes, we need to address this. I'm prescribing Atorvastatin 40mg at bedtime.
Patient: I've heard statins can cause muscle pain?
Doctor: That's a possible side effect in about 10% of patients. Let me know if you experience any muscle aches.
I'm also recommending Omega-3 supplements, 1000mg twice daily, to help with your triglycerides.
Patient: Should I change my diet too?
Doctor: Absolutely. Reduce saturated fats and increase fiber. The Atorvastatin works by reducing cholesterol production in your liver.
"""
        },
        {
            "id": 6,
            "patient_id": "P001",
            "date": "2024-02-10",
            "doctor": "Dr. Smith",
            "src": """
Doctor: How are you doing with the Zoloft and Trazodone?
Patient: Much better! Sleeping well now, but I'm feeling a bit drowsy during the day.
Doctor: That's the Trazodone. We can reduce it to 25mg since your sleep has improved.
Patient: The Zoloft nausea is gone now.
Doctor: Good! That's typical. Let's continue the Zoloft at 50mg. Your anxiety seems well-controlled.
Patient: Yes, I'm feeling much calmer overall.
Doctor: Excellent. Continue both medications and we'll reassess in a month.
"""
        }
    ]
    
    # Save to JSON file
    filename = "test_medical_transcripts.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(transcripts, f, indent=2, ensure_ascii=False)
    
    # Dataset created: {filename}
    return filename

def create_initial_messages(instruction_prompt: str, query: str, dataset_samples: Dict[str, Any]) -> tuple:
    """
    Create initial message list for pipeline generation using PIPELINE_GENERATION_PROMPT.
    
    Args:
        instruction_prompt: Base instruction for pipeline generation (INSTRUCTION_PROMPT)
        query: Natural language query
        dataset_samples: Dictionary with {file_path: sample_data} format
        
    Returns:
        List of message dictionaries
    """
    # Format dataset profiles string
    profiles_str = ""
    for file_path, sample_data in dataset_samples.items():
        profiles_str += f"File: {file_path}\n"
        profiles_str += f"Sample Data:\n{json.dumps(sample_data, indent=2)}\n\n"
    
    # Use PIPELINE_GENERATION_PROMPT template
    user_content = PIPELINE_GENERATION_PROMPT_TEMPLATE.safe_substitute(
        query=query,
        profiles_str=profiles_str
    )
    # Return messages with the initial prompt
    initial_prompt = instruction_prompt + user_content
    
    return initial_prompt, [
        {
            "role": "user",
            "content": instruction_prompt + user_content
        }
    ]

def add_error_message(messages: List[Dict[str, str]], error_type: str, error_msg: str, 
                      previous_pipeline: str = None, sample_output: str = None) -> List[Dict[str, str]]:
    """
    Add error feedback message to conversation history.
    
    Args:
        messages: Current message list
        error_type: Type of error ('execution' or 'validation')
        error_msg: Error message details
        previous_pipeline: Previous pipeline YAML that failed
        sample_output: Sample output from the failed pipeline (for validation errors)
        
    Returns:
        Updated message list
    """
    if error_type == "execution":
        feedback = f"The pipeline failed to execute with the following error:\n\n{error_msg}\n\nPlease fix the error and regenerate the pipeline."
    else:  # validation
        feedback = f"The pipeline executed but failed validation:\n\n{error_msg}"
        
        if sample_output:
            feedback += f"\n\nSample output from the pipeline:\n{sample_output}"
        
        if previous_pipeline:
            feedback += f"\n\nPrevious pipeline that failed:\n```yaml\n{previous_pipeline}\n```"
        
        feedback += "\n\nPlease generate a pipeline that correctly answers the query."
    
    messages.append({"role": "user", "content": feedback})
    return messages

def llm_call_with_messages(messages: List[Dict[str, str]]) -> str:
    """
    Call LLM API with message list format.

    Args:
        messages: List of message dictionaries with 'role' and 'content'

    Returns:
        LLM response string
    """
    try:
        # Call Azure GPT-4o with message list
        response = gpt_4o_azure(
            prompt=messages,
            max_tokens=4000,
            temperature=0.3
        )
        return response
    except Exception as e:
        # Error calling Azure GPT-4o: {e}
        raise


def llm_call_with_schema(messages: List[Dict[str, str]], parameters: Dict[str, Any], system_prompt: str = "") -> str:
    """
    Call LLM API with JSON schema parameters to ensure structured output.

    Args:
        messages: List of message dictionaries with 'role' and 'content'
        parameters: JSON schema parameters to constrain the output format
        system_prompt: Optional system prompt to prepend

    Returns:
        LLM response string (structured JSON)
    """
    try:
        # Prepare messages with optional system prompt
        if system_prompt:
            full_messages = [{"role": "system", "content": system_prompt}] + messages
        else:
            full_messages = messages

        # Prepare response format for structured output
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "structured_output",
                "strict": True,
                "schema": {
                    **parameters,
                    "additionalProperties": False
                }
            }
        }

        # Call Azure GPT-4o with structured output
        response = gpt_4o_azure(
            prompt=full_messages,
            max_tokens=4000,
            temperature=0.3,
            response_format=response_format
        )
        return response
    except Exception as e:
        # Error calling Azure GPT-4o with schema: {e}
        raise

def llm_call_wrapper(prompt: str) -> str:
    """Legacy wrapper function for backward compatibility"""
    messages = [
        {"role": "system", "content": "You are an expert at generating DocETL pipeline configurations."},
        {"role": "user", "content": prompt}
    ]
    return llm_call_with_messages(messages)

# Removed old clean_and_truncate_value function - replaced with intelligent DataTruncator

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
    
    return dataset_samples


def find_output_path(pipeline_file: str) -> Optional[str]:
    """
    Find the output path from pipeline configuration.
    
    Args:
        pipeline_file: Path to the pipeline YAML file
        
    Returns:
        Output file path if found, None otherwise
    """
    try:
        with open(pipeline_file, 'r') as f:
            pipeline_config = yaml.safe_load(f)
        
        # First check if output path is directly specified in pipeline
        pipeline_output = pipeline_config.get('pipeline', {}).get('output', {})
        if isinstance(pipeline_output, dict) and 'path' in pipeline_output:
            return pipeline_output['path']
        
        # Otherwise look for output path in datasets (for backward compatibility)
        for dataset in pipeline_config.get('datasets', []):
            output_name = pipeline_config.get('pipeline', {}).get('output', {}).get('path')
            if dataset.get('name') == output_name:
                return dataset.get('path')
        return None
    except Exception:
        return None

def load_output_data(pipeline_file: str, max_length: int = 2000) -> str:
    """
    Load pipeline output data for validation.
    
    Args:
        pipeline_file: Path to the pipeline YAML file
        max_length: Maximum length of returned string
        
    Returns:
        JSON string representation of output data
    """
    output_path = find_output_path(pipeline_file)
    
    if not output_path or not os.path.exists(output_path):
        return None
    
    try:
        with open(output_path, 'r', encoding='utf-8') as f:
            output_json = json.load(f)
            # Take sample of output for validation
            if isinstance(output_json, list) and len(output_json) > 2:
                sample_json = output_json[:2]
            else:
                sample_json = output_json
            output_str = json.dumps(sample_json, indent=2)
            return output_str[:max_length] + "..." if len(output_str) > max_length else output_str
    except Exception as e:
        # print(f"  Warning: Could not load output data for validation: {e}")
        return None

def validate_answer_with_llm(
    query: str, 
    output_data: str, 
    original_data: str, 
    llm_call: callable
) -> tuple:
    """
    Use LLM to validate if the pipeline output answers the original query
    
    Args:
        query: The original natural language query
        output_data: The pipeline's output result
        original_data: The original dataset content (sample)
        llm_call: LLM function to call for validation
    
    Returns:
        Tuple of (is_valid, validation_message)
    """
    # Create output validation prompt using Template
    validation_prompt = OUTPUT_VALIDATION_PROMPT_TEMPLATE.safe_substitute(
        query=query,
        original_data=original_data,
        output_data=output_data
    )
    # print ("  Validation prompt for LLM:")
    # print(validation_prompt)
    try:
        response = llm_call(validation_prompt)
        
        # Parse the response
        lines = response.strip().split('\n')
        assessment = None
        explanation = ""
        
        for line in lines:
            if line.startswith('ASSESSMENT:'):
                assessment = line.replace('ASSESSMENT:', '').strip().upper()
            elif line.startswith('EXPLANATION:'):
                explanation = line.replace('EXPLANATION:', '').strip()
        
        # If parsing fails, try to find VALID/INVALID in the response
        if not assessment:
            if 'VALID' in response.upper():
                assessment = 'VALID'
            elif 'INVALID' in response.upper():
                assessment = 'INVALID'
            else:
                assessment = 'INVALID'  # Default to invalid if unclear
            explanation = response
        
        is_valid = assessment == 'VALID'
        return is_valid, explanation
        
    except Exception as e:
        # print(f"Error during answer validation: {e}")
        return False, f"Validation failed due to error: {str(e)}"

def execute_single_pipeline(pipeline_file: str) -> tuple:
    """
    Execute a single pipeline and return success status with error details.
    
    Args:
        pipeline_file: Path to the pipeline YAML file
        
    Returns:
        Tuple of (success: bool, error_message: str or None)
    """
    try:
        # Load .env file
        cwd = os.getcwd()
        env_file = os.path.join(cwd, ".env")
        if os.path.exists(env_file):
            load_dotenv(env_file)
        
        # Execute the pipeline
        print("Reading: ", pipeline_file)
        runner = DSLRunner.from_yaml(str(pipeline_file), max_threads=10)
        runner.load_run_save()
        return True, None
        
    except Exception as e:
        error_msg = f"Pipeline execution error: {str(e)}\n{traceback.format_exc()}"
        print(f"  ✗ Pipeline execution failed: {str(e)}")
        print(f"  Full traceback:\n{traceback.format_exc()}")
        return False, error_msg

def validate_pipeline_output(
    pipeline_file: str,
    query: str,
    dataset_paths: List[str],
    llm_call: callable
) -> tuple:
    """
    Validate pipeline output using LLM and return validation result with sample output.
    
    Args:
        pipeline_file: Path to the pipeline YAML file
        query: Original natural language query
        dataset_paths: List of dataset file paths
        llm_call: LLM function for validation
        
    Returns:
        Tuple of (is_valid: bool, validation_message: str, sample_output: str)
    """
    # print("  Validating answer with LLM...")
    
    # Load data for validation
    original_data_dict = load_sample_data(dataset_paths, max_length=1500, max_string_length=200)
    # Convert to string format for validation
    original_data = json.dumps(original_data_dict)
    sample_output = load_output_data(pipeline_file)
    
    # If we couldn't load output, treat as validation failure
    if sample_output is None:
        validation_message = "Could not load pipeline output for validation. The pipeline may not have produced output."
        # print(f"  Validation result: ✗ INVALID")
        # print(f"  Validation explanation: {validation_message}")
        return False, validation_message, "No output produced"
    
    # Validate the answer
    is_valid, validation_message = validate_answer_with_llm(
        query=query,
        output_data=sample_output,
        original_data=original_data,
        llm_call=llm_call
    )
    
    status_text = '✓ VALID' if is_valid else '✗ INVALID'
    # print(f"  Validation result: {status_text}")
    # print(f"  Validation explanation: {validation_message}")
    
    # Return with sample output for use in error messages
    return is_valid, validation_message, sample_output

def validate_generated_pipeline(pipeline_yaml: str) -> None:
    """
    Validate the basic structure of generated pipeline YAML.
    
    Args:
        pipeline_yaml: Generated pipeline YAML string
        
    Raises:
        ValueError: If pipeline structure is invalid
    """
    pipeline_config = yaml.safe_load(pipeline_yaml)
    if not pipeline_config:
        raise ValueError("Generated pipeline YAML is empty")
    
    required_keys = ['default_model', 'datasets', 'operations', 'pipeline']
    missing_keys = [k for k in required_keys if k not in pipeline_config]
    if missing_keys:
        raise ValueError(f"Generated pipeline missing required keys: {missing_keys}")

def generate_and_execute_pipeline_with_messages(
    instruction_prompt: str,
    query: str,
    dataset_paths: List[str],
    pipeline_file: str,
    max_attempts: int = 5,
    validate_answer: bool = True
) -> bool:
    """
    Generate and execute pipeline using message-based conversation history.
    
    Args:
        instruction_prompt: Base instruction for pipeline generation
        query: Natural language query
        dataset_paths: List of dataset file paths
        pipeline_file: Path to save the pipeline YAML
        max_attempts: Maximum total attempts (generation + retry)
        validate_answer: Whether to validate the answer
        
    Returns:
        True if successful, False otherwise
    """
    # Load dataset samples for all files
    dataset_samples = load_sample_data(dataset_paths, max_length=2500, max_string_length=500)
    
    # Initialize conversation with system and user messages
    messages = create_initial_messages(instruction_prompt, query, dataset_samples)
    
    # Track conversation history
    pipeline_history = []
    
    for attempt in range(max_attempts):
        # print(f"\n  Attempt {attempt + 1}/{max_attempts}...")
        
        try:
            # Step 1: Generate pipeline
            # print("  Generating pipeline...")
            response = llm_call_with_messages(messages)
            
            # Add assistant's response to history
            messages.append({"role": "assistant", "content": response})
            
            # Extract pipeline YAML from response
            pipeline_yaml = extract_yaml_from_response(response)
            if not pipeline_yaml:
                raise ValueError("No valid YAML found in response")
            
            # Validate pipeline structure
            validate_generated_pipeline(pipeline_yaml)
            
            # Save pipeline to file
            with open(pipeline_file, "w") as f:
                f.write(pipeline_yaml)
            # print("  ✓ Pipeline generated successfully")
            
            # Step 2: Execute pipeline
            # print("  Executing pipeline...")
            success, error_msg = execute_single_pipeline(pipeline_file)
            
            if not success:
                # Execution failed - add error feedback
                # print(f"  ✗ Execution failed: {error_msg.split(chr(10))[0]}")
                messages = add_error_message(messages, "execution", error_msg)
                pipeline_history.append(FailedPipeline(
                    pipeline_yaml=pipeline_yaml,
                    error_type='execution',
                    error_message=error_msg
                ))
                continue
            
            # print("  ✓ Pipeline executed successfully")
            
            # Step 3: Validate answer (if enabled)
            if validate_answer:
                # print("  Validating answer...")
                is_valid, validation_msg, sample_output = validate_pipeline_output(
                    pipeline_file, query, dataset_paths, llm_call_with_messages
                )
                
                if not is_valid:
                    # Validation failed - add feedback with pipeline and sample output
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
                
                # print("  ✓ Answer validation passed!")
            
            # Success!
            return True
            
        except Exception as e:
            # Generation or parsing error
            error_msg = f"Pipeline generation/parsing error: {str(e)}"
            
            # Add error feedback for next attempt
            messages = add_error_message(messages, "execution", error_msg)
            pipeline_history.append(FailedPipeline(
                pipeline_yaml="",
                error_type='generation',
                error_message=error_msg
            ))
    
    # All attempts failed
    # print(f"\n  Failed after {max_attempts} attempts")
    _print_conversation_summary(messages, pipeline_history)
    return False

def extract_yaml_from_response(response: str) -> Optional[str]:
    """
    Extract YAML content from LLM response.
    
    Args:
        response: LLM response that may contain YAML
        
    Returns:
        Extracted YAML string or None if not found
    """
    # Try to find YAML between ```yaml and ``` markers
    import re
    yaml_pattern = r'```yaml\n(.*?)\n```'
    match = re.search(yaml_pattern, response, re.DOTALL)
    if match:
        return match.group(1)
    
    # Try to find YAML between ``` markers (without yaml tag)
    generic_pattern = r'```\n(.*?)\n```'
    match = re.search(generic_pattern, response, re.DOTALL)
    if match:
        content = match.group(1)
        # Check if it looks like YAML
        if 'default_model:' in content or 'operations:' in content:
            return content
    
    # If no markers, assume the entire response is YAML
    if 'default_model:' in response or 'operations:' in response:
        return response
    
    return None

def _print_conversation_summary(messages: List[Dict[str, str]], pipeline_history: List[FailedPipeline]):
    """Print a summary of the conversation and failed attempts"""
    # print("\n  Conversation Summary:")
    # print("  " + "-" * 50)
    # print(f"  Total messages: {len(messages)}")
    # print(f"  Failed pipelines: {len(pipeline_history)}")
    
    if pipeline_history:
        # print("\n  Failed attempts:")
        for i, failed in enumerate(pipeline_history, 1):
            error_summary = failed.error_message.split('\n')[0] if failed.error_message else "Unknown"
            # print(f"    {i}. {failed.error_type}: {error_summary[:80]}...")

def test_llm2pipeline_with_messages(max_attempts: int = 3):
    """
    Main test function using message-based conversation approach.
    
    Args:
        max_attempts: Maximum total attempts for generation and retry
    """
    # print("=" * 60)
    # print("LLM2Pipeline4DocETL Test - Medical Transcript Analysis")
    # print(f"(Message-based approach with {max_attempts} max attempts)")
    # print("=" * 60)
    
    # Step 1: Create sample dataset
    # print("\n1. Creating sample medical transcripts dataset...")
    dataset_path = create_sample_medical_transcripts()
    
    # Step 2: Define natural language query
    query = """
    Fine all unique medications prescribed across all patients, along with their dosages and frequencies. 
    """
    
    # print("\n2. Natural Language Query:")
    # print("-" * 40)
    # print(query)
    # print("-" * 40)
    
    # Step 3 & 4: Generate and execute pipeline with integrated retry
    # print("\n3. Generating and executing pipeline with message-based retry...")
    
    pipeline_file = "generated_medical_pipeline.yaml"
    success = generate_and_execute_pipeline_with_messages(
        instruction_prompt=INSTRUCTION_PROMPT,
        query=query,
        dataset_paths=[dataset_path],
        pipeline_file=pipeline_file,
        max_attempts=max_attempts,
        validate_answer=True
    )
    
    if success:
        # print("\n✓ Pipeline successfully generated, executed and validated!")
        # print(f"\nPipeline saved to: {pipeline_file}")
        # print("Results have been saved to the output directory.")
        pass
    else:
        # print("\n✗ Failed to generate a working pipeline after all attempts.")
        exit(1)
    
    # print("\n" + "=" * 60)
    # print("Test completed!")
    # print("=" * 60)

if __name__ == "__main__":
    test_llm2pipeline_with_messages()