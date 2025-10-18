#!/usr/bin/env python3
"""
LLM2Pipeline4DocETL
"""

import os
import sys
import json
from dotenv import load_dotenv
import yaml
from typing import List, Dict, Any, Optional, NamedTuple
import traceback

from docetl.runner import DSLRunner

# Data structure for pipeline-error pairs
class FailedPipeline(NamedTuple):
    """Store a failed pipeline with its error information"""
    pipeline_yaml: str
    error_type: str  # 'execution' or 'validation'
    error_message: str

# Add parent directory to path to import azuregpt4o
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.litellm_client import llm_call
from .prompt import (
    INSTRUCTION_PROMPT,
    PIPELINE_GENERATION_PROMPT_TEMPLATE,
    VALIDATION_PROMPT_TEMPLATE,
    OUTPUT_VALIDATION_PROMPT_TEMPLATE
)
from .data_utils import load_sample_data

def create_initial_messages(instruction_prompt: str, query: str, dataset_samples: Any) -> tuple:
    """
    Create initial message list for pipeline generation using PIPELINE_GENERATION_PROMPT.

    Args:
        instruction_prompt: Base instruction for pipeline generation (INSTRUCTION_PROMPT)
        query: Natural language query
        dataset_samples: Dictionary with {file_path: sample_data} format or a list/data structure for single dataset

    Returns:
        List of message dictionaries
    """
    # Format dataset profiles string
    profiles_str = ""

    # Handle both dictionary and direct data (list or other structure) inputs
    if isinstance(dataset_samples, dict):
        # Multiple files case - dataset_samples is a dictionary
        for file_path, sample_data in dataset_samples.items():
            profiles_str += f"File: {file_path}\n"
            profiles_str += f"Sample Data:\n{json.dumps(sample_data, indent=2)}\n\n"
    else:
        # Single file case - dataset_samples is the actual data (list or other structure)
        profiles_str += f"Sample Data:\n{json.dumps(dataset_samples, indent=2)}\n\n"
    
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
        # Call LLM with message list
        response = llm_call(
            messages=messages,
            max_tokens=4000,
            temperature=0.3
        )
        return response
    except Exception as e:
        # Error calling Azure GPT-4o: {e}
        raise


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
