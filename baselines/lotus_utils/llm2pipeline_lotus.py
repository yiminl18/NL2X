#!/usr/bin/env python3
"""
LLM2Pipeline4LOTUS Test Script
Based on llm2pipeline_docetl.py, generates LOTUS pipeline from natural language query
"""

import os
import sys
import json
from dotenv import load_dotenv
import subprocess
from typing import List, Dict, Any, Optional, NamedTuple
import traceback
import re
import pandas as pd

# Data structure for pipeline-error pairs
class FailedPipeline(NamedTuple):
    """Store a failed pipeline with its error information"""
    pipeline_code: str
    error_type: str  # 'execution' or 'validation'
    error_message: str

# Add parent directory to path to import llm_call
sys.path.append('/Users/chiyuh/Workspace/NL2X/model')
from litellm_client import llm_call
from .prompt import INSTRUCTION_PROMPT, PIPELINE_GENERATION_PROMPT

# Import smart truncation utility  
from ..utils import DataTruncator

def convert_json_to_csv(json_path: str) -> str:
    """
    Convert a JSON file to CSV format and save it with .csv extension.
    
    Args:
        json_path: Path to the JSON file
        
    Returns:
        Path to the created CSV file
    """
    if not json_path.endswith('.json'):
        return json_path
    
    csv_path = json_path.replace('.json', '.csv')
    
    try:
        # Load JSON data
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Convert to DataFrame
        if isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, dict):
            # If dict, try to convert to DataFrame
            df = pd.DataFrame([data])
        else:
            # If other type, wrap in list
            df = pd.DataFrame([{'data': data}])
        
        # Save as CSV
        df.to_csv(csv_path, index=False, encoding='utf-8')
        # Converted {json_path} to {csv_path}
        return csv_path
        
    except Exception as e:
        # Failed to convert {json_path} to CSV: {e}
        return json_path  # Return original path if conversion fails

def create_sample_medical_transcripts() -> str:
    """Create sample medical transcripts dataset for testing. Returns CSV file path."""
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
    
    # Save to JSON file first
    json_filename = "test_medical_transcripts.json"
    with open(json_filename, "w", encoding="utf-8") as f:
        json.dump(transcripts, f, indent=2, ensure_ascii=False)
    
    # Convert to CSV for consistent handling
    csv_filename = convert_json_to_csv(json_filename)
    
    # Dataset created: {csv_filename}
    return csv_filename

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
    user_content = PIPELINE_GENERATION_PROMPT.format(
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
        previous_pipeline: Previous pipeline code that failed
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
            feedback += f"\n\nPrevious pipeline that failed:\n```python\n{previous_pipeline}\n```"
        
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

def llm_call_wrapper(prompt: str) -> str:
    """Legacy wrapper function for backward compatibility"""
    messages = [
        {"role": "system", "content": "You are an expert at generating LOTUS pipeline code."},
        {"role": "user", "content": prompt}
    ]
    return llm_call_with_messages(messages)

def load_sample_data(dataset_paths: List[str], max_length: int = 1500, max_string_length: int = 200, max_plain_text_length: int = 5000) -> Dict[str, Any]:
    """
    Load samples from all dataset files with clean formatting.
    Converts JSON files to CSV first for consistent handling.
    
    Args:
        dataset_paths: List of dataset file paths
        max_length: Maximum total length for each file's sample
        max_string_length: Maximum length for individual string fields
        max_plain_text_length: Maximum length for plain text content
        
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
    
    # Convert JSON files to CSV first for LOTUS compatibility
    converted_paths = []
    for path in dataset_paths:
        if path.endswith('.json'):
            csv_path = convert_json_to_csv(path)
            converted_paths.append(csv_path)
        else:
            converted_paths.append(path)
    
    for file_path in converted_paths:
        try:
            if file_path.endswith('.csv'):
                data = pd.read_csv(file_path)
                # Convert to list of dictionaries
                data = data.to_dict('records')
            else:
                # For other file types, try to read as text
                with open(file_path, 'r', encoding='utf-8') as txt_f:
                    content = txt_f.read()
                    # Clean up the content and treat as plain text
                    content = ' '.join(content.split())
                    # Use plain text truncation for text files
                    truncated_content = truncator.truncate_text(content, is_plain_text=True)
                    data = [{"text": truncated_content}]
            
            # Apply intelligent truncation (except for plain text which was already handled)
            if file_path.endswith('.csv'):
                truncated_data = truncator.truncate_data(data)
                dataset_samples[file_path] = truncated_data
            else:
                dataset_samples[file_path] = data
            
        except Exception as e:
            # print(f"  Warning: Could not load data from {file_path}: {e}")
            dataset_samples[file_path] = f"Error loading file: {str(e)}"
    
    return dataset_samples

def execute_single_pipeline(pipeline_file: str, dataset_paths: List[str]) -> tuple:
    """
    Execute a single LOTUS pipeline and return success status with error details.
    Converts JSON files to CSV before execution for consistent DataFrame handling.
    
    Args:
        pipeline_file: Path to the pipeline Python file
        dataset_paths: List of paths to input datasets
        
    Returns:
        Tuple of (success: bool, error_message: str or None, output_data: Any or None)
    """
    try:
        # Load .env file
        cwd = os.getcwd()
        env_file = os.path.join(cwd, ".env")
        if os.path.exists(env_file):
            load_dotenv(env_file)
        
        # Convert JSON files to CSV for consistent handling
        converted_paths = []
        for path in dataset_paths:
            if path.endswith('.json'):
                csv_path = convert_json_to_csv(path)
                converted_paths.append(csv_path)
            else:
                converted_paths.append(path)
        
        # Define output file path
        output_file = "pipeline_output.json"
        
        # Remove any existing output file
        if os.path.exists(output_file):
            os.remove(output_file)
        
        # Execute the pipeline Python file with all dataset paths (now CSV files)
        input_data_str = ','.join(converted_paths)
        result = subprocess.run(
            [sys.executable, pipeline_file],
            capture_output=True,
            text=True,
            timeout=120,
            env={**os.environ, 'INPUT_DATA': input_data_str, 'OUTPUT_FILE': output_file}
        )
        
        if result.returncode != 0:
            error_msg = f"Pipeline execution error:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
            print(f"  ✗ Pipeline execution failed: {result.stderr.split(chr(10))[0] if result.stderr else 'Unknown error'}")
            print(f"  STDOUT: {result.stdout}")
            print(f"  STDERR: {result.stderr}")
            return False, error_msg, None
        
        # Load output data from the saved file
        output_data = None
        if os.path.exists(output_file):
            try:
                with open(output_file, 'r') as f:
                    output_data = json.load(f)
                # print(f"  ✓ Output loaded from {output_file}")
            except Exception as e:
                # print(f"  ✗ Failed to load output file: {e}")
                return False, f"Failed to load output file: {e}", None
        else:
            # print(f"  ✗ No output file generated at {output_file}")
            return False, f"Pipeline did not generate output file at {output_file}", None
        
        return True, None, output_data
        
    except subprocess.TimeoutExpired:
        error_msg = "Pipeline execution timeout (exceeded 120 seconds)"
        print(f"  ✗ {error_msg}")
        return False, error_msg, None
    except Exception as e:
        error_msg = f"Pipeline execution error: {str(e)}\n{traceback.format_exc()}"
        print(f"  ✗ Pipeline execution failed: {str(e)}")
        print(f"  Full traceback:\n{traceback.format_exc()}")
        return False, error_msg, None

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
    validation_prompt = f"""
You are an expert data analyst. Please evaluate whether the provided output correctly answers the given query based on the original data.

ORIGINAL QUERY:
{query}

ORIGINAL DATA (sample):
{original_data}

PIPELINE OUTPUT (sample):
{output_data}

Please analyze and note:
0. Fields starting with "_" are metadata and can be ignored.
1. Don't need to verify if every information in the output (it's sample) is from the original data (it's also sample).
2. Are all required elements from the query addressed in the output? (Fields existing in the query should be reflected in the output, value-missing is acceptable)
3. Is the output format appropriate and complete?

Respond with:
- "VALID" if the output correctly answers the query
- "INVALID" if the output does not answer the query or contains errors

- Provide a 1-2 sentence brief explanation for your assessment.

Format your response as:
ASSESSMENT: [VALID/INVALID]
EXPLANATION: [Your explanation here]
"""
    # print("  Validation prompt for LLM:")
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

def validate_pipeline_output(
    query: str,
    dataset_paths: List[str],
    output_data: Any,
    llm_call: callable
) -> tuple:
    """
    Validate pipeline output using LLM and return validation result with sample output.
    
    Args:
        query: Original natural language query
        dataset_paths: List of dataset file paths
        output_data: Output data from pipeline execution
        llm_call: LLM function for validation
        
    Returns:
        Tuple of (is_valid: bool, validation_message: str, sample_output: str)
    """
    # print("  Validating answer with LLM...")
    
    # Load data for validation
    original_data_dict = load_sample_data(dataset_paths, max_length=1500, max_string_length=200)
    # Convert to string format for validation
    original_data = json.dumps(original_data_dict)
    
    # Format output data for validation
    if output_data is None:
        validation_message = "Could not capture pipeline output for validation. The pipeline may not have produced output."
        # print(f"  Validation result: ✗ INVALID")
        # print(f"  Validation explanation: {validation_message}")
        return False, validation_message, "No output produced"
    
    # Convert output to string if necessary
    if isinstance(output_data, (dict, list)):
        sample_output = json.dumps(output_data, indent=2)
    else:
        sample_output = str(output_data)
    
    # Truncate if too long
    max_output_length = 2000
    if len(sample_output) > max_output_length:
        sample_output = sample_output[:max_output_length] + "..."
    
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

def extract_python_from_response(response: str) -> Optional[str]:
    """
    Extract Python code from LLM response.
    
    Args:
        response: LLM response that may contain Python code
        
    Returns:
        Extracted Python code string or None if not found
    """
    # Try to find Python between ```python and ``` markers
    python_pattern = r'```python\n(.*?)\n```'
    match = re.search(python_pattern, response, re.DOTALL)
    if match:
        return match.group(1)
    
    # Try to find code between ``` markers (without python tag)
    generic_pattern = r'```\n(.*?)\n```'
    match = re.search(generic_pattern, response, re.DOTALL)
    if match:
        content = match.group(1)
        # Check if it looks like Python
        if 'import' in content or 'df' in content or 'lotus' in content:
            return content
    
    # If no markers, check if the entire response looks like Python code
    if 'import' in response and ('lotus' in response or 'pandas' in response):
        return response
    
    return None

def create_wrapped_pipeline_code(pipeline_code: str, dataset_paths: List[str], output_file: str = "pipeline_output.json") -> str:
    """
    Create wrapped pipeline code for execution.
    
    Args:
        pipeline_code: The core pipeline code to wrap
        dataset_paths: List of dataset file paths
        output_file: Path for output file
        
    Returns:
        Wrapped executable pipeline code
    """
    input_data_str = ','.join(dataset_paths)
    
    wrapped_code = f"""#!/usr/bin/env python3
import os
import sys
import json
import pandas as pd

# Get input data paths from environment or command line
input_data_paths = os.environ.get('INPUT_DATA', '{input_data_str}').split(',')
output_file = os.environ.get('OUTPUT_FILE', '{output_file}')

data_dict = {{}}
for path in input_data_paths:
    path = path.strip()
    if os.path.exists(path):
        # Use filename as key in data_dict
        dataset_name = os.path.basename(path)
        
        # Load based on file extension
        if path.endswith('.csv'):
            # CSV files are loaded as pandas DataFrame
            data_dict[dataset_name] = pd.read_csv(path)
        else:
            # Raise error for unsupported file types
            raise ValueError(f"Unsupported file type: {{path}}")

{pipeline_code}

# Save the result to a file
if 'result' in locals():
    if isinstance(result, pd.DataFrame):
        # Convert DataFrame to dict for JSON serialization
        output_data = result.to_dict(orient='records')
    else:
        output_data = result
    
    # Save to output file
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    # print(f"✓ Pipeline executed successfully. Output saved to {{output_file}}")
    # print(f"✓ Output contains {{len(output_data) if isinstance(output_data, list) else 1}} records")
else:
    # print("✗ Error: Pipeline must produce a variable named 'result' containing a pandas DataFrame")
    sys.exit(1)
"""
    # print(wrapped_code)
    return wrapped_code

def validate_generated_pipeline(pipeline_code: str) -> None:
    """
    Validate the basic structure of generated pipeline Python code.
    
    Args:
        pipeline_code: Generated pipeline Python code string
        
    Raises:
        ValueError: If pipeline structure is invalid
    """
    if not pipeline_code:
        raise ValueError("Generated pipeline code is empty")
    
    # Check for basic required elements
    required_elements = ['import', 'lotus', 'pd']
    missing_elements = [elem for elem in required_elements if elem not in pipeline_code]
    if missing_elements:
        raise ValueError(f"Generated pipeline missing required elements: {missing_elements}")
    
    # Try to compile the code to check for syntax errors
    try:
        compile(pipeline_code, '<string>', 'exec')
    except SyntaxError as e:
        raise ValueError(f"Generated pipeline has syntax error: {e}")

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
    Converts JSON files to CSV for consistent DataFrame handling.
    
    Args:
        instruction_prompt: Base instruction for pipeline generation
        query: Natural language query
        dataset_paths: List of dataset file paths (will be converted to CSV if JSON)
        pipeline_file: Path to save the pipeline Python file
        max_attempts: Maximum total attempts (generation + retry)
        validate_answer: Whether to validate the answer
        
    Returns:
        True if successful, False otherwise
    """
    # Convert JSON files to CSV for consistent handling
    converted_paths = []
    for path in dataset_paths:
        if path.endswith('.json'):
            csv_path = convert_json_to_csv(path)
            converted_paths.append(csv_path)
        else:
            converted_paths.append(path)
    
    # Use converted paths from now on
    dataset_paths = converted_paths
    
    # Load dataset samples for all files
    dataset_samples = load_sample_data(dataset_paths, max_length=1500, max_string_length=200)
    
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
            
            # Extract pipeline Python code from response
            pipeline_code = extract_python_from_response(response)
            if not pipeline_code:
                raise ValueError("No valid Python code found in response")
            # Write extracted code to local file
            # print("  Extracted pipeline code.")
            # print(pipeline_code)

            # Validate pipeline structure
            validate_generated_pipeline(pipeline_code)
            
            # Add necessary imports and wrapper code if not present
            if "__name__" not in pipeline_code:
                pipeline_code = create_wrapped_pipeline_code(pipeline_code, dataset_paths)
            
            # Save pipeline to file
            with open(pipeline_file, "w") as f:
                f.write(pipeline_code)
            os.chmod(pipeline_file, 0o755)  # Make executable
            # print("  ✓ Pipeline generated successfully")
            
            # Step 2: Execute pipeline
            # print("  Executing pipeline...")
            success, error_msg, output_data = execute_single_pipeline(pipeline_file, dataset_paths)
            
            if not success:
                # Execution failed - add error feedback
                # print(f"  ✗ Execution failed: {error_msg.split(chr(10))[0]}")
                messages = add_error_message(messages, "execution", error_msg)
                pipeline_history.append(FailedPipeline(
                    pipeline_code=pipeline_code,
                    error_type='execution',
                    error_message=error_msg
                ))
                continue
            
            # print("  ✓ Pipeline executed successfully")
            
            # Step 3: Validate answer (if enabled)
            if validate_answer:
                # print("  Validating answer...")
                is_valid, validation_msg, sample_output = validate_pipeline_output(
                    query, dataset_paths, output_data, llm_call_with_messages
                )
                
                if not is_valid:
                    # Validation failed - add feedback with pipeline and sample output
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
                
                # print("  ✓ Answer validation passed!")
            
            # Success!
            return True
            
        except Exception as e:
            # Generation or parsing error
            error_msg = f"Pipeline generation/parsing error: {str(e)}"
            # print(f"  ✗ {error_msg}")
            
            # Add error feedback for next attempt
            messages = add_error_message(messages, "execution", error_msg)
            pipeline_history.append(FailedPipeline(
                pipeline_code="",
                error_type='generation',
                error_message=error_msg
            ))
    
    # All attempts failed
    # print(f"\n  Failed after {max_attempts} attempts")
    _print_conversation_summary(messages, pipeline_history)
    return False

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
    # print("LLM2Pipeline4LOTUS Test - Medical Transcript Analysis")
    # print(f"(Message-based approach with {max_attempts} max attempts)")
    # print("=" * 60)
    
    # Step 1: Create sample dataset
    # print("\n1. Creating sample medical transcripts dataset...")
    dataset_path = create_sample_medical_transcripts()
    
    # Step 2: Define natural language query
    query = """
    Find all unique medications prescribed across all patients, along with their dosages and frequencies. 
    """
    
    # print("\n2. Natural Language Query:")
    # print("-" * 40)
    # print(query)
    # print("-" * 40)
    
    # Step 3 & 4: Generate and execute pipeline with integrated retry
    # print("\n3. Generating and executing pipeline with message-based retry...")
    
    pipeline_file = "generated_lotus_pipeline.py"
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
        # print("Results have been printed to stdout.")
        pass
    else:
        # print("\n✗ Failed to generate a working pipeline after all attempts.")
        exit(1)
    
    # print("\n" + "=" * 60)
    # print("Test completed!")
    # print("=" * 60)

if __name__ == "__main__":
    test_llm2pipeline_with_messages(max_attempts=3)