#!/usr/bin/env python3
"""
Optimized Dynamic Checker for DocETL pipelines using Requirement Specification Graph.

This implements a 4-step methodology:
1. Requirement Specification: Convert query to structured requirement graph with tasks, constraints, and dataflow
2. Implementation Specification: Extract structured operator specifications from pipeline
3. Specification Matching: Match requirement tasks to operator implementations
4. Semantic Diffing: Verify task coverage, constraint fulfillment, and dataflow consistency

Usage:
    from dynamic_checker_opt import check_pipeline_dynamic_opt
    
    result = check_pipeline_dynamic_opt(
        question="Report the average number of reported identity thefts...",
        pipeline_path="pipeline.yaml",
        use_azure_gpt4=True
    )
"""

import json
import yaml
import sys
import os
from typing import Dict, Any, List, Optional
from string import Template
from pathlib import Path
from datetime import datetime

# Import Azure OpenAI if available
try:
    from openai import AzureOpenAI
    AZURE_OPENAI_AVAILABLE = True
except ImportError:
    AZURE_OPENAI_AVAILABLE = False

# ===== Step 1: Query to Requirement Specification Graph =====
REQUIREMENT_SPECIFICATION_PROMPT = Template("""
# [Role]
You are a senior data scientist and requirements analyst. Your task is to analyze a natural language query and generate a structured Requirement Specification Graph that captures tasks, constraints, and dataflow dependencies.

# [Task]
Convert the natural language query into a structured graph representation that clearly defines:
1. Individual tasks that need to be performed
2. Constraints that must be satisfied
3. Data dependencies between tasks

# [Analysis Instructions]
1. **Task Decomposition**: Break down the query into atomic, well-defined tasks
2. **Constraint Extraction**: Identify all explicit and implicit constraints
3. **Dependency Analysis**: Determine the logical flow and dependencies between tasks
4. **Output Requirements**: Specify what each task should produce

# [Output Format]
Produce a JSON object with the following structure. Do not add any extra commentary outside of the JSON object.


{
  "tasks": [
    {
      "task_id": "T1",
      "description": "Clear description of what this task does",
      "required_inputs": ["list of required input fields"],
      "required_outputs": ["list of output fields this task must produce"],
      "task_type": "extraction|transformation|filtering|aggregation|other"
    }
  ],
  "constraints": [
    {
      "constraint_id": "C1",
      "applies_to_tasks": ["T1", "T2"],
      "description": "Exact constraint description",
      "constraint_type": "filtering|calculation|formatting|normalization|other",
      "is_critical": true
    }
  ],
  "dataflow": [
    {
      "from": "T1",
      "to": "T2",
      "data_passed": ["list of data fields passed between tasks"]
    }
  ],
  "final_output": {
    "description": "Description of the expected final result",
    "format": "Expected format or structure"
  }
}


# [Query to Analyze]
$query
""")

# ===== Step 2: Operator Implementation Specification =====
OPERATOR_SPECIFICATION_PROMPT = Template("""
# [Role]
You are an expert code reviewer and pipeline analyst. Your task is to analyze a single operator from a data pipeline and extract its implementation specification.

# [Task]
Analyze the provided operator and create a detailed specification that captures its primary function, side effects, inputs, outputs, and constraints handled.

# [Operator Information]
Operator Name: $operator_name
Operator Type: $operator_type
Original Query: $original_query

# [Operator Definition]
$operator_definition

# [Analysis Instructions]
1. **Primary Function**: The main purpose of this operator
2. **Side Effects**: Any additional operations or transformations performed
3. **Input/Output Analysis**: Precise data flow specification
4. **Constraint Implementation**: Which specific constraints from the query are handled

# [Output Format]
Produce a JSON object with the following structure. Do not add any extra commentary outside of the JSON object.


{
  "operator_name": "$operator_name",
  "operator_type": "$operator_type",
  "primary_function": "Main purpose of this operator in one sentence",
  "side_effects": [
    "List of any additional operations performed"
  ],
  "inputs": [
    {
      "name": "input field name",
      "schema": "data type/structure",
      "source_operator": "previous operator or dataset"
    }
  ],
  "outputs": [
    {
      "name": "output field name",
      "schema": "data type/structure"
    }
  ],
  "constraints_handled": [
    "List of specific constraints from the original query that this operator implements"
  ],
  "logic_description": "Detailed description of the operator's logic, especially for complex operations"
}

""")

# ===== Step 3: Task Coverage Matching =====
TASK_COVERAGE_MATCHING_PROMPT = Template("""
# [Role]
You are a verification specialist. Your task is to match requirement tasks with operator implementations.

# [Task]
For each task in the requirement specification, identify which operator(s) implement it.

# [Requirement Tasks]
$requirement_tasks_json

# [Operator Specifications]
$operator_specs_json

# [Matching Instructions]
1. For each requirement task, find the operator(s) that implement its functionality
2. Consider both primary functions and side effects of operators
3. A task may be implemented by multiple operators working together
4. An operator may implement multiple tasks
5. Assign confidence scores (0.0-1.0) based on:
   - 1.0: Perfect match with clear evidence
   - 0.8-0.99: Strong match with minor differences
   - 0.6-0.79: Partial match with some gaps
   - 0.4-0.59: Weak match with significant gaps
   - Below 0.4: Poor or no match

# [Output Format]
Produce a JSON object with the following structure. Do not add any extra commentary outside of the JSON object.


{
  "task_coverage": [
    {
      "task_id": "T1",
      "task_description": "Description from requirement",
      "implemented_by": ["operator_name1", "operator_name2"],
      "coverage_status": "fully_covered|partially_covered|not_covered",
      "coverage_confidence": 0.95,
      "coverage_details": "Explanation of how the task is covered",
      "evidence": "Specific evidence from operator specification that matches this task"
    }
  ],
  "coverage_score": 0.0,
  "uncovered_tasks": ["List of task IDs that are not covered"],
  "redundant_operators": ["List of operators that don't match any task"]
}

""")

# ===== Step 4: Constraint Fulfillment Verification =====
CONSTRAINT_FULFILLMENT_PROMPT = Template("""
# [Role]
You are a compliance auditor. Your task is to verify that all constraints are properly handled.

# [Task]
Check if each constraint from the requirement specification is correctly implemented in the operators.

# [Requirement Constraints]
$requirement_constraints_json

# [Task Coverage Mapping]
$task_coverage_json

# [Operator Specifications]
$operator_specs_json

# [Verification Instructions]
1. For each constraint, identify which tasks it applies to
2. Find the operators implementing those tasks
3. Verify if the constraint is mentioned in the operator's constraints_handled or logic_description
4. Check if the implementation is correct (not just mentioned)
5. Assign confidence scores (0.0-1.0) based on:
   - 1.0: Constraint explicitly and correctly implemented with clear evidence
   - 0.8-0.99: Constraint implemented but with minor differences (e.g., >= vs >)
   - 0.6-0.79: Constraint partially implemented or indirectly handled
   - 0.4-0.59: Constraint mentioned but implementation unclear
   - Below 0.4: Constraint not implemented or incorrectly implemented

# [Output Format]
Produce a JSON object with the following structure. Do not add any extra commentary outside of the JSON object.

{
  "constraint_verification": [
    {
      "constraint_id": "C1",
      "constraint_description": "Description from requirement",
      "applies_to_tasks": ["T1"],
      "implemented_in_operators": ["operator_name"],
      "verification_status": "satisfied|partially_satisfied|not_satisfied|incorrectly_implemented",
      "verification_confidence": 0.95,
      "verification_details": "Explanation of how the constraint is handled",
      "evidence": "Specific code snippet or prompt text that implements this constraint"
    }
  ],
  "fulfillment_score": 0.0,
  "critical_violations": ["List of critical constraints that are not satisfied"]
}
""")

# ===== Step 5: Dataflow Consistency Check =====
DATAFLOW_CONSISTENCY_PROMPT = Template("""
# [Role]
You are a data flow analyst. Your task is to verify the consistency of data flow between requirement specification and implementation.

# [Task]
Check if the actual pipeline dataflow matches the required dataflow dependencies.

# [Required Dataflow]
$required_dataflow_json

# [Task Coverage Mapping]
$task_coverage_json

# [Operator Specifications]
$operator_specs_json

# [Verification Instructions]
1. Map the required dataflow edges to actual operator connections
2. Check if data dependencies are preserved
3. Identify any broken or missing connections
4. Verify that data transformations maintain required fields
5. Assign confidence scores (0.0-1.0) based on:
   - 1.0: Perfect dataflow match with all required fields preserved
   - 0.8-0.99: Strong dataflow consistency with minor field variations
   - 0.6-0.79: Partial dataflow preserved but some fields missing/transformed
   - 0.4-0.59: Weak dataflow with significant gaps
   - Below 0.4: Broken or missing dataflow

# [Output Format]
Produce a JSON object with the following structure. Do not add any extra commentary outside of the JSON object.


{
  "dataflow_verification": [
    {
      "required_edge": {"from": "T1", "to": "T2"},
      "implemented_edge": {"from": "operator1", "to": "operator2"},
      "data_passed": ["fields"],
      "consistency_status": "consistent|inconsistent|missing",
      "consistency_confidence": 0.95,
      "details": "Explanation",
      "evidence": "Specific input/output mappings that confirm the dataflow"
    }
  ],
  "dataflow_score": 0.0,
  "broken_dependencies": ["List of broken dataflow dependencies"],
  "data_integrity_issues": ["List of potential data integrity problems"]
}

""")


class AzureGPT4Client:
    """Azure OpenAI GPT-4o client for LLM calls."""
    
    def __init__(self, api_key_path: str = '/Users/chiyuh/Workspace/NL2X/model/azuregpt4o.txt',
                 endpoint_url: str = 'https://text-db.openai.azure.com/',
                 api_version: str = '2025-01-01-preview',
                 model_name: str = 'gpt-4o',
                 max_tokens: int = 2000,
                 temperature: float = 0.0):
        """Initialize Azure GPT-4o client."""
        if not AZURE_OPENAI_AVAILABLE:
            raise ImportError("Azure OpenAI library not available. Install with: pip install openai")
            
        self.api_key_path = api_key_path
        self.endpoint_url = endpoint_url
        self.api_version = api_version
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        # Read API key
        try:
            with open(api_key_path, 'r') as f:
                self.api_key = f.read().strip()
        except FileNotFoundError:
            raise FileNotFoundError(f"API key file not found at {api_key_path}")
        
        # Initialize client
        self.client = AzureOpenAI(
            azure_endpoint=os.getenv("ENDPOINT_URL", self.endpoint_url),
            api_key=self.api_key,
            api_version=self.api_version,
        )
    
    def complete(self, prompt: str):
        """Make a completion request."""
        try:
            messages = [{"role": "user", "content": prompt}]
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                stream=False
            )
            return completion.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"Azure OpenAI API call failed: {str(e)}")


class DocETLDynamicCheckerOpt:
    """Optimized dynamic checker using Requirement Specification Graph methodology."""
    
    def __init__(self, llm_client=None, save_intermediate=True):
        """Initialize the checker with an LLM client.
        
        Args:
            llm_client: LLM client for API calls
            save_intermediate: Whether to save intermediate responses to files
        """
        self.llm_client = llm_client
        self.save_intermediate = save_intermediate
        self.intermediate_dir = None
        
        if self.save_intermediate:
            # Create checker_intermediate directory if it doesn't exist
            self.intermediate_base_dir = Path("checker_intermediate")
            self.intermediate_base_dir.mkdir(exist_ok=True)
    
    def check(self, question: str, pipeline_yaml: str = None, pipeline_path: str = None) -> Dict[str, Any]:
        """
        Check pipeline against question using graph-based methodology.
        
        Returns:
            Dict containing scores and detailed analysis results
        """
        try:
            # Create session-specific intermediate directory
            if self.save_intermediate:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                self.intermediate_dir = self.intermediate_base_dir / timestamp
                self.intermediate_dir.mkdir(exist_ok=True)
                
                # Save the question
                self._save_intermediate("question.txt", question)
            
            # Load pipeline content
            if pipeline_yaml is not None:
                pipeline_content = pipeline_yaml
            elif pipeline_path is not None:
                with open(pipeline_path, 'r', encoding='utf-8') as f:
                    pipeline_content = f.read()
            else:
                raise ValueError("Either pipeline_yaml or pipeline_path must be provided")
            
            # Parse pipeline YAML
            pipeline_dict = yaml.safe_load(pipeline_content)
            
            # Save pipeline content
            if self.save_intermediate:
                self._save_intermediate("pipeline.yaml", pipeline_content)

            # Step 1: Generate Requirement Specification Graph
            requirement_spec = self._generate_requirement_specification(question)
            if "error" in requirement_spec:
                return self._error_result("Step 1 failed", requirement_spec, {})
            
            # Step 2: Extract Operator Specifications
            operator_specs = self._extract_operator_specifications(pipeline_dict, question)
            if "error" in operator_specs:
                return self._error_result("Step 2 failed", operator_specs, 
                                        {"step1_requirement_spec": requirement_spec})
            
            # Step 3: Match Tasks to Operators (Task Coverage)
            task_coverage = self._match_task_coverage(requirement_spec, operator_specs)
            if "error" in task_coverage:
                return self._error_result("Step 3 failed", task_coverage,
                                        {"step1_requirement_spec": requirement_spec,
                                         "step2_operator_specs": operator_specs})
            
            # Step 4: Verify Constraint Fulfillment
            constraint_verification = self._verify_constraint_fulfillment(
                requirement_spec, task_coverage, operator_specs)
            if "error" in constraint_verification:
                return self._error_result("Step 4 failed", constraint_verification,
                                        {"step1_requirement_spec": requirement_spec,
                                         "step2_operator_specs": operator_specs,
                                         "step3_task_coverage": task_coverage})
            
            # Step 5: Check Dataflow Consistency
            dataflow_verification = self._verify_dataflow_consistency(
                requirement_spec, task_coverage, operator_specs)
            if "error" in dataflow_verification:
                return self._error_result("Step 5 failed", dataflow_verification,
                                        {"step1_requirement_spec": requirement_spec,
                                         "step2_operator_specs": operator_specs,
                                         "step3_task_coverage": task_coverage,
                                         "step4_constraint_verification": constraint_verification})
            
            # Calculate overall scores
            coverage_score = task_coverage.get("coverage_score", 0.0)
            fulfillment_score = constraint_verification.get("fulfillment_score", 0.0)
            dataflow_score = dataflow_verification.get("dataflow_score", 0.0)
            
            # Weighted combination: Coverage (40%), Constraints (40%), Dataflow (20%)
            overall_score = 0.4 * coverage_score + 0.4 * fulfillment_score + 0.2 * dataflow_score
            
            # Calculate confidence summary
            confidence_summary = self._calculate_confidence_summary(
                task_coverage, constraint_verification, dataflow_verification)
            
            final_result = {
                "overall_score": overall_score,
                "component_scores": {
                    "task_coverage": coverage_score,
                    "constraint_fulfillment": fulfillment_score,
                    "dataflow_consistency": dataflow_score
                },
                "confidence_summary": confidence_summary,
                "step_results": {
                    "step1_requirement_spec": requirement_spec,
                    "step2_operator_specs": operator_specs,
                    "step3_task_coverage": task_coverage,
                    "step4_constraint_verification": constraint_verification,
                    "step5_dataflow_verification": dataflow_verification
                },
                "critical_issues": self._extract_critical_issues(
                    task_coverage, constraint_verification, dataflow_verification)
            }
            
            # Save final result
            if self.save_intermediate:
                self._save_intermediate("final_result.json", final_result)
                print(f"\nAll intermediate files saved to: {self.intermediate_dir}")
            
            return final_result
            
        except Exception as e:
            return {
                "overall_score": 0.0,
                "error": f"Checker error: {str(e)}",
                "step_results": {}
            }
    
    def _error_result(self, error_msg: str, error_step: Dict, completed_steps: Dict) -> Dict[str, Any]:
        """Generate error result structure."""
        return {
            "overall_score": 0.0,
            "error": f"{error_msg}: {error_step.get('error', 'Unknown error')}",
            "step_results": {**completed_steps, f"error_step": error_step}
        }
    
    def _calculate_confidence_summary(self, task_coverage: Dict, constraint_verification: Dict,
                                      dataflow_verification: Dict) -> Dict[str, Any]:
        """Calculate confidence statistics across all verifications."""
        summary = {
            "average_confidence": 0.0,
            "min_confidence": 1.0,
            "max_confidence": 0.0,
            "confidence_distribution": {
                "high": [],  # >= 0.8
                "medium": [],  # 0.6-0.79
                "low": []  # < 0.6
            }
        }
        
        all_confidences = []
        
        # Collect task coverage confidences
        for task in task_coverage.get("task_coverage", []):
            conf = task.get("coverage_confidence", 0)
            all_confidences.append(conf)
            item = f"Task {task['task_id']}: {conf:.2f}"
            if conf >= 0.8:
                summary["confidence_distribution"]["high"].append(item)
            elif conf >= 0.6:
                summary["confidence_distribution"]["medium"].append(item)
            else:
                summary["confidence_distribution"]["low"].append(item)
        
        # Collect constraint verification confidences
        for constraint in constraint_verification.get("constraint_verification", []):
            conf = constraint.get("verification_confidence", 0)
            all_confidences.append(conf)
            item = f"Constraint {constraint['constraint_id']}: {conf:.2f}"
            if conf >= 0.8:
                summary["confidence_distribution"]["high"].append(item)
            elif conf >= 0.6:
                summary["confidence_distribution"]["medium"].append(item)
            else:
                summary["confidence_distribution"]["low"].append(item)
        
        # Collect dataflow confidences
        for flow in dataflow_verification.get("dataflow_verification", []):
            conf = flow.get("consistency_confidence", 0)
            all_confidences.append(conf)
            edge = flow.get("required_edge", {})
            item = f"Dataflow {edge.get('from')}->{edge.get('to')}: {conf:.2f}"
            if conf >= 0.8:
                summary["confidence_distribution"]["high"].append(item)
            elif conf >= 0.6:
                summary["confidence_distribution"]["medium"].append(item)
            else:
                summary["confidence_distribution"]["low"].append(item)
        
        # Calculate statistics
        if all_confidences:
            summary["average_confidence"] = sum(all_confidences) / len(all_confidences)
            summary["min_confidence"] = min(all_confidences)
            summary["max_confidence"] = max(all_confidences)
        
        return summary
    
    def _extract_critical_issues(self, task_coverage: Dict, constraint_verification: Dict, 
                                 dataflow_verification: Dict) -> List[str]:
        """Extract critical issues from verification results."""
        issues = []
        
        # Uncovered tasks
        if "uncovered_tasks" in task_coverage and task_coverage["uncovered_tasks"]:
            issues.append(f"Uncovered tasks: {', '.join(task_coverage['uncovered_tasks'])}")
        
        # Low confidence task coverage
        for task in task_coverage.get("task_coverage", []):
            if task.get("coverage_confidence", 0) < 0.6:
                issues.append(f"Low confidence ({task['coverage_confidence']:.2f}) for task {task['task_id']}: {task['task_description']}")
        
        # Critical constraint violations
        if "critical_violations" in constraint_verification and constraint_verification["critical_violations"]:
            issues.append(f"Critical constraints violated: {', '.join(constraint_verification['critical_violations'])}")
        
        # Low confidence constraint verification
        for constraint in constraint_verification.get("constraint_verification", []):
            if constraint.get("verification_confidence", 0) < 0.6 and constraint.get("is_critical", False):
                issues.append(f"Low confidence ({constraint['verification_confidence']:.2f}) for critical constraint {constraint['constraint_id']}: {constraint['constraint_description']}")
        
        # Broken dependencies
        if "broken_dependencies" in dataflow_verification and dataflow_verification["broken_dependencies"]:
            issues.append(f"Broken dataflow: {', '.join(dataflow_verification['broken_dependencies'])}")
        
        # Low confidence dataflow
        for flow in dataflow_verification.get("dataflow_verification", []):
            if flow.get("consistency_confidence", 0) < 0.6:
                edge = flow.get("required_edge", {})
                issues.append(f"Low confidence ({flow['consistency_confidence']:.2f}) for dataflow from {edge.get('from')} to {edge.get('to')}")
        
        return issues
    
    def _save_intermediate(self, filename: str, content: Any):
        """Save intermediate content to file."""
        if not self.save_intermediate or not self.intermediate_dir:
            return
        
        file_path = self.intermediate_dir / filename
        
        # Handle different content types
        if isinstance(content, (dict, list)):
            content_str = json.dumps(content, indent=2, ensure_ascii=False)
        else:
            content_str = str(content)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content_str)
        
        print(f"  Saved intermediate: {file_path}")
    
    def _trim_json_response(self, response: str) -> str:
        """Trim response to extract only the JSON content between first { and last }."""
        if not response:
            return response
        
        # Find the first { and last }
        first_brace = response.find('{')
        last_brace = response.rfind('}')
        
        if first_brace != -1 and last_brace != -1 and first_brace < last_brace:
            return response[first_brace:last_brace + 1]
        
        return response
    
    def _generate_requirement_specification(self, query: str) -> Dict[str, Any]:
        """Step 1: Generate Requirement Specification Graph from query."""
        print("Step 1: Generate Requirement Specification Graph from query.")
        try:
            prompt = REQUIREMENT_SPECIFICATION_PROMPT.substitute(query=query)
            
            # Save prompt
            if self.save_intermediate:
                self._save_intermediate("step1_prompt.txt", prompt)
            
            response = self._call_llm(prompt)
            print(f"DEBUG: Response received, length: {len(response) if response else 'None'}")
            
            # Save raw response
            if self.save_intermediate:
                self._save_intermediate("step1_response_raw.txt", response)
            
            # Trim response to JSON content
            trimmed_response = self._trim_json_response(response)
            
            # Save trimmed response
            if self.save_intermediate:
                self._save_intermediate("step1_response_trimmed.txt", trimmed_response)
            
            result = json.loads(trimmed_response)
            
            # Save parsed result
            if self.save_intermediate:
                self._save_intermediate("step1_result.json", result)
            
            return result
        except json.JSONDecodeError as e:
            print(f"DEBUG: Raw response was: {response[:100] if response else 'None'}")
            if 'trimmed_response' in locals():
                print(f"DEBUG: Trimmed response was: {trimmed_response[:100] if trimmed_response else 'None'}")
            error_result = {"error": f"Failed to parse LLM response as JSON: {str(e)}"}
            if self.save_intermediate:
                self._save_intermediate("step1_error.json", error_result)
            return error_result
        except Exception as e:
            error_result = {"error": f"Step 1 error: {str(e)}"}
            if self.save_intermediate:
                self._save_intermediate("step1_error.json", error_result)
            return error_result
    
    def _extract_operator_specifications(self, pipeline_dict: dict, original_query: str) -> Dict[str, Any]:
        """Step 2: Extract implementation specifications for each operator."""
        print("Step 2: Extract implementation specifications for each operator.")
        try:
            operator_specs = []
            
            # Extract operations from pipeline
            operations = pipeline_dict.get("operations", [])
            
            for op in operations:
                # Get operator details
                op_name = op.get("name", "unnamed")
                op_type = op.get("type", "unknown")
                
                # Format operator definition based on type
                if op_type.startswith("code_"):
                    # Code operator - include the code
                    op_definition = f"Code:\npython\n{op.get('code', 'No code provided')}\n"
                else:
                    # LLM operator - include the prompt
                    op_definition = f"Prompt:\n{op.get('prompt', 'No prompt provided')}"
                    if "output" in op and "schema" in op["output"]:
                        op_definition += f"\n\nOutput Schema: {json.dumps(op['output']['schema'])}"
                    if "reduce_key" in op:
                        op_definition += f"\n\nReduce Key: {op['reduce_key']}"
                
                # Extract specification for this operator
                prompt = OPERATOR_SPECIFICATION_PROMPT.substitute(
                    operator_name=op_name,
                    operator_type=op_type,
                    original_query=original_query,
                    operator_definition=op_definition
                )
                
                # Save operator prompt
                if self.save_intermediate:
                    self._save_intermediate(f"step2_prompt_operator_{op_name}.txt", prompt)
                
                response = self._call_llm(prompt)
                
                # Save operator response
                if self.save_intermediate:
                    self._save_intermediate(f"step2_response_operator_{op_name}_raw.txt", response)
                
                # Trim response to JSON content
                trimmed_response = self._trim_json_response(response)
                
                # Save trimmed response
                if self.save_intermediate:
                    self._save_intermediate(f"step2_response_operator_{op_name}_trimmed.txt", trimmed_response)
                
                spec = json.loads(trimmed_response)
                
                # Save parsed operator spec
                if self.save_intermediate:
                    self._save_intermediate(f"step2_result_operator_{op_name}.json", spec)
                
                operator_specs.append(spec)
            
            # Also build operator dataflow from pipeline structure
            dataflow = self._extract_pipeline_dataflow(pipeline_dict)
            
            result = {
                "operator_specifications": operator_specs,
                "pipeline_dataflow": dataflow
            }
            
            # Save complete step 2 result
            if self.save_intermediate:
                self._save_intermediate("step2_result_complete.json", result)
            
            return result
            
        except json.JSONDecodeError as e:
            error_result = {"error": f"Failed to parse operator specification: {str(e)}"}
            if self.save_intermediate:
                self._save_intermediate("step2_error.json", error_result)
            return error_result
        except Exception as e:
            error_result = {"error": f"Step 2 error: {str(e)}"}
            if self.save_intermediate:
                self._save_intermediate("step2_error.json", error_result)
            return error_result
    
    def _extract_pipeline_dataflow(self, pipeline_dict: dict) -> List[Dict[str, str]]:
        """Extract actual dataflow from pipeline structure."""
        dataflow = []
        operations = pipeline_dict.get("operations", [])
        
        # Simple sequential flow assumption (can be enhanced)
        for i in range(len(operations) - 1):
            dataflow.append({
                "from": operations[i].get("name", f"op_{i}"),
                "to": operations[i+1].get("name", f"op_{i+1}")
            })
        
        return dataflow
    
    def _match_task_coverage(self, requirement_spec: Dict, operator_specs: Dict) -> Dict[str, Any]:
        """Step 3: Match requirement tasks to operator implementations."""
        print("Step 3: Match requirement tasks to operator implementations.")
        try:
            req_tasks_json = json.dumps(requirement_spec.get("tasks", []), indent=2)
            op_specs_json = json.dumps(operator_specs.get("operator_specifications", []), indent=2)
            
            prompt = TASK_COVERAGE_MATCHING_PROMPT.substitute(
                requirement_tasks_json=req_tasks_json,
                operator_specs_json=op_specs_json
            )
            
            # Save prompt
            if self.save_intermediate:
                self._save_intermediate("step3_prompt.txt", prompt)
            
            response = self._call_llm(prompt)
            
            # Save raw response
            if self.save_intermediate:
                self._save_intermediate("step3_response_raw.txt", response)
            
            # Trim response to JSON content
            trimmed_response = self._trim_json_response(response)
            
            # Save trimmed response
            if self.save_intermediate:
                self._save_intermediate("step3_response_trimmed.txt", trimmed_response)
            
            result = json.loads(trimmed_response)
            
            # Save parsed result
            if self.save_intermediate:
                self._save_intermediate("step3_result.json", result)
            
            return result
            
        except json.JSONDecodeError as e:
            error_result = {"error": f"Failed to parse task coverage: {str(e)}"}
            if self.save_intermediate:
                self._save_intermediate("step3_error.json", error_result)
            return error_result
        except Exception as e:
            error_result = {"error": f"Step 3 error: {str(e)}"}
            if self.save_intermediate:
                self._save_intermediate("step3_error.json", error_result)
            return error_result
    
    def _verify_constraint_fulfillment(self, requirement_spec: Dict, task_coverage: Dict, 
                                      operator_specs: Dict) -> Dict[str, Any]:
        """Step 4: Verify all constraints are properly fulfilled."""
        print("Step 4: Verify all constraints are properly fulfilled.")
        try:
            constraints_json = json.dumps(requirement_spec.get("constraints", []), indent=2)
            coverage_json = json.dumps(task_coverage, indent=2)
            op_specs_json = json.dumps(operator_specs.get("operator_specifications", []), indent=2)
            
            prompt = CONSTRAINT_FULFILLMENT_PROMPT.substitute(
                requirement_constraints_json=constraints_json,
                task_coverage_json=coverage_json,
                operator_specs_json=op_specs_json
            )
            
            # Save prompt
            if self.save_intermediate:
                self._save_intermediate("step4_prompt.txt", prompt)
            
            response = self._call_llm(prompt)
            
            # Save raw response
            if self.save_intermediate:
                self._save_intermediate("step4_response_raw.txt", response)
            
            # Trim response to JSON content
            trimmed_response = self._trim_json_response(response)
            
            # Save trimmed response
            if self.save_intermediate:
                self._save_intermediate("step4_response_trimmed.txt", trimmed_response)
            
            result = json.loads(trimmed_response)
            
            # Save parsed result
            if self.save_intermediate:
                self._save_intermediate("step4_result.json", result)
            
            return result
            
        except json.JSONDecodeError as e:
            error_result = {"error": f"Failed to parse constraint verification: {str(e)}"}
            if self.save_intermediate:
                self._save_intermediate("step4_error.json", error_result)
            return error_result
        except Exception as e:
            error_result = {"error": f"Step 4 error: {str(e)}"}
            if self.save_intermediate:
                self._save_intermediate("step4_error.json", error_result)
            return error_result
    
    def _verify_dataflow_consistency(self, requirement_spec: Dict, task_coverage: Dict,
                                    operator_specs: Dict) -> Dict[str, Any]:
        """Step 5: Verify dataflow consistency between requirement and implementation."""
        print("Step 5: Verify dataflow consistency between requirement and implementation.")
        try:
            required_flow_json = json.dumps(requirement_spec.get("dataflow", []), indent=2)
            coverage_json = json.dumps(task_coverage, indent=2)
            op_specs_json = json.dumps(operator_specs, indent=2)
            
            prompt = DATAFLOW_CONSISTENCY_PROMPT.substitute(
                required_dataflow_json=required_flow_json,
                task_coverage_json=coverage_json,
                operator_specs_json=op_specs_json
            )
            
            # Save prompt
            if self.save_intermediate:
                self._save_intermediate("step5_prompt.txt", prompt)
            
            response = self._call_llm(prompt)
            
            # Save raw response
            if self.save_intermediate:
                self._save_intermediate("step5_response_raw.txt", response)
            
            # Trim response to JSON content
            trimmed_response = self._trim_json_response(response)
            
            # Save trimmed response
            if self.save_intermediate:
                self._save_intermediate("step5_response_trimmed.txt", trimmed_response)
            
            result = json.loads(trimmed_response)
            
            # Save parsed result
            if self.save_intermediate:
                self._save_intermediate("step5_result.json", result)
            
            return result
            
        except json.JSONDecodeError as e:
            error_result = {"error": f"Failed to parse dataflow verification: {str(e)}"}
            if self.save_intermediate:
                self._save_intermediate("step5_error.json", error_result)
            return error_result
        except Exception as e:
            error_result = {"error": f"Step 5 error: {str(e)}"}
            if self.save_intermediate:
                self._save_intermediate("step5_error.json", error_result)
            return error_result
    
    def _call_llm(self, prompt: str) -> str:
        """Call LLM with the given prompt."""
        if self.llm_client is None:
            # Return mock responses for testing
            return self._get_mock_response(prompt)
        
        # Use actual LLM client
        if hasattr(self.llm_client, 'complete'):
            return self.llm_client.complete(prompt)
        else:
            raise ValueError("LLM client must have 'complete' method")
    
    def _get_mock_response(self, prompt: str) -> str:
        """Generate mock responses for testing."""
        if "Requirement Specification Graph" in prompt:
            return '''{
                "tasks": [
                    {"task_id": "T1", "description": "Extract data from HTML", "required_inputs": [], 
                     "required_outputs": ["name", "population_2010", "population_2020", "identity_thefts"], "task_type": "extraction"},
                    {"task_id": "T2", "description": "Interpolate 2023 population", "required_inputs": ["population_2010", "population_2020"], 
                     "required_outputs": ["population_2023"], "task_type": "transformation"},
                    {"task_id": "T3", "description": "Filter areas > 1 million", "required_inputs": ["population_2023"], 
                     "required_outputs": ["filtered_areas"], "task_type": "filtering"},
                    {"task_id": "T4", "description": "Calculate average", "required_inputs": ["identity_thefts"], 
                     "required_outputs": ["average"], "task_type": "aggregation"}
                ],
                "constraints": [
                    {"constraint_id": "C1", "applies_to_tasks": ["T3"], "description": "Population > 1,000,000", 
                     "constraint_type": "filtering", "is_critical": true},
                    {"constraint_id": "C2", "applies_to_tasks": ["T4"], "description": "Round to 4 decimals", 
                     "constraint_type": "formatting", "is_critical": true}
                ],
                "dataflow": [
                    {"from": "T1", "to": "T2", "data_passed": ["population_2010", "population_2020"]},
                    {"from": "T2", "to": "T3", "data_passed": ["population_2023"]},
                    {"from": "T3", "to": "T4", "data_passed": ["identity_thefts"]}
                ],
                "final_output": {"description": "Average identity thefts", "format": "number"}
            }'''
        elif "operator_name" in prompt and "Operator Specification" in prompt:
            return '''{
                "operator_name": "filter_large_metro",
                "operator_type": "code_filter",
                "primary_function": "Filters metropolitan areas with population greater than 1 million",
                "side_effects": [],
                "inputs": [{"name": "population_2023", "schema": "integer", "source_operator": "previous"}],
                "outputs": [{"name": "filtered_data", "schema": "boolean"}],
                "constraints_handled": ["population > 1,000,000"],
                "logic_description": "Returns true if population_2023 > 1_000_000"
            }'''
        elif "task_coverage" in prompt:
            return '''{
                "task_coverage": [
                    {"task_id": "T1", "task_description": "Extract data", "implemented_by": ["extract_op"], 
                     "coverage_status": "fully_covered", "coverage_confidence": 0.95,
                     "coverage_details": "Fully implemented", "evidence": "Extract operator processes HTML and extracts all required fields"},
                    {"task_id": "T2", "task_description": "Interpolate", "implemented_by": ["interpolate_op"], 
                     "coverage_status": "fully_covered", "coverage_confidence": 0.98,
                     "coverage_details": "Fully implemented", "evidence": "Linear interpolation formula correctly implemented"},
                    {"task_id": "T3", "task_description": "Filter", "implemented_by": ["filter_op"], 
                     "coverage_status": "fully_covered", "coverage_confidence": 1.0,
                     "coverage_details": "Fully implemented", "evidence": "Code shows: return doc['population_2023'] > 1_000_000"},
                    {"task_id": "T4", "task_description": "Average", "implemented_by": ["reduce_op"], 
                     "coverage_status": "fully_covered", "coverage_confidence": 0.92,
                     "coverage_details": "Fully implemented", "evidence": "Reduce operation calculates average with rounding"}
                ],
                "coverage_score": 0.96,
                "uncovered_tasks": [],
                "redundant_operators": []
            }'''
        elif "constraint_verification" in prompt:
            return '''{
                "constraint_verification": [
                    {"constraint_id": "C1", "constraint_description": "Population > 1,000,000", 
                     "applies_to_tasks": ["T3"], "implemented_in_operators": ["filter_op"],
                     "verification_status": "satisfied", "verification_confidence": 1.0,
                     "verification_details": "Correctly implemented", 
                     "evidence": "Code explicitly shows: doc['population_2023'] > 1_000_000"},
                    {"constraint_id": "C2", "constraint_description": "Round to 4 decimals", 
                     "applies_to_tasks": ["T4"], "implemented_in_operators": ["reduce_op"],
                     "verification_status": "satisfied", "verification_confidence": 0.85,
                     "verification_details": "Correctly implemented",
                     "evidence": "Prompt includes instruction to round to 4 decimal places"}
                ],
                "fulfillment_score": 0.93,
                "critical_violations": []
            }'''
        elif "dataflow_verification" in prompt:
            return '''{
                "dataflow_verification": [
                    {"required_edge": {"from": "T1", "to": "T2"}, 
                     "implemented_edge": {"from": "extract_op", "to": "interpolate_op"},
                     "data_passed": ["population_2010", "population_2020"],
                     "consistency_status": "consistent", "consistency_confidence": 0.98,
                     "details": "Dataflow preserved",
                     "evidence": "Extract outputs population fields, interpolate receives them as inputs"},
                    {"required_edge": {"from": "T2", "to": "T3"}, 
                     "implemented_edge": {"from": "interpolate_op", "to": "filter_op"},
                     "data_passed": ["population_2023"],
                     "consistency_status": "consistent", "consistency_confidence": 1.0,
                     "details": "Perfect dataflow match",
                     "evidence": "Interpolate outputs population_2023, filter uses it directly"},
                    {"required_edge": {"from": "T3", "to": "T4"}, 
                     "implemented_edge": {"from": "filter_op", "to": "reduce_op"},
                     "data_passed": ["identity_thefts"],
                     "consistency_status": "consistent", "consistency_confidence": 0.95,
                     "details": "Dataflow maintained through filtering",
                     "evidence": "Filter preserves identity_thefts field for averaging"}
                ],
                "dataflow_score": 0.98,
                "broken_dependencies": [],
                "data_integrity_issues": []
            }'''
        else:
            return '{"mock_response": "Unable to determine prompt type"}'


def check_pipeline_dynamic_opt(question: str, 
                               pipeline_path: str = None,
                               pipeline_yaml: str = None,
                               llm_client=None,
                               use_azure_gpt4: bool = False,
                               api_key_path: str = '/Users/chiyuh/Workspace/NL2X/model/azuregpt4o.txt',
                               save_intermediate: bool = True) -> Dict[str, Any]:
    """
    Convenience function to check pipeline using optimized graph-based methodology.
    
    Args:
        question: Natural language query
        pipeline_path: Path to pipeline YAML file
        pipeline_yaml: Pipeline YAML content as string
        llm_client: LLM client for API calls
        use_azure_gpt4: Whether to use Azure GPT-4o
        api_key_path: Path to Azure API key file
        save_intermediate: Whether to save intermediate responses to files
    
    Returns:
        Dict with overall_score and detailed results
    """
    # Create LLM client if needed
    if llm_client is None and use_azure_gpt4:
        llm_client = AzureGPT4Client(api_key_path)
    checker = DocETLDynamicCheckerOpt(llm_client, save_intermediate=save_intermediate)
    return checker.check(question, pipeline_yaml, pipeline_path)


def main():
    """Command-line interface for the optimized dynamic checker."""
    if len(sys.argv) < 3:
        print("Usage: python3 dynamic_checker_opt.py <question> <pipeline_yaml_path> [--use-azure-gpt4] [--api-key-path <path>] [--no-save-intermediate]")
        print("Example: python3 dynamic_checker_opt.py 'What is the average?' pipeline.yaml --use-azure-gpt4")
        sys.exit(1)
    
    question = sys.argv[1]
    pipeline_path = sys.argv[2]
    use_azure_gpt4 = True
    api_key_path = '/Users/chiyuh/Workspace/NL2X/model/azuregpt4o.txt'
    save_intermediate = True
    
    # Parse arguments
    i = 3
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == '--use-azure-gpt4':
            use_azure_gpt4 = True
        elif arg == '--no-save-intermediate':
            save_intermediate = False
        elif arg == '--api-key-path':
            if i + 1 < len(sys.argv):
                api_key_path = sys.argv[i + 1]
                i += 1
            else:
                print("Error: --api-key-path requires a path argument")
                sys.exit(1)
        i += 1
    
    try:
        # Create appropriate client based on flags
        if use_azure_gpt4:
            llm_client = AzureGPT4Client(api_key_path)
        else:
            llm_client = None
        
        result = check_pipeline_dynamic_opt(
            question=question,
            pipeline_path=pipeline_path,
            llm_client=llm_client,
            save_intermediate=save_intermediate
        )
        
        print(json.dumps(result, indent=2))
        
        # Exit with appropriate code
        overall_score = result.get("overall_score", 0.0)
        sys.exit(0 if overall_score > 0.5 else 1)
        
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()