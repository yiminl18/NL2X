import json
import os
from typing import Any, Dict, List
from pathlib import Path

from model.litellm_client import llm_call
from .base import BenchmarkInterface, BenchmarkSample, EvaluationResult, ContentDataType
from . import register_benchmark

# Constants for KramaBench
KRAMABENCH_PATH = "./benchmarks/Kramabench"
KRAMABENCH_DATA_PATH = os.path.join(KRAMABENCH_PATH, "data")
KRAMABENCH_WORKLOAD_PATH = os.path.join(KRAMABENCH_PATH, "workload")

@register_benchmark("kramabench")
class KramaBenchBenchmark(BenchmarkInterface):
    def _setup(self):
        # Available datasets in KramaBench
        self.available_datasets = [
            "legal", "wildfire", "environment", "biomedical", 
            "astronomy", "archeology"
        ]
        
        # Select dataset based on config or default
        self.dataset_name = getattr(self.config, 'dataset', 'legal')
        if self.dataset_name not in self.available_datasets:
            self.dataset_name = 'legal'
            
        # Determine workload file
        if hasattr(self.config, 'workload_file'):
            self.workload_file = self.config.workload_file
        else:
            # Use tiny version for testing if max_samples is small
            if self.config.max_samples and self.config.max_samples <= 10:
                self.workload_file = f"{self.dataset_name}-tiny.json"
            else:
                self.workload_file = f"{self.dataset_name}.json"
        
        self.workload_path = os.path.join(KRAMABENCH_WORKLOAD_PATH, self.workload_file)
        self.data_path = os.path.join(KRAMABENCH_DATA_PATH, self.dataset_name, "input")
        
        # GPT-4o pricing for evaluation
        self.input_cost_per_1k = 0.0015
        self.output_cost_per_1k = 0.006
        self.total_cost = 0.0
        
        if self.config.verbose:
            self.logger.info(f"Initialized KramaBench benchmark")
            self.logger.info(f"Dataset: {self.dataset_name}")
            self.logger.info(f"Workload: {self.workload_file}")
            self.logger.info(f"Data path: {self.data_path}")

    def _load_data(self) -> List[BenchmarkSample]:
        """Load data from KramaBench workload files."""
        # Load workload
        with open(self.workload_path, 'r') as f:
            workload = json.load(f)
        
        samples = []
        num_samples = min(self.config.max_samples, len(workload)) if self.config.max_samples else len(workload)
        
        for i in range(num_samples):
            task = workload[i]
            
            # Create context from data sources
            context = {}
            if "data_sources" in task:
                for data_source in task["data_sources"]:
                    # Handle different types of data sources
                    if data_source.endswith("/*"):
                        # Directory with all files (recursive)
                        dir_name = data_source[:-2]
                        dir_path = os.path.join(self.data_path, dir_name)
                        if os.path.exists(dir_path):
                            self._add_directory_files(dir_path, dir_name, context, recursive=True)
                    elif data_source.endswith("/"):
                        # Directory reference
                        dir_name = data_source[:-1] if data_source != "/" else ""
                        dir_path = os.path.join(self.data_path, dir_name)
                        if os.path.exists(dir_path):
                            self._add_directory_files(dir_path, dir_name, context, recursive=False)
                    else:
                        # Single file reference
                        file_path = os.path.join(self.data_path, data_source)
                        if os.path.exists(file_path):
                            if os.path.isdir(file_path):
                                # Handle case where it's actually a directory
                                self._add_directory_files(file_path, data_source, context, recursive=False)
                            else:
                                context[data_source] = self._create_content_type(file_path)
            
            # Create sample
            sample = BenchmarkSample(
                id=task["id"],
                query=task["query"],
                context=context,
                ground_truth={
                    "answer": task["answer"],
                    "answer_type": task.get("answer_type", "text")
                },
                metadata={
                    "dataset": self.dataset_name,
                    "runtime": task.get("runtime", 1),
                    "subtasks": task.get("subtasks", [])
                }
            )
            samples.append(sample)
        
        self._samples = samples
        
        if self.config.verbose:
            self.logger.info(f"Loaded {len(self._samples)} samples from KramaBench")
            
        return self._samples
    
    def _add_directory_files(self, dir_path: str, dir_name: str, context: Dict[str, ContentDataType], recursive: bool = False):
        """Add files from a directory to context."""
        if recursive:
            # Recursively walk through subdirectories
            for root, _, files in os.walk(dir_path):
                for file_name in files:
                    file_path = os.path.join(root, file_name)
                    # Get relative path from dir_path
                    rel_path = os.path.relpath(file_path, dir_path)
                    if dir_name:
                        key = f"{dir_name}/{rel_path}".replace("\\", "/")
                    else:
                        key = rel_path.replace("\\", "/")
                    context[key] = self._create_content_type(file_path)
        else:
            # Only process files in the immediate directory
            for file_name in os.listdir(dir_path):
                file_path = os.path.join(dir_path, file_name)
                if os.path.isfile(file_path):
                    if dir_name:
                        key = f"{dir_name}/{file_name}"
                    else:
                        key = file_name
                    context[key] = self._create_content_type(file_path)
    
    def _create_content_type(self, file_path: str) -> ContentDataType:
        """Create ContentDataType based on file extension."""
        ext = Path(file_path).suffix.lower()
        
        if ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
            return ContentDataType("image", path=file_path)
        elif ext in ['.xlsx', '.xls', '.xlsm', '.xlsb']:
            return ContentDataType("excel", path=file_path)
        elif ext in ['.csv']:
            return ContentDataType("csv", path=file_path)
        elif ext in ['.json']:
            return ContentDataType("json", path=file_path)
        elif ext in ['.html', '.htm']:
            return ContentDataType("html", path=file_path)
        elif ext in ['.pdf']:
            return ContentDataType("pdf", path=file_path)
        elif ext in ['.txt', '.text', '.md', '.lst', '.dat']:
            return ContentDataType("text", path=file_path)
        elif ext in ['.npz', '.npy']:
            # NumPy files
            return ContentDataType("numpy", path=file_path)
        elif ext in ['.tle']:
            # TLE files (Two Line Element for satellites)
            return ContentDataType("text", path=file_path)
        elif ext in ['.cdf']:
            # Common Data Format files
            return ContentDataType("cdf", path=file_path)
        elif ext in ['.gpkg']:
            # GeoPackage files
            return ContentDataType("geo", path=file_path)
        elif ext in ['.sp3', '.hdr']:
            # SP3 orbit files
            return ContentDataType("text", path=file_path)
        else:
            # Default to text for unknown extensions
            return ContentDataType("text", path=file_path)
    
    def get_samples(self) -> List[BenchmarkSample]:
        return self._samples
    
    def evaluate_sample(self, sample: BenchmarkSample, prediction: Any) -> EvaluationResult:
        """Evaluate a single sample prediction against ground truth."""
        ground_truth = sample.ground_truth
        answer_type = ground_truth.get("answer_type", "text")
        
        # Extract answer from prediction
        if isinstance(prediction, dict):
            predicted_answer = prediction.get('answer', '')
        else:
            predicted_answer = str(prediction)
        
        # Evaluate based on answer type
        if answer_type == "numeric_approximate":
            score = self._evaluate_numeric_approximate(
                predicted_answer, 
                ground_truth["answer"]
            )
        elif answer_type == "list_exact":
            score = self._evaluate_list_exact(
                predicted_answer,
                ground_truth["answer"]
            )
        elif answer_type == "text":
            score = self._evaluate_text_with_llm(
                sample.query,
                predicted_answer,
                ground_truth["answer"]
            )
        else:
            # Default to text evaluation
            score = self._evaluate_text_with_llm(
                sample.query,
                predicted_answer,
                ground_truth["answer"]
            )
        
        return EvaluationResult(
            sample_id=sample.id,
            prediction=prediction,
            ground_truth=ground_truth,
            metrics={
                "accuracy": score,
                "answer_type": answer_type
            },
            metadata=sample.metadata
        )
    
    def _evaluate_numeric_approximate(self, predicted: Any, target: Any, tolerance: float = 0.01) -> float:
        """Evaluate numeric answers with tolerance."""
        try:
            pred_val = float(str(predicted).strip())
            target_val = float(target)
            
            # Calculate relative error
            if target_val != 0:
                relative_error = abs(pred_val - target_val) / abs(target_val)
                return 1.0 if relative_error <= tolerance else 0.0
            else:
                # For zero target, use absolute error
                return 1.0 if abs(pred_val - target_val) <= tolerance else 0.0
        except (ValueError, TypeError):
            return 0.0
    
    def _evaluate_list_exact(self, predicted: Any, target: List) -> float:
        """Evaluate list answers for exact match."""
        try:
            if isinstance(predicted, list):
                pred_list = predicted
            elif isinstance(predicted, str):
                # Try to parse as JSON list
                import json
                try:
                    pred_list = json.loads(predicted)
                except:
                    # Split by common separators
                    pred_list = [item.strip() for item in predicted.split(',')]
            else:
                pred_list = [str(predicted)]
            
            # Normalize and compare
            pred_set = set(str(item).strip().lower() for item in pred_list)
            target_set = set(str(item).strip().lower() for item in target)
            
            if pred_set == target_set:
                return 1.0
            else:
                # Partial credit based on overlap
                if len(target_set) > 0:
                    overlap = len(pred_set & target_set)
                    return overlap / len(target_set)
                return 0.0
        except:
            return 0.0
    
    def _evaluate_text_with_llm(self, query: str, predicted: str, target: Any) -> float:
        """Use GPT-4o to evaluate text-based answers."""
        target_str = str(target)
        predicted_str = str(predicted).strip()
        
        # Special case: if predicted contains "I don't know" or similar
        if any(phrase in predicted_str.lower() for phrase in ["i don't know", "cannot determine", "unable to answer"]):
            return 0.0
        
        # Use GPT-4o for evaluation
        prompt = f"""Please evaluate whether the predicted answer is correct compared to the ground truth.

Question: {query}
Ground Truth Answer: {target_str}
Predicted Answer: {predicted_str}

Evaluate if the predicted answer conveys the same information as the ground truth. 
Consider the answer correct if:
1. It contains the essential information from the ground truth
2. Minor formatting or wording differences are acceptable
3. Additional context is fine as long as the core answer is correct

Reply with only "CORRECT" or "INCORRECT".
"""
        
        try:
            messages = [
                {"role": "system", "content": "You are an answer evaluation assistant."},
                {"role": "user", "content": prompt}
            ]
            
            response, usage = llm_call(messages, return_usage=True, max_tokens=10)
            
            # Update cost
            input_cost = (usage.prompt_tokens / 1000) * self.input_cost_per_1k
            output_cost = (usage.completion_tokens / 1000) * self.output_cost_per_1k
            self.total_cost += input_cost + output_cost
            
            # Parse response
            if "CORRECT" in response.upper():
                return 1.0
            else:
                return 0.0
                
        except Exception as e:
            if self.config.verbose:
                self.logger.warning(f"Error in LLM evaluation: {e}")
            # Fallback to simple string comparison
            return 1.0 if predicted_str.lower() == target_str.lower() else 0.0
    
    def compute_aggregate_metrics(self, results: List[EvaluationResult]) -> Dict[str, float]:
        """Compute aggregate metrics across all evaluation results."""
        if not results:
            return {}
        
        n = len(results)
        
        # Calculate accuracy by answer type
        answer_type_scores = {}
        for result in results:
            answer_type = result.metrics.get("answer_type", "unknown")
            if answer_type not in answer_type_scores:
                answer_type_scores[answer_type] = []
            answer_type_scores[answer_type].append(result.metrics.get("accuracy", 0.0))
        
        # Compute overall and per-type metrics
        aggregate = {
            "total_samples": n,
            "overall_accuracy": sum(r.metrics.get("accuracy", 0.0) for r in results) / n,
            "evaluation_cost": self.total_cost
        }
        
        # Add per-type accuracy
        for answer_type, scores in answer_type_scores.items():
            aggregate[f"accuracy_{answer_type}"] = sum(scores) / len(scores)
            aggregate[f"count_{answer_type}"] = len(scores)
        
        return aggregate