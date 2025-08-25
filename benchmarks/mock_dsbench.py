import random
from typing import Any, Dict, List, Optional
from .base import BenchmarkInterface, BenchmarkSample, EvaluationResult, BenchmarkConfig
from . import register_benchmark

@register_benchmark("mock_dsbench")
class MockDSBenchBenchmark(BenchmarkInterface):
    def _setup(self):
        self.task_types = ["data_analysis", "data_modeling", "visualization", "statistical_test"]
        self.data_formats = ["csv", "xlsx", "json", "parquet"]
        if self.config.verbose:
            self.logger.info(f"Initialized MockDSBench benchmark with config: {self.config}")
    
    def _load_data(self):
        num_samples = self.config.max_samples or 50
        
        for i in range(num_samples):
            task_type = random.choice(self.task_types)
            data_format = random.choice(self.data_formats)
            num_rows = random.randint(100, 10000)
            num_cols = random.randint(5, 50)
            
            sample = BenchmarkSample(
                id=f"dsbench_{i:04d}",
                query=f"Perform {task_type} on dataset with {num_rows} rows and {num_cols} columns",
                context={
                    "task_type": task_type,
                    "data_format": data_format,
                    "dataset_shape": (num_rows, num_cols),
                    "has_missing_values": random.choice([True, False]),
                    "requires_preprocessing": random.choice([True, False])
                },
                ground_truth={
                    "result": random.uniform(0, 100) if task_type == "statistical_test" else f"Result_{i}",
                    "method_used": random.choice(["regression", "classification", "clustering", "analysis"]),
                    "accuracy": random.uniform(0.7, 0.99)
                },
                metadata={
                    "competition": f"mock_competition_{random.randint(1, 10)}",
                    "domain": random.choice(["finance", "healthcare", "retail", "manufacturing"])
                }
            )
            self._samples.append(sample)
    
    def get_samples(self) -> List[BenchmarkSample]:
        return self._samples
    
    def evaluate_sample(self, sample: BenchmarkSample, prediction: Any) -> EvaluationResult:
        ground_truth = sample.ground_truth
        
        if sample.context["task_type"] == "statistical_test":
            if isinstance(prediction, (int, float)):
                pred_value = float(prediction)
                true_value = float(ground_truth["result"])
                error = abs(pred_value - true_value)
                relative_error = error / max(abs(true_value), 1e-10)
                accuracy = max(0, 1 - relative_error)
            else:
                accuracy = 0.0
        else:
            accuracy = random.uniform(0.5, 1.0)
        
        metrics = {
            "accuracy": accuracy,
            "precision": random.uniform(0.6, 0.95),
            "recall": random.uniform(0.6, 0.95),
            "f1_score": random.uniform(0.6, 0.95)
        }
        
        if sample.context["task_type"] == "data_modeling":
            metrics["auc"] = random.uniform(0.7, 0.99)
            metrics["rmse"] = random.uniform(0.1, 1.0)
        
        return EvaluationResult(
            sample_id=sample.id,
            prediction=prediction,
            ground_truth=ground_truth,
            metrics=metrics,
            metadata={
                "task_type": sample.context["task_type"],
                "domain": sample.metadata["domain"]
            }
        )
    
    def compute_aggregate_metrics(self, results: List[EvaluationResult]) -> Dict[str, float]:
        if not results:
            return {}
        
        aggregate = {
            "accuracy": sum(r.metrics["accuracy"] for r in results) / len(results),
            "precision": sum(r.metrics["precision"] for r in results) / len(results),
            "recall": sum(r.metrics["recall"] for r in results) / len(results),
            "f1_score": sum(r.metrics["f1_score"] for r in results) / len(results),
            "total_samples": len(results)
        }
        
        modeling_results = [r for r in results if r.metadata.get("task_type") == "data_modeling"]
        if modeling_results:
            aggregate["avg_auc"] = sum(r.metrics.get("auc", 0) for r in modeling_results) / len(modeling_results)
            aggregate["avg_rmse"] = sum(r.metrics.get("rmse", 0) for r in modeling_results) / len(modeling_results)
        
        by_task = {}
        for result in results:
            task = result.metadata.get("task_type", "unknown")
            if task not in by_task:
                by_task[task] = []
            by_task[task].append(result.metrics["accuracy"])
        
        for task, scores in by_task.items():
            aggregate[f"accuracy_{task}"] = sum(scores) / len(scores)
        
        return aggregate