import random
from typing import Any, Dict, List, Optional
from .base import BenchmarkInterface, BenchmarkSample, EvaluationResult, BenchmarkConfig
from . import register_benchmark

@register_benchmark("mock_crag")
class MockCRAGBenchmark(BenchmarkInterface):
    def _setup(self):
        self.categories = ["finance", "movie", "music", "sports", "open"]
        if self.config.verbose:
            self.logger.info(f"Initialized MockCRAG benchmark with config: {self.config}")
    
    def _load_data(self):
        num_samples = self.config.max_samples or 100
        
        for i in range(num_samples):
            category = random.choice(self.categories)
            sample = BenchmarkSample(
                id=f"crag_{i:04d}",
                query=f"Mock CRAG question {i} about {category}?",
                context={
                    "category": category,
                    "has_knowledge_graph": random.choice([True, False]),
                    "requires_retrieval": random.choice([True, False])
                },
                ground_truth={
                    "answer": f"Ground truth answer for question {i}",
                    "supporting_facts": [f"Fact {j}" for j in range(random.randint(1, 3))]
                },
                metadata={
                    "question_type": random.choice(["factual", "reasoning", "complex"]),
                    "source": "mock_crag_dataset"
                }
            )
            self._samples.append(sample)
    
    def get_samples(self) -> List[BenchmarkSample]:
        return self._samples
    
    def evaluate_sample(self, sample: BenchmarkSample, prediction: Any) -> EvaluationResult:
        ground_truth = sample.ground_truth["answer"]
        
        if isinstance(prediction, dict):
            pred_answer = prediction.get("answer", "")
        else:
            pred_answer = str(prediction)
        
        exact_match = float(pred_answer.lower().strip() == ground_truth.lower().strip())
        
        pred_words = set(pred_answer.lower().split())
        truth_words = set(ground_truth.lower().split())
        if len(truth_words) > 0:
            f1_score = 2 * len(pred_words & truth_words) / (len(pred_words) + len(truth_words))
        else:
            f1_score = 0.0
        
        metrics = {
            "exact_match": exact_match,
            "f1_score": f1_score,
            "similarity": random.uniform(0.3, 1.0)
        }
        
        return EvaluationResult(
            sample_id=sample.id,
            prediction=prediction,
            ground_truth=sample.ground_truth,
            metrics=metrics,
            metadata={
                "category": sample.context["category"]
            }
        )
    
    def compute_aggregate_metrics(self, results: List[EvaluationResult]) -> Dict[str, float]:
        if not results:
            return {}
        
        aggregate = {
            "exact_match": sum(r.metrics["exact_match"] for r in results) / len(results),
            "f1_score": sum(r.metrics["f1_score"] for r in results) / len(results),
            "similarity": sum(r.metrics["similarity"] for r in results) / len(results),
            "total_samples": len(results)
        }
        
        by_category = {}
        for result in results:
            cat = result.metadata.get("category", "unknown")
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(result.metrics["f1_score"])
        
        for cat, scores in by_category.items():
            aggregate[f"f1_{cat}"] = sum(scores) / len(scores)
        
        return aggregate