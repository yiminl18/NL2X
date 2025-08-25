import random
from typing import Any, Dict, List, Optional
from .base import BenchmarkInterface, BenchmarkSample, EvaluationResult, BenchmarkConfig
from . import register_benchmark

@register_benchmark("mock_kramabench")
class MockKramaBenchmark(BenchmarkInterface):
    def _setup(self):
        self.domains = ["archeology", "astronomy", "biomedical", "environment", "legal", "wildfire"]
        self.doc_types = ["pdf", "html", "text", "mixed"]
        if self.config.verbose:
            self.logger.info(f"Initialized MockKramaBench benchmark with config: {self.config}")
    
    def _load_data(self):
        num_samples = self.config.max_samples or 80
        
        for i in range(num_samples):
            domain = random.choice(self.domains)
            doc_type = random.choice(self.doc_types)
            num_documents = random.randint(1, 20)
            total_pages = random.randint(10, 500)
            
            sample = BenchmarkSample(
                id=f"krama_{domain}_{i:03d}",
                query=f"Complex question about {domain} requiring analysis of {num_documents} documents",
                context={
                    "domain": domain,
                    "document_type": doc_type,
                    "num_documents": num_documents,
                    "total_pages": total_pages,
                    "requires_multi_hop": random.choice([True, False]),
                    "requires_numerical": random.choice([True, False]),
                    "requires_extraction": True
                },
                ground_truth={
                    "answer": f"Ground truth answer for {domain} question {i}",
                    "evidence": [f"Evidence from doc {j}" for j in range(random.randint(1, 3))],
                    "numerical_value": random.uniform(0, 1000) if random.random() > 0.5 else None
                },
                metadata={
                    "question_complexity": random.choice(["simple", "moderate", "complex", "very_complex"]),
                    "answer_type": random.choice(["extractive", "abstractive", "numerical", "boolean"]),
                    "source_year": random.randint(2020, 2024)
                }
            )
            self._samples.append(sample)
    
    def get_samples(self) -> List[BenchmarkSample]:
        return self._samples
    
    def evaluate_sample(self, sample: BenchmarkSample, prediction: Any) -> EvaluationResult:
        ground_truth = sample.ground_truth
        
        metrics = {}
        
        if isinstance(prediction, dict):
            pred_answer = prediction.get("answer", "")
        else:
            pred_answer = str(prediction)
        
        exact_match = float(pred_answer.lower().strip() == ground_truth["answer"].lower().strip())
        metrics["exact_match"] = exact_match
        
        pred_words = set(pred_answer.lower().split())
        truth_words = set(ground_truth["answer"].lower().split())
        if len(truth_words) > 0:
            overlap = len(pred_words & truth_words)
            precision = overlap / len(pred_words) if pred_words else 0
            recall = overlap / len(truth_words)
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        else:
            f1 = 0.0
        
        metrics["f1_score"] = f1
        
        if ground_truth.get("numerical_value") is not None:
            try:
                import re
                numbers = re.findall(r'[-+]?\d*\.?\d+', pred_answer)
                if numbers:
                    pred_num = float(numbers[0])
                    true_num = float(ground_truth["numerical_value"])
                    rel_error = abs(pred_num - true_num) / max(abs(true_num), 1)
                    metrics["numerical_accuracy"] = max(0, 1 - rel_error)
                else:
                    metrics["numerical_accuracy"] = 0.0
            except:
                metrics["numerical_accuracy"] = 0.0
        
        metrics["semantic_similarity"] = random.uniform(0.4, 1.0)
        
        if isinstance(prediction, dict) and "evidence" in prediction:
            metrics["evidence_quality"] = random.uniform(0.5, 1.0)
        
        return EvaluationResult(
            sample_id=sample.id,
            prediction=prediction,
            ground_truth=ground_truth,
            metrics=metrics,
            metadata={
                "domain": sample.context["domain"],
                "answer_type": sample.metadata["answer_type"]
            }
        )
    
    def compute_aggregate_metrics(self, results: List[EvaluationResult]) -> Dict[str, float]:
        if not results:
            return {}
        
        aggregate = {
            "exact_match": sum(r.metrics["exact_match"] for r in results) / len(results),
            "f1_score": sum(r.metrics["f1_score"] for r in results) / len(results),
            "semantic_similarity": sum(r.metrics["semantic_similarity"] for r in results) / len(results),
            "total_samples": len(results)
        }
        
        numerical_results = [r for r in results if "numerical_accuracy" in r.metrics]
        if numerical_results:
            aggregate["numerical_accuracy"] = sum(r.metrics["numerical_accuracy"] for r in numerical_results) / len(numerical_results)
        
        evidence_results = [r for r in results if "evidence_quality" in r.metrics]
        if evidence_results:
            aggregate["evidence_quality"] = sum(r.metrics["evidence_quality"] for r in evidence_results) / len(evidence_results)
        
        by_domain = {}
        
        for result in results:
            domain = result.metadata.get("domain", "unknown")
            if domain not in by_domain:
                by_domain[domain] = []
            by_domain[domain].append(result.metrics["f1_score"])
        
        for domain, scores in by_domain.items():
            aggregate[f"f1_{domain}"] = sum(scores) / len(scores)
        
        return aggregate