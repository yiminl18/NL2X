import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import argparse
from utils import parse_args
from baselines import get_baseline, list_baselines
from baselines.base import BaselineConfig
from benchmarks import get_benchmark, list_benchmarks
from benchmarks.base import BenchmarkConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EvaluationFramework:
    def __init__(self, output_dir: str = "./results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = []
    
    def evaluate(
        self,
        baseline_name: str,
        benchmark_name: str,
        baseline_config: Optional[Dict[str, Any]] = None,
        benchmark_config: Optional[Dict[str, Any]] = None,
        save_results: bool = True
    ) -> Dict[str, Any]:
        logger.info(f"Starting evaluation: {baseline_name} on {benchmark_name}")
        
        baseline_cfg = BaselineConfig(
            name=baseline_name,
            **(baseline_config or {})
        )
        baseline = get_baseline(baseline_name, config=baseline_cfg)
        
        benchmark_cfg = BenchmarkConfig(
            name=benchmark_name,
            **(benchmark_config or {})
        )
        benchmark = get_benchmark(benchmark_name, config=benchmark_cfg)
        
        logger.info(f"Loaded baseline: {baseline}")
        logger.info(f"Loaded benchmark: {benchmark}")
        logger.info(f"Benchmark statistics: {benchmark.get_statistics()}")
        
        samples = benchmark.get_samples()
        logger.info(f"Processing {len(samples)} samples...")
        
        eval_results = []
        start_time = time.time()
        
        for i, sample in enumerate(samples):
            if i % 10 == 0:
                logger.info(f"Processing sample {i+1}/{len(samples)}")
            
            baseline_result = baseline.process(
                query=sample.query,
                context=sample.context
            )
            
            eval_result = benchmark.evaluate_sample(
                sample=sample,
                prediction=baseline_result.response
            )
            eval_results.append(eval_result)
        
        total_time = time.time() - start_time
        
        aggregate_metrics = benchmark.compute_aggregate_metrics(eval_results)
        baseline_metrics = baseline.get_metrics()
        
        final_results = {
            "baseline": baseline_name,
            "benchmark": benchmark_name,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_samples": len(samples),
            "total_time": total_time,
            "avg_time_per_sample": total_time / len(samples),
            "aggregate_metrics": aggregate_metrics,
            "baseline_metrics": baseline_metrics,
            "config": {
                "baseline": baseline_config,
                "benchmark": benchmark_config
            }
        }
        
        if save_results:
            self._save_results(final_results, baseline_name, benchmark_name)
        
        self.results.append(final_results)
        
        logger.info("Evaluation completed!")
        logger.info(f"Aggregate metrics: {aggregate_metrics}")
        
        return final_results
    
    def _save_results(self, results: Dict[str, Any], baseline: str, benchmark: str):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{baseline}_{benchmark}_{timestamp}.json"
        filepath = self.output_dir / filename
        
        # Convert TaskDifficulty enums to strings for JSON serialization
        results_json = json.loads(json.dumps(results, default=str))
        
        with open(filepath, 'w') as f:
            json.dump(results_json, f, indent=2)
        
        logger.info(f"Results saved to: {filepath}")
    
    def compare_baselines(
        self,
        baselines: List[str],
        benchmark: str,
        benchmark_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        comparison_results = {}
        
        for baseline in baselines:
            logger.info(f"Evaluating {baseline}...")
            results = self.evaluate(
                baseline_name=baseline,
                benchmark_name=benchmark,
                benchmark_config=benchmark_config
            )
            comparison_results[baseline] = results
        
        comparison_summary = self._generate_comparison_summary(comparison_results)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"comparison_{benchmark}_{timestamp}.json"
        filepath = self.output_dir / filename
        
        # Convert any enums to strings for JSON serialization
        comparison_data = {
            "results": comparison_results,
            "summary": comparison_summary
        }
        comparison_json = json.loads(json.dumps(comparison_data, default=str))
        
        with open(filepath, 'w') as f:
            json.dump(comparison_json, f, indent=2)
        
        logger.info(f"Comparison results saved to: {filepath}")
        
        return comparison_summary
    
    def _generate_comparison_summary(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        summary = {
            "baselines": list(results.keys()),
            "metrics_comparison": {},
            "metric_availability": {}
        }
        
        all_metrics = set()
        for baseline_results in results.values():
            all_metrics.update(baseline_results["aggregate_metrics"].keys())
        
        for metric in all_metrics:
            summary["metrics_comparison"][metric] = {}
            available_baselines = []
            
            for baseline, baseline_results in results.items():
                value = baseline_results["aggregate_metrics"].get(metric, None)
                summary["metrics_comparison"][metric][baseline] = value
                if value is not None:
                    available_baselines.append(baseline)
            
            summary["metric_availability"][metric] = {
                "available_baselines": available_baselines,
                "coverage": len(available_baselines) / len(results) if results else 0.0,
                "missing_baselines": [b for b in results.keys() if b not in available_baselines]
            }
        
        for metric, values in summary["metrics_comparison"].items():
            valid_values = [(k, v) for k, v in values.items() if v is not None and isinstance(v, (int, float))]
            
            if valid_values:
                best_baseline, best_value = max(valid_values, key=lambda x: x[1])
                worst_baseline, worst_value = min(valid_values, key=lambda x: x[1])
                
                summary["metrics_comparison"][metric].update({
                    "best": best_baseline,
                    "best_value": best_value,
                    "worst": worst_baseline,
                    "worst_value": worst_value,
                    "mean": sum(v for _, v in valid_values) / len(valid_values),
                    "std": (sum((v - sum(v2 for _, v2 in valid_values) / len(valid_values)) ** 2 
                               for _, v in valid_values) / len(valid_values)) ** 0.5 if len(valid_values) > 1 else 0.0,
                    "valid_comparisons": len(valid_values)
                })
            else:
                summary["metrics_comparison"][metric].update({
                    "best": None,
                    "best_value": None,
                    "worst": None,
                    "worst_value": None,
                    "mean": None,
                    "std": None,
                    "valid_comparisons": 0
                })
        
        summary["comparison_quality"] = {
            "total_metrics": len(all_metrics),
            "fully_comparable_metrics": len([m for m in all_metrics 
                                           if summary["metric_availability"][m]["coverage"] == 1.0]),
            "partially_comparable_metrics": len([m for m in all_metrics 
                                               if 0 < summary["metric_availability"][m]["coverage"] < 1.0]),
            "incomparable_metrics": len([m for m in all_metrics 
                                       if summary["metric_availability"][m]["coverage"] == 0.0])
        }
        
        return summary

def main():
    args = parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    framework = EvaluationFramework(output_dir=args.output_dir)
    
    baseline_config = {
        "seed": args.seed,
        "verbose": args.verbose
    }
    
    benchmark_config = {
        "seed": args.seed,
        "max_samples": args.max_samples,
        "verbose": args.verbose
    }
    
    if args.compare:
        logger.info("Running baseline comparison...")
        results = framework.compare_baselines(
            baselines=list_baselines(),
            benchmark=args.benchmark,
            benchmark_config=benchmark_config
        )
        
        print("\n" + "="*50)
        print("COMPARISON RESULTS")
        print("="*50)
        print(json.dumps(results, indent=2))
    else:
        results = framework.evaluate(
            baseline_name=args.baseline,
            benchmark_name=args.benchmark,
            baseline_config=baseline_config,
            benchmark_config=benchmark_config
        )
        
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        print(f"Baseline: {results['baseline']}")
        print(f"Benchmark: {results['benchmark']}")
        print(f"Total samples: {results['total_samples']}")
        print(f"Total time: {results['total_time']:.2f}s")
        print(f"Avg time per sample: {results['avg_time_per_sample']:.3f}s")
        print("\nAggregate Metrics:")
        for metric, value in results['aggregate_metrics'].items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: {value}")

if __name__ == "__main__":
    main()