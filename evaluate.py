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

            # Confirm mode: prompt for confirmation before processing
            if baseline_cfg.confirm:
                print("\n" + "="*80)
                mode_text = "[DEBUG MODE]" if baseline_cfg.debug else "[CONFIRM MODE]"
                print(f"{mode_text} Sample {i+1}/{len(samples)}")
                print("="*80)
                print(f"\n📝 Query:")
                print("-"*40)
                print(sample.query)
                print("-"*40)

                if sample.context:
                    print(f"\n📂 Data Sources:")
                    print("-"*40)
                    for key, value in sample.context.items():
                        # Display data source information
                        if hasattr(value, 'path'):
                            print(f"  • {key}: {value.path}")
                        elif hasattr(value, 'type'):
                            print(f"  • {key}: {value.type} (embedded content)")
                        else:
                            print(f"  • {key}: {type(value).__name__}")

                        # Show preview of content if available and not too large
                        if hasattr(value, 'content') and value.content:
                            content_str = str(value.content)
                            if len(content_str) > 500:
                                print(f"    Preview: {content_str[:500]}...")
                            else:
                                print(f"    Content: {content_str}")
                    print("-"*40)
                else:
                    print("\n📂 Data Sources: None")

                print("\n➡️ Continue processing this sample? (Y/n): ", end="")
                user_input = input().strip().lower()

                if user_input and user_input != 'y':
                    logger.info("User aborted debug mode execution")
                    print("\n❌ Execution aborted by user")
                    return {
                        "baseline": baseline_name,
                        "benchmark": benchmark_name,
                        "status": "aborted",
                        "aborted_at_sample": i+1,
                        "total_samples": len(samples),
                        "total_time": 0,
                        "avg_time_per_sample": 0,
                        'aggregate_metrics': {},
                        "message": "User aborted during debug mode"
                    }

            baseline_result = baseline.process(
                query=sample.query,
                context=sample.context
            )

            eval_result = benchmark.evaluate_sample(
                sample=sample,
                prediction=baseline_result.response
            )
            eval_results.append(eval_result)

            # 立即打印当前题目的答案（如果开启了debug或verbose模式）
            if baseline_cfg.debug or baseline_cfg.verbose:
                print(f"\n✅ Sample {i+1} Answer:")
                print(f"{baseline_result.response}")
                print("-"*40)
        
        total_time = time.time() - start_time
        print(eval_results)
        try:
            aggregate_metrics = benchmark.compute_aggregate_metrics(eval_results)
        except Exception as e:
            logger.error(f"Error computing aggregate metrics for benchmark {benchmark_name}: {e}")
            aggregate_metrics = {"error": f"Failed to compute metrics: {str(e)}"}
        
        try:
            baseline_metrics = baseline.get_metrics()
        except Exception as e:
            logger.error(f"Error getting baseline metrics for {baseline_name}: {e}")
            baseline_metrics = {"error": f"Failed to get baseline metrics: {str(e)}"}
        
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
        baseline_config: Optional[Dict[str, Any]] = None,
        benchmark_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        comparison_results = {}
        
        for baseline in baselines:
            logger.info(f"Evaluating {baseline}...")
            results = self.evaluate(
                baseline_name=baseline,
                benchmark_name=benchmark,
                baseline_config=baseline_config,
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
            if isinstance(baseline_results.get("aggregate_metrics"), dict):
                all_metrics.update(baseline_results["aggregate_metrics"].keys())
            else:
                logger.warning(f"Invalid aggregate_metrics format for baseline {baseline_results.get('baseline', 'unknown')}")
        
        for metric in all_metrics:
            summary["metrics_comparison"][metric] = {}
            available_baselines = []
            
            for baseline, baseline_results in results.items():
                if isinstance(baseline_results.get("aggregate_metrics"), dict):
                    value = baseline_results["aggregate_metrics"].get(metric, None)
                    summary["metrics_comparison"][metric][baseline] = value
                    if value is not None:
                        available_baselines.append(baseline)
                else:
                    summary["metrics_comparison"][metric][baseline] = None
            
            summary["metric_availability"][metric] = {
                "available_baselines": available_baselines,
                "coverage": len(available_baselines) / len(results) if results else 0.0,
                "missing_baselines": [b for b in results.keys() if b not in available_baselines]
            }
        
        for metric, values in summary["metrics_comparison"].items():
            # Filter for valid numeric values only
            valid_values = []
            for k, v in values.items():
                if v is not None:
                    try:
                        # Try to convert to float to ensure it's numeric
                        float_v = float(v)
                        valid_values.append((k, float_v))
                    except (ValueError, TypeError):
                        # Skip non-numeric values
                        logger.warning(f"Skipping non-numeric metric value for {metric} in baseline {k}: {v}")
                        continue
            
            if valid_values:
                best_baseline, best_value = max(valid_values, key=lambda x: x[1])
                worst_baseline, worst_value = min(valid_values, key=lambda x: x[1])
                mean_value = sum(v for _, v in valid_values) / len(valid_values)
                
                # Calculate standard deviation
                if len(valid_values) > 1:
                    variance = sum((v - mean_value) ** 2 for _, v in valid_values) / len(valid_values)
                    std_value = variance ** 0.5
                else:
                    std_value = 0.0
                
                summary["metrics_comparison"][metric].update({
                    "best": best_baseline,
                    "best_value": best_value,
                    "worst": worst_baseline,
                    "worst_value": worst_value,
                    "mean": mean_value,
                    "std": std_value,
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
        "verbose": args.verbose,
        "max_attempts": args.max_attempts,
        "validate_answer": args.validate_answer,
        "debug": args.debug,
        "confirm": args.confirm or args.debug  # debug mode automatically enables confirm
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
            baseline_config=baseline_config,
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