import argparse
from baselines import list_baselines
from benchmarks import list_benchmarks

def parse_args():
    parser = argparse.ArgumentParser(description="Modular Evaluation Framework")
    
    parser.add_argument(
        "--baseline",
        type=str,
        required=False,
        help=f"Baseline algorithm to evaluate. Available: {list_baselines()}"
    )
    
    parser.add_argument(
        "--benchmark",
        type=str,
        required=True,
        help=f"Benchmark to evaluate on. Available: {list_benchmarks()}"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    parser.add_argument(
        "--max-samples",
        type=int,
        default=1,
        help="Maximum number of samples to evaluate"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results",
        help="Directory to save results"
    )
    
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare all available baselines"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.compare and args.baseline is None:
        parser.error("--baseline is required when not using --compare")
    
    return args



