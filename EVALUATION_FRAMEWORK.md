# Modular Evaluation Framework

## Overview

This modular evaluation framework allows you to easily evaluate different baselines (algorithms) on various benchmarks. The framework is designed to be extensible - you can add new baselines and benchmarks by implementing the appropriate interfaces.

## Directory Structure

```
├── evaluate.py          # Main evaluation script
├── baselines/          # Baseline implementations
│   ├── __init__.py     # Baseline registry
│   ├── base.py         # Abstract baseline interface
│   ├── mock_unify.py   # Mock Unify baseline
│   ├── mock_docetl.py  # Mock DocETL baseline
│   └── mock_lotus.py   # Mock Lotus baseline
├── benchmarks/         # Benchmark implementations
│   ├── __init__.py     # Benchmark registry
│   ├── base.py         # Abstract benchmark interface
│   ├── mock_crag.py    # Mock CRAG benchmark
│   ├── mock_dsbench.py # Mock DSBench benchmark
│   └── mock_kramabench.py # Mock KramaBench benchmark
└── results/            # Evaluation results (auto-created)
```

## Usage

### Basic Evaluation

Evaluate a single baseline on a benchmark:

```bash
python evaluate.py --baseline mock_unify --benchmark mock_crag
```

### With Options

```bash
python evaluate.py \
    --baseline mock_lotus \
    --benchmark mock_kramabench \
    --max-samples 50 \
    --seed 42 \
    --verbose
```

### Compare All Baselines

Compare all available baselines on a benchmark:

```bash
python evaluate.py \
    --baseline mock_unify \
    --benchmark mock_crag \
    --compare \
    --max-samples 20
```

## Command Line Arguments

- `--baseline`: Baseline algorithm to evaluate (required)
- `--benchmark`: Benchmark to evaluate on (required)
- `--seed`: Random seed for reproducibility (default: 42)
- `--max-samples`: Maximum number of samples to evaluate
- `--output-dir`: Directory to save results (default: ./results)
- `--compare`: Compare all available baselines
- `--verbose`: Enable verbose logging

## Adding New Baselines

To add a new baseline, create a new file in `baselines/` that:

1. Imports the base classes and registry:
```python
from .base import BaselineInterface, BaselineResult, BaselineConfig
from . import register_baseline
```

2. Implements the `BaselineInterface`:
```python
@register_baseline("your_baseline_name")
class YourBaseline(BaselineInterface):
    def _setup(self):
        # Initialize your baseline
        pass
    
    def process(self, query: str, context: Optional[Dict] = None) -> BaselineResult:
        # Process a single query
        pass
    
    def batch_process(self, queries: List[str], contexts: Optional[List[Dict]] = None) -> List[BaselineResult]:
        # Process multiple queries
        pass
    
    def get_metrics(self) -> Dict[str, Any]:
        # Return baseline-specific metrics
        pass
```

3. Import it in `baselines/__init__.py`:
```python
from .your_baseline import YourBaseline
```

## Adding New Benchmarks

To add a new benchmark, create a new file in `benchmarks/` that:

1. Imports the base classes and registry:
```python
from .base import BenchmarkInterface, BenchmarkSample, EvaluationResult, BenchmarkConfig
from . import register_benchmark
```

2. Implements the `BenchmarkInterface`:
```python
@register_benchmark("your_benchmark_name")
class YourBenchmark(BenchmarkInterface):
    def _setup(self):
        # Initialize benchmark settings
        pass
    
    def _load_data(self):
        # Load benchmark data into self._samples
        pass
    
    def get_samples(self) -> List[BenchmarkSample]:
        # Return all samples
        return self._samples
    
    def evaluate_sample(self, sample: BenchmarkSample, prediction: Any) -> EvaluationResult:
        # Evaluate a single prediction
        pass
    
    def compute_aggregate_metrics(self, results: List[EvaluationResult]) -> Dict[str, float]:
        # Compute aggregate metrics across all results
        pass
```

3. Import it in `benchmarks/__init__.py`:
```python
from .your_benchmark import YourBenchmark
```

## Output Format

Results are saved as JSON files in the `results/` directory with the following structure:

```json
{
  "baseline": "baseline_name",
  "benchmark": "benchmark_name",
  "timestamp": "2024-01-01 12:00:00",
  "total_samples": 100,
  "total_time": 45.2,
  "avg_time_per_sample": 0.452,
  "aggregate_metrics": {
    "accuracy": 0.85,
    "f1_score": 0.82,
    ...
  },
  "baseline_metrics": {
    "total_queries": 100,
    "cache_hits": 20,
    ...
  },
  "config": {
    "baseline": {...},
    "benchmark": {...}
  }
}
```

## Integration with Real Baselines

To integrate with the actual baseline implementations (Unify, DocETL, Lotus):

1. Replace the mock implementations with real ones that call the actual systems
2. Update the `_setup()` method to initialize connections/configurations
3. Implement proper `process()` methods that interact with the real systems
4. Add any necessary dependencies to requirements.txt

## Integration with Real Benchmarks

To integrate with the actual benchmarks (CRAG, DSBench, KramaBench):

1. Update `_load_data()` to load real data from the benchmark folders
2. Implement proper evaluation metrics specific to each benchmark
3. Add data preprocessing as needed
4. Handle benchmark-specific file formats and structures

## Notes

- The current implementation uses mock baselines and benchmarks for demonstration
- Results are automatically saved with timestamps to prevent overwriting
- The framework supports parallel evaluation through batch processing
- Metrics are computed both at the sample level and aggregate level
- The comparison feature automatically evaluates all registered baselines