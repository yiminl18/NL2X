from typing import Dict, Type
from .base import BenchmarkInterface

_BENCHMARK_REGISTRY: Dict[str, Type[BenchmarkInterface]] = {}

def register_benchmark(name: str):
    def decorator(cls):
        _BENCHMARK_REGISTRY[name] = cls
        return cls
    return decorator

def get_benchmark(name: str, **kwargs) -> BenchmarkInterface:
    if name not in _BENCHMARK_REGISTRY:
        raise ValueError(f"Benchmark '{name}' not found. Available benchmarks: {list(_BENCHMARK_REGISTRY.keys())}")
    return _BENCHMARK_REGISTRY[name](**kwargs)

def list_benchmarks():
    return list(_BENCHMARK_REGISTRY.keys())

from .mock_crag import MockCRAGBenchmark
from .mock_dsbench import MockDSBenchBenchmark
from .mock_kramabench import MockKramaBenchmark
from .dsbench import DSBenchBenchmark