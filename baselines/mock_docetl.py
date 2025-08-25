import time
import random
from typing import Any, Dict, List, Optional
from .base import BaselineInterface, BaselineResult, BaselineConfig
from . import register_baseline

@register_baseline("mock_docetl")
class MockDocETLBaseline(BaselineInterface):
    def _setup(self):
        self.total_queries = 0
        self.total_time = 0
        self.cache = {}
        self.pipeline_stats = {
            "operations_executed": 0,
            "documents_processed": 0
        }
        
        if self.config.verbose:
            self.logger.info(f"Initialized MockDocETL baseline with config: {self.config}")
    
    def process(self, query: str, context: Optional[Dict[str, Any]] = None) -> BaselineResult:
        start_time = time.time()
        
        cache_key = f"{query}_{str(context)}"
        if cache_key in self.cache:
            result = self.cache[cache_key]
            result.metadata['cache_hit'] = True
            return result
        
        time.sleep(random.uniform(0.15, 0.35))
        
        num_docs = random.randint(10, 100)
        self.pipeline_stats["documents_processed"] += num_docs
        self.pipeline_stats["operations_executed"] += random.randint(3, 8)
        
        mock_response = {
            "answer": f"Mock DocETL pipeline result for: {query[:50]}...",
            "confidence": random.uniform(0.75, 0.98),
            "pipeline": {
                "name": "mock_etl_pipeline",
                "operations": [
                    {"type": "extract", "status": "completed"},
                    {"type": "map", "status": "completed"},
                    {"type": "reduce", "status": "completed"},
                    {"type": "resolve", "status": "completed"}
                ],
                "documents_processed": num_docs,
                "transformations_applied": random.randint(2, 5)
            },
            "optimizations": {
                "enabled": True,
                "strategies": ["parallel_map", "reduce_optimization", "caching"]
            }
        }
        
        execution_time = time.time() - start_time
        self.total_queries += 1
        self.total_time += execution_time
        
        result = BaselineResult(
            query=query,
            response=mock_response,
            metadata={
                "baseline": "mock_docetl",
                "context_provided": context is not None,
                "cache_hit": False,
                "documents_processed": num_docs,
                "pipeline_operations": len(mock_response["pipeline"]["operations"])
            },
            execution_time=execution_time
        )
        
        self.cache[cache_key] = result
        return result
    
    def batch_process(self, queries: List[str], contexts: Optional[List[Dict[str, Any]]] = None) -> List[BaselineResult]:
        if contexts is None:
            contexts = [None] * len(queries)
        
        results = []
        for query, context in zip(queries, contexts):
            results.append(self.process(query, context))
        
        return results
    
    def get_metrics(self) -> Dict[str, Any]:
        return {
            "total_queries": self.total_queries,
            "total_time": self.total_time,
            "average_time": self.total_time / max(1, self.total_queries),
            "cache_size": len(self.cache),
            "pipeline_stats": self.pipeline_stats,
            "baseline_type": "mock_docetl"
        }