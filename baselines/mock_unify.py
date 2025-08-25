import time
import random
from typing import Any, Dict, List, Optional
from .base import BaselineInterface, BaselineResult, BaselineConfig
from . import register_baseline

@register_baseline("mock_unify")
class MockUnifyBaseline(BaselineInterface):
    def _setup(self):
        self.total_queries = 0
        self.total_time = 0
        self.cache = {}
        
        if self.config.verbose:
            self.logger.info(f"Initialized MockUnify baseline with config: {self.config}")
    
    def process(self, query: str, context: Optional[Dict[str, Any]] = None) -> BaselineResult:
        start_time = time.time()
        
        cache_key = f"{query}_{str(context)}"
        if cache_key in self.cache:
            result = self.cache[cache_key]
            result.metadata['cache_hit'] = True
            return result
        
        time.sleep(random.uniform(0.1, 0.3))
        
        mock_response = {
            "answer": f"Mock Unify response for: {query[:50]}...",
            "confidence": random.uniform(0.7, 0.95),
            "operations": ["scan", "filter", "extract", "aggregate"],
            "plan": {
                "steps": [
                    {"op": "scan", "target": "documents"},
                    {"op": "filter", "condition": "relevance > 0.5"},
                    {"op": "extract", "fields": ["key_info"]},
                    {"op": "aggregate", "method": "summarize"}
                ]
            }
        }
        
        execution_time = time.time() - start_time
        self.total_queries += 1
        self.total_time += execution_time
        
        result = BaselineResult(
            query=query,
            response=mock_response,
            metadata={
                "baseline": "mock_unify",
                "context_provided": context is not None,
                "cache_hit": False,
                "operations_count": len(mock_response["operations"])
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
            "baseline_type": "mock_unify"
        }