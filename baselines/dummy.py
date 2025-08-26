import time
import random
from typing import Any, Dict, List, Optional
from .base import BaselineInterface, BaselineResult, BaselineConfig
from . import register_baseline

@register_baseline("dummy")
class DummyBaseline(BaselineInterface):
    def _setup(self):
        self.total_queries = 0
        self.total_time = 0
        
        if self.config.verbose:
            self.logger.info(f"Initialized Dummy baseline with config: {self.config}")

    def process(self, query: str, context: Optional[Dict[str, Any]] = None) -> BaselineResult:
        start_time = time.time()
        print("=== Dummy Baseline ===")
        print(f"Query: {query}")
        # print(f"Context: {context}")
        print("==============================")
        response = {
            "answer": f"I don't know :)",
        }
        
        execution_time = time.time() - start_time
        self.total_queries += 1
        self.total_time += execution_time
        
        result = BaselineResult(
            query=query,
            response=response,
            metadata={
                "baseline": "dummy",
                "context_provided": context is not None,
            },
            execution_time=execution_time
        )

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
            "baseline_type": "dummy"
        }