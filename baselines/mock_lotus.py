import time
import random
from typing import Any, Dict, List, Optional
from .base import BaselineInterface, BaselineResult, BaselineConfig
from . import register_baseline

@register_baseline("mock_lotus")
class MockLotusBaseline(BaselineInterface):
    def _setup(self):
        self.total_queries = 0
        self.total_time = 0
        self.cache = {}
        self.semantic_ops_stats = {
            "filters_applied": 0,
            "joins_performed": 0,
            "aggregations_computed": 0
        }
        
        if self.config.verbose:
            self.logger.info(f"Initialized MockLotus baseline with config: {self.config}")
    
    def process(self, query: str, context: Optional[Dict[str, Any]] = None) -> BaselineResult:
        start_time = time.time()
        
        cache_key = f"{query}_{str(context)}"
        if cache_key in self.cache:
            result = self.cache[cache_key]
            result.metadata['cache_hit'] = True
            return result
        
        time.sleep(random.uniform(0.12, 0.32))
        
        num_rows = random.randint(100, 1000)
        semantic_ops = random.choice([
            ["sem_filter", "sem_agg"],
            ["sem_map", "sem_filter", "sem_topk"],
            ["sem_join", "sem_filter", "sem_extract"],
            ["sem_search", "sem_cluster", "sem_agg"]
        ])
        
        self.semantic_ops_stats["filters_applied"] += semantic_ops.count("sem_filter")
        self.semantic_ops_stats["joins_performed"] += semantic_ops.count("sem_join")
        self.semantic_ops_stats["aggregations_computed"] += semantic_ops.count("sem_agg")
        
        mock_response = {
            "answer": f"Mock Lotus semantic result for: {query[:50]}...",
            "confidence": random.uniform(0.72, 0.96),
            "dataframe_shape": (num_rows, random.randint(3, 10)),
            "semantic_operations": semantic_ops,
            "execution_plan": {
                "optimizer": "cascade",
                "model": "gpt-4o-mini",
                "operations": [
                    {"op": op, "rows_processed": random.randint(50, num_rows)}
                    for op in semantic_ops
                ]
            },
            "nl_expression": f"Filter where relevance > 0.5 then aggregate by category"
        }
        
        execution_time = time.time() - start_time
        self.total_queries += 1
        self.total_time += execution_time
        
        result = BaselineResult(
            query=query,
            response=mock_response,
            metadata={
                "baseline": "mock_lotus",
                "context_provided": context is not None,
                "cache_hit": False,
                "rows_processed": num_rows,
                "semantic_ops_count": len(semantic_ops)
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
            "semantic_ops_stats": self.semantic_ops_stats,
            "baseline_type": "mock_lotus"
        }