from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class BenchmarkConfig:
    name: str
    data_dir: Optional[str] = None
    split: str = "test"
    max_samples: Optional[int] = None
    seed: int = 42
    verbose: bool = False
    metrics: List[str] = None

@dataclass
class BenchmarkSample:
    id: str
    query: str
    context: Optional[Dict[str, Any]] = None
    ground_truth: Optional[Any] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class EvaluationResult:
    sample_id: str
    prediction: Any
    ground_truth: Any
    metrics: Dict[str, float]
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class ContentDataType:
    type: str
    path: Optional[str] = None

class BenchmarkInterface(ABC):
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._samples = []
        self._setup()
        self._load_data()
    
    @abstractmethod
    def _setup(self):
        pass
    
    @abstractmethod
    def _load_data(self):
        pass
    
    @abstractmethod
    def get_samples(self) -> List[BenchmarkSample]:
        pass
    
    @abstractmethod
    def evaluate_sample(self, sample: BenchmarkSample, prediction: Any) -> EvaluationResult:
        pass
    
    @abstractmethod
    def compute_aggregate_metrics(self, results: List[EvaluationResult]) -> Dict[str, float]:
        pass
    
    def get_sample_by_id(self, sample_id: str) -> Optional[BenchmarkSample]:
        for sample in self._samples:
            if sample.id == sample_id:
                return sample
        return None
    
    def get_statistics(self) -> Dict[str, Any]:
        return {
            "name": self.config.name,
            "total_samples": len(self._samples),
            "split": self.config.split,
        }
    
    def __repr__(self):
        return f"{self.__class__.__name__}(config={self.config.name}, samples={len(self._samples)})"