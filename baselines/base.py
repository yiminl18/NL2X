from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class BaselineConfig:
    name: str
    model_config: Optional[Dict[str, Any]] = None
    seed: int = 42
    timeout: int = 60
    cache_dir: Optional[str] = None
    verbose: bool = False
    max_attempts: int = 1
    validate_answer: bool = False
    debug: bool = False
    confirm: bool = False

@dataclass
class BaselineResult:
    query: str
    response: Any
    metadata: Dict[str, Any]
    execution_time: float
    error: Optional[str] = None

class BaselineInterface(ABC):
    def __init__(self, config: BaselineConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._setup()
    
    @abstractmethod
    def _setup(self):
        pass
    
    @abstractmethod
    def process(self, query: str, context: Optional[Dict[str, Any]] = None) -> BaselineResult:
        pass
    
    @abstractmethod
    def batch_process(self, queries: List[str], contexts: Optional[List[Dict[str, Any]]] = None) -> List[BaselineResult]:
        pass
    
    @abstractmethod
    def get_metrics(self) -> Dict[str, Any]:
        pass
    
    def reset(self):
        self._setup()
    
    def __repr__(self):
        return f"{self.__class__.__name__}(config={self.config.name})"