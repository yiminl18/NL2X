from typing import Dict, Type
from .base import BaselineInterface

_BASELINE_REGISTRY: Dict[str, Type[BaselineInterface]] = {}

def register_baseline(name: str):
    def decorator(cls):
        _BASELINE_REGISTRY[name] = cls
        return cls
    return decorator

def get_baseline(name: str, **kwargs) -> BaselineInterface:
    if name not in _BASELINE_REGISTRY:
        raise ValueError(f"Baseline '{name}' not found. Available baselines: {list(_BASELINE_REGISTRY.keys())}")
    return _BASELINE_REGISTRY[name](**kwargs)

def list_baselines():
    return list(_BASELINE_REGISTRY.keys())

from .mock_docetl import MockDocETLBaseline
from .mock_lotus import MockLotusBaseline
from .gpt4o_base import GPT4oBaseBaseline
from .gpt_4o_zero_shot import GPT4oZeroShotBaseline
from .dummy import DummyBaseline