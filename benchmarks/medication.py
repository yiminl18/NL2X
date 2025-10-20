import random
from typing import Any, Dict, List, Optional
import json

from model.litellm_client import llm_call
from .base import BenchmarkInterface, BenchmarkSample, EvaluationResult, BenchmarkConfig, ContentDataType
from . import register_benchmark
import os
import base64
import pandas as pd
import tiktoken

MEDICATION_PATH = "./benchmarks/medication/"
MEDICATION_DATA_PATH = MEDICATION_PATH + "medical_transcripts.json"

class MedicationDataLoader:
    def __init__(self, data_path: str):
        # Read medical_transcripts.json
        with open(data_path, 'r') as f:
            data = json.load(f)

        documents = []
        ground_truths = []

        for item in data:
            # Each item has 'src' (transcript), 'tgt' (clinical note), and 'file' (identifier)
            documents.append({'src': item['src'], 'file': item['file']})

            # Extract ground truth medications from the clinical note
            # The tgt field contains structured clinical notes with medication information
            ground_truths.append([])

        self.documents = documents
        self.ground_truths = ground_truths

        self.question = """Analyze the following medical transcript and perform the following tasks:

1. Extract all medications mentioned in the transcript
2. For each unique medication (resolve similar medication names to a standardized form):
   - Identify any mentioned side effects
   - Identify the therapeutic uses (what conditions/symptoms the medication was prescribed for)

Return your answer as a JSON object with the following structure:
{
  "medications": [
    {
      "medication": "standardized medication name",
      "side_effects": "summary of side effects mentioned",
      "uses": "summary of therapeutic uses mentioned"
    }
  ]
}

Base your analysis solely on information from the provided transcript. Include quotes from the transcript where relevant."""
@register_benchmark("medication")
class MedicationBenchmark(BenchmarkInterface):
    def _setup(self):
        self.data_formats = ["json"]
        if self.config.verbose:
            self.logger.info(f"Initialized Medication benchmark with config: {self.config}")

    def _load_data(self) -> List[BenchmarkSample]:
        samples = []
        data_loader = MedicationDataLoader(MEDICATION_DATA_PATH)

        # Determine number of samples to use
        num_samples = min(self.config.max_samples, len(data_loader.documents))

        # Collect all documents into a single list
        docs = []
        ground_truths = []
        for idx in range(num_samples):
            doc = data_loader.documents[idx]
            ground_truth = data_loader.ground_truths[idx]
            docs.append(doc)
            ground_truths.append(ground_truth)

        # Create single sample with all documents
        context = {"medical_transcripts.json": ContentDataType("json", content=docs)}
        samples.append(BenchmarkSample(
            id='0',
            query=data_loader.question,
            context=context,
            ground_truth=ground_truths
        ))

        self._samples = samples
        return self._samples

    
    def get_samples(self) -> List[BenchmarkSample]:
        return self._samples
    
    def evaluate_sample(self, sample: BenchmarkSample, prediction: Any) -> EvaluationResult:
        """Evaluate medication extraction against ground truth clinical notes."""
        
        return EvaluationResult(
            sample_id=sample.id,
            prediction=prediction,
            ground_truth=[],
            metrics={"f1": 0, "precision": 0, "recall": 0}
        )
    
    def compute_aggregate_metrics(self, results: List[EvaluationResult]) -> Dict[str, float]:
        if not results:
            return {}
        
        aggregate = {
            "f1": sum(r.metrics["f1"] for r in results) / len(results),
            "precision": sum(r.metrics["precision"] for r in results) / len(results),
            "recall": sum(r.metrics["recall"] for r in results) / len(results),
        }
        
        return aggregate

if __name__ == "__main__":
    # Test the medication data loader
    data_loader = MedicationDataLoader(MEDICATION_DATA_PATH)
    print(f"Loaded {len(data_loader.documents)} medical transcripts")
    print(f"Sample transcript file: {data_loader.documents[0]['file']}")
    print(f"\nSample query:\n{data_loader.question[:200]}...")