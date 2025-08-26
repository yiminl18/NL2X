import random
from typing import Any, Dict, List, Optional

from benchmarks.CRAG.local_evaluation import get_system_message, load_data_in_batches, parse_response
from model.azuregpt4o import gpt_4o_azure
from .base import BenchmarkInterface, BenchmarkSample, ContentDataType, EvaluationResult, BenchmarkConfig
from . import register_benchmark

@register_benchmark("crag")
class CRAGBenchmark(BenchmarkInterface):
    def _setup(self):
        self.DATASET_PATH = "/Users/chiyuh/Workspace/NL2X/benchmarks/CRAG/example_data/dev_data.jsonl.bz2"
        # GPT-4o pricing (per 1K tokens)
        self.input_cost_per_1k = 0.0015  # $0.0015 per 1K input tokens
        self.output_cost_per_1k = 0.006  # $0.006 per 1K output tokens
        self.total_cost = 0.0

        if self.config.verbose:
            self.logger.info(f"Initialized CRAG benchmark with config: {self.config}")

    def _load_data(self):
        queries, contents, ground_truths = [], [], []
        for batch in load_data_in_batches(self.DATASET_PATH, 1):
            """
            batch (Dict[str, Any]): A dictionary containing a batch of input queries with the following keys:
                - 'interaction_id' (List[str]): List of interaction_ids for the associated queries
                - 'query' (List[str]): List of user queries.
                - 'search_results' (List[List[Dict]]): List of search result lists, each corresponding
                                                    to a query.
                - 'query_time' (List[str]): List of timestamps (represented as a string), each corresponding to when a query was made.
            """
            queries.extend(batch["query"])
            contents.extend(batch["search_results"])
            ground_truths.extend(batch.pop("answer"))
            if self.config.max_samples and len(queries) >= self.config.max_samples:
                break

        print(len(contents[0]))
        # Print structure (no value) of contents[0]
        print({k: type(v) for k, v in contents[0][0].items()})

        self._samples = [BenchmarkSample(id=f"crag_{i:04d}", query=q, ground_truth={"answer": gt})
                         for i, (q, gt) in enumerate(zip(queries, ground_truths))]
        for sample, content in zip(self._samples, contents):
            sample.context = {}
            for idx, item in enumerate(content):
                sample.context[f"content_{idx}"] = ContentDataType("text", content=str(item))

        return self._samples

    def get_samples(self) -> List[BenchmarkSample]:
        return self._samples
    
    def evaluate_sample(self, sample: BenchmarkSample, prediction: Any) -> EvaluationResult:
        n_miss, n_correct = 0, 0
        system_message = get_system_message()

        query = sample.query
        ground_truths = sample.ground_truth["answer"].strip()
        # trim prediction to 75 tokens using Llama2 tokenizer
        prediction = prediction['answer']
        prediction = prediction.strip()

        if "i don't know" in prediction.lower():
            n_miss += 1

        accuracy = -1

        for ground_truth in ground_truths:
            ground_truth_lowercase = ground_truth.lower()
            prediction_lowercase = prediction.lower()
            messages = [
                {"role": "system", "content": system_message},
                {
                    "role": "user",
                    "content": f"Question: {query}\n Ground truth: {ground_truth}\n Prediction: {prediction}\n",
                },
            ]
            if prediction_lowercase == ground_truth_lowercase:
                # exact correct
                accuracy = 1
                break
            elif "invalid" in prediction_lowercase and "invalid" in ground_truth_lowercase:
                accuracy = 1
                break
            elif "invalid" in prediction_lowercase and "invalid" not in ground_truth_lowercase:
                # hallucination
                accuracy = 0
                continue
            elif "invalid" not in prediction_lowercase and "invalid" in ground_truth_lowercase:
                # hallucination
                accuracy = 0
                continue
            else:
                # need to use the OpenAI evaluation model to get the accuracy result (0 means wrong, 1 means correct)
                response, usage = gpt_4o_azure(messages, return_usage=True)
                # Calculate cost for this API call
                input_cost = (usage.prompt_tokens / 1000) * self.input_cost_per_1k
                output_cost = (usage.completion_tokens / 1000) * self.output_cost_per_1k
                call_cost = input_cost + output_cost
                self.total_cost += call_cost
                
                _, accuracy = parse_response(response)

        if accuracy == 1:
            status = "correct"
        elif n_miss == 1:
            status = "missing"
        else:
            status = "hallucination"

        results = {
            "score": (2 * accuracy + n_miss) - 1,
            "status": status
        }
        return results
    
    def compute_aggregate_metrics(self, results: List[EvaluationResult]) -> Dict[str, float]:
        """
            "score": (2 * n_correct + n_miss) / n - 1,
            "accuracy": n_correct / n,
            "hallucination": (n - n_correct - n_miss) / n,
            "missing": n_miss / n,
            "n_miss": n_miss,
            "n_correct": n_correct,
            "n_hallucination": n - n_correct - n_miss,
            "total": n,
        """
        
        if not results:
            return {}
        n = len(results)

        n_miss = sum(1 for result in results if result["status"] == "missing")
        n_correct = sum(1 for result in results if result["status"] == "correct")
        n_hallucination = n - n_correct - n_miss

        aggregate = {
            "score": (2 * n_correct + n_miss) / n - 1,
            "accuracy": n_correct / n,
            "hallucination": n_hallucination / n,
            "missing": n_miss / n,
            "n_miss": n_miss,
            "n_correct": n_correct,
            "n_hallucination": n_hallucination,
            "total": n,
            "total_cost": self.total_cost,
        }

        return aggregate