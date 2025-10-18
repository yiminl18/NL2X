import time
from typing import Any, Dict, List, Optional

from .base import BaselineInterface, BaselineResult
from . import register_baseline
from .utils import (
    encode_image,
    prepare_context_data
)
from model.litellm_client import llm_call
import tiktoken

# Model configuration
MODEL_NAME = "gpt-4o"
MODEL_LIMITS = 128000
TOKENS_FOR_GENERATION = 6000

# Initialize tokenizer
encoding = tiktoken.encoding_for_model(MODEL_NAME)

@register_baseline("gpt4o_zero_shot")
class GPT4oZeroShotBaseline(BaselineInterface):
    """
    GPT-4o Zero-shot baseline implementation for KramaBench.
    This baseline directly queries GPT-4o without any examples or demonstrations.
    """
    
    def _setup(self):
        self.total_queries = 0
        self.total_time = 0
        self.total_cost = 0
        
        # Pricing for GPT-4o
        self.input_cost_per_1k = 0.0015
        self.output_cost_per_1k = 0.006
        
        if self.config.verbose:
            self.logger.info(f"Initialized GPT-4o Zero-shot baseline with config: {self.config}")
    
    def _prepare_context(self, context: Dict[str, Any]) -> tuple:
        """Prepare context data for the model."""
        return prepare_context_data(context, self.config.verbose, self.logger)
    
    def _create_prompt(self, query: str, text_content: str, images: List[str]) -> List[Dict]:
        """Create the prompt for GPT-4o."""
        # Prepare image codes
        image_codes = []
        for image_path in images:
            try:
                base64_image = encode_image(image_path)
                image_codes.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                })
            except Exception as e:
                if self.config.verbose:
                    self.logger.warning(f"Failed to encode image {image_path}: {e}")
        
        # Build the prompt
        system_prompt = """You are a highly capable data analyst and problem solver. 
You will be given data and a question. Analyze the data carefully and provide a precise answer.

Important instructions:
1. Read and understand all provided data before answering
2. Provide clear, accurate answers based on the data
3. For numeric answers, calculate precisely and show the final number
4. For list answers, provide all relevant items
5. Always end your response with 'Answer: {your_answer}' on a new line

Examples of answer formats:
- For numeric: Answer: 42.5
- For lists: Answer: ["item1", "item2", "item3"]
- For text: Answer: The explanation is...
"""
        
        user_content = []
        
        # Add text content if available
        if text_content:
            user_content.append({
                "type": "text",
                "text": f"Data:\n{text_content}\n\nQuestion:\n{query}\n\nPlease analyze the data and answer the question."
            })
        else:
            user_content.append({
                "type": "text",
                "text": f"Question:\n{query}\n\nPlease provide your answer."
            })
        
        # Add images
        user_content.extend(image_codes)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]
        
        return messages
    
    def process(self, query: str, context: Optional[Dict[str, Any]] = None) -> BaselineResult:
        """Process a single query with optional context."""
        start_time = time.time()
        
        try:
            # Prepare context
            text_content, images = self._prepare_context(context)
            print(f"Prepared text content length: {len(text_content)}")
            # Truncate text if too long
            if text_content:
                token_count = len(encoding.encode(text_content))
                if token_count > MODEL_LIMITS - TOKENS_FOR_GENERATION:
                    # Truncate to fit within limits
                    truncated_tokens = encoding.encode(text_content)[:MODEL_LIMITS - TOKENS_FOR_GENERATION]
                    text_content = encoding.decode(truncated_tokens)
                    if self.config.verbose:
                        self.logger.warning(f"Truncated context from {token_count} to {len(truncated_tokens)} tokens")
            
            print(f"Prepared text content length2: {len(text_content)}")
            # Create prompt
            messages = self._create_prompt(query, text_content, images)
            
            # Call GPT-4o
            response, usage = llm_call(messages, max_tokens=2256, return_usage=True)
            
            # Calculate cost
            input_cost = (usage.prompt_tokens / 1000) * self.input_cost_per_1k
            output_cost = (usage.completion_tokens / 1000) * self.output_cost_per_1k
            call_cost = input_cost + output_cost
            self.total_cost += call_cost
            
            # Parse answer from response
            answer = response
            if "Answer:" in response:
                # Extract everything after "Answer:"
                answer = response.split("Answer:")[-1].strip()
            
            execution_time = time.time() - start_time
            self.total_queries += 1
            self.total_time += execution_time
            
            result = BaselineResult(
                query=query,
                response={"answer": answer, "full_response": response},
                metadata={
                    "baseline": "gpt_4o_zero_shot",
                    "context_provided": context is not None,
                    "num_images": len(images),
                    "text_length": len(text_content) if text_content else 0,
                    "tokens_used": usage.total_tokens,
                    "cost": call_cost
                },
                execution_time=execution_time
            )
            
            if self.config.verbose:
                self.logger.info(f"Processed query in {execution_time:.2f}s, cost: ${call_cost:.4f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing query: {e}")
            
            execution_time = time.time() - start_time
            self.total_queries += 1
            self.total_time += execution_time
            
            return BaselineResult(
                query=query,
                response={"answer": "", "error": str(e)},
                metadata={
                    "baseline": "gpt_4o_zero_shot",
                    "context_provided": context is not None,
                    "error": True
                },
                execution_time=execution_time,
                error=str(e)
            )
    
    def batch_process(self, queries: List[str], contexts: Optional[List[Dict[str, Any]]] = None) -> List[BaselineResult]:
        """Process multiple queries in batch."""
        if contexts is None:
            contexts = [None] * len(queries)
        
        results = []
        for query, context in zip(queries, contexts):
            results.append(self.process(query, context))
        
        return results
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get baseline metrics."""
        return {
            "total_queries": self.total_queries,
            "total_time": self.total_time,
            "average_time": self.total_time / max(1, self.total_queries),
            "total_cost": self.total_cost,
            "average_cost": self.total_cost / max(1, self.total_queries),
            "baseline_type": "gpt_4o_zero_shot"
        }