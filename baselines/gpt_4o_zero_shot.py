import time
from typing import Any, Dict, List, Optional
import os
import base64
import pandas as pd

from .base import BaselineInterface, BaselineResult
from . import register_baseline
from model.azuregpt4o import gpt_4o_azure
import tiktoken

# Model configuration
MODEL_NAME = "gpt-4o"
MODEL_LIMITS = 128000
TOKENS_FOR_GENERATION = 6000

# Initialize tokenizer
encoding = tiktoken.encoding_for_model(MODEL_NAME)

def encode_image(image_path):
    """Encode image to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def read_excel(file_path):
    """Read all sheets from an Excel file."""
    xls = pd.ExcelFile(file_path)
    sheets = {}
    for sheet_name in xls.sheet_names:
        sheets[sheet_name] = xls.parse(sheet_name)
    return sheets

def dataframe_to_text(df):
    """Convert DataFrame to readable text."""
    return df.to_string(index=False)

def combine_sheets_text(sheets):
    """Combine all Excel sheets into a single text."""
    combined_text = ""
    for sheet_name, df in sheets.items():
        sheet_text = dataframe_to_text(df)
        combined_text += f"Sheet name: {sheet_name}\n{sheet_text}\n\n"
    return combined_text

def read_csv(file_path):
    """Read CSV file and return as text."""
    df = pd.read_csv(file_path)
    return dataframe_to_text(df)

def read_json(file_path):
    """Read JSON file and return as formatted text."""
    import json
    with open(file_path, 'r') as f:
        data = json.load(f)
    return json.dumps(data, indent=2)

def read_text(file_path):
    """Read text file."""
    with open(file_path, "r") as f:
        return f.read()

@register_baseline("gpt_4o_zero_shot")
class GPT4oZeroShotBaseline(BaselineInterface):
    """
    GPT-4o Zero-shot baseline implementation for KramaBench.
    This baseline directly queries GPT-4o without any examples or demonstrations.
    """
    
    def _setup(self):
        self.total_queries = 0
        self.total_time = 0
        self.total_cost = 0
        self.cache = {}  # Initialize cache
        
        # Pricing for GPT-4o
        self.input_cost_per_1k = 0.0015
        self.output_cost_per_1k = 0.006
        
        if self.config.verbose:
            self.logger.info(f"Initialized GPT-4o Zero-shot baseline with config: {self.config}")
    
    def _prepare_context(self, context: Dict[str, Any]) -> tuple:
        """Prepare context data for the model."""
        images = []
        text_content = ""
        
        if context:
            for key, value in context.items():
                file_type = value.type
                file_path = value.path if hasattr(value, 'path') else None
                
                if not file_path and hasattr(value, 'content'):
                    # Handle inline content
                    text_content += f"\n[{key}]:\n{value.content}\n"
                    continue
                
                if not file_path or not os.path.exists(file_path):
                    continue
                
                # Process based on file type
                if file_type == "image":
                    images.append(file_path)
                elif file_type == "excel":
                    try:
                        sheets = read_excel(file_path)
                        excel_text = combine_sheets_text(sheets)
                        text_content += f"\n[Excel file: {key}]\n{excel_text}\n"
                    except Exception as e:
                        if self.config.verbose:
                            self.logger.warning(f"Failed to read Excel file {file_path}: {e}")
                elif file_type == "csv":
                    try:
                        csv_text = read_csv(file_path)
                        text_content += f"\n[CSV file: {key}]\n{csv_text}\n"
                    except Exception as e:
                        if self.config.verbose:
                            self.logger.warning(f"Failed to read CSV file {file_path}: {e}")
                elif file_type == "json":
                    try:
                        json_text = read_json(file_path)
                        text_content += f"\n[JSON file: {key}]\n{json_text}\n"
                    except Exception as e:
                        if self.config.verbose:
                            self.logger.warning(f"Failed to read JSON file {file_path}: {e}")
                elif file_type in ["text", "html"]:
                    try:
                        file_text = read_text(file_path)
                        text_content += f"\n[{key}]\n{file_text}\n"
                    except Exception as e:
                        if self.config.verbose:
                            self.logger.warning(f"Failed to read text file {file_path}: {e}")
                elif file_type == "pdf":
                    # For PDF files, we'll just note their presence
                    # Full PDF processing would require additional libraries
                    text_content += f"\n[PDF file: {key}] (PDF processing not implemented - file present but not read)\n"
                else:
                    # Try to read as text for unknown types
                    try:
                        file_text = read_text(file_path)
                        text_content += f"\n[{key}]\n{file_text}\n"
                    except:
                        pass
        
        return text_content, images
    
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
        
        # Generate cache key
        cache_key = f"{query}:{str(sorted(context.keys()) if context else [])}"
        if cache_key in self.cache:
            cached_result = self.cache[cache_key]
            cached_result.metadata["from_cache"] = True
            return cached_result
        
        try:
            # Prepare context
            text_content, images = self._prepare_context(context)
            
            # Truncate text if too long
            if text_content:
                token_count = len(encoding.encode(text_content))
                if token_count > MODEL_LIMITS - TOKENS_FOR_GENERATION:
                    # Truncate to fit within limits
                    truncated_tokens = encoding.encode(text_content)[:MODEL_LIMITS - TOKENS_FOR_GENERATION]
                    text_content = encoding.decode(truncated_tokens)
                    if self.config.verbose:
                        self.logger.warning(f"Truncated context from {token_count} to {len(truncated_tokens)} tokens")
            
            # Create prompt
            messages = self._create_prompt(query, text_content, images)
            
            # Call GPT-4o
            response, usage = gpt_4o_azure(messages, max_tokens=2256, return_usage=True)
            
            # Calculate cost
            input_cost = (usage.prompt_tokens / 1000) * self.input_cost_per_1k
            output_cost = (usage.completion_tokens / 1000) * self.output_cost_per_1k
            call_cost = input_cost + output_cost
            self.total_cost += call_cost
            
            # Parse answer from response
            answer = response
            if "Answer:" in response:
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
                    "cost": call_cost,
                },
                execution_time=execution_time
            )
            
            # Cache the result
            if hasattr(self, 'cache'):
                self.cache[cache_key] = result
            
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
                    "from_cache": False
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
            "cache_size": len(self.cache) if hasattr(self, 'cache') else 0,
            "baseline_type": "gpt_4o_zero_shot"
        }