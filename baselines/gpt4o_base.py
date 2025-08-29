import time
import random
from typing import Any, Dict, List, Optional

import cost
from .base import BaselineInterface, BaselineResult, BaselineConfig
from . import register_baseline
from .utils import (
    encode_image,
    read_excel,
    combine_sheets_text,
    read_txt
)
from model.azuregpt4o import gpt_4o_azure
import tiktoken

MODEL_LIMITS = 128000
TOKENS4GENERATION = 6000

model_name = "gpt-4o"
encoding = tiktoken.encoding_for_model(model_name)

def get_gpt_res(text, images):
    image_codes = []
    for image in images:
        base64_image = encode_image(image)
        image_codes.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"}
        })
    prompt = [
            {
              "role": "system",
              "content": [
                {
                  "type": "text",
                  "text": "You are a data analyst. I will give you a background introduction, data analysis question, and a workbook. You must answer the question. You must clearly answer the question using 'Answer: {answer}' at the end, for example, 'Answer: B' and 'Answer: 1919'"
                }
              ]
            },
            {
              "role": "user",
              "content": [
                {
                  "type": "text",
                  "text": text
                },
                *image_codes
              ]
            }
          ]
    return gpt_4o_azure(prompt, max_tokens=2256)

@register_baseline("gpt4o_base")
class GPT4oBaseBaseline(BaselineInterface):
    def _setup(self):
        self.total_queries = 0
        self.total_time = 0
        self.cache = {}
        self.total_cost = 0
        if self.config.verbose:
            self.logger.info(f"Initialized GPT4oBase baseline with config: {self.config}")
    
    def process(self, query: str, context: Optional[Dict[str, Any]] = None) -> BaselineResult:
        start_time = time.time()
        
        # get the images and excel files
        images = []
        excel_content = ""
        text_content = ""
        introduction = ""
        print(context)
        for key, value in context.items():
            if value.type == "image":
                images.append(value.path)
            elif value.type == "excel":
                sheets = read_excel(value.path)
                combined_text = combine_sheets_text(sheets)
                excel_content += f"The excel file {key} is: " + combined_text
            elif value.type == "text" and key == "introduction.txt":
                introduction = read_txt(value.path)
            elif value.type == "text":
                text_content += f"The text file {key} is: " + read_txt(value.path)

        text = ""
        if excel_content != "":
            text += f"The workbook is detailed as follows. {excel_content} \n"

        if text_content != "":
            text += f"The text content is detailed as follows. \n {text_content} \n"

        if introduction != "":
            text += f"The introduction is detailed as follows. \n {introduction} \n"

        text += f"The question is detailed as follows. \n {query} \n"

        cut_text = encoding.decode(encoding.encode(text)[TOKENS4GENERATION - MODEL_LIMITS:])
        response = get_gpt_res(cut_text, images)
        print(response)
        self.total_cost += cost.cost(model_name, cost.count_tokens(cut_text, model_name), cost.count_tokens(response, model_name)) 
        print("Total cost: ", self.total_cost)

        # parse the response
        answer = response.split("Answer")[-1]
        question_response = {
            "answer": f"{answer}",
        }
        
        execution_time = time.time() - start_time
        self.total_queries += 1
        self.total_time += execution_time
        
        result = BaselineResult(
            query=query,
            response=question_response,
            metadata={
                "baseline": "gpt4o_base",
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
            "cache_size": len(self.cache),
            "baseline_type": "gpt4o_base"
        }