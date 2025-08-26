import time
import random
from typing import Any, Dict, List, Optional

import cost
from .base import BaselineInterface, BaselineResult, BaselineConfig
from . import register_baseline
from model.azuregpt4o import gpt_4o_azure
import tiktoken
import os
import base64
import pandas as pd

MODEL_LIMITS = 128000
TOKENS4GENERATION = 6000

model_name = "gpt-4o"
encoding = tiktoken.encoding_for_model(model_name)

def find_jpg_files(directory):
    jpg_files = [file for file in os.listdir(directory) if file.lower().endswith('.jpg') or file.lower().endswith('.png')]
    return jpg_files if jpg_files else None

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def find_excel_files(directory):
    jpg_files = [file for file in os.listdir(directory) if (file.lower().endswith('xlsx') or file.lower().endswith('xlsb') or file.lower().endswith('xlsm')) and not "answer" in file.lower()]
    return jpg_files if jpg_files else None

def read_excel(file_path):
    # 读取Excel文件中的所有sheet
    xls = pd.ExcelFile(file_path)
    sheets = {}
    for sheet_name in xls.sheet_names:
        sheets[sheet_name] = xls.parse(sheet_name)
    return sheets

def dataframe_to_text(df):
    # 将DataFrame转换为文本
    text = df.to_string(index=False)
    return text

def combine_sheets_text(sheets):
    # 将所有sheet的文本内容组合起来
    combined_text = ""
    for sheet_name, df in sheets.items():
        sheet_text = dataframe_to_text(df)
        combined_text += f"Sheet name: {sheet_name}\n{sheet_text}\n\n"
    return combined_text

def read_txt(path):
    with open(path, "r") as f:
        return f.read()

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