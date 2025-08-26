import random
from typing import Any, Dict, List, Optional

from model.azuregpt4o import gpt_4o_azure
from .base import BenchmarkInterface, BenchmarkSample, EvaluationResult, BenchmarkConfig, ContentDataType
from . import register_benchmark
import os
import base64
import pandas as pd
import tiktoken

DSBench_PATH = "./benchmarks/DSBench/data_analysis/"
DSBench_DATA_PATH = DSBench_PATH + "data/"

def find_jpg_files(directory):
    jpg_files = [file for file in os.listdir(directory) if file.lower().endswith('.jpg') or file.lower().endswith('.png')]
    return jpg_files if jpg_files else []

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def find_excel_files(directory):
    jpg_files = [file for file in os.listdir(directory) if (file.lower().endswith('xlsx') or file.lower().endswith('xlsb') or file.lower().endswith('xlsm')) and not "answer" in file.lower()]
    return jpg_files if jpg_files else []

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


@register_benchmark("dsbench")
class DSBenchBenchmark(BenchmarkInterface):
    def _setup(self):
        self.data_formats = ["csv", "xlsx", "json", "parquet"]
        if self.config.verbose:
            self.logger.info(f"Initialized DSBench benchmark with config: {self.config}")

    def _load_data(self) -> List[BenchmarkSample]:
        raw_samples = []
        samples = []
        with open(DSBench_PATH + "data.json", "r") as f:
            for line in f:
                raw_samples.append(eval(line.strip()))
        num_samples = min(self.config.max_samples, len(raw_samples))
        for id in range(num_samples):
            sample =raw_samples[id]
            content = {}
            if len(sample["questions"]) > 0:
                # find the image
                image_paths = find_jpg_files(os.path.join(DSBench_DATA_PATH, sample["id"]))
                for image_path in image_paths:
                    content[image_path] = ContentDataType("image", os.path.join(DSBench_DATA_PATH, sample["id"], image_path))
                # find the excel files
                excels_paths = find_excel_files(os.path.join(DSBench_DATA_PATH, sample["id"]))
                for excels_path in excels_paths:
                    content[excels_path] = ContentDataType("excel", os.path.join(DSBench_DATA_PATH,  sample["id"], excels_path))

                content["introduction.txt"] = ContentDataType("text", os.path.join(DSBench_DATA_PATH, sample["id"], "introduction.txt"))

                # for each question, we directly read the content from the file
                questions = []
                for question_name in sample["questions"]:
                    questions.append(read_txt(os.path.join(DSBench_DATA_PATH, sample["id"], question_name+".txt")))
                
                answers = sample["answers"]

                for question, answer in zip(questions, answers):
                    samples.append(BenchmarkSample(
                        id=sample["id"],
                        query=question,
                        context=content,
                        ground_truth=answer
                    ))
                    
        self._samples = samples
        # for sample in samples:
        #     print(sample)
        return self._samples

    
    def get_samples(self) -> List[BenchmarkSample]:
        return self._samples
    
    def evaluate_sample(self, sample: BenchmarkSample, prediction: Any) -> EvaluationResult:
        if not isinstance(prediction['answer'], str):
            return EvaluationResult(
                sample_id=sample.id,
                prediction=prediction,
                ground_truth=sample.ground_truth,
                metrics={"accuracy": 0.0}
            )
        
        prompt = (f"Please judge whether the generated answer is right or wrong. We require that the correct answer "
              f"to the prediction gives a clear answer, not just a calculation process or a disassembly of ideas. "
              f"The question is {sample.query}. The true answer is \n {sample.ground_truth}. \n The predicted answer is \n {prediction['answer']}.\n "
              f"If the predicted answer is right, please output True. Otherwise output Flase. "
              f"Don't output any other text content. You only can output True or False.")
        response = gpt_4o_azure(prompt)
        if "True" in response:
            accuracy = 1.0
        else:
            accuracy = 0.0
        
        return EvaluationResult(
            sample_id=sample.id,
            prediction=prediction,
            ground_truth=sample.ground_truth,
            metrics={"accuracy": accuracy}
        )
    
    def compute_aggregate_metrics(self, results: List[EvaluationResult]) -> Dict[str, float]:
        if not results:
            return {}
        
        aggregate = {
            "accuracy": sum(r.metrics["accuracy"] for r in results) / len(results),
        }
        
        return aggregate