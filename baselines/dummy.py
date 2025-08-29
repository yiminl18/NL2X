import time
import random
from typing import Any, Dict, List, Optional
from .base import BaselineInterface, BaselineResult, BaselineConfig
from . import register_baseline
import os
import base64
import pandas as pd
import json

def encode_image(image_path):
    """Encode image to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def read_excel(file_path):
    """Read all sheets from an Excel file."""
    try:
        xls = pd.ExcelFile(file_path)
        sheets = {}
        for sheet_name in xls.sheet_names:
            sheets[sheet_name] = xls.parse(sheet_name)
        return sheets
    except Exception as e:
        return None

def dataframe_to_text(df, max_rows=1000):
    """Convert DataFrame to readable text with row limit."""
    if len(df) > max_rows:
        df = df.head(max_rows)
        text = df.to_string(index=False)
        text += f"\n... (showing first {max_rows} of {len(df)} rows)"
    else:
        text = df.to_string(index=False)
    return text

def combine_sheets_text(sheets):
    """Combine all Excel sheets into a single text."""
    combined_text = ""
    for sheet_name, df in sheets.items():
        sheet_text = dataframe_to_text(df)
        combined_text += f"Sheet name: {sheet_name}\n{sheet_text}\n\n"
    return combined_text

def read_csv(file_path):
    """Read CSV file and return as text."""
    try:
        df = pd.read_csv(file_path)
        return dataframe_to_text(df)
    except Exception as e:
        # Try with different parameters
        try:
            df = pd.read_csv(file_path, encoding='latin-1')
            return dataframe_to_text(df)
        except:
            return f"Error reading CSV: {str(e)}"

def read_json(file_path):
    """Read JSON file and return as formatted text."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return json.dumps(data, indent=2)
    except Exception as e:
        return f"Error reading JSON: {str(e)}"

def read_text(file_path):
    """Read text file with encoding fallback."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        # Try with different encoding
        try:
            with open(file_path, "r", encoding="latin-1") as f:
                return f.read()
        except:
            return "(Unable to decode file content)"
    except Exception as e:
        return f"Error reading file: {str(e)}"

    
@register_baseline("dummy")
class DummyBaseline(BaselineInterface):
    def _setup(self):
        self.total_queries = 0
        self.total_time = 0
        
        if self.config.verbose:
            self.logger.info(f"Initialized Dummy baseline with config: {self.config}")


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
                    if file_path:
                        text_content += f"\n[{key}]: (File not found: {file_path})\n"
                    continue
                
                # Process based on file type
                if file_type == "image":
                    images.append(file_path)
                elif file_type == "excel":
                    try:
                        sheets = read_excel(file_path)
                        if sheets:
                            excel_text = combine_sheets_text(sheets)
                            text_content += f"\n[Excel file: {key}]\n{excel_text}\n"
                        else:
                            text_content += f"\n[Excel file: {key}] (Could not read file)\n"
                    except Exception as e:
                        text_content += f"\n[Excel file: {key}] (Error: {str(e)})\n"
                        if self.config.verbose:
                            self.logger.warning(f"Failed to read Excel file {file_path}: {e}")
                elif file_type == "csv":
                    try:
                        csv_text = read_csv(file_path)
                        text_content += f"\n[CSV file: {key}]\n{csv_text}\n"
                    except Exception as e:
                        text_content += f"\n[CSV file: {key}] (Error: {str(e)})\n"
                        if self.config.verbose:
                            self.logger.warning(f"Failed to read CSV file {file_path}: {e}")
                elif file_type == "json":
                    try:
                        json_text = read_json(file_path)
                        text_content += f"\n[JSON file: {key}]\n{json_text}\n"
                    except Exception as e:
                        text_content += f"\n[JSON file: {key}] (Error: {str(e)})\n"
                        if self.config.verbose:
                            self.logger.warning(f"Failed to read JSON file {file_path}: {e}")
                elif file_type in ["text", "html"]:
                    try:
                        file_text = read_text(file_path)
                        text_content += f"\n[{key}]\n{file_text}\n"
                    except Exception as e:
                        text_content += f"\n[{key}] (Error: {str(e)})\n"
                        if self.config.verbose:
                            self.logger.warning(f"Failed to read text file {file_path}: {e}")
                elif file_type == "pdf":
                    # For PDF files, we'll just note their presence
                    # Full PDF processing would require additional libraries like PyPDF2
                    text_content += f"\n[PDF file: {key}] (PDF processing not implemented - file present but content not extracted)\n"
                elif file_type in ["numpy", "npz", "npy"]:
                    # NumPy files would require numpy library
                    text_content += f"\n[NumPy file: {key}] (NumPy file detected - requires numpy library for processing)\n"
                elif file_type == "cdf":
                    # CDF files would require specific scientific libraries
                    text_content += f"\n[CDF file: {key}] (Common Data Format file - requires specialized library)\n"
                elif file_type == "geo":
                    # GeoPackage files would require geopandas or similar
                    text_content += f"\n[GeoPackage file: {key}] (Geographic data file - requires spatial data library)\n"
                else:
                    # Try to read as text for unknown types
                    try:
                        file_text = read_text(file_path)
                        text_content += f"\n[{key}]\n{file_text}\n"
                    except:
                        # If can't read as text, note the file presence
                        text_content += f"\n[Binary file: {key}] (File present but could not read as text)\n"
        
        return text_content, images

    def process(self, query: str, context: Optional[Dict[str, Any]] = None) -> BaselineResult:
        start_time = time.time()
        print("=== Dummy Baseline ===")
        print(f"Query: {query}")
        print(f"Context: {context}")
        print("==============================")
        text_content, images = self._prepare_context(context)
        response = {
            "answer": f"I don't know :)",
        }
        
        execution_time = time.time() - start_time
        self.total_queries += 1
        self.total_time += execution_time
        
        result = BaselineResult(
            query=query,
            response=response,
            metadata={
                "baseline": "dummy",
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
            "baseline_type": "dummy"
        }