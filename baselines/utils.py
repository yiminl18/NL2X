"""
Common utility functions for baseline implementations.
This module contains shared functions used across multiple baseline classes.
"""

import os
import base64
import json
import pandas as pd
from typing import Any, Dict, List, Optional


def encode_image(image_path: str) -> str:
    """
    Encode image to base64 string.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Base64 encoded string of the image
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def read_excel(file_path: str) -> Optional[Dict[str, pd.DataFrame]]:
    """
    Read all sheets from an Excel file.
    
    Args:
        file_path: Path to the Excel file
        
    Returns:
        Dictionary mapping sheet names to DataFrames, or None if error
    """
    try:
        xls = pd.ExcelFile(file_path)
        sheets = {}
        for sheet_name in xls.sheet_names:
            sheets[sheet_name] = xls.parse(sheet_name)
        return sheets
    except Exception:
        return None


def dataframe_to_text(df: pd.DataFrame, max_rows: int = 1000) -> str:
    """
    Convert DataFrame to readable text with optional row limit.
    
    Args:
        df: The DataFrame to convert
        max_rows: Maximum number of rows to include
        
    Returns:
        Text representation of the DataFrame
    """
    if len(df) > max_rows:
        df = df.head(max_rows)
        text = df.to_string(index=False)
        text += f"\n... (showing first {max_rows} of {len(df)} rows)"
    else:
        text = df.to_string(index=False)
    return text


def combine_sheets_text(sheets: Dict[str, pd.DataFrame]) -> str:
    """
    Combine all Excel sheets into a single text representation.
    
    Args:
        sheets: Dictionary mapping sheet names to DataFrames
        
    Returns:
        Combined text representation of all sheets
    """
    combined_text = ""
    for sheet_name, df in sheets.items():
        sheet_text = dataframe_to_text(df)
        combined_text += f"Sheet name: {sheet_name}\n{sheet_text}\n\n"
    return combined_text


def read_csv(file_path: str) -> str:
    """
    Read CSV file and return as text with encoding fallback.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        Text representation of the CSV data
    """
    try:
        df = pd.read_csv(file_path)
        return dataframe_to_text(df)
    except Exception as e:
        # Try with different encoding
        try:
            df = pd.read_csv(file_path, encoding='latin-1')
            return dataframe_to_text(df)
        except:
            return f"Error reading CSV: {str(e)}"


def read_json(file_path: str) -> str:
    """
    Read JSON file and return as formatted text.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Formatted JSON string or error message
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return json.dumps(data, indent=2)
    except Exception as e:
        return f"Error reading JSON: {str(e)}"


def read_text(file_path: str) -> str:
    """
    Read text file with encoding fallback.
    
    Args:
        file_path: Path to the text file
        
    Returns:
        Content of the text file or error message
    """
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


def prepare_context_data(context: Optional[Dict[str, Any]], verbose: bool = False, logger = None) -> tuple[str, List[str]]:
    """
    Prepare context data for model processing.
    
    Args:
        context: Context dictionary containing file information
        verbose: Whether to log verbose information
        logger: Logger instance for verbose output
        
    Returns:
        Tuple of (text_content, image_paths)
    """
    images = []
    text_content = ""
    
    if context:
        for key, value in context.items():
            file_type = getattr(value, 'type', None)
            file_path = getattr(value, 'path', None) if hasattr(value, 'path') else None
            
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
                    if verbose and logger:
                        logger.warning(f"Failed to read Excel file {file_path}: {e}")
            elif file_type == "csv":
                try:
                    csv_text = read_csv(file_path)
                    text_content += f"\n[CSV file: {key}]\n{csv_text}\n"
                except Exception as e:
                    text_content += f"\n[CSV file: {key}] (Error: {str(e)})\n"
                    if verbose and logger:
                        logger.warning(f"Failed to read CSV file {file_path}: {e}")
            elif file_type == "json":
                try:
                    json_text = read_json(file_path)
                    text_content += f"\n[JSON file: {key}]\n{json_text}\n"
                except Exception as e:
                    text_content += f"\n[JSON file: {key}] (Error: {str(e)})\n"
                    if verbose and logger:
                        logger.warning(f"Failed to read JSON file {file_path}: {e}")
            elif file_type in ["text", "html"]:
                try:
                    file_text = read_text(file_path)
                    text_content += f"\n[{key}]\n{file_text}\n"
                except Exception as e:
                    text_content += f"\n[{key}] (Error: {str(e)})\n"
                    if verbose and logger:
                        logger.warning(f"Failed to read text file {file_path}: {e}")
            elif file_type == "pdf":
                # For PDF files, we'll just note their presence
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


def find_jpg_files(directory: str) -> Optional[List[str]]:
    """
    Find JPG and PNG files in a directory.
    
    Args:
        directory: Directory path to search
        
    Returns:
        List of image files or None if none found
    """
    jpg_files = [file for file in os.listdir(directory) 
                 if file.lower().endswith('.jpg') or file.lower().endswith('.png')]
    return jpg_files if jpg_files else None


def find_excel_files(directory: str) -> Optional[List[str]]:
    """
    Find Excel files in a directory, excluding answer files.
    
    Args:
        directory: Directory path to search
        
    Returns:
        List of Excel files or None if none found
    """
    excel_files = [file for file in os.listdir(directory) 
                   if (file.lower().endswith('xlsx') or 
                       file.lower().endswith('xlsb') or 
                       file.lower().endswith('xlsm')) and 
                   "answer" not in file.lower()]
    return excel_files if excel_files else None


def read_txt(file_path: str) -> str:
    """
    Simple text file reader (kept for backward compatibility).
    
    Args:
        file_path: Path to the text file
        
    Returns:
        Content of the file
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()