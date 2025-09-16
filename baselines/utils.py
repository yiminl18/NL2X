"""
Common utility functions for baseline implementations.
This module contains shared functions used across multiple baseline classes.
"""

import os
import base64
import json
import pandas as pd
import re
import tempfile
from typing import Any, Dict, List, Optional
from bs4 import BeautifulSoup


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


def parse_html_to_dict(html_content: str) -> Dict[str, Any]:
    """
    Parse HTML content into a structured dictionary.

    Args:
        html_content: HTML string content

    Returns:
        Dictionary with parsed HTML structure
    """
    soup = BeautifulSoup(html_content, 'html.parser')

    # Extract metadata
    metadata = {}
    if soup.title:
        metadata['title'] = soup.title.string

    # Extract meta tags
    meta_tags = {}
    for meta in soup.find_all('meta'):
        if meta.get('name'):
            meta_tags[meta.get('name')] = meta.get('content', '')
        elif meta.get('property'):
            meta_tags[meta.get('property')] = meta.get('content', '')
    if meta_tags:
        metadata['meta'] = meta_tags

    # Extract main content
    result = {
        'metadata': metadata,
        'text': soup.get_text(separator=' ', strip=True),
        'structure': {}
    }

    # Extract structured content
    # Headers
    headers = []
    for i in range(1, 7):
        for header in soup.find_all(f'h{i}'):
            headers.append({
                'level': i,
                'text': header.get_text(strip=True)
            })
    if headers:
        result['structure']['headers'] = headers

    # Links
    links = []
    for link in soup.find_all('a', href=True):
        links.append({
            'text': link.get_text(strip=True),
            'href': link['href']
        })
    if links:
        result['structure']['links'] = links

    # Images
    images = []
    for img in soup.find_all('img'):
        img_info = {
            'src': img.get('src', ''),
            'alt': img.get('alt', '')
        }
        if img.get('title'):
            img_info['title'] = img['title']
        images.append(img_info)
    if images:
        result['structure']['images'] = images

    # Tables
    tables = []
    for table in soup.find_all('table'):
        table_data = []
        for row in table.find_all('tr'):
            row_data = []
            for cell in row.find_all(['td', 'th']):
                row_data.append(cell.get_text(strip=True))
            if row_data:
                table_data.append(row_data)
        if table_data:
            tables.append(table_data)
    if tables:
        result['structure']['tables'] = tables

    # Lists
    lists = []
    for list_elem in soup.find_all(['ul', 'ol']):
        list_items = []
        for li in list_elem.find_all('li'):
            list_items.append(li.get_text(strip=True))
        if list_items:
            lists.append({
                'type': list_elem.name,
                'items': list_items
            })
    if lists:
        result['structure']['lists'] = lists

    return result


def convert_txt_to_json(txt_content: str, filename: str = "text_file",
                       head_chars: int = 2500, tail_chars: int = 2500) -> Dict[str, Any]:
    """
    Convert text content to JSON format with optional head/tail preservation.

    Args:
        txt_content: Text file content
        filename: Name of the original file
        head_chars: Number of characters to keep from beginning (default 2500, None = all)
        tail_chars: Number of characters to keep from end (default 2500, None = all)

    Returns:
        Dictionary with file_name and content
    """
    original_length = len(txt_content)

    # Apply character limits (both head_chars and tail_chars default to 2500, totaling 5000)
    if head_chars is None and tail_chars is None:
        # Keep all content if both are explicitly set to None
        content = txt_content
    elif head_chars is not None and tail_chars is not None:
        # Both limits specified
        total_chars = head_chars + tail_chars
        if original_length > total_chars:
            # Keep head and tail characters
            content = txt_content[:head_chars] + "\n...\n" + txt_content[-tail_chars:]
        else:
            content = txt_content
    elif head_chars is not None:
        # Only head limit specified
        if original_length > head_chars:
            content = txt_content[:head_chars] + "\n..."
        else:
            content = txt_content
    elif tail_chars is not None:
        # Only tail limit specified
        if original_length > tail_chars:
            content = "...\n" + txt_content[-tail_chars:]
        else:
            content = txt_content
    else:
        # Shouldn't reach here given defaults, but keep for safety
        content = txt_content

    return {
        "file_name": filename,
        "content": content,
        "original_character_count": original_length
    }


def convert_xlsx_to_csv(xlsx_path: str, output_dir: str = None,
                        sheet_name: str = None) -> List[str]:
    """
    Convert XLSX file to CSV format.

    Args:
        xlsx_path: Path to the XLSX file
        output_dir: Directory to save CSV files (defaults to temp directory)
        sheet_name: Specific sheet to convert (None = all sheets)

    Returns:
        List of paths to generated CSV files

    Note:
        Requires 'openpyxl' package for Excel file handling.
        Install with: pip install openpyxl
    """
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="xlsx_to_csv_")

    csv_paths = []

    try:
        # Try to import openpyxl to check if it's available
        try:
            import openpyxl
        except ImportError:
            raise ImportError(
                "The 'openpyxl' package is required for Excel file handling. "
                "Please install it with: pip install openpyxl"
            )

        # Read Excel file
        xls = pd.ExcelFile(xlsx_path)

        # Get base filename without extension
        base_name = os.path.splitext(os.path.basename(xlsx_path))[0]

        # Process specified sheet or all sheets
        if sheet_name:
            if sheet_name in xls.sheet_names:
                df = pd.read_excel(xlsx_path, sheet_name=sheet_name)
                csv_filename = f"{base_name}_{sheet_name}.csv"
                csv_path = os.path.join(output_dir, csv_filename)
                df.to_csv(csv_path, index=False, encoding='utf-8')
                csv_paths.append(csv_path)
            else:
                raise ValueError(f"Sheet '{sheet_name}' not found in {xlsx_path}")
        else:
            # Convert all sheets
            for sheet in xls.sheet_names:
                df = pd.read_excel(xlsx_path, sheet_name=sheet)
                # Clean sheet name for filename
                safe_sheet_name = re.sub(r'[^\w\s-]', '_', sheet)
                csv_filename = f"{base_name}_{safe_sheet_name}.csv" if len(xls.sheet_names) > 1 else f"{base_name}.csv"
                csv_path = os.path.join(output_dir, csv_filename)
                df.to_csv(csv_path, index=False, encoding='utf-8')
                csv_paths.append(csv_path)

    except Exception as e:
        raise Exception(f"Failed to convert XLSX to CSV: {str(e)}")

    return csv_paths


def convert_csv_to_json(csv_path: str, output_path: str = None) -> str:
    """
    Convert CSV file to JSON format.

    Args:
        csv_path: Path to the CSV file
        output_path: Output path for JSON file (optional, defaults to same name with .json extension)

    Returns:
        Path to the generated JSON file
    """
    import pandas as pd

    try:
        # Read CSV file
        df = pd.read_csv(csv_path)

        # Convert to list of dictionaries
        data = df.to_dict('records')

        # Determine output path
        if output_path is None:
            output_path = os.path.splitext(csv_path)[0] + '.json'

        # Write JSON file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        return output_path

    except Exception as e:
        raise Exception(f"Failed to convert CSV to JSON: {str(e)}")


def xlsx_to_csv_content(xlsx_content: bytes, sheet_name: str = None) -> str:
    """
    Convert XLSX content directly to CSV string.

    Args:
        xlsx_content: XLSX file content as bytes
        sheet_name: Specific sheet to convert (None = first sheet)

    Returns:
        CSV content as string

    Note:
        Requires 'openpyxl' package for Excel file handling.
        Install with: pip install openpyxl
    """
    try:
        # Try to import openpyxl to check if it's available
        try:
            import openpyxl
        except ImportError:
            raise ImportError(
                "The 'openpyxl' package is required for Excel file handling. "
                "Please install it with: pip install openpyxl"
            )
        # Create a temporary file to write the content
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp_file:
            tmp_file.write(xlsx_content)
            tmp_path = tmp_file.name

        # Read the Excel file
        if sheet_name:
            df = pd.read_excel(tmp_path, sheet_name=sheet_name)
        else:
            df = pd.read_excel(tmp_path)

        # Convert to CSV string
        csv_content = df.to_csv(index=False)

        # Clean up temp file
        os.unlink(tmp_path)

        return csv_content

    except Exception as e:
        # Clean up temp file if it exists
        if 'tmp_path' in locals():
            try:
                os.unlink(tmp_path)
            except:
                pass
        raise Exception(f"Failed to convert XLSX content to CSV: {str(e)}")


class DataTruncator:
    """Intelligent data truncator for handling large JSON and text data."""
    
    def __init__(self, 
                 max_total_length: int = 2000,
                 max_string_length: int = 200,
                 max_plain_text_length: int = 5000,
                 max_array_items: int = 3,
                 max_dict_keys: int = 10,
                 preserve_head_tail_ratio: float = 0.3):
        """
        Initialize the data truncator.
        
        Args:
            max_total_length: Maximum total length for serialized data
            max_string_length: Maximum length for individual string fields
            max_plain_text_length: Maximum length for plain text content
            max_array_items: Maximum number of array items to keep
            max_dict_keys: Maximum number of dictionary keys to keep
            preserve_head_tail_ratio: Ratio of head/tail content to preserve in long texts
        """
        self.max_total_length = max_total_length
        self.max_string_length = max_string_length
        self.max_plain_text_length = max_plain_text_length
        self.max_array_items = max_array_items
        self.max_dict_keys = max_dict_keys
        self.preserve_head_tail_ratio = preserve_head_tail_ratio
    
    def truncate_text(self, text: str, max_length: Optional[int] = None, is_plain_text: bool = False) -> str:
        """
        Intelligently truncate long text while preserving important head and tail content.
        
        Args:
            text: Input text to truncate
            max_length: Maximum length, uses default if not specified
            is_plain_text: Whether this is plain text content (uses longer limit)
            
        Returns:
            Truncated text with preserved structure
        """
        if max_length is None:
            max_length = self.max_plain_text_length if is_plain_text else self.max_string_length
        
        if len(text) <= max_length:
            return text
        
        # Calculate head and tail preservation length
        preserve_length = int(max_length * self.preserve_head_tail_ratio)
        middle_space = max_length - 2 * preserve_length - 30  # 30 chars for truncation markers
        
        if middle_space < 0:
            # If not enough space, only preserve the beginning
            return text[:max_length-3] + "..."
        
        # Try to break at appropriate positions (periods, newlines, etc.)
        head_text = text[:preserve_length]
        tail_text = text[-preserve_length:]
        
        # Find suitable break points
        head_break_points = [m.start() for m in re.finditer(r'[.!?\n]', head_text)]
        if head_break_points:
            head_end = head_break_points[-1] + 1
            head_text = head_text[:head_end]
        
        tail_break_points = [m.start() for m in re.finditer(r'[.!?\n]', tail_text)]
        if tail_break_points:
            tail_start = tail_break_points[0]
            tail_text = tail_text[tail_start:]
        
        # Calculate number of truncated characters
        truncated_chars = len(text) - len(head_text) - len(tail_text)
        
        return f"{head_text}...[truncated {truncated_chars} chars]...{tail_text}"
    
    def truncate_list(self, data_list: List[Any]) -> List[Any]:
        """
        Intelligently truncate lists by preserving representative samples from head, middle, and tail.
        
        Args:
            data_list: Input list to truncate
            
        Returns:
            Truncated list with representative items
        """
        if len(data_list) <= self.max_array_items:
            return [self._truncate_value(item) for item in data_list]
        
        # Select representative samples: head, middle, tail
        if self.max_array_items >= 3:
            indices = [
                0,  # Head
                len(data_list) // 2,  # Middle
                -1  # Tail
            ]
            # If more samples allowed, add more representative positions
            if self.max_array_items > 3:
                additional_count = self.max_array_items - 3
                step = len(data_list) // (additional_count + 1)
                for i in range(1, additional_count + 1):
                    indices.append(i * step)
            
            # Remove duplicates and sort
            indices = sorted(list(set(indices)))[:self.max_array_items-1]
            
            result = [self._truncate_value(data_list[i]) for i in indices]
            
            # Add truncation marker
            truncated_count = len(data_list) - len(result)
            result.append(f"...[{truncated_count} more items truncated]...")
            
        else:
            # If only few items allowed, prioritize head items
            result = [self._truncate_value(data_list[i]) for i in range(self.max_array_items)]
        
        return result
    
    def truncate_dict(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Intelligently truncate dictionary by preserving keys based on length and alphabetical order.
        
        Args:
            data_dict: Input dictionary to truncate
            
        Returns:
            Truncated dictionary
        """
        if len(data_dict) <= self.max_dict_keys:
            return {k: self._truncate_value(v) for k, v in data_dict.items()}
        
        # Sort by key name length and alphabetical order (prioritize shorter keys for readability)
        keys = list(data_dict.keys())
        sorted_keys = sorted(keys, key=lambda x: (len(x), x.lower()))
        
        # Select keys to preserve
        selected_keys = sorted_keys[:self.max_dict_keys-1]
        result = {k: self._truncate_value(data_dict[k]) for k in selected_keys}
        
        # Add truncation information
        if len(keys) > len(selected_keys):
            truncated_count = len(keys) - len(selected_keys)
            remaining_keys = [k for k in keys if k not in selected_keys]
            # Show first 3 truncated keys as sample
            truncated_keys_sample = remaining_keys[:3]
            result['__truncated__'] = f"Truncated {truncated_count} keys: {truncated_keys_sample}..."
        
        return result
    
    def _truncate_value(self, value: Any) -> Any:
        """
        Recursively truncate a single value.
        
        Args:
            value: Input value to truncate
            
        Returns:
            Truncated value
        """
        if isinstance(value, str):
            return self.truncate_text(value)
        elif isinstance(value, list):
            return self.truncate_list(value)
        elif isinstance(value, dict):
            return self.truncate_dict(value)
        else:
            # Numbers, booleans, etc. remain unchanged
            return value
    
    def truncate_data(self, data: Any) -> Any:
        """
        Main truncation method for intelligently truncating any type of data.
        
        Args:
            data: Input data to truncate
            
        Returns:
            Truncated data
        """
        truncated = self._truncate_value(data)
        
        # Check if total length exceeds limit
        serialized = json.dumps(truncated, ensure_ascii=False, separators=(',', ':'))
        if len(serialized) > self.max_total_length:
            # If still too long, perform more aggressive truncation
            return self._aggressive_truncate(truncated)
        
        return truncated
    
    def _aggressive_truncate(self, data: Any) -> Any:
        """
        Aggressive truncation: further reduce data volume.
        
        Args:
            data: Input data to truncate
            
        Returns:
            Aggressively truncated data
        """
        # Store original parameters
        old_max_string = self.max_string_length
        old_max_array = self.max_array_items
        old_max_dict = self.max_dict_keys
        
        # Temporarily adjust parameters to be more aggressive
        self.max_string_length = min(100, self.max_string_length // 2)
        self.max_array_items = max(2, self.max_array_items // 2)
        self.max_dict_keys = max(3, self.max_dict_keys // 2)
        
        result = self._truncate_value(data)
        
        # Restore original parameters
        self.max_string_length = old_max_string
        self.max_array_items = old_max_array
        self.max_dict_keys = old_max_dict
        
        return result


def smart_truncate_json(data: Any, 
                       max_total_length: int = 2000,
                       max_string_length: int = 200,
                       max_plain_text_length: int = 5000,
                       max_array_items: int = 3) -> Any:
    """
    Convenient JSON intelligent truncation function.
    
    Args:
        data: Data to truncate
        max_total_length: Maximum total length
        max_string_length: Maximum length for individual strings
        max_plain_text_length: Maximum length for plain text content
        max_array_items: Maximum number of array items
        
    Returns:
        Truncated data
    """
    truncator = DataTruncator(
        max_total_length=max_total_length,
        max_string_length=max_string_length,
        max_plain_text_length=max_plain_text_length,
        max_array_items=max_array_items
    )
    return truncator.truncate_data(data)