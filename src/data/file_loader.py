import pandas as pd
from typing import Optional, Tuple
import io

MAX_FILE_SIZE_MB = 50
MAX_ROWS = 1000000

def validate_and_load_uploaded_file(uploaded_file) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Securely loads a Streamlit UploadedFile into a Pandas DataFrame.
    Returns (df, error_message).
    Never writes to the filesystem.
    """
    if uploaded_file is None:
        return None, "No file uploaded."
    
    # Check size (uploaded_file.size is in bytes)
    file_size = getattr(uploaded_file, "size", 0)
    size_mb = file_size / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        return None, f"File size ({size_mb:.1f} MB) exceeds maximum allowed ({MAX_FILE_SIZE_MB} MB)."

    if file_size == 0:
        return None, "The uploaded file is empty (0 bytes)."
        
    try:
        # Load based on extension without writing to disk
        filename = getattr(uploaded_file, "name", "").lower()
        if filename.endswith(".csv"):
            # Try multiple encodings
            for enc in ("utf-8", "latin1", "cp1252"):
                try:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, encoding=enc)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                return None, "Could not decode CSV file. Please ensure it is UTF-8 encoded."
        elif filename.endswith((".xls", ".xlsx")):
            uploaded_file.seek(0)
            df = pd.read_excel(uploaded_file)
        else:
            return None, "Unsupported file format. Please upload a .csv or .xlsx file."
            
        if df.empty:
            return None, "The uploaded file contains no data."
            
        if len(df) > MAX_ROWS:
            return None, f"The file contains {len(df)} rows, which exceeds the limit of {MAX_ROWS}."
            
        return df, None
        
    except Exception:
        return None, "Failed to parse file. Please verify that the file is a properly formatted, valid CSV or Excel document."
