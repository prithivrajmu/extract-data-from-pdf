#!/usr/bin/env python3
"""
Unified extraction service that routes to appropriate OCR/AI method based on user selection.
"""

import os
import tempfile
from typing import List, Dict, Optional, Callable
from pathlib import Path


def extract_with_local_model(pdf_path: str, model_name: str = 'datalab-to/chandra', use_cpu: bool = False, use_pretty: bool = False) -> List[Dict[str, str]]:
    """
    Extract data using local OCR model (Chandra or CPU variant).
    
    Args:
        pdf_path: Path to PDF file
        model_name: Model name/identifier (currently only 'datalab-to/chandra' supported)
        use_cpu: If True, force CPU mode
        use_pretty: If True, use pretty output formatter
        
    Returns:
        List of extracted row dictionaries
    """
    if use_cpu:
        from extract_ec_data_cpu import extract_data_from_pdf
    elif use_pretty:
        from extract_ec_data_pretty import extract_data_from_pdf
    else:
        from extract_ec_data import extract_data_from_pdf
    
    rows = extract_data_from_pdf(pdf_path)
    
    # Normalize field names
    normalized_rows = []
    for row in rows:
        if 'Plot No./' in row:
            row['Plot No.'] = row.pop('Plot No./')
        if 'Survey No./' in row:
            row['Survey No.'] = row.pop('Survey No./')
        if 'Plot No' in row and 'Plot No.' not in row:
            row['Plot No.'] = row.pop('Plot No')
        if 'Survey No' in row and 'Survey No.' not in row:
            row['Survey No.'] = row.pop('Survey No')
        normalized_rows.append(row)
    
    return normalized_rows


def extract_with_easyocr(pdf_path: str) -> List[Dict[str, str]]:
    """
    Extract data using EasyOCR.
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        List of extracted row dictionaries
    """
    from extract_ec_data_easyocr import extract_data_from_pdf
    
    rows = extract_data_from_pdf(pdf_path)
    
    # Normalize field names to match standard format
    normalized_rows = []
    for row in rows:
        # EasyOCR uses 'Plot No' instead of 'Plot No.'
        if 'Plot No' in row and 'Plot No.' not in row:
            row['Plot No.'] = row.pop('Plot No')
        if 'Survey No' in row and 'Survey No.' not in row:
            row['Survey No.'] = row.pop('Survey No')
        if 'Survey No./' in row:
            row['Survey No.'] = row.pop('Survey No./')
        if 'Plot No./' in row:
            row['Plot No.'] = row.pop('Plot No./')
        normalized_rows.append(row)
    
    return normalized_rows


def extract_with_huggingface(pdf_path: str, api_key: str, model_name: str = "datalab-to/chandra") -> List[Dict[str, str]]:
    """
    Extract data using HuggingFace API.
    
    Args:
        pdf_path: Path to PDF file
        api_key: HuggingFace API key
        model_name: Model name to use (not used directly, but kept for compatibility)
        
    Returns:
        List of extracted row dictionaries
    """
    from extract_ec_data_hf_api import extract_data_from_pdf
    
    # Use the existing extract function
    rows = extract_data_from_pdf(pdf_path, api_key)
    
    # Normalize field names
    for row in rows:
        if 'Plot No./' in row:
            row['Plot No.'] = row.pop('Plot No./')
        if 'Survey No./' in row:
            row['Survey No.'] = row.pop('Survey No./')
    
    return rows


def extract_with_datalab_api(pdf_path: str, api_key: str) -> List[Dict[str, str]]:
    """
    Extract data using Datalab API.
    
    Args:
        pdf_path: Path to PDF file
        api_key: Datalab API key
        
    Returns:
        List of extracted row dictionaries
    """
    from extract_ec_data_api import extract_data_from_pdf
    
    rows = extract_data_from_pdf(pdf_path, api_key)
    
    # Normalize field names
    for row in rows:
        if 'Plot No./' in row:
            row['Plot No.'] = row.pop('Plot No./')
        if 'Survey No./' in row:
            row['Survey No.'] = row.pop('Survey No./')
    
    return rows


def extract_with_gemini(pdf_path: str, api_key: str, custom_fields: Optional[List[str]] = None) -> List[Dict[str, str]]:
    """
    Extract data using Gemini API.
    
    Args:
        pdf_path: Path to PDF file
        api_key: Gemini API key
        custom_fields: Optional list of custom field names to extract
        
    Returns:
        List of extracted row dictionaries
    """
    import google.generativeai as genai
    
    # Configure Gemini
    genai.configure(api_key=api_key)
    
    # Setup model
    model_name = os.getenv('GEMINI_MODEL', 'gemini-2.5-flash')
    model_name_clean = model_name.replace('models/', '')
    model = genai.GenerativeModel(model_name_clean)
    model._model_name = model_name_clean
    
    # Extract data with custom fields support
    from extract_ec_data_gemini import extract_data_from_pdf_gemini_custom
    return extract_data_from_pdf_gemini_custom(pdf_path, model, custom_fields)


def extract_with_deepseek(pdf_path: str, api_key: str, custom_fields: Optional[List[str]] = None) -> List[Dict[str, str]]:
    """
    Extract data using Deepseek API.
    
    Args:
        pdf_path: Path to PDF file
        api_key: Deepseek API key
        custom_fields: Optional list of custom field names to extract
        
    Returns:
        List of extracted row dictionaries
    """
    from deepseek_api import extract_data_from_pdf_deepseek
    return extract_data_from_pdf_deepseek(pdf_path, api_key, custom_fields)


def extract_data(
    pdf_path: str,
    method: str,
    api_keys: Dict[str, str],
    model_name: Optional[str] = None,
    custom_fields: Optional[List[str]] = None,
    local_model_options: Optional[Dict] = None
) -> List[Dict[str, str]]:
    """
    Main router function that calls appropriate extraction method.
    
    Args:
        pdf_path: Path to PDF file
        method: Extraction method ('easyocr', 'huggingface', 'datalab', 'gemini', 'deepseek')
        api_keys: Dictionary of API keys for different providers
        model_name: Optional model name (for HuggingFace)
        custom_fields: Optional list of custom field names to extract
        
    Returns:
        List of extracted row dictionaries
    """
    method = method.lower()
    
    if method == 'local' or method == 'local_model':
        # Local model extraction (downloads model on first use)
        options = local_model_options or {}
        return extract_with_local_model(
            pdf_path,
            model_name or 'datalab-to/chandra',
            use_cpu=options.get('use_cpu', False),
            use_pretty=options.get('use_pretty', False)
        )
    
    elif method == 'easyocr':
        # EasyOCR doesn't support custom fields dynamically, returns all fields
        return extract_with_easyocr(pdf_path)
    
    elif method == 'huggingface' or method == 'hf':
        api_key = api_keys.get('huggingface') or api_keys.get('hf')
        if not api_key:
            raise ValueError("HuggingFace API key is required")
        model = model_name or "datalab-to/chandra"
        # HuggingFace also doesn't support custom fields in current implementation
        return extract_with_huggingface(pdf_path, api_key, model)
    
    elif method == 'datalab':
        api_key = api_keys.get('datalab')
        if not api_key:
            raise ValueError("Datalab API key is required")
        # Datalab also doesn't support custom fields in current implementation
        return extract_with_datalab_api(pdf_path, api_key)
    
    elif method == 'gemini':
        api_key = api_keys.get('gemini')
        if not api_key:
            raise ValueError("Gemini API key is required")
        return extract_with_gemini(pdf_path, api_key, custom_fields)
    
    elif method == 'deepseek':
        api_key = api_keys.get('deepseek')
        if not api_key:
            raise ValueError("Deepseek API key is required")
        return extract_with_deepseek(pdf_path, api_key, custom_fields)
    
    else:
        raise ValueError(f"Unknown extraction method: {method}")


def process_multiple_files(
    pdf_files: List[str],
    method: str,
    api_keys: Dict[str, str],
    model_name: Optional[str] = None,
    progress_callback: Optional[Callable] = None
) -> Dict[str, List[Dict[str, str]]]:
    """
    Process multiple PDF files sequentially.
    
    Args:
        pdf_files: List of PDF file paths
        method: Extraction method
        api_keys: Dictionary of API keys
        model_name: Optional model name
        progress_callback: Optional callback function for progress updates
        
    Returns:
        Dictionary mapping filename to list of extracted rows
    """
    results = {}
    
    for i, pdf_path in enumerate(pdf_files):
        filename = os.path.basename(pdf_path)
        
        if progress_callback:
            progress_callback(i + 1, len(pdf_files), f"Processing {filename}...")
        
        try:
            rows = extract_data(pdf_path, method, api_keys, model_name)
            results[filename] = rows
        except Exception as e:
            if progress_callback:
                progress_callback(i + 1, len(pdf_files), f"Error processing {filename}: {str(e)}")
            results[filename] = []
    
    return results


def save_uploaded_file(uploaded_file, temp_dir: Optional[str] = None) -> str:
    """
    Save Streamlit uploaded file to temporary location.
    
    Args:
        uploaded_file: Streamlit UploadedFile object
        temp_dir: Optional temporary directory path
        
    Returns:
        Path to saved file
    """
    if temp_dir is None:
        temp_dir = tempfile.gettempdir()
    
    os.makedirs(temp_dir, exist_ok=True)
    
    file_path = os.path.join(temp_dir, uploaded_file.name)
    
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    return file_path

