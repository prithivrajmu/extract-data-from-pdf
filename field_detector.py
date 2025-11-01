#!/usr/bin/env python3
"""
Field detection module for automatically identifying fields in PDF documents.
Supports both AI-based (Gemini/Deepseek) and OCR-based detection methods.
"""

import os
import json
import re
import time
from typing import List, Dict, Optional, Tuple, Set
from pathlib import Path


def create_field_detection_prompt() -> str:
    """
    Create prompt for detecting fields/columns from PDF table headers.
    
    Returns:
        Prompt string for AI field detection
    """
    prompt = """Analyze this PDF document and identify all column headers/field names from any tables present.

TASK: Identify the column headers/field names in the table(s) shown in this document.

INSTRUCTIONS:
1. Look for table headers in the document
2. Extract the exact column names/field names as they appear
3. Handle variations in spelling, spacing, punctuation, or language
4. Return ONLY a JSON array of field names (strings), nothing else

RETURN FORMAT: Valid JSON array of strings only, no explanations, no markdown.

Example output format:
["Sr.No", "Document No.& Year", "Name of Executant(s)", "Name of Claimant(s)", "Survey No.", "Plot No."]

Return ONLY the JSON array of field names."""
    return prompt


def parse_json_response(response_text: str) -> List[str]:
    """
    Robustly parse JSON response containing field names.
    
    Args:
        response_text: Raw response text from AI
        
    Returns:
        Parsed list of field names
    """
    # Clean up response text
    text = response_text.strip()
    
    # Remove markdown code blocks if present
    if text.startswith("```json"):
        text = text[7:].strip()
    if text.startswith("```"):
        text = text[3:].strip()
    if text.endswith("```"):
        text = text[:-3].strip()
    
    # Strategy 1: Try direct JSON parsing
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(field).strip() for field in parsed if field]
        elif isinstance(parsed, dict):
            # Sometimes AI returns {"fields": [...]}
            if 'fields' in parsed:
                fields = parsed['fields']
                if isinstance(fields, list):
                    return [str(field).strip() for field in fields if field]
    except json.JSONDecodeError:
        pass
    
    # Strategy 2: Try to find JSON array in the text
    json_match = re.search(r'\[[\s\S]*\]', text)
    if json_match:
        try:
            parsed = json.loads(json_match.group())
            if isinstance(parsed, list):
                return [str(field).strip() for field in parsed if field]
        except json.JSONDecodeError:
            pass
    
    # Strategy 3: Try to extract field names from text (fallback)
    # Look for quoted strings
    field_pattern = r'["\']([^"\']+)["\']'
    fields = re.findall(field_pattern, text)
    if fields:
        return [f.strip() for f in fields if f.strip()]
    
    # If all strategies fail, return empty list
    print(f"⚠️  Could not parse field detection response. Preview: {text[:200]}...")
    return []


def detect_fields_with_ai(pdf_path: str, api_key: str, provider: str = 'gemini', max_retries: int = 3) -> List[str]:
    """
    Detect fields from PDF using AI (Gemini or Deepseek).
    
    Args:
        pdf_path: Path to PDF file
        api_key: API key for the provider
        provider: 'gemini' or 'deepseek'
        max_retries: Maximum number of retry attempts
        
    Returns:
        List of detected field names
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    prompt = create_field_detection_prompt()
    
    if provider.lower() == 'gemini':
        return _detect_fields_with_gemini(pdf_path, api_key, prompt, max_retries)
    elif provider.lower() == 'deepseek':
        return _detect_fields_with_deepseek(pdf_path, api_key, prompt, max_retries)
    else:
        raise ValueError(f"Unsupported AI provider: {provider}. Use 'gemini' or 'deepseek'.")


def _detect_fields_with_gemini(pdf_path: str, api_key: str, prompt: str, max_retries: int) -> List[str]:
    """Detect fields using Gemini API."""
    import google.generativeai as genai
    
    genai.configure(api_key=api_key)
    
    # Setup model
    model_name = os.getenv('GEMINI_MODEL', 'gemini-2.5-flash')
    model_name_clean = model_name.replace('models/', '')
    model = genai.GenerativeModel(model_name_clean)
    
    pdf_file = None
    
    try:
        # Upload PDF file
        for retry in range(max_retries):
            try:
                pdf_file = genai.upload_file(path=pdf_path)
                break
            except Exception as e:
                if retry < max_retries - 1:
                    wait_time = (retry + 1) * 5
                    print(f"   ⚠️  Upload failed (attempt {retry + 1}/{max_retries}): {e}")
                    time.sleep(wait_time)
                else:
                    raise Exception(f"Failed to upload PDF after {max_retries} attempts: {e}")
        
        # Wait for file to be processed
        max_wait_time = 60
        wait_interval = 2
        elapsed_time = 0
        
        while pdf_file.state.name == "PROCESSING" and elapsed_time < max_wait_time:
            time.sleep(wait_interval)
            elapsed_time += wait_interval
            pdf_file = genai.get_file(pdf_file.name)
        
        if pdf_file.state.name == "PROCESSING":
            raise Exception(f"File processing timeout after {max_wait_time} seconds")
        if pdf_file.state.name == "FAILED":
            raise Exception("PDF file upload failed")
        
        # Get response
        response = None
        for retry in range(max_retries):
            try:
                response = model.generate_content(
                    [prompt, pdf_file],
                    generation_config={
                        "temperature": 0.1,
                        "response_mime_type": "application/json",
                    }
                )
                break
            except Exception as e:
                if retry < max_retries - 1:
                    wait_time = (retry + 1) * 5
                    print(f"   ⚠️  Request failed (attempt {retry + 1}/{max_retries}): {e}")
                    time.sleep(wait_time)
                else:
                    raise Exception(f"Failed to get response after {max_retries} attempts: {e}")
        
        if response is None:
            raise Exception("Failed to get response from Gemini API")
        
        response_text = response.text.strip()
        detected_fields = parse_json_response(response_text)
        
        return detected_fields
        
    finally:
        # Clean up uploaded file
        if pdf_file:
            try:
                genai.delete_file(pdf_file.name)
            except:
                pass


def _detect_fields_with_deepseek(pdf_path: str, api_key: str, prompt: str, max_retries: int) -> List[str]:
    """Detect fields using Deepseek API."""
    import requests
    import base64
    from pdf2image import convert_from_path
    import io
    
    base_url = "https://api.deepseek.com"
    api_endpoint = f"{base_url}/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Convert PDF to images
    try:
        images = convert_from_path(pdf_path, dpi=200)
        
        # Convert first page to base64 (for field detection, first page usually has headers)
        image_parts = []
        if images:
            buffered = io.BytesIO()
            images[0].save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            image_parts.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{img_base64}"
                }
            })
    except Exception as e:
        raise Exception(f"Failed to process PDF images: {e}")
    
    # Create messages
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt}
            ] + image_parts
        }
    ]
    
    # Make API request
    response = None
    for retry in range(max_retries):
        try:
            payload = {
                "model": "deepseek-chat",
                "messages": messages,
                "temperature": 0.1
            }
            
            api_response = requests.post(api_endpoint, json=payload, headers=headers, timeout=120)
            api_response.raise_for_status()
            
            result = api_response.json()
            if 'choices' in result and len(result['choices']) > 0:
                response_text = result['choices'][0]['message']['content']
                detected_fields = parse_json_response(response_text)
                return detected_fields
            else:
                raise Exception("Invalid response from Deepseek API")
                
        except Exception as e:
            if retry < max_retries - 1:
                wait_time = (retry + 1) * 5
                print(f"   ⚠️  Request failed (attempt {retry + 1}/{max_retries}): {e}")
                time.sleep(wait_time)
            else:
                raise Exception(f"Failed to get response after {max_retries} attempts: {e}")
    
    return []


def detect_fields_with_ocr(pdf_path: str, ocr_method: str = 'chandra') -> List[str]:
    """
    Detect fields from PDF using OCR by extracting table headers.
    
    Args:
        pdf_path: Path to PDF file
        ocr_method: OCR method to use ('chandra', 'easyocr', 'pytesseract')
        
    Returns:
        List of detected field names
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    # Extract text using OCR
    text = None
    
    if ocr_method == 'chandra':
        try:
            from extract_ec_data import extract_text_from_pdf
        except ImportError:
            from examples.extract_ec_data import extract_text_from_pdf
        result = extract_text_from_pdf(pdf_path, use_structure=False)
        if isinstance(result, tuple):
            text = result[0]
        else:
            text = result
    elif ocr_method == 'easyocr':
        try:
            from extract_ec_data_easyocr import extract_text_from_pdf_easyocr
        except ImportError:
            from examples.extract_ec_data_easyocr import extract_text_from_pdf_easyocr
        text = extract_text_from_pdf_easyocr(pdf_path)
    elif ocr_method == 'pytesseract':
        try:
            from extract_ec_data_pytesseract import extract_text_from_pdf_pytesseract
        except ImportError:
            from examples.extract_ec_data_pytesseract import extract_text_from_pdf_pytesseract
        text = extract_text_from_pdf_pytesseract(pdf_path)
    else:
        raise ValueError(f"Unsupported OCR method: {ocr_method}")
    
    if not text:
        return []
    
    # Parse headers from OCR text
    return _parse_headers_from_ocr_text(text)


def _parse_headers_from_ocr_text(text: str) -> List[str]:
    """
    Parse table headers from OCR text.
    
    Args:
        text: OCR extracted text
        
    Returns:
        List of detected header/field names
    """
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    detected_fields = []
    
    # Common patterns for headers
    header_patterns = [
        r'Sr\.?\s*No\.?',
        r'Serial\s+No\.?',
        r'Document\s+No\.?\s*[&/]?\s*Year',
        r'Doc\.?\s*No\.?',
        r'Name\s+of\s+Executant',
        r'Executant',
        r'Name\s+of\s+Claimant',
        r'Claimant',
        r'Survey\s+No\.?',
        r'Plot\s+No\.?',
        r'Plot\s+Number',
    ]
    
    # Look for lines that match header patterns (usually in first few lines or lines with multiple matches)
    header_candidates = []
    
    for i, line in enumerate(lines[:50]):  # Check first 50 lines
        line_lower = line.lower()
        matches = []
        
        for pattern in header_patterns:
            if re.search(pattern, line_lower, re.IGNORECASE):
                matches.append(pattern)
        
        if matches:
            # Extract potential header text
            # Try to split by common delimiters
            parts = re.split(r'[|\t\n\r]+', line)
            for part in parts:
                part = part.strip()
                if part and len(part) > 2:
                    header_candidates.append(part)
    
    # If we found candidates, use them
    if header_candidates:
        detected_fields = list(dict.fromkeys(header_candidates))  # Remove duplicates, preserve order
    else:
        # Fallback: Look for common field names in text
        common_fields = [
            'Sr.No', 'Sr No', 'Serial No',
            'Document No.& Year', 'Document No', 'Doc No',
            'Name of Executant(s)', 'Executant',
            'Name of Claimant(s)', 'Claimant',
            'Survey No.', 'Survey No',
            'Plot No.', 'Plot No', 'Plot Number'
        ]
        
        for field in common_fields:
            if field.lower() in text.lower():
                detected_fields.append(field)
    
    # Normalize and clean
    detected_fields = normalize_field_names(detected_fields)
    
    return detected_fields


def normalize_field_names(fields: List[str]) -> List[str]:
    """
    Normalize field names to standard formats.
    
    Args:
        fields: List of field names
        
    Returns:
        Normalized list of field names
    """
    normalized = []
    seen = set()
    
    # Mapping of variations to standard names
    field_mappings = {
        'sr.no': 'Sr.No',
        'sr no': 'Sr.No',
        'serial no': 'Sr.No',
        'serial number': 'Sr.No',
        'document no.& year': 'Document No.& Year',
        'document no': 'Document No.& Year',
        'doc no': 'Document No.& Year',
        'document number': 'Document No.& Year',
        'name of executant(s)': 'Name of Executant(s)',
        'executant': 'Name of Executant(s)',
        'executants': 'Name of Executant(s)',
        'name of executant': 'Name of Executant(s)',
        'name of claimant(s)': 'Name of Claimant(s)',
        'claimant': 'Name of Claimant(s)',
        'claimants': 'Name of Claimant(s)',
        'name of claimant': 'Name of Claimant(s)',
        'survey no.': 'Survey No.',
        'survey no': 'Survey No.',
        'survey number': 'Survey No.',
        'plot no.': 'Plot No.',
        'plot no': 'Plot No.',
        'plot number': 'Plot No.',
        'plot no./': 'Plot No.',
        'survey no./': 'Survey No.',
    }
    
    for field in fields:
        field = field.strip()
        if not field:
            continue
        
        # Normalize using mappings
        field_lower = field.lower()
        if field_lower in field_mappings:
            normalized_field = field_mappings[field_lower]
        else:
            # Try fuzzy matching
            normalized_field = _fuzzy_match_field(field, field_mappings)
        
        # Avoid duplicates
        if normalized_field and normalized_field not in seen:
            normalized.append(normalized_field)
            seen.add(normalized_field)
    
    return normalized


def _fuzzy_match_field(field: str, mappings: Dict[str, str]) -> str:
    """
    Fuzzy match field name to standard name.
    
    Args:
        field: Field name to match
        mappings: Dictionary of normalized field mappings
        
    Returns:
        Matched field name or original if no match
    """
    field_lower = field.lower()
    
    # Try partial matches
    for key, value in mappings.items():
        if key in field_lower or field_lower in key:
            return value
    
    # Return original if no match found
    return field


def merge_field_sets(field_sets: List[List[str]]) -> List[str]:
    """
    Merge multiple field sets by taking union of all fields.
    
    Args:
        field_sets: List of field name lists
        
    Returns:
        Merged list of unique field names
    """
    all_fields = []
    for fields in field_sets:
        all_fields.extend(fields)
    
    # Normalize and remove duplicates
    normalized = normalize_field_names(all_fields)
    
    return normalized


def detect_fields_from_pdf(
    pdf_path: str,
    method: str = 'auto',
    extraction_method: Optional[str] = None,
    api_keys: Optional[Dict[str, str]] = None
) -> List[str]:
    """
    Main function to detect fields from a PDF file.
    
    Args:
        pdf_path: Path to PDF file
        method: Detection method ('auto', 'ai', 'ocr')
        extraction_method: Extraction method to use ('gemini', 'deepseek', 'chandra', etc.)
        api_keys: Dictionary of API keys {'gemini': '...', 'deepseek': '...'}
        
    Returns:
        List of detected field names
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    api_keys = api_keys or {}
    
    # Determine method
    if method == 'auto':
        # Try AI first if available, fallback to OCR
        if extraction_method in ['gemini', 'deepseek']:
            method = 'ai'
        else:
            method = 'ocr'
    
    # Detect using chosen method
    if method == 'ai':
        provider = extraction_method or 'gemini'
        api_key = api_keys.get(provider, '')
        
        if not api_key:
            # Fallback to OCR if no API key
            print("⚠️  No API key provided, falling back to OCR detection")
            ocr_method = extraction_method or 'chandra'
            return detect_fields_with_ocr(pdf_path, ocr_method)
        
        return detect_fields_with_ai(pdf_path, api_key, provider)
    
    elif method == 'ocr':
        ocr_method = extraction_method or 'chandra'
        return detect_fields_with_ocr(pdf_path, ocr_method)
    
    else:
        raise ValueError(f"Unknown detection method: {method}. Use 'auto', 'ai', or 'ocr'.")


def detect_fields_batch(
    pdf_files: List[str],
    method: str = 'auto',
    extraction_method: Optional[str] = None,
    api_keys: Optional[Dict[str, str]] = None,
    mode: str = 'unified'
) -> Dict[str, List[str]]:
    """
    Detect fields from multiple PDF files.
    
    Args:
        pdf_files: List of PDF file paths
        method: Detection method ('auto', 'ai', 'ocr')
        extraction_method: Extraction method to use
        api_keys: Dictionary of API keys
        mode: Detection mode
            - 'unified': Detect from first file, return unified field set
            - 'per_file': Detect per-file, return union of all fields + per-file mapping
            
    Returns:
        Dictionary with:
        - 'fields': Unified list of all detected fields
        - 'per_file': (if mode='per_file') Dictionary mapping filename to detected fields
    """
    if not pdf_files:
        return {'fields': []}
    
    api_keys = api_keys or {}
    
    if mode == 'unified':
        # Detect from first file only (assuming similar structure)
        first_file = pdf_files[0]
        detected_fields = detect_fields_from_pdf(
            first_file,
            method=method,
            extraction_method=extraction_method,
            api_keys=api_keys
        )
        return {'fields': detected_fields}
    
    elif mode == 'per_file':
        # Detect from each file
        per_file_fields = {}
        all_fields = []
        
        for pdf_path in pdf_files:
            filename = os.path.basename(pdf_path)
            try:
                fields = detect_fields_from_pdf(
                    pdf_path,
                    method=method,
                    extraction_method=extraction_method,
                    api_keys=api_keys
                )
                per_file_fields[filename] = fields
                all_fields.append(fields)
            except Exception as e:
                print(f"⚠️  Error detecting fields from {filename}: {e}")
                per_file_fields[filename] = []
        
        # Merge all fields
        unified_fields = merge_field_sets(all_fields)
        
        return {
            'fields': unified_fields,
            'per_file': per_file_fields
        }
    
    else:
        raise ValueError(f"Unknown detection mode: {mode}. Use 'unified' or 'per_file'.")

