#!/usr/bin/env python3
"""
Unified extraction service that routes to appropriate OCR/AI method based on user selection.

This module provides a unified interface for extracting structured data from PDF documents
using multiple OCR and AI methods. It acts as a router, calling the appropriate extraction
function based on the selected method.

Supported Extraction Methods:
    - Local Model (Chandra OCR): High-accuracy local OCR with GPU/CPU support
    - PyTesseract: Google's Tesseract OCR engine
    - EasyOCR: Fast CPU-based OCR with lightweight models
    - HuggingFace: Cloud-based OCR via Inference API
    - Datalab API: High-accuracy OCR via Datalab Marker API
    - Gemini AI: Google's AI-powered extraction with context understanding
    - Deepseek AI: AI-powered extraction with vision capabilities

Key Functions:
    - extract_data(): Main router function for single PDF extraction
    - process_multiple_files(): Batch processing for multiple PDFs
    - extract_with_*(): Method-specific extraction functions

Example:
    >>> api_keys = {'gemini': 'your-api-key'}
    >>> rows = extract_data('document.pdf', method='gemini', api_keys=api_keys)
    >>> print(f"Extracted {len(rows)} rows")
"""

import os
import tempfile
from collections.abc import Callable
from typing import Any

from logging_config import get_logger

logger = get_logger(__name__)


__version__ = "1.1.0"


def extract_with_local_model(
    pdf_path: str,
    model_name: str = "datalab-to/chandra",
    use_cpu: bool = False,
    use_pretty: bool = False,
) -> list[dict[str, str]]:
    """
    Extract data using local OCR model.

    Supports:
    - Chandra (datalab-to/chandra) - via Chandra CLI (fully supported)
    - TrOCR models - via transformers library (experimental)
    - Other models - attempted via transformers (may need custom implementation)

    Args:
        pdf_path: Path to PDF file
        model_name: Model name/identifier
        use_cpu: If True, force CPU mode
        use_pretty: If True, use pretty output formatter (only for Chandra)

    Returns:
        List of extracted row dictionaries
    """
    from model_loaders import get_model_loader, is_model_supported

    # Check if model is supported
    is_supported, reason = is_model_supported(model_name)

    if not is_supported:
        raise ValueError(
            f"Model '{model_name}' is not supported: {reason}\n"
            f"Please install required dependencies or use a supported model."
        )

    # Get the appropriate loader
    loader_func, loader_type = get_model_loader(model_name)

    # For Chandra models, use existing extraction scripts (they handle parsing)
    if loader_type == "chandra_cli":
        if use_cpu:
            from extract_ec_data_cpu import extract_data_from_pdf
        elif use_pretty:
            from extract_ec_data_pretty import extract_data_from_pdf
        else:
            from extract_ec_data import extract_data_from_pdf

        rows = extract_data_from_pdf(pdf_path)

    # For other models, use generic extraction
    else:
        # Extract text using the model
        text, structured_data = loader_func(model_name, pdf_path, use_cpu)

        # Parse the extracted text (reuse parsing logic from EasyOCR)
        from extract_ec_data_easyocr import parse_table_rows

        rows = parse_table_rows(text)

        # Add filename
        filename = os.path.basename(pdf_path)
        for row in rows:
            row["filename"] = filename

    # Normalize field names
    normalized_rows = []
    for row in rows:
        if "Plot No./" in row:
            row["Plot No."] = row.pop("Plot No./")
        if "Survey No./" in row:
            row["Survey No."] = row.pop("Survey No./")
        if "Plot No" in row and "Plot No." not in row:
            row["Plot No."] = row.pop("Plot No")
        if "Survey No" in row and "Survey No." not in row:
            row["Survey No."] = row.pop("Survey No")
        normalized_rows.append(row)

    return normalized_rows


def extract_with_pytesseract(pdf_path: str) -> list[dict[str, str]]:
    """
    Extract data using PyTesseract (Google's Tesseract OCR).

    Args:
        pdf_path: Path to PDF file

    Returns:
        List of extracted row dictionaries
    """
    from extract_ec_data_pytesseract import extract_data_from_pdf

    rows = extract_data_from_pdf(pdf_path)

    # Normalize field names to match standard format
    normalized_rows = []
    for row in rows:
        if "Plot No" in row and "Plot No." not in row:
            row["Plot No."] = row.pop("Plot No")
        if "Survey No" in row and "Survey No." not in row:
            row["Survey No."] = row.pop("Survey No")
        if "Survey No./" in row:
            row["Survey No."] = row.pop("Survey No./")
        if "Plot No./" in row:
            row["Plot No."] = row.pop("Plot No./")
        normalized_rows.append(row)

    return normalized_rows


def extract_with_easyocr(pdf_path: str) -> list[dict[str, str]]:
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
        if "Plot No" in row and "Plot No." not in row:
            row["Plot No."] = row.pop("Plot No")
        if "Survey No" in row and "Survey No." not in row:
            row["Survey No."] = row.pop("Survey No")
        if "Survey No./" in row:
            row["Survey No."] = row.pop("Survey No./")
        if "Plot No./" in row:
            row["Plot No."] = row.pop("Plot No./")
        normalized_rows.append(row)

    return normalized_rows


def extract_with_huggingface(
    pdf_path: str, api_key: str, model_name: str = "datalab-to/chandra"
) -> list[dict[str, str]]:
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
        if "Plot No./" in row:
            row["Plot No."] = row.pop("Plot No./")
        if "Survey No./" in row:
            row["Survey No."] = row.pop("Survey No./")

    return rows


def extract_with_datalab_api(pdf_path: str, api_key: str) -> list[dict[str, str]]:
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
        if "Plot No./" in row:
            row["Plot No."] = row.pop("Plot No./")
        if "Survey No./" in row:
            row["Survey No."] = row.pop("Survey No./")

    return rows


def extract_with_gemini(
    pdf_path: str, api_key: str, custom_fields: list[str] | None = None
) -> list[dict[str, str]]:
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
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    model_name_clean = model_name.replace("models/", "")
    model = genai.GenerativeModel(model_name_clean)
    model._model_name = model_name_clean

    # Extract data with custom fields support
    from extract_ec_data_gemini import extract_data_from_pdf_gemini_custom

    return extract_data_from_pdf_gemini_custom(pdf_path, model, custom_fields)


def extract_with_deepseek(
    pdf_path: str, api_key: str, custom_fields: list[str] | None = None
) -> list[dict[str, str]]:
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
    api_keys: dict[str, str],
    model_name: str | None = None,
    custom_fields: list[str] | None = None,
    local_model_options: dict[str, Any] | None = None,
    auto_detect_fields: bool = False,
    detected_fields: list[str] | None = None,
) -> list[dict[str, str]]:
    """
    Main router function that calls appropriate extraction method based on user selection.

    This is the primary entry point for extracting structured data from PDF documents.
    It automatically routes to the correct extraction method and handles field detection
    if enabled.

    Args:
        pdf_path: Path to PDF file to extract data from
        method: Extraction method name. Valid values:
            - 'local' or 'local_model': Local OCR model (Chandra)
            - 'pytesseract' or 'tesseract': PyTesseract OCR
            - 'easyocr': EasyOCR library
            - 'huggingface' or 'hf': HuggingFace Inference API
            - 'datalab': Datalab Marker API
            - 'gemini': Google Gemini AI
            - 'deepseek': Deepseek AI
        api_keys: Dictionary mapping provider names to API keys.
            Keys: 'huggingface', 'datalab', 'gemini', 'deepseek'
        model_name: Optional model identifier (mainly for HuggingFace).
            Default: 'datalab-to/chandra'
        custom_fields: Optional list of custom field names to extract.
            Only supported by AI methods (Gemini, Deepseek).
            Example: ['Village Name', 'Registration Date', 'Property Type']
        local_model_options: Optional dictionary with local model configuration:
            - 'use_cpu': Force CPU mode even if GPU available (bool)
            - 'use_pretty': Use formatted output for Chandra (bool)
        auto_detect_fields: If True, automatically detect field names from PDF
            before extraction. Works best with AI methods.
        detected_fields: Pre-detected field names list. Used when
            auto_detect_fields=False but fields were detected separately.

    Returns:
        List of dictionaries, where each dictionary represents one extracted row.
        Each row contains field names as keys and extracted values as strings.
        Always includes 'filename' field with the PDF filename.

        Example:
            [
                {
                    'filename': 'document.pdf',
                    'Plot No.': '123',
                    'Survey No.': '456',
                    'Name of Executant(s)': 'John Doe'
                },
                ...
            ]

    Raises:
        ValueError: If method is unknown or required API key is missing
        FileNotFoundError: If PDF file doesn't exist
        Exception: Various extraction errors depending on method

    Example:
        >>> # Extract using Gemini AI with custom fields
        >>> api_keys = {'gemini': 'your-api-key'}
        >>> rows = extract_data(
        ...     'document.pdf',
        ...     method='gemini',
        ...     api_keys=api_keys,
        ...     custom_fields=['Village Name', 'Plot No.']
        ... )

        >>> # Extract using local model with CPU mode
        >>> rows = extract_data(
        ...     'document.pdf',
        ...     method='local',
        ...     api_keys={},
        ...     local_model_options={'use_cpu': True}
        ... )
    """
    method = method.lower()

    # Auto-detect fields if requested
    if auto_detect_fields:
        try:
            from field_detector import detect_fields_from_pdf

            detection_method = "ai" if method in ["gemini", "deepseek"] else "ocr"
            detected_fields_list = detect_fields_from_pdf(
                pdf_path,
                method=detection_method,
                extraction_method=method,
                api_keys=api_keys,
            )
            if detected_fields_list:
                # Use detected fields as custom_fields for AI methods, or store for OCR methods
                if method in ["gemini", "deepseek"]:
                    custom_fields = detected_fields_list
                detected_fields = detected_fields_list
        except Exception as e:
            logger.warning(
                "Field detection failed: %s. Proceeding with default fields.", e
            )

    # Use pre-detected fields if provided (and no custom_fields specified)
    if detected_fields and not custom_fields and method in ["gemini", "deepseek"]:
        custom_fields = detected_fields

    if method == "local" or method == "local_model":
        # Local model extraction (downloads model on first use)
        options = local_model_options or {}
        return extract_with_local_model(
            pdf_path,
            model_name or "datalab-to/chandra",
            use_cpu=options.get("use_cpu", False),
            use_pretty=options.get("use_pretty", False),
        )

    elif method == "pytesseract" or method == "tesseract":
        # PyTesseract doesn't support custom fields dynamically, returns all fields
        return extract_with_pytesseract(pdf_path)

    elif method == "easyocr":
        # EasyOCR doesn't support custom fields dynamically, returns all fields
        return extract_with_easyocr(pdf_path)

    elif method == "huggingface" or method == "hf":
        api_key = api_keys.get("huggingface") or api_keys.get("hf")
        if not api_key:
            raise ValueError("HuggingFace API key is required")
        model = model_name or "datalab-to/chandra"
        # HuggingFace also doesn't support custom fields in current implementation
        return extract_with_huggingface(pdf_path, api_key, model)

    elif method == "datalab":
        api_key = api_keys.get("datalab")
        if not api_key:
            raise ValueError("Datalab API key is required")
        # Datalab also doesn't support custom fields in current implementation
        return extract_with_datalab_api(pdf_path, api_key)

    elif method == "gemini":
        api_key = api_keys.get("gemini")
        if not api_key:
            raise ValueError("Gemini API key is required")
        return extract_with_gemini(pdf_path, api_key, custom_fields=custom_fields)

    elif method == "deepseek":
        api_key = api_keys.get("deepseek")
        if not api_key:
            raise ValueError("Deepseek API key is required")
        return extract_with_deepseek(pdf_path, api_key, custom_fields=custom_fields)

    else:
        raise ValueError(f"Unknown extraction method: {method}")


def process_multiple_files(
    pdf_files: list[str],
    method: str,
    api_keys: dict[str, str],
    model_name: str | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
    auto_detect_fields: bool = False,
    field_detection_mode: str = "unified",
) -> dict[str, list[dict[str, str]]]:
    """
    Process multiple PDF files sequentially with progress tracking.

    This function processes multiple PDF files one by one, aggregating results
    into a dictionary. It supports progress callbacks for UI updates and can
    detect fields either once from the first file or per-file.

    Args:
        pdf_files: List of absolute or relative paths to PDF files to process
        method: Extraction method name (same as extract_data())
        api_keys: Dictionary mapping provider names to API keys
        model_name: Optional model identifier (mainly for HuggingFace)
        progress_callback: Optional callback function called after each file.
            Signature: callback(current_file_index, total_files, filename)
            Example: lambda idx, total, name: print(f"Processing {idx}/{total}: {name}")
        auto_detect_fields: If True, automatically detect fields before extraction.
            When True, uses field_detection_mode to determine detection strategy.
        field_detection_mode: Field detection strategy when auto_detect_fields=True:
            - 'unified': Detect fields once from the first file, use for all files
            - 'per_file': Detect fields from each file individually and merge

    Returns:
        Dictionary mapping PDF filenames (basename) to lists of extracted rows.
        Each value is a list of dictionaries, same format as extract_data().

        Example:
            {
                'file1.pdf': [
                    {'filename': 'file1.pdf', 'Plot No.': '123', ...},
                    {'filename': 'file1.pdf', 'Plot No.': '456', ...}
                ],
                'file2.pdf': [
                    {'filename': 'file2.pdf', 'Plot No.': '789', ...}
                ]
            }

    Raises:
        ValueError: If method is unknown or required API key is missing
        FileNotFoundError: If any PDF file doesn't exist
        Exception: Various extraction errors depending on method

    Example:
        >>> files = ['doc1.pdf', 'doc2.pdf', 'doc3.pdf']
        >>> api_keys = {'datalab': 'your-api-key'}
        >>> results = process_multiple_files(
        ...     files,
        ...     method='datalab',
        ...     api_keys=api_keys,
        ...     progress_callback=lambda i, t, n: print(f"{i}/{t}: {n}")
        ... )
        >>> total_rows = sum(len(rows) for rows in results.values())
        >>> print(f"Extracted {total_rows} total rows from {len(results)} files")
    """
    detected_fields = None

    # Auto-detect fields if requested
    if auto_detect_fields and pdf_files:
        try:
            from field_detector import detect_fields_batch

            detection_method = "ai" if method in ["gemini", "deepseek"] else "ocr"
            detection_result = detect_fields_batch(
                pdf_files,
                method=detection_method,
                extraction_method=method,
                api_keys=api_keys,
                mode=field_detection_mode,
            )
            detected_fields = detection_result.get("fields", [])

            if progress_callback and detected_fields:
                progress_callback(
                    0,
                    len(pdf_files),
                    f"Detected {len(detected_fields)} fields: {', '.join(detected_fields[:5])}...",
                )
        except Exception as e:
            logger.warning(
                "Field detection failed: %s. Proceeding with default fields.", e
            )

    results = {}

    for i, pdf_path in enumerate(pdf_files):
        filename = os.path.basename(pdf_path)

        if progress_callback:
            progress_callback(i + 1, len(pdf_files), f"Processing {filename}...")

        try:
            # Pass detected_fields to extract_data
            rows = extract_data(
                pdf_path,
                method,
                api_keys,
                model_name,
                custom_fields=(
                    detected_fields if method in ["gemini", "deepseek"] else None
                ),
                detected_fields=detected_fields,
            )
            results[filename] = rows
        except Exception as e:
            if progress_callback:
                progress_callback(
                    i + 1, len(pdf_files), f"Error processing {filename}: {str(e)}"
                )
            results[filename] = []

    return results


def save_uploaded_file(uploaded_file, temp_dir: str | None = None) -> str:
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
