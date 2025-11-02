# API Documentation

This document provides detailed API documentation for the core modules of the PDF Data Extraction project.

## Table of Contents

- [extraction_service](#extraction_service)
- [utils](#utils)
- [field_detector](#field_detector)
- [api_key_manager](#api_key_manager)

---

## extraction_service

Main service module for routing PDF extraction to appropriate OCR/AI methods.

### `extract_data()`

Primary entry point for extracting structured data from a single PDF document.

```python
def extract_data(
    pdf_path: str,
    method: str,
    api_keys: Dict[str, str],
    model_name: Optional[str] = None,
    custom_fields: Optional[List[str]] = None,
    local_model_options: Optional[Dict[str, Any]] = None,
    auto_detect_fields: bool = False,
    detected_fields: Optional[List[str]] = None
) -> List[Dict[str, str]]
```

**Parameters:**

- `pdf_path` (str): Path to PDF file to extract data from
- `method` (str): Extraction method. Valid values:
  - `'local'` or `'local_model'`: Local OCR model (Chandra)
  - `'pytesseract'` or `'tesseract'`: PyTesseract OCR
  - `'easyocr'`: EasyOCR library
  - `'huggingface'` or `'hf'`: HuggingFace Inference API
  - `'datalab'`: Datalab Marker API
  - `'gemini'`: Google Gemini AI
  - `'deepseek'`: Deepseek AI
- `api_keys` (Dict[str, str]): Dictionary mapping provider names to API keys
  - Keys: `'huggingface'`, `'datalab'`, `'gemini'`, `'deepseek'`
- `model_name` (Optional[str]): Model identifier (mainly for HuggingFace). Default: `'datalab-to/chandra'`
- `custom_fields` (Optional[List[str]]): Custom field names to extract. Only supported by AI methods (Gemini, Deepseek)
- `local_model_options` (Optional[Dict[str, Any]]): Local model configuration:
  - `'use_cpu'`: Force CPU mode even if GPU available (bool)
  - `'use_pretty'`: Use formatted output for Chandra (bool)
- `auto_detect_fields` (bool): Automatically detect field names from PDF before extraction
- `detected_fields` (Optional[List[str]]): Pre-detected field names list

**Returns:**

- `List[Dict[str, str]]`: List of dictionaries, each representing one extracted row. Always includes `'filename'` field.

**Raises:**

- `ValueError`: If method is unknown or required API key is missing
- `FileNotFoundError`: If PDF file doesn't exist
- `Exception`: Various extraction errors depending on method

**Example:**

```python
from extraction_service import extract_data

# Extract using Gemini AI with custom fields
api_keys = {'gemini': 'your-api-key'}
rows = extract_data(
    'document.pdf',
    method='gemini',
    api_keys=api_keys,
    custom_fields=['Village Name', 'Plot No.']
)

print(f"Extracted {len(rows)} rows")
for row in rows:
    print(row)
```

### `process_multiple_files()`

Process multiple PDF files sequentially with progress tracking.

```python
def process_multiple_files(
    pdf_files: List[str],
    method: str,
    api_keys: Dict[str, str],
    model_name: Optional[str] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    auto_detect_fields: bool = False,
    field_detection_mode: str = 'unified'
) -> Dict[str, List[Dict[str, str]]]
```

**Parameters:**

- `pdf_files` (List[str]): List of absolute or relative paths to PDF files
- `method` (str): Extraction method name (same as `extract_data()`)
- `api_keys` (Dict[str, str]): Dictionary mapping provider names to API keys
- `model_name` (Optional[str]): Model identifier (mainly for HuggingFace)
- `progress_callback` (Optional[Callable[[int, int, str], None]]): Callback function called after each file
  - Signature: `callback(current_file_index, total_files, filename)`
- `auto_detect_fields` (bool): Automatically detect fields before extraction
- `field_detection_mode` (str): Field detection strategy:
  - `'unified'`: Detect fields once from the first file, use for all files
  - `'per_file'`: Detect fields from each file individually and merge

**Returns:**

- `Dict[str, List[Dict[str, str]]]`: Dictionary mapping PDF filenames (basename) to lists of extracted rows

**Example:**

```python
from extraction_service import process_multiple_files

files = ['doc1.pdf', 'doc2.pdf', 'doc3.pdf']
api_keys = {'datalab': 'your-api-key'}

def progress_callback(idx, total, filename):
    print(f"Processing {idx}/{total}: {filename}")

results = process_multiple_files(
    files,
    method='datalab',
    api_keys=api_keys,
    progress_callback=progress_callback
)

total_rows = sum(len(rows) for rows in results.values())
print(f"Extracted {total_rows} total rows from {len(results)} files")
```

---

## utils

Utility functions for data processing, validation, and formatting.

### `filter_fields()`

Filter dictionary list to only include selected fields.

```python
def filter_fields(data: List[Dict], selected_fields: Set[str]) -> List[Dict]
```

**Parameters:**

- `data` (List[Dict]): List of dictionaries with extracted data
- `selected_fields` (Set[str]): Set of field names to include

**Returns:**

- `List[Dict]`: Filtered list of dictionaries. Always includes `'filename'` field even if not selected.

**Example:**

```python
from utils import filter_fields

data = [
    {'filename': 'doc1.pdf', 'Plot No.': '123', 'Extra': 'value'},
    {'filename': 'doc2.pdf', 'Plot No.': '456', 'Extra': 'ignore'}
]

filtered = filter_fields(data, {'Plot No.'})
# Result: [
#     {'filename': 'doc1.pdf', 'Plot No.': '123'},
#     {'filename': 'doc2.pdf', 'Plot No.': '456'}
# ]
```

### `validate_pdf_file()`

Comprehensive PDF file validation including signature, structure, and content checks.

```python
def validate_pdf_file(file) -> Tuple[bool, str]
```

**Parameters:**

- `file`: Uploaded file object (Streamlit UploadedFile or file-like object)

**Returns:**

- `Tuple[bool, str]`: (is_valid, error_message)
  - `is_valid` (bool): True if PDF is valid, False otherwise
  - `error_message` (str): Empty string if valid, error description if invalid

**Validation Checks:**

1. File extension check (must be `.pdf`)
2. File size check (max 50 MB)
3. PDF signature check (must start with `%PDF`)
4. EOF marker check (must contain `%%EOF` in last 2KB)
5. Structural validation using PyPDF2:
   - Parses PDF structure
   - Verifies page count > 0
   - Catches parsing errors

**Example:**

```python
from utils import validate_pdf_file

is_valid, error = validate_pdf_file(uploaded_file)
if not is_valid:
    print(f"Invalid PDF: {error}")
```

### `get_default_fields()`

Get list of default extraction fields.

```python
def get_default_fields() -> List[str]
```

**Returns:**

- `List[str]`: List of default field names including:
  - `filename`
  - `Sr.No`
  - `Document No.& Year`
  - `Name of Executant(s)`
  - `Name of Claimant(s)`
  - `Survey No.`
  - `Plot No.`

---

## field_detector

Module for automatically detecting field names from PDF documents.

### `detect_fields_batch()`

Detect field names from multiple PDF files.

```python
def detect_fields_batch(
    pdf_files: List[str],
    method: str = 'ai',
    extraction_method: str = 'gemini',
    api_keys: Dict[str, str] = None,
    mode: str = 'unified'
) -> Dict[str, Any]
```

**Parameters:**

- `pdf_files` (List[str]): List of PDF file paths
- `method` (str): Detection method: `'ai'` or `'ocr'`
- `extraction_method` (str): Extraction method to use for detection
- `api_keys` (Dict[str, str]): Dictionary of API keys
- `mode` (str): Detection mode:
  - `'unified'`: Detect once from first file
  - `'per_file'`: Detect from each file and merge

**Returns:**

- `Dict[str, Any]`: Dictionary containing:
  - `'fields'`: List of detected field names
  - `'per_file'`: Dictionary mapping filenames to their detected fields (if mode='per_file')

---

## api_key_manager

Module for managing API keys securely.

### `save_api_key()`

Save API key to `.env` file.

```python
def save_api_key(provider: str, api_key: str) -> bool
```

**Parameters:**

- `provider` (str): Provider name (`'datalab'`, `'huggingface'`, `'gemini'`, `'deepseek'`)
- `api_key` (str): API key value

**Returns:**

- `bool`: True if successful, False otherwise

### `load_api_key()`

Load API key from `.env` file.

```python
def load_api_key(provider: str) -> Optional[str]
```

**Parameters:**

- `provider` (str): Provider name

**Returns:**

- `Optional[str]`: API key value or None if not found

### `save_all_api_keys()`

Save multiple API keys at once.

```python
def save_all_api_keys(keys: Dict[str, str]) -> bool
```

**Parameters:**

- `keys` (Dict[str, str]): Dictionary with provider names as keys and API keys as values

**Returns:**

- `bool`: True if all successful, False otherwise

---

## Error Handling

All functions follow consistent error handling patterns:

- **Validation Errors**: Functions validate inputs and raise `ValueError` for invalid parameters
- **File Errors**: File operations raise `FileNotFoundError` or `OSError` as appropriate
- **API Errors**: API-related functions catch and log errors, returning error messages where applicable
- **Extraction Errors**: Extraction functions may raise various exceptions depending on the method

---

## Best Practices

1. **Always validate PDFs**: Use `validate_pdf_file()` before processing
2. **Handle missing API keys**: Check for required keys before calling extraction functions
3. **Use progress callbacks**: For batch processing, provide progress callbacks for better UX
4. **Error handling**: Wrap extraction calls in try-except blocks
5. **Field filtering**: Use `filter_fields()` to reduce data to only needed fields

---

## See Also

- [README.md](README.md) - Project overview and quick start guide
- [CONTRIBUTING.md](CONTRIBUTING.md) - Contribution guidelines
- Module source code for additional implementation details

