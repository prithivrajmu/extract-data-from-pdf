# Files to Remove Analysis

Based on REPOSITORY_ANALYSIS.md and verification of Streamlit app functionality.

## Files to Remove

### 1. Redundant Gemini Test Scripts ✅ VERIFIED

These standalone test scripts have their functionality fully integrated into the Streamlit app via `gemini_test_modules.py`:

- **`test_gemini_2.5_flash.py`**
  - Functionality: Tests Gemini API basic functionality (model creation, text generation, JSON output)
  - Streamlit integration: `gemini_test_modules.test_gemini_basic()` used in streamlit_app.py (lines 715, 798)
  - Status: ✅ Can be removed - functionality available in UI under "Advanced Testing > Test 1: Basic"

- **`test_gemini_file_upload.py`**
  - Functionality: Tests Gemini API with PDF file upload
  - Streamlit integration: `gemini_test_modules.test_gemini_file_upload()` used in streamlit_app.py (lines 740, 803)
  - Status: ✅ Can be removed - functionality available in UI under "Advanced Testing > Test 2: File Upload"

- **`test_gemini_json_upload.py`**
  - Functionality: Tests Gemini API with PDF upload and JSON output format
  - Streamlit integration: `gemini_test_modules.test_gemini_json_upload()` used in streamlit_app.py (lines 767, 808)
  - Status: ✅ Can be removed - functionality available in UI under "Advanced Testing > Test 3: JSON Upload"

### 2. Redundant GPU Check Script ✅ VERIFIED

- **`check_gpu.py`**
  - Functionality: Standalone CLI script to check GPU availability
  - Streamlit integration: `gpu_check_utils.check_gpu_comprehensive()` used in streamlit_app.py (line 596)
  - Status: ✅ Can be removed - functionality available in UI under "GPU Status > Check GPU"

### 3. Log Files

- **`ec2_extraction.log`** (if it exists)
  - Status: ❌ Already removed or doesn't exist (verified via find command)
  - Note: This was explicitly mentioned in REPOSITORY_ANALYSIS.md as needing deletion

## Files to Keep (Functionality Not in Streamlit or Still Needed)

### CLI Utility Scripts (Not in Streamlit)

- **`list_gemini_models.py`**
  - Purpose: Standalone CLI utility to list available Gemini models
  - Status: ✅ Keep - Simple utility useful for command-line users
  - Note: Not integrated into Streamlit, but useful for CLI debugging

### Diagnostic Utilities

- **`diagnose_chandra.py`**
  - Purpose: Diagnostic tool for troubleshooting Chandra OCR model issues
  - Status: ✅ Keep - Useful for troubleshooting model download/loading issues
  - Note: Not integrated into Streamlit, but valuable for debugging

### Core Extraction Scripts (Used by Streamlit)

All `extract_ec_data*.py` files are **KEPT** because they are imported by `extraction_service.py`, which is used by the Streamlit app:
- `extract_ec_data.py` - Main extraction (used via extraction_service)
- `extract_ec_data_cpu.py` - CPU-only extraction (used via extraction_service)
- `extract_ec_data_pretty.py` - Formatted output (used via extraction_service)
- `extract_ec_data_easyocr.py` - EasyOCR extraction (used via extraction_service)
- `extract_ec_data_pytesseract.py` - PyTesseract extraction (used via extraction_service)
- `extract_ec_data_api.py` - Datalab API extraction (used via extraction_service)
- `extract_ec_data_hf_api.py` - HuggingFace API extraction (used via extraction_service)
- `extract_ec_data_gemini.py` - Gemini extraction (used via extraction_service)

### Project-Specific Utilities

- **`merge_combine_outputs.py`**
  - Purpose: Utility to merge outputs from multiple directories
  - Status: ✅ Keep - Project-specific utility (mentioned in REPOSITORY_ANALYSIS.md as acceptable)

## Summary

**Files to Remove:**
1. `test_gemini_2.5_flash.py` ✅
2. `test_gemini_file_upload.py` ✅
3. `test_gemini_json_upload.py` ✅
4. `check_gpu.py` ✅

**Total: 4 files**

All removed files have verified functionality available in the Streamlit web app.

