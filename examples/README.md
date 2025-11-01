# Example CLI Scripts

This directory contains example command-line scripts for extracting data from EC (Encumbrance Certificate) PDF files using various OCR methods.

## Available Scripts

- **`extract_ec_data.py`** - Uses Chandra OCR (GPU/CPU automatic detection)
- **`extract_ec_data_cpu.py`** - Forces CPU-only mode for Chandra OCR
- **`extract_ec_data_pretty.py`** - Chandra OCR with formatted, user-friendly output
- **`extract_ec_data_api.py`** - Uses Datalab Marker API (no local models required)
- **`extract_ec_data_hf_api.py`** - Uses HuggingFace Inference Providers API
- **`extract_ec_data_easyocr.py`** - Fast CPU-only extraction using EasyOCR

## Usage

All scripts support the same command-line interface:

```bash
# Use default test file (test_file/RG EC 103 3.pdf)
python examples/extract_ec_data.py

# Use a custom PDF file
python examples/extract_ec_data.py --file path/to/your/file.pdf

# Alternative syntax
python examples/extract_ec_data.py --input path/to/your/file.pdf
```

## Default Behavior

By default, all scripts will look for `test_file/RG EC 103 3.pdf` relative to the project root. If you provide a custom file path, it can be:
- An absolute path
- A relative path from the current working directory
- A relative path from the project root

## Output

Each script will:
1. Extract data from the PDF using the specified OCR method
2. Display the extracted data in a formatted table
3. Save results to:
   - CSV file: `{input_filename}_extracted.csv`
   - Excel file: `{input_filename}_extracted.xlsx`

## Requirements

Each script has different requirements:

- **Chandra OCR scripts** (`extract_ec_data*.py` except `api.py` and `easyocr.py`):
  - Requires `chandra-ocr` package
  - First run downloads ~2GB models
  - GPU recommended for faster processing

- **API scripts** (`extract_ec_data_api.py`, `extract_ec_data_hf_api.py`):
  - Require API keys (see script help messages)
  - No local model downloads
  - Internet connection required

- **EasyOCR script** (`extract_ec_data_easyocr.py`):
  - Requires `easyocr` package
  - Fast CPU processing
  - No GPU required

## Getting Help

Run any script with `--help` to see usage information:

```bash
python examples/extract_ec_data.py --help
```

