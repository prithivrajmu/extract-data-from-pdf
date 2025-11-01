# EC Data Extraction Script

This script extracts data from EC (Encumbrance Certificate) PDF files using OCR.

## Requirements

- Python 3.8+
- Chandra OCR (installed via pip/uv)

### Setup with uv (Recommended)

This project uses [uv](https://github.com/astral-sh/uv) for fast package management.

1. Install uv (if not already installed):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"  # Add to ~/.bashrc for persistence
```

2. Create virtual environment and install dependencies:
```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
```

### Alternative: Setup with pip

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```


## Usage

### Option 1: Using Local Model (CPU/GPU)

Make sure your virtual environment is activated:
```bash
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
python extract_ec_data_cpu.py  # CPU-only version
# or
python extract_ec_data.py     # Auto-detects GPU if available
```

### Option 2: Using Datalab API (No Local Model Required) - Recommended

Uses Datalab Marker API - no need to download models locally!

1. Get your free API key from [Datalab](https://datalab.to)
2. Create a `.env` file:
   ```bash
   cp .env.example .env
   # Edit .env and add: DATALAB_API_KEY=your_key_here
   ```
3. Run the API script:
   ```bash
   python extract_ec_data_api.py
   ```

**Benefits of Datalab API:**
- No local model download (~18GB saved)
- Faster setup
- Works on any machine
- Free tier available

**Documentation:** https://documentation.datalab.to/docs/recipes/marker/conversion-api-overview

### Option 3: Using HuggingFace Inference Providers API (Experimental)

Uses the new HuggingFace Inference Providers API endpoint.

⚠️ **Note:** The Chandra model may not be available on HuggingFace Inference Providers API.
If you encounter errors, use Option 2 (Datalab API) instead.

1. Get your free API token from [HuggingFace](https://huggingface.co/settings/tokens)
2. Create/update `.env` file:
   ```bash
   cp .env.example .env
   # Edit .env and add: HF_API_KEY=your_token_here
   ```
3. Run the HF API script:
   ```bash
   python extract_ec_data_hf_api.py
   ```

**Documentation:** https://huggingface.co/docs/inference-providers

### What the scripts do:

1. Process the PDF file `ec/RG EC 103 3.pdf`
2. Extract text using OCR (local model or API)
3. Parse table rows that contain Plot No./ information
4. Export results to CSV and Excel files

## Extracted Fields

- filename
- Sr.No
- Document No.& Year
- Name of Executant(s)
- Name of Claimant(s)
- Survey No./
- Plot No./

## Output Files

- `*_ocr_text.txt`: Raw OCR text for debugging
- `*_extracted.csv`: Extracted data in CSV format
- `*_extracted.xlsx`: Extracted data in Excel format

