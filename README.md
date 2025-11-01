# PDF Data Extraction

Extract structured data from PDF documents using multiple OCR and AI methods. Available as both a **Streamlit web application** and command-line scripts.

## 🚀 Quick Start - Streamlit Web App (Recommended)

The easiest way to use this tool is through the Streamlit web interface.

### Prerequisites

- Python 3.8+
- All dependencies from `requirements.txt`

### Installation

1. **Clone or navigate to the project directory:**
   ```bash
   cd extract_tn_ec
   ```

2. **Set up virtual environment (using uv - recommended):**
   ```bash
   # Install uv if not already installed
   curl -LsSf https://astral.sh/uv/install.sh | sh
   export PATH="$HOME/.local/bin:$PATH"
   
   # Create virtual environment
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   
   # Install dependencies
   uv pip install -r requirements.txt
   ```

   **Or using pip:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app:**
   ```bash
   streamlit run streamlit_app.py
   ```

4. **Access the app:**
   - The app will automatically open in your default browser
   - Default URL: `http://localhost:8501`
   - If it doesn't open automatically, copy the URL shown in the terminal

### Streamlit App Features

✨ **User-Friendly Interface:**
- Clean, modern UI with intuitive controls
- Real-time progress tracking during extraction
- Interactive results display

📄 **Multiple Extraction Methods:**
- **EasyOCR** - Fast local OCR (no API key needed)
- **HuggingFace** - Cloud-based OCR with model selection
- **Datalab API** - High-accuracy OCR via API (recommended)
- **Gemini AI** - Google's AI-powered extraction
- **Deepseek AI** - AI-powered extraction with vision

🔑 **API Key Management:**
- Test API keys before use
- Secure local storage in `.env` file
- Support for multiple providers (HuggingFace, Datalab, Gemini, Deepseek)

📋 **Flexible Field Selection:**
- Pre-defined field templates
- Custom field extraction (define your own fields)
- Selective field export

💾 **Multiple Output Formats:**
- CSV
- Excel (XLSX)
- JSON
- Markdown (MD)
- Select multiple formats at once

📤 **Batch Processing:**
- Upload single or multiple PDF files
- Process files sequentially with progress tracking
- Results aggregated across all files

### Using the Streamlit App

1. **Configure Extraction Method:**
   - Select from: EasyOCR, HuggingFace, Datalab API, Gemini AI, or Deepseek AI
   - For HuggingFace: Enter model name (default: `datalab-to/chandra`)
   - For API methods: Enter and test your API keys

2. **Set Up API Keys (if using API methods):**
   - Enter API keys in the sidebar
   - Click "Test" buttons to verify keys work
   - Click "💾 Save All Keys Locally" to persist keys
   - ⚠️ **Important:** Make a backup copy of the `.env` file

3. **Select Fields to Extract:**
   - Choose from default fields, or
   - Click "➕ Add Custom Fields" to define your own fields
   - Custom fields work best with Gemini AI and Deepseek AI

4. **Choose Output Format(s):**
   - Select one or more formats: CSV, Excel, JSON, Markdown
   - ⚠️ **Required:** At least one format must be selected

5. **Upload PDF Files:**
   - Drag and drop or click to upload
   - Supports single or multiple files
   - File validation ensures PDFs are valid

6. **Extract Data:**
   - Click "🚀 Extract Data" button
   - Watch real-time progress with detailed status updates
   - View metrics: Files Processed, Rows Extracted, Success Rate

7. **Download Results:**
   - View extracted data in interactive table
   - Download in your selected format(s)
   - Each format has its own download button

### Getting API Keys

**Datalab API (Recommended for OCR):**
- Sign up at [https://datalab.to](https://datalab.to)
- Get your API key from the dashboard
- Free tier available

**HuggingFace:**
- Sign up at [https://huggingface.co](https://huggingface.co)
- Go to Settings → Access Tokens
- Create a new token with read permissions

**Gemini AI:**
- Get API key from [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)
- Create a new API key

**Deepseek AI:**
- Sign up at [https://platform.deepseek.com](https://platform.deepseek.com)
- Get your API key from the dashboard

---

## 📟 Command-Line Usage

For script-based automation, you can use the CLI scripts directly.

### Option 1: Using Local Model (CPU/GPU)

Make sure your virtual environment is activated:
```bash
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
python extract_ec_data_cpu.py  # CPU-only version
# or
python extract_ec_data.py     # Auto-detects GPU if available
```

### Option 2: Using Datalab API (No Local Model Required)

Uses Datalab Marker API - no need to download models locally!

1. Get your free API key from [Datalab](https://datalab.to)
2. Create a `.env` file:
   ```bash
   echo "DATALAB_API_KEY=your_key_here" > .env
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
   echo "HF_API_KEY=your_token_here" >> .env
   ```
3. Run the HF API script:
   ```bash
   python extract_ec_data_hf_api.py
   ```

**Documentation:** https://huggingface.co/docs/inference-providers

### Option 4: Using Gemini AI

1. Get your API key from [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Create/update `.env` file:
   ```bash
   echo "GEMINI_API_KEY=your_key_here" >> .env
   ```
3. Run the Gemini script:
   ```bash
   python extract_ec_data_gemini.py
   ```

### What the CLI scripts do:

1. Process PDF file(s) from specified directories or files
2. Extract text using OCR (local model or API)
3. Parse table rows that contain relevant information
4. Export results to CSV and Excel files

---

## 📋 Default Extracted Fields

The following fields are extracted by default:

- `filename` - Source PDF filename
- `Sr.No` - Serial number
- `Document No.& Year` - Document number and year
- `Name of Executant(s)` - Name(s) of executant(s)
- `Name of Claimant(s)` - Name(s) of claimant(s)
- `Survey No.` - Survey number
- `Plot No.` - Plot number (required field)

**Note:** You can define custom fields in the Streamlit app when using AI-based extraction methods (Gemini, Deepseek).

---

## 📁 Output Files

**CLI Scripts:**
- `*_ocr_text.txt`: Raw OCR text for debugging
- `*_extracted.csv`: Extracted data in CSV format
- `*_extracted.xlsx`: Extracted data in Excel format

**Streamlit App:**
- Download in selected format(s): CSV, Excel, JSON, or Markdown
- Results shown in interactive table before download

---

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to contribute to this project.

---

## 🔧 Requirements

See `requirements.txt` for complete dependency list. Key dependencies:

- `streamlit>=1.28.0` - Web application framework
- `pandas>=2.0.0` - Data manipulation
- `openpyxl>=3.1.0` - Excel file support
- `easyocr>=1.7.0` - EasyOCR library
- `pdf2image>=1.16.3` - PDF to image conversion
- `google-generativeai>=0.3.0` - Gemini AI support
- `requests>=2.31.0` - API requests
- `python-dotenv>=1.0.0` - Environment variable management

---

## 🛠️ Troubleshooting

### Streamlit App Issues

**App won't start:**
```bash
# Make sure dependencies are installed
pip install -r requirements.txt

# Check if port 8501 is in use
# Change port if needed:
streamlit run streamlit_app.py --server.port 8502
```

**API key not working:**
- Use the "Test" buttons in the sidebar to verify keys
- Check that keys are saved in `.env` file
- Ensure `.env` file is in the project root directory

**No results extracted:**
- Try a different extraction method
- For AI methods (Gemini/Deepseek), ensure API keys are valid
- Check PDF file quality - some PDFs may require better OCR

### CLI Script Issues

See `TROUBLESHOOTING.md` and `MODEL_LOADING_ISSUES.md` for detailed troubleshooting guides.

---

## 📝 License

See `LICENSE` file for details.

---

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines on how to contribute to this project.

---

## 📚 Additional Documentation

- `GPU_SETUP.md` - GPU configuration guide
- `MODEL_LOADING_ISSUES.md` - Troubleshooting model loading
- `TROUBLESHOOTING.md` - General troubleshooting guide
