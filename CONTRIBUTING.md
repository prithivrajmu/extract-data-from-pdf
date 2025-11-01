# Contributing to PDF Data Extraction

Thank you for your interest in contributing to this project! This document provides guidelines and instructions for contributing.

## 🤝 How to Contribute

### Reporting Issues

If you find a bug or have a feature request:

1. Check if the issue already exists in the repository
2. Create a new issue with:
   - Clear title and description
   - Steps to reproduce (for bugs)
   - Expected vs actual behavior
   - Environment details (OS, Python version, etc.)

### Submitting Pull Requests

1. **Fork the repository** and clone your fork
2. **Create a branch** for your feature/fix:
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/your-bug-fix
   ```
3. **Make your changes** following the coding standards below
4. **Test your changes** thoroughly
5. **Commit with clear messages**:
   ```bash
   git commit -m "Add feature: description of what you added"
   ```
6. **Push to your fork** and create a Pull Request
7. **Wait for review** and address any feedback

## 📝 Coding Standards

### Python Style

- Follow PEP 8 style guide
- Use meaningful variable and function names
- Add docstrings to functions and classes
- Keep functions focused and single-purpose
- Maximum line length: 88-100 characters

### File Naming Conventions

All Python files must follow consistent naming patterns based on their purpose:

#### Example/CLI Scripts
- **Pattern**: `extract_ec_data_<method>.py`
- **Location**: `examples/` directory (required)
- **Examples**: 
  - `extract_ec_data.py` - Default/main extraction script
  - `extract_ec_data_cpu.py` - CPU-only variant
  - `extract_ec_data_api.py` - API-based extraction
  - `extract_ec_data_easyocr.py` - EasyOCR method
  - `extract_ec_data_gemini.py` - Gemini AI method
  - `extract_ec_data_pytesseract.py` - PyTesseract method
- **Requirements**:
  - Must be located in `examples/` directory
  - Must support CLI arguments via `argparse`
  - Default to `test_file/RG EC 103 3.pdf`
  - Accept `--file` or `--input` for custom paths
  - Use path resolution to find files relative to project root

#### Service Modules
- **Pattern**: `*_service.py`
- **Purpose**: Core business logic and service orchestration
- **Examples**: `extraction_service.py`

#### Utility Modules
- **Pattern**: `*_utils.py` or `utils.py`
- **Purpose**: Reusable helper functions and utilities
- **Examples**: 
  - `utils.py` - General utilities
  - `gpu_check_utils.py` - GPU-specific utilities
  - `prompt_utils.py` - Prompt generation utilities

#### Manager Modules
- **Pattern**: `*_manager.py`
- **Purpose**: Manage resources, configurations, or state
- **Examples**: `api_key_manager.py`

#### Formatter Modules
- **Pattern**: `*_formatter.py`
- **Purpose**: Format output or display data
- **Location**: Root directory (if used by core modules) or `tools/` (if standalone utility)
- **Examples**: `chandra_output_formatter.py` (standalone utility - can be moved to tools/)

#### API Client Modules
- **Pattern**: `*_api.py`
- **Purpose**: External API integrations (not extraction scripts)
- **Examples**: `deepseek_api.py`
- **Note**: Extraction scripts using APIs go in `examples/` directory

#### Test Modules
- **Pattern**: `test_*.py`
- **Purpose**: Testing utilities and test scripts
- **Examples**: `test_api_keys.py`

#### Module Collections
- **Pattern**: `*_modules.py`
- **Purpose**: Collections of related modules or test modules
- **Examples**: `gemini_test_modules.py`

#### Model-Related Modules
- **Pattern**: `model_*.py`
- **Purpose**: Model management and information
- **Examples**:
  - `model_info.py` - Model information
  - `model_loaders.py` - Model loading utilities
  - `model_fetcher.py` - Model downloading/fetching

#### Diagnostic Modules
- **Pattern**: `diagnose_*.py`
- **Purpose**: Diagnostic and debugging tools
- **Location**: `tools/` directory for standalone diagnostic utilities
- **Examples**: `tools/diagnose_chandra.py`

#### List/Listing Modules
- **Pattern**: `list_*.py`
- **Purpose**: List resources or information
- **Examples**: `list_gemini_models.py`

#### Monitor Modules
- **Pattern**: `monitor_*.py`
- **Purpose**: Monitoring and progress tracking
- **Examples**: `monitor_download.py`

#### Merge/Combine Modules
- **Pattern**: `merge_*.py` or `combine_*.py`
- **Purpose**: Merging or combining data/files
- **Examples**: `merge_combine_outputs.py`

#### Main Application
- **Pattern**: `streamlit_app.py`, `app.py`, or `main.py`
- **Purpose**: Main entry point for applications
- **Examples**: `streamlit_app.py`

#### General Rules

1. **Use lowercase with underscores**: `my_module.py` ✅, not `MyModule.py` ❌
2. **Be descriptive**: File names should clearly indicate their purpose
3. **Avoid abbreviations**: Use full words unless they're widely understood (e.g., `api`, `utils`)
4. **Group related functionality**: Use suffixes to indicate module type (`_service`, `_utils`, etc.)
5. **CLI scripts in examples/**: All runnable example scripts belong in `examples/` directory
6. **Single purpose**: Each file should have a clear, single responsibility

### Code Structure

- Place related functionality in appropriate modules
- Keep imports organized (standard library, third-party, local)
- Add type hints where possible
- Include error handling for user-facing functions

### Example:

```python
def extract_data_from_pdf(pdf_path: str, method: str = 'easyocr') -> List[Dict[str, str]]:
    """
    Extract structured data from a PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        method: Extraction method to use
        
    Returns:
        List of dictionaries containing extracted data
        
    Raises:
        FileNotFoundError: If PDF file doesn't exist
        ValueError: If method is not supported
    """
    # Implementation
    pass
```

## 🧪 Testing

- Test your changes with different PDF files
- Test error cases (invalid files, missing API keys, etc.)
- Ensure existing functionality still works
- For Streamlit changes, test in the UI

### Test Files

- Use files from `test_file/` directory for testing
- Don't commit large test PDFs (>5MB) directly
- Use sample PDFs or provide links to test files

## 📚 Documentation

- Update README.md if you add new features
- Add docstrings to new functions/classes
- Update relevant documentation sections
- Keep comments clear and concise

## 🎯 Areas for Contribution

We welcome contributions in these areas:

### Features
- New OCR extraction methods
- Additional output formats
- Enhanced error handling
- Performance optimizations
- UI/UX improvements

### Bug Fixes
- Fixing extraction errors
- Resolving API integration issues
- Improving error messages
- Handling edge cases

### Documentation
- Improving README clarity
- Adding usage examples
- Creating tutorials
- Translating documentation

### Testing
- Adding unit tests
- Improving test coverage
- Creating test utilities

## 🔍 Code Review Process

1. All PRs will be reviewed by maintainers
2. Reviewers may request changes
3. Address feedback promptly
4. Be open to suggestions and discussion
5. Maintain a respectful and collaborative tone

## 📋 Commit Message Guidelines

Use clear, descriptive commit messages:

- **Good**: `Add support for JSON output format`
- **Good**: `Fix API key validation error handling`
- **Avoid**: `Fixed bug`
- **Avoid**: `Updates`

### Commit Types

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `style:` Code style changes (formatting)
- `refactor:` Code refactoring
- `test:` Adding or updating tests
- `chore:` Maintenance tasks

Example:
```
feat: Add Deepseek AI extraction method
fix: Handle empty PDF pages gracefully
docs: Update README with new API key instructions
```

## 🛠️ Development Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/extract_tn_ec.git
   cd extract_tn_ec
   ```

2. **Set up virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit app**:
   ```bash
   streamlit run streamlit_app.py
   ```

## ❓ Questions?

If you have questions or need clarification:

1. Check existing issues and discussions
2. Create a new issue with the `question` label
3. Reach out to maintainers

## 📄 License

By contributing, you agree that your contributions will be licensed under the same license as the project.

---

Thank you for contributing! 🎉

