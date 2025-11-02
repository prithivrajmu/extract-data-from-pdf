# Development Guide

This guide covers development setup, code quality tools, and contribution workflows.

## 🛠️ Development Setup

### Prerequisites

- Python 3.12+
- Git
- Virtual environment (venv or uv)

### Initial Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd extract_tn_ec
   ```

2. **Create virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install pre-commit hooks (optional but recommended):**
   ```bash
   pre-commit install
   ```

## 🔧 Code Quality Tools

### Black (Code Formatter)

**Black** automatically formats your code to ensure consistent style.

```bash
# Check formatting
black --check .

# Format all files
black .

# Format specific file
black path/to/file.py
```

**Configuration:** See `pyproject.toml` under `[tool.black]`

### Ruff (Fast Linter)

**Ruff** is a modern, fast Python linter that replaces multiple tools.

```bash
# Check for issues
ruff check .

# Auto-fix issues
ruff check --fix .

# Check specific file
ruff check path/to/file.py
```

**Configuration:** See `pyproject.toml` under `[tool.ruff]`

### Flake8 (Linter)

**Flake8** checks code quality and style.

```bash
# Run flake8
flake8

# Check specific file
flake8 path/to/file.py
```

**Configuration:** See `.flake8`

### MyPy (Type Checker)

**MyPy** performs static type checking to catch type errors before runtime.

```bash
# Type check all files
mypy . --ignore-missing-imports

# Type check specific file
mypy path/to/file.py
```

**Configuration:** See `pyproject.toml` under `[tool.mypy]`

## 🧪 Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/unit/test_utils.py

# Run specific test
pytest tests/unit/test_utils.py::test_filter_fields_keeps_selected_and_filename

# Run with coverage report
pytest --cov=. --cov-report=html --cov-report=term
```

### Coverage Reports

After running tests with coverage, view HTML report:
```bash
pytest --cov=. --cov-report=html
open htmlcov/index.html  # On macOS
# Or open htmlcov/index.html in your browser
```

## 📋 Pre-commit Hooks

Pre-commit hooks automatically run code quality checks before each commit.

### Setup

```bash
pre-commit install
```

### Manual Run

```bash
# Run on all files
pre-commit run --all-files

# Run on staged files only (default)
pre-commit run
```

### What Gets Checked

- Trailing whitespace
- End of file fixes
- YAML/JSON/TOML validation
- Large file detection
- Merge conflict markers
- Black formatting
- Ruff linting
- MyPy type checking
- Pytest tests

## 🏗️ Project Structure

```
extract_tn_ec/
├── streamlit_app.py          # Main Streamlit application
├── extraction_service.py     # Core extraction logic
├── utils.py                  # Utility functions
├── field_detector.py         # Field detection module
├── api_key_manager.py        # API key management
├── streamlit_ui/             # Streamlit UI modules
│   ├── sidebar.py
│   ├── results.py
│   └── state.py
├── tests/                    # Test suite
│   ├── unit/
│   │   ├── test_utils.py
│   │   ├── test_extraction_service.py
│   │   └── test_field_detector.py
│   └── fixtures/
│       └── pdf_samples.py
├── examples/                 # Example scripts
├── tools/                    # Utility tools
├── .github/workflows/        # CI/CD workflows
├── pyproject.toml           # Tool configurations
├── .pre-commit-config.yaml  # Pre-commit hooks
└── requirements.txt         # Dependencies
```

## 📝 Code Style Guidelines

### General Rules

1. **Line Length:** Maximum 88 characters (Black default)
2. **Import Order:** 
   - Standard library
   - Third-party packages
   - Local application imports
3. **Type Hints:** Use type hints for all function signatures
4. **Docstrings:** Use Google-style docstrings
5. **Naming:**
   - Functions: `snake_case`
   - Classes: `PascalCase`
   - Constants: `UPPER_SNAKE_CASE`

### Example

```python
from typing import List, Dict, Optional

def process_data(
    items: List[Dict[str, str]], 
    filter_value: Optional[str] = None
) -> List[Dict[str, str]]:
    """
    Process a list of data items.
    
    Args:
        items: List of data dictionaries to process
        filter_value: Optional value to filter items by
        
    Returns:
        Processed list of data dictionaries
    """
    # Implementation
    pass
```

## 🚀 Continuous Integration

The CI pipeline (`.github/workflows/ci.yml`) automatically:

1. Checks code formatting with Black
2. Runs linting with flake8
3. Runs all unit tests
4. Generates coverage reports
5. Uploads coverage to Codecov (if configured)

All checks must pass for pull requests to be merged.

## 🐛 Debugging

### Running with Verbose Output

```bash
# Streamlit app with debug
streamlit run streamlit_app.py --logger.level=debug

# Tests with verbose output
pytest -vv

# Tests with print statements visible
pytest -s
```

### Common Issues

**Import Errors:**
- Ensure virtual environment is activated
- Run `pip install -r requirements.txt`

**Type Checking Errors:**
- MyPy may show false positives for third-party libraries
- Use `# type: ignore` comments sparingly for known false positives

**Formatting Issues:**
- Run `black .` to auto-format
- Pre-commit hooks will format automatically

## 📚 Additional Resources

- [Black Documentation](https://black.readthedocs.io/)
- [Ruff Documentation](https://docs.astral.sh/ruff/)
- [MyPy Documentation](https://mypy.readthedocs.io/)
- [Pytest Documentation](https://docs.pytest.org/)
- [Pre-commit Documentation](https://pre-commit.com/)

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed contribution guidelines.

Before submitting a PR:
1. ✅ Run `black .` to format code
2. ✅ Run `flake8` to check linting
3. ✅ Run `pytest` to ensure all tests pass
4. ✅ Update documentation if needed

**Note:** Ruff and MyPy can be run manually for additional checks but are not required for CI.

