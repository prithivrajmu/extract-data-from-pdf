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

