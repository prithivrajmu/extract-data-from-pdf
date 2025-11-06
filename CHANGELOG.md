# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Field preset system for common document types
- Encumbrance Certificate preset (default)
- Preset selection in Streamlit UI
- Generic field detection (removed EC-specific bias)
- Support for customizable field extraction
- Additional presets: Invoice, Receipt, and more
- Unit tests for preset system
- Performance benchmarking utilities
- Capability detection for local OCR models (CPU/GPU, pretty output hints)
- Unit tests covering Chandra and transformer-based local extraction paths

### Changed

- Field detection now uses generic examples instead of EC-specific examples
- Extraction prompts now support presets and custom fields
- Default fields now come from preset system
- `get_default_fields()` now accepts `preset_name` parameter
- Streamlit sidebar exposes local model options based on detected capabilities
- Field detection automatically maps local model identifiers to supported OCR engines

### Migration Notes

- Existing code continues to work (defaults to "encumbrance" preset)
- Field detection is now more general and unbiased
- Users can continue using `get_default_fields()` without changes; preset-aware version available with parameter

## [1.1.0] - 2025-11-02

### Changed

- Migrated to `uv` package manager for faster dependency management and improved CI/CD performance
- Updated CI/CD pipeline to use `uv` for dependency installation and test execution
- Improved code formatting consistency across the codebase
- Enhanced type checking coverage with MyPy fixes

### Fixed

- Fixed code formatting issues identified by Black formatter
- Resolved Flake8 linting warnings and errors
- Fixed Ruff linting issues for improved code quality
- Corrected exception clause indentation for better code readability
- Fixed unit test failures and improved test stability
- Improved error handling and code structure across extraction modules

### Development

- Removed pre-commit hooks in favor of CI/CD-based linting
- Streamlined development workflow with `uv` integration
- Improved code quality checks in continuous integration

## [1.0.0] - 2025-11-02

### Added

- Streamlit web application for PDF data extraction
- Multiple extraction backends (EasyOCR, PyTesseract, Chandra OCR, Gemini, Deepseek, HuggingFace, Datalab)
- API key management utilities and validation
- Comprehensive PDF validation helpers
- Unit tests, fixtures, and continuous integration pipeline
- Developer tooling (Black, Ruff, MyPy, pytest-cov, pre-commit)
- Documentation suite (README, API reference, troubleshooting guides)

### Security

- Established security reporting policy (`.github/SECURITY.md`)

[1.1.0]: https://github.com/prithivrajmu/extract-data-from-pdf/releases/tag/v1.1.0
[Unreleased]: https://github.com/prithivrajmu/extract-data-from-pdf/compare/v1.1.0...HEAD
[1.0.0]: https://github.com/prithivrajmu/extract-data-from-pdf/releases/tag/v1.0.0

