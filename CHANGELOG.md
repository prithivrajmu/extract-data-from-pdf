# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

- _Nothing yet._

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

