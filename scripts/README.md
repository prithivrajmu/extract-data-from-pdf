# Scripts Directory

Utility scripts for project maintenance and releases.

## Available Scripts

### `bump_version.py`

Automated version bumping utility for releases.

**Usage:**
```bash
python scripts/bump_version.py <new_version>
```

**Example:**
```bash
python scripts/bump_version.py 1.0.1
```

**What it does:**
1. Updates `version` in `pyproject.toml`
2. Updates `__version__` in `extraction_service.py`
3. Adds a new section to `CHANGELOG.md` for the new version
4. Updates version comparison links in CHANGELOG

**Requirements:**
- Python 3.8+
- Valid semantic version format (e.g., `1.0.1`)

**Optional Arguments:**
- Custom release date: `python scripts/bump_version.py 1.0.1 2024-11-15`

