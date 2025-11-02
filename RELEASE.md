# Release Process

This document describes the process for creating new releases of this project.

## Automated Release Process

### Prerequisites

1. Ensure you have push access to the repository
2. All tests pass (`pytest`)
3. CI pipeline is green
4. CHANGELOG.md has been updated with new features/bug fixes

### Step-by-Step Release

1. **Bump Version**
   ```bash
   python scripts/bump_version.py <new_version>
   ```
   
   Example:
   ```bash
   python scripts/bump_version.py 1.0.1
   ```

2. **Review Changes**
   ```bash
   git diff
   ```

3. **Commit Version Bump**
   ```bash
   git add pyproject.toml extraction_service.py CHANGELOG.md
   git commit -m "chore(release): v<new_version>"
   ```

4. **Create and Push Tag**
   ```bash
   git tag -a v<new_version> -m "Release v<new_version>"
   git push origin main
   git push origin v<new_version>
   ```

5. **GitHub Actions Will Automatically:**
   - Verify version matches in pyproject.toml
   - Build the package
   - Create GitHub Release with notes from CHANGELOG
   - Upload release artifacts

## Manual Release (Alternative)

If you prefer to create the release manually:

1. Follow steps 1-4 above
2. Go to [GitHub Releases](https://github.com/prithivrajmu/extract-data-from-pdf/releases)
3. Click "Draft a new release"
4. Select the tag you just pushed
5. Copy the relevant section from CHANGELOG.md into the release description
6. Click "Publish release"

## Version Numbering

This project follows [Semantic Versioning](https://semver.org/):
- **MAJOR** (1.0.0): Breaking changes
- **MINOR** (0.1.0): New features, backward compatible
- **PATCH** (0.0.1): Bug fixes, backward compatible

## Release Checklist

- [ ] Update CHANGELOG.md with all changes
- [ ] Ensure all tests pass
- [ ] Run `black .` to format code
- [ ] Run `ruff check .` and fix any issues
- [ ] Update version using `bump_version.py`
- [ ] Commit version bump
- [ ] Create and push tag
- [ ] Verify GitHub Release was created
- [ ] Test installation from PyPI (if published)

## Codecov Setup

To enable Codecov coverage badges:

1. **Sign up for Codecov:**
   - Go to [codecov.io](https://codecov.io)
   - Sign in with your GitHub account
   - Authorize access to your repository

2. **Get Codecov Token (Optional):**
   - The workflow uses `GITHUB_TOKEN` by default, which should work for public repos
   - For private repos, you may need to add `CODECOV_TOKEN` secret in GitHub Settings → Secrets

3. **Verify Badge:**
   - The badge in README will automatically update once coverage data is uploaded
   - Badge URL: `https://codecov.io/gh/prithivrajmu/extract-data-from-pdf`

4. **View Coverage:**
   - Go to [codecov.io/gh/prithivrajmu/extract-data-from-pdf](https://codecov.io/gh/prithivrajmu/extract-data-from-pdf)
   - Coverage reports are uploaded after each CI run

## Post-Release

After a release:

1. Update any documentation that references version numbers
2. Announce the release (if applicable)
3. Monitor for issues reported by users
4. Verify Codecov badge is showing coverage data

## Troubleshooting

**Version mismatch error:**
- Ensure `pyproject.toml` and `extraction_service.py` both have the same version
- The tag should match the version exactly (e.g., tag `v1.0.1` = version `1.0.1`)

**Release not created:**
- Check GitHub Actions workflow logs
- Ensure `GITHUB_TOKEN` has write permissions
- Verify the tag was pushed: `git ls-remote --tags origin`

**Codecov badge not showing:**
- Wait for CI to complete (coverage uploads only on Python 3.11 job)
- Check if Codecov has access to your repository
- Verify coverage.xml is being generated in CI logs
- The badge may take a few minutes to update after first upload

