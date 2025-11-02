# Quick Start: Creating Your First Release (v1.0.0)

This is a step-by-step guide to create your first GitHub release.

## Prerequisites

- You have push access to the repository
- All tests pass (`pytest`)
- CI pipeline is green

## Steps

### 1. Review Current State

Current version is already set to `1.0.0` in:
- `pyproject.toml`
- `extraction_service.py`
- `CHANGELOG.md` has a v1.0.0 entry

### 2. Commit Any Uncommitted Changes

```bash
git status
git add .
git commit -m "chore: prepare for v1.0.0 release"
```

### 3. Create and Push the Tag

```bash
git tag -a v1.0.0 -m "Release v1.0.0 - Initial release"
git push origin main
git push origin v1.0.0
```

### 4. Verify Release Created

1. Go to: https://github.com/prithivrajmu/extract_tn_ec/releases
2. You should see "Release 1.0.0" created automatically by GitHub Actions
3. The release notes will be populated from CHANGELOG.md

## For Future Releases (v1.0.1, v1.0.2, etc.)

Use the automated version bumping script:

```bash
# Bump to next version
python scripts/bump_version.py 1.0.1

# Review changes
git diff

# Commit
git add pyproject.toml extraction_service.py CHANGELOG.md
git commit -m "chore(release): v1.0.1"

# Tag and push
git tag -a v1.0.1 -m "Release v1.0.1"
git push origin main
git push origin v1.0.1
```

GitHub Actions will automatically create the release!

## Codecov Setup (One-time)

To enable the Codecov badge:

1. Visit: https://codecov.io
2. Sign in with GitHub
3. Add repository: `prithivrajmu/extract_tn_ec`
4. Wait for next CI run - badge will update automatically

**Note:** For public repositories, no token is needed. The badge will work with `GITHUB_TOKEN` automatically.

## Need Help?

See [RELEASE.md](RELEASE.md) for detailed documentation.

