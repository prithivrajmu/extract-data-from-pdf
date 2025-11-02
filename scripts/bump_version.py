#!/usr/bin/env python3
"""
Version bumping utility for releases.

Usage:
    python scripts/bump_version.py <new_version>

Example:
    python scripts/bump_version.py 1.0.1
"""

import re
import sys
from pathlib import Path
from datetime import date

PROJECT_ROOT = Path(__file__).parent.parent


def update_pyproject_toml(new_version: str) -> None:
    """Update version in pyproject.toml."""
    pyproject_path = PROJECT_ROOT / "pyproject.toml"
    content = pyproject_path.read_text()
    
    # Replace version in [project] section
    content = re.sub(
        r'^version = "[^"]*"',
        f'version = "{new_version}"',
        content,
        flags=re.MULTILINE,
    )
    
    pyproject_path.write_text(content)
    print(f"✅ Updated pyproject.toml to version {new_version}")


def update_extraction_service(new_version: str) -> None:
    """Update __version__ in extraction_service.py."""
    service_path = PROJECT_ROOT / "extraction_service.py"
    content = service_path.read_text()
    
    # Replace __version__
    content = re.sub(
        r'__version__ = "[^"]*"',
        f'__version__ = "{new_version}"',
        content,
    )
    
    service_path.write_text(content)
    print(f"✅ Updated extraction_service.py to version {new_version}")


def update_changelog(new_version: str, release_date: str = None) -> None:
    """Add new version section to CHANGELOG.md."""
    if release_date is None:
        release_date = date.today().isoformat()
    
    changelog_path = PROJECT_ROOT / "CHANGELOG.md"
    content = changelog_path.read_text()
    
    # Find the first [Unreleased] section and add new version after it
    unreleased_pattern = r'## \[Unreleased\]\n\n([^\n]+)\n'
    
    # Get the current version from existing changelog
    version_match = re.search(r'## \[(\d+\.\d+\.\d+)\]', content)
    current_version = version_match.group(1) if version_match else "1.0.0"
    
    # Create new version entry
    new_section = f"""## [Unreleased]

- _Nothing yet._

## [{new_version}] - {release_date}

### Added

- _Add new features here_

### Changed

- _Add changes here_

### Fixed

- _Add bug fixes here_

### Security

- _Add security fixes here_

"""
    
    # Replace Unreleased section
    content = re.sub(
        r'## \[Unreleased\]\n\n[^\n]+\n\n',
        new_section,
        content,
        count=1,
    )
    
    # Update version links at bottom
    if f"[{new_version}]:" not in content:
        # Add new version link
        repo_url = "https://github.com/prithivrajmu/extract_tn_ec"
        version_link = f"[{new_version}]: {repo_url}/releases/tag/v{new_version}\n"
        
        # Insert before [Unreleased] link
        unreleased_link_pattern = r'(\[Unreleased\]: [^\n]+\n)'
        content = re.sub(
            unreleased_link_pattern,
            f"{version_link}\\1",
            content,
        )
        
        # Update Unreleased link to compare from new version
        content = re.sub(
            r'\[Unreleased\]: [^\n]+',
            f"[Unreleased]: {repo_url}/compare/v{new_version}...HEAD",
            content,
        )
    
    changelog_path.write_text(content)
    print(f"✅ Updated CHANGELOG.md with version {new_version}")


def main():
    """Main version bumping function."""
    if len(sys.argv) < 2:
        print("Usage: python scripts/bump_version.py <new_version>")
        print("Example: python scripts/bump_version.py 1.0.1")
        sys.exit(1)
    
    new_version = sys.argv[1]
    
    # Validate version format (semver)
    if not re.match(r'^\d+\.\d+\.\d+$', new_version):
        print(f"❌ Error: Invalid version format '{new_version}'. Use semantic versioning (e.g., 1.0.1)")
        sys.exit(1)
    
    print(f"🚀 Bumping version to {new_version}...\n")
    
    # Get release date
    release_date = sys.argv[2] if len(sys.argv) > 2 else None
    
    try:
        update_pyproject_toml(new_version)
        update_extraction_service(new_version)
        update_changelog(new_version, release_date)
        
        print(f"\n✅ Version bumped to {new_version} successfully!")
        print("\nNext steps:")
        print(f"  1. Review changes: git diff")
        print(f"  2. Commit: git commit -am 'chore(release): v{new_version}'")
        print(f"  3. Tag: git tag -a v{new_version} -m 'Release v{new_version}'")
        print(f"  4. Push: git push && git push --tags")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

