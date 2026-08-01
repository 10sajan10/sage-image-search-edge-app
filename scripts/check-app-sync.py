#!/usr/bin/env python3
"""Verify that monorepo app sources match their standalone repositories."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
IGNORED_NAMES = {
    ".git",
    ".gitignore",
    ".pytest_cache",
    "README.md",
    "__pycache__",
    "sage.yaml",
}


def comparable_files(root: Path) -> dict[Path, Path]:
    return {
        path.relative_to(root): path
        for path in root.rglob("*")
        if path.is_file() and not any(part in IGNORED_NAMES for part in path.relative_to(root).parts)
    }


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(role: str, standalone: Path) -> list[str]:
    source = ROOT / "apps" / role
    if not standalone.is_dir():
        return [f"{role}: standalone repository not found: {standalone}"]

    source_files = comparable_files(source)
    standalone_files = comparable_files(standalone)
    errors: list[str] = []
    for relative in sorted(source_files.keys() | standalone_files.keys()):
        if relative not in source_files:
            errors.append(f"{role}: standalone-only file: {relative}")
        elif relative not in standalone_files:
            errors.append(f"{role}: missing standalone file: {relative}")
        elif digest(source_files[relative]) != digest(standalone_files[relative]):
            errors.append(f"{role}: content differs: {relative}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parent = ROOT.parent
    parser.add_argument(
        "--ingest-repo",
        type=Path,
        default=parent / "sage-image-search-ingest",
    )
    parser.add_argument(
        "--search-repo",
        type=Path,
        default=parent / "sage-image-search-search",
    )
    args = parser.parse_args()

    errors = compare("ingest", args.ingest_repo.resolve())
    errors.extend(compare("search", args.search_repo.resolve()))
    if errors:
        print("App synchronization check failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    print("App sources match both standalone repositories.")
    print("README.md, sage.yaml, and repository metadata are intentionally excluded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
