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
# sage.yaml cannot be compared byte-for-byte -- these three keys name the
# repository the image is built from and are legitimately different. Every
# other key, `version` above all, is the deployment contract and must match.
SAGE_REPO_SPECIFIC_KEYS = {"homepage", "url", "directory"}


def sage_yaml_contract(path: Path) -> list[str]:
    """Flatten sage.yaml to comparable lines, dropping repo-specific keys."""
    lines = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key = line.split(":", 1)[0].strip().lstrip("- ").strip('"')
        if key in SAGE_REPO_SPECIFIC_KEYS:
            continue
        lines.append(line)
    return lines


def compare_sage_yaml(role: str, standalone: Path) -> list[str]:
    source = ROOT / "apps" / role / "sage.yaml"
    target = standalone / "sage.yaml"
    if not target.is_file():
        return [f"{role}: standalone sage.yaml not found: {target}"]
    expected = sage_yaml_contract(source)
    actual = sage_yaml_contract(target)
    if expected == actual:
        return []
    return [
        f"{role}: sage.yaml contract differs (ignoring "
        f"{sorted(SAGE_REPO_SPECIFIC_KEYS)}):\n"
        + "\n".join(
            f"    - monorepo:   {line}"
            for line in expected
            if line not in actual
        )
        + "\n"
        + "\n".join(
            f"    - standalone: {line}"
            for line in actual
            if line not in expected
        )
    ]


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
    errors.extend(compare_sage_yaml(role, standalone))
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
    print(
        "README.md and repository metadata are intentionally excluded; "
        f"sage.yaml is compared except for {sorted(SAGE_REPO_SPECIFIC_KEYS)}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
