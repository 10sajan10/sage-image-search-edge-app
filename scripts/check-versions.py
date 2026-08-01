#!/usr/bin/env python3
"""Verify that every declared version agrees with apps/*/sage.yaml.

`sage.yaml:version` is the single source of truth: Sage ECR builds that exact
tag, and every deployment artifact must reference the same one. Nothing
generates these references, so this check is what keeps them honest.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ROLE_IMAGES = {"ingest": "image-search-ingest", "search": "image-search-api"}


def sage_version(path: Path) -> str:
    match = re.search(r'^version:\s*"([^"]+)"', path.read_text(encoding="utf-8"), re.M)
    if not match:
        raise SystemExit(f"{path}: no version field")
    return match.group(1)


def check() -> list[str]:
    errors: list[str] = []
    versions = {
        role: sage_version(ROOT / "apps" / role / "sage.yaml")
        for role in ROLE_IMAGES
    }
    if len(set(versions.values())) != 1:
        errors.append(
            f"apps/*/sage.yaml versions disagree: {versions}. The two apps are "
            "released together against one collection contract."
        )
    expected = versions["ingest"]

    # The FastAPI app advertises its version on /docs and /openapi.json.
    api = (ROOT / "apps" / "search" / "api.py").read_text(encoding="utf-8")
    api_version = re.search(r'^\s*version="([^"]+)",', api, re.M)
    if not api_version:
        errors.append("apps/search/api.py: no FastAPI version= found")
    elif api_version.group(1) != versions["search"]:
        errors.append(
            f"apps/search/api.py declares {api_version.group(1)} but "
            f"apps/search/sage.yaml declares {versions['search']}"
        )

    # Any `name:tag` reference in a deploy artifact or script must be current.
    tracked = [
        *(ROOT / "deploy").rglob("*.yaml"),
        *(ROOT / "scripts").glob("*.sh"),
    ]
    for path in sorted(tracked):
        text = path.read_text(encoding="utf-8")
        # Docker tag charset only, so surrounding shell/sed syntax such as
        # `${VAR:-...:0.3.4}` or `s|...:0.3.4|$REF|` is not swallowed.
        for image, tag in re.findall(
            r"(image-search-ingest|image-search-api):([0-9][0-9A-Za-z._-]*)", text
        ):
            if tag != expected:
                errors.append(
                    f"{path.relative_to(ROOT)}: {image}:{tag} should be "
                    f"{image}:{expected}"
                )
    return errors


def main() -> int:
    errors = check()
    if errors:
        print("Version consistency check failed:")
        for error in errors:
            print(f"- {error}")
        print(
            "\nBump apps/*/sage.yaml, then run this check and update every "
            "reference it reports."
        )
        return 1
    print(f"All version references agree: {sage_version(ROOT / 'apps/ingest/sage.yaml')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
