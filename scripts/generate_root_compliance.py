#!/usr/bin/env python3
"""
Generate (or refresh) the project-root compliance_matrix.yaml from the
canonical source at fedpatterns/privacy/compliance/compliance_matrix.yaml.

Usage:
    python scripts/generate_root_compliance.py
"""

import pathlib
import shutil
import yaml

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
CANONICAL = PROJECT_ROOT / "fedpatterns" / "privacy" / "compliance" / "compliance_matrix.yaml"
ROOT_COPY = PROJECT_ROOT / "compliance_matrix.yaml"


def main() -> None:
    if not CANONICAL.exists():
        raise FileNotFoundError(f"Canonical source not found: {CANONICAL}")

    # Validate YAML and pattern count before copying
    with CANONICAL.open() as f:
        data = yaml.safe_load(f)

    patterns = data.get("patterns", [])
    if len(patterns) != 8:
        raise ValueError(
            f"Expected 8 patterns in canonical source, got {len(patterns)}. "
            "Check fedpatterns/privacy/compliance/compliance_matrix.yaml."
        )

    shutil.copy2(CANONICAL, ROOT_COPY)
    print(f"[OK] Root compliance_matrix.yaml generated — {len(patterns)} patterns present.")
    print(f"     Source : {CANONICAL}")
    print(f"     Written: {ROOT_COPY}")


if __name__ == "__main__":
    main()
