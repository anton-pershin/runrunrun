"""Fetch pinned external source trees and build the Blocks World generator.

Run from this study directory: python scripts/bootstrap.py
No Python packages are installed.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor"
REPOS = {
    "pddl-generators": (
        "https://github.com/AI-Planning/pddl-generators.git",
        "e6277fbd0b516e345c06f21be9816af63ab816c2",
    ),
    "pyperplan": (
        "https://github.com/aibasel/pyperplan.git",
        "586b99f060bfe457f359aa8b4bae1a299fa4e8c7",
    ),
    "VAL": (
        "https://github.com/KCL-Planning/VAL.git",
        "3c7a1f330bdab0ba28a4762bb45c3f06c27fb6d4",
    ),
}


def run(args: list[str], cwd: Path | None = None) -> None:
    subprocess.run(args, cwd=cwd, check=True)


def main() -> None:
    VENDOR.mkdir(exist_ok=True)
    for name, (url, revision) in REPOS.items():
        path = VENDOR / name
        if not path.exists():
            run(["git", "clone", url, str(path)])
        run(["git", "-C", str(path), "checkout", "--detach", revision])
    run(["make", "-j2"], cwd=VENDOR / "pddl-generators" / "blocksworld")
    print("Pinned source trees ready; Blocks World generator built.")
    print("VAL build requires CMake and was not attempted by this script.")


if __name__ == "__main__":
    main()
