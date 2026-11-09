"""Repository content checks, run by CI (CONTRIBUTING.md, "Repository contents").

Reports, and exits with status 1 on:

- documentation history pages (``*history*.md`` under ``docs/``): documentation describes the current state;
- machine-specific absolute paths or machine names in text files;
- files larger than ``MAX_FILE_BYTES``, unless listed in ``ALLOWED_LARGE_FILES``;
- issue or merge-request numbers in Python code.

usage: python tools/check_repo.py [root]
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

MAX_FILE_BYTES = 1 << 20
ALLOWED_LARGE_FILES: set[str] = set()  # repository-relative paths
SKIP_DIRS = {".git", ".cache", ".ruff_cache", ".pytest_cache", "__pycache__", "artifacts", ".venv", "venv",
             "build", "dist"}
MACHINE_PATH = re.compile(r"(?<![\w.])/(?:opt|mnt|workspace)/|\bdevbox\b", re.IGNORECASE)
ISSUE_REFERENCE = re.compile(r"\b[\w.-]+/[\w.-]+#\d+|\b(?:issue|MR|PR)\s*[#!]\d+", re.IGNORECASE)


def _files(root: Path):
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if any(part in SKIP_DIRS or part.endswith(".egg-info") for part in relative.parts[:-1]):
            continue
        if path.is_file():
            yield path, relative.as_posix()


def violations(root: Path) -> list[str]:
    """Return one ``path[:line]: problem`` entry per violation under ``root``."""
    this_file = Path(__file__).resolve()
    found = []
    for path, name in _files(root):
        if name.startswith("docs/") and name.endswith(".md") and "history" in Path(name).name.lower():
            found.append(f"{name}: history page; documentation describes the current state")
        if path.stat().st_size > MAX_FILE_BYTES and name not in ALLOWED_LARGE_FILES:
            found.append(f"{name}: larger than {MAX_FILE_BYTES} bytes")
            continue
        if path.resolve() == this_file:
            continue
        data = path.read_bytes()
        if b"\0" in data:
            continue
        for number, line in enumerate(data.decode("utf-8", errors="replace").splitlines(), 1):
            if MACHINE_PATH.search(line):
                found.append(f"{name}:{number}: machine-specific path or machine name")
            if name.endswith(".py") and ISSUE_REFERENCE.search(line):
                found.append(f"{name}:{number}: issue or merge-request number in code")
    return found


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    found = violations(root)
    for entry in found:
        print(entry)
    print(f"check_repo: {len(found)} violation(s)")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
