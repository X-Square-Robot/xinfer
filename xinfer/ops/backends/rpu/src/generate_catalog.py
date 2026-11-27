# Copyright 2026 X-Square Robot
# SPDX-License-Identifier: Apache-2.0
"""Generate native metadata from the RPU catalog, without device assets."""

import argparse
import hashlib
import json
from pathlib import Path
import re


def generate(catalog: Path, output: Path) -> None:
    raw = catalog.read_bytes()
    document = json.loads(raw)
    names = set()
    entries = []
    for entry in document["kernels"]:
        name = entry["name"]
        if not re.fullmatch(r"[a-z][a-z0-9_]*", name) or name in names:
            raise ValueError(f"invalid or duplicate kernel name: {name}")
        if entry.get("execution_status") or entry.get("admission", "") not in ("", "candidate"):
            raise ValueError(f"unsupported kernel admission: {name}")
        names.add(name)
        entries.append(f'    {{"{name}", "{name}.ref", "{entry.get("admission", "")}", ""}},')
    text = "\n".join([
        "// Generated catalog metadata; contains no compiled device assets.",
        "#pragma once",
        "#include <cstddef>",
        "namespace xinfer::rpu::metadata {",
        f"inline constexpr unsigned abi_version = {int(document['abi_version'])};",
        f'inline constexpr char catalog_sha256[] = "{hashlib.sha256(raw).hexdigest()}";',
        "struct Asset { const char* name; const char* filename; const char* admission;",
        "               const char* execution_status; };",
        "inline constexpr Asset assets[] = {",
        *entries,
        "};",
        "inline constexpr std::size_t count = sizeof(assets) / sizeof(assets[0]);",
        "}  // namespace xinfer::rpu::metadata",
        "",
    ])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    generate(arguments.catalog, arguments.output)
