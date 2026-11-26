# Copyright 2026 X-Square Robot
# SPDX-License-Identifier: Apache-2.0
"""Compile cataloged RPU sources with the external hxcc interface.

The result is a separate asset directory, never bytes embedded in a library or
an import-time build. The compiler and Rhino Launch SDK are external tools.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import tempfile

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "xinfer/ops/backends/rpu/src"
ENTRY_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source(root: Path, name: str, suffix: str | None = None) -> Path:
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts or "\\" in name:
        raise ValueError(f"source path must stay inside the source tree: {name!r}")
    path = root / relative
    if not path.resolve().is_relative_to(root) or path.is_symlink() or not path.is_file():
        raise ValueError(f"source must be a regular file inside the source tree: {name!r}")
    if suffix is not None and path.suffix != suffix:
        raise ValueError(f"source must have suffix {suffix}: {name!r}")
    return path


def build(source_root: Path, output: Path, compiler: str, kernels: list[str] | None = None) -> dict:
    """Build an explicit catalog snapshot and atomically publish a new directory.

    An existing nonempty output directory is never replaced. Every catalog
    entry and dependency is validated before invoking the compiler, even when
    only a subset is selected. Compiler output is not an ABI inspection claim.
    """
    source_root = source_root.resolve(strict=True)
    catalog_path = _source(source_root, "common/rpu/catalog.json", ".json")
    catalog_hash = _digest(catalog_path)
    catalog = json.loads(catalog_path.read_text())
    abi = catalog["abi_version"]
    if isinstance(abi, bool) or not isinstance(abi, int) or abi < 1:
        raise ValueError("catalog abi_version must be a positive integer")
    entries, snapshots = {}, {}
    for entry in catalog["kernels"]:
        name = entry["name"]
        if not isinstance(name, str) or not ENTRY_NAME.fullmatch(name) or name in entries:
            raise ValueError(f"invalid or duplicate catalog kernel name: {name!r}")
        if entry.get("execution_status"):
            raise ValueError(f"catalog entry is not executable: {name}")
        _source(source_root, entry["source"], ".rc")
        for relative in (entry["source"], *entry.get("dependencies", [])):
            path = _source(source_root, relative)
            snapshots[relative] = _digest(path)
        entries[name] = entry
    if not entries:
        raise ValueError("the source catalog is empty")
    selected = sorted(entries if kernels is None else set(kernels))
    if not selected or set(selected) - entries.keys():
        raise ValueError("kernel selection must be a nonempty subset of the catalog")
    executable = Path(shutil.which(compiler) or compiler).resolve(strict=True)
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise ValueError("compiler must be an executable file")
    compiler_hash = _digest(executable)
    output = output.absolute()
    if output.is_symlink() or (output.exists() and (not output.is_dir() or any(output.iterdir()))):
        raise ValueError("output must not exist or must be an empty directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {"abi_version": abi, "catalog_sha256": catalog_hash, "assets": {}}
    with tempfile.TemporaryDirectory(prefix=".xinfer-rpu-assets-", dir=output.parent) as temporary:
        staging = Path(temporary)
        for name in selected:
            entry = entries[name]
            source = source_root / entry["source"]
            asset = staging / f"{name}.ref"
            includes = sorted({source_root, source.parent,
                               *((source_root / dep).parent for dep in entry.get("dependencies", []))})
            command = [str(executable), "-O2", "-c", str(source), "-o", str(asset)]
            for include in includes:
                command.extend(("-I", str(include)))
            subprocess.run(command, cwd=source_root, check=True)
            if asset.is_symlink() or not asset.is_file() or asset.stat().st_size == 0:
                raise ValueError(f"compiler did not produce a nonempty regular asset: {name}")
            manifest["assets"][name] = {"sha256": _digest(asset)}
        if _digest(catalog_path) != catalog_hash or _digest(executable) != compiler_hash:
            raise ValueError("catalog or compiler changed during the build")
        if any(_digest(source_root / name) != digest for name, digest in snapshots.items()):
            raise ValueError("kernel source changed during the build")
        receipt = {"format_version": 1, "compiler": {"sha256": compiler_hash},
                   "sources": snapshots, "kernels": selected}
        (staging / "build.json").write_text(json.dumps(receipt, indent=2) + "\n")
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        # Rename only after every asset and its manifest are complete. This also
        # refuses a concurrently populated output directory instead of deleting it.
        staging.replace(output)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compiler", default=os.environ.get("HXCC", "hxcc"))
    parser.add_argument("--kernels", nargs="+", default=None)
    args = parser.parse_args()
    manifest = build(args.source_root, args.output, args.compiler, args.kernels)
    print(f"Built {len(manifest['assets'])} RPU assets; ABI {manifest['abi_version']}")


if __name__ == "__main__":
    main()
