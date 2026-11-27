# Copyright 2026 X-Square Robot
# SPDX-License-Identifier: Apache-2.0
"""Validate external REF assets against their ABI/catalog-bound manifest."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from pathlib import Path

ENV = "XINFER_RPU_ASSETS"


def _regular_file(path):
    try:
        info = path.lstat()
    except FileNotFoundError as error:
        raise FileNotFoundError(f"RPU asset file does not exist: {path}") from error
    if not stat.S_ISREG(info.st_mode) or info.st_size == 0:
        raise ValueError(f"RPU asset must be a nonempty regular file without symlinks: {path}")


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate RPU asset manifest key: {key}")
        result[key] = value
    return result


def validate_assets(metadata, kernel_names):
    """Verify all declared REF bytes; native loading still enforces per-kernel admission.

    The manifest supplies expected hashes, not an independent signature or a
    numerical qualification. Its provenance must be established by the caller.
    """
    location = os.environ.get(ENV)
    if not location:
        raise FileNotFoundError(f"Set {ENV} to the external RPU assets directory containing manifest.json and *.ref")
    directory = Path(location)
    if not directory.is_dir():
        raise FileNotFoundError(f"{ENV} is not a directory: {directory}")
    directory = directory.resolve()
    manifest = directory / "manifest.json"
    _regular_file(manifest)
    try:
        manifest_bytes = manifest.read_bytes()
        data = json.loads(manifest_bytes, object_pairs_hook=_unique_keys)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid RPU asset manifest: {manifest}") from error
    if not isinstance(data, dict):
        raise ValueError("RPU asset manifest must be an object")
    if type(data.get("abi_version")) is not int or data["abi_version"] != metadata["abi_version"]:
        raise ValueError("RPU asset manifest ABI does not match the loaded native library")
    if data.get("catalog_sha256") != metadata["catalog_sha256"]:
        raise ValueError("RPU asset manifest catalog SHA256 does not match the loaded native library")
    assets = data.get("assets")
    if not isinstance(assets, dict) or not assets:
        raise ValueError("RPU asset manifest must declare a nonempty assets object")
    undeclared = {path.stem for path in directory.glob("*.ref")} - set(assets)
    if undeclared:
        raise ValueError(f"RPU asset files are absent from the manifest: {sorted(undeclared)}")
    for kernel, entry in assets.items():
        if re.fullmatch(r"[A-Za-z0-9_]+", kernel) is None or kernel not in kernel_names:
            raise ValueError(f"RPU asset kernel is outside the native catalog: {kernel!r}")
        if not isinstance(entry, dict) or set(entry) != {"sha256"}:
            raise ValueError(f"RPU asset {kernel!r} must declare only sha256; file paths are fixed by the catalog")
        expected = entry["sha256"]
        if not isinstance(expected, str) or re.fullmatch(r"[0-9a-f]{64}", expected) is None:
            raise ValueError(f"RPU asset {kernel!r} has an invalid SHA256")
        path = directory / f"{kernel}.ref"
        _regular_file(path)
        if path.resolve().parent != directory:
            raise ValueError(f"RPU asset escapes its directory: {path}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"RPU asset SHA256 mismatch: {path}")
    return {"manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest()}
