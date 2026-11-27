# Copyright 2026 X-Square Robot
# SPDX-License-Identifier: Apache-2.0
"""Explicit, lazy loading of the RPU AOT TVM-FFI module; never builds code."""

from __future__ import annotations

import hashlib
import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path

ABI_VERSION = 11
ENV = "XINFER_RPU_LIBRARY"


@dataclass(frozen=True)
class _Loaded:
    module: object
    path: Path
    metadata: dict


_loaded = None
_lock = threading.Lock()


def _library_path():
    explicit = os.environ.get(ENV)
    path = Path(explicit).expanduser() if explicit else Path(__file__).with_name("xinfer_rpu.so")
    if not path.is_file():
        raise FileNotFoundError(
            f"RPU AOT library does not exist: {path}. Build xinfer_rpu.so explicitly and set {ENV}; "
            "importing the backend never compiles or falls back to Torch."
        )
    return path.resolve()


def _function(module, name):
    try:
        return module.get_function(name)
    except AttributeError:
        return None


def _query(module, name):
    function = _function(module, name)
    if function is None:
        raise RuntimeError(f"RPU AOT library lacks required metadata export {name}; rebuild xinfer_rpu.so")
    return function()


def _load():
    global _loaded
    path = _library_path()
    with _lock:
        if _loaded is not None:
            if path != _loaded.path:
                raise RuntimeError(
                    f"{ENV} changed after native loading; use a new process to select another RPU library")
            return _loaded
        try:
            import tvm_ffi
        except ImportError as error:
            raise ImportError("The RPU AOT library requires the apache-tvm-ffi Python package") from error
        try:
            module = tvm_ffi.load_module(str(path))
        except Exception as error:
            raise RuntimeError(
                f"Cannot load RPU AOT library {path}: {error}. A runtime build requires the external "
                "Rhino Launch SDK and its shared libraries; a host-only build supports planning only."
            ) from error
        abi = int(_query(module, "rpu_abi_version"))
        if abi != ABI_VERSION:
            raise RuntimeError(f"RPU ABI mismatch: library has {abi}, Python bindings require {ABI_VERSION}")
        backends = tuple(str(_query(module, "compiled_backends")).split(";"))
        if backends != ("rpu",):
            raise RuntimeError(f"Expected an independent RPU library, got compiled_backends={backends!r}")
        catalog = str(_query(module, "rpu_catalog_sha256"))
        if re.fullmatch(r"[0-9a-f]{64}", catalog) is None:
            raise RuntimeError("RPU library returned an invalid catalog SHA256")
        metadata = {
            "library": "xinfer_rpu.so",
            "library_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "compiled_backends": list(backends),
            "abi_version": abi,
            "catalog_sha256": catalog,
            "runtime_available": bool(_query(module, "rpu_runtime_available")),
        }
        _loaded = _Loaded(module, path, metadata)
        return _loaded


def provenance():
    """Identify the actual loaded AOT module, without constructing a device context."""
    metadata = _load().metadata
    return {**metadata, "compiled_backends": list(metadata["compiled_backends"])}


def require_runtime_assets():
    """Reject host-only modules and invalid external assets before device allocation."""
    loaded = _load()
    if not loaded.metadata["runtime_available"]:
        raise RuntimeError(
            "This xinfer_rpu.so is host-only and cannot execute RPU operations; "
            "build with XINFER_RPU_HOST_ONLY=OFF and the external Rhino Launch SDK"
        )
    from ._assets import validate_assets

    names = _query(loaded.module, "rpu_kernel_names")
    return validate_assets(loaded.metadata, {str(name) for name in names})


class _Library:
    """Defer the shared module load until a function is actually requested."""

    def __init__(self):
        self._functions = {}

    def get_function(self, name):
        function = self._functions.get(name)
        if function is None:
            function = _function(_load().module, name)
            if function is not None:
                self._functions[name] = function
        return function

    def __getattr__(self, name):
        function = self.get_function(name)
        if function is None:
            raise RuntimeError(f"RPU AOT library lacks {name}; build xinfer_rpu.so with the required runtime bindings")
        return function


lib = _Library()
