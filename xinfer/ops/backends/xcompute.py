"""Load the xcompute TVM-FFI library and register the fused BF16 operators used by Wall-OSS.

Set ``XINFER_XCOMPUTE_LIBRARY`` to the path of an xcompute shared library or install an xcompute Python package
with its matching library. The backend uses the unified entries on architectures targeted by the build.
MXFP8 operators run on their PyTorch references.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path

BACKEND = "xcompute"
ENV = "XINFER_XCOMPUTE_LIBRARY"


class _Library:
    """A loaded ``xcompute.so`` TVM-FFI module with cached function handles.

    Args:
        path: path to the library; resolved, hashed and loaded as is.

    Attributes:
        path: resolved absolute path.
        sha256: hex SHA-256 of the library file.

    Raises:
        FileNotFoundError: ``path`` does not exist.
        ImportError: ``tvm_ffi`` is not installed.
    """

    def __init__(self, path: str):
        import tvm_ffi

        self.path = str(Path(path).resolve(strict=True))
        self.sha256 = hashlib.sha256(Path(self.path).read_bytes()).hexdigest()
        self._module = tvm_ffi.load_module(self.path)
        self._fns = {}

    def has_function(self, name: str) -> bool:
        """Whether the library exports function ``name``."""
        try:
            self.function(name)
            return True
        except Exception:  # noqa: BLE001 - missing symbol
            return False

    def function(self, name):
        """The TVM-FFI function ``name`` (looked up once, then cached)."""
        if name not in self._fns:
            self._fns[name] = self._module.get_function(name)
        return self._fns[name]


def register(registry) -> list[str]:
    """Register what the xcompute library provides (the backend protocol of ``OpRegistry.load_plugins``).

    Loads the library (``$XINFER_XCOMPUTE_LIBRARY``, else the installed ``xcompute`` package's) and registers its
    unified entries (``register_unified``).

    Args:
        registry: the ``OpRegistry`` to register into.

    Returns:
        The registered operator names; none without a library.
    """
    from xinfer.ops.backends.xcompute_unified import register_unified

    path = os.environ.get(ENV) or _package_library()
    if not path:
        return []
    register_unified(registry, _Library(path))
    return registry.backend_ops(BACKEND)


def _package_library() -> str | None:
    """The library of the installed ``xcompute`` package, or None when the package is missing or finds no library."""
    if importlib.util.find_spec("xcompute") is None:
        return None
    os.environ.setdefault("XCOMPUTE_SKIP_AUTO_BUILD", "1")  # load the built library; never build at import
    try:
        import xcompute
    except Exception:  # noqa: BLE001 - installed without a loadable library
        return None
    return xcompute.KERNEL_LIB_PATH
