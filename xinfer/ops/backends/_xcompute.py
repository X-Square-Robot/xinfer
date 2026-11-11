"""The xcompute library, shared by the backend modules that call it: ``xc`` is its Python package, or None.

Set ``XINFER_XCOMPUTE_LIBRARY`` to the path of an xcompute shared library or install an xcompute Python package with
its matching library. The operators registered with ``XCOMPUTE`` need the Python package of the same build (the
entries in ``UNIFIED``) and run on the architectures the build targets; without it (or with ``XINFER_XCOMPUTE=0``)
they are not registered and run on their torch references. The architectures are CUDA ones (``sm120``) as long as
the builds target CUDA only.
"""

from __future__ import annotations

import hashlib
import importlib.util
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

ENV = "XINFER_XCOMPUTE_LIBRARY"
UNIFIED = ("gemm", "gemm_glu", "qkv_proj_rope", "prepack_weight", "selected_backend", "pack_rope_table", "AdaRms",
           "rmsnorm", "add_rmsnorm", "attention_kvcache", "attention_kvcache_backend", "denoise_output_euler")
LIBRARY_ENTRIES = ("build_target_archs", "gemm_envelope_table")


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
        self._tvm_ffi = tvm_ffi
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

    def call(self, name, *args):
        """Call a library function on the current torch stream."""
        with self._tvm_ffi.use_torch_stream():
            return self.function(name)(*args)


def _import():
    """``(package, library, archs, version)`` of the usable xcompute build, or None (the reason is logged).

    Raises:
        FileNotFoundError: ``$XINFER_XCOMPUTE_LIBRARY`` does not exist.
    """
    if os.environ.get("XINFER_XCOMPUTE", "1") == "0":
        return None
    path = os.environ.get(ENV) or _package_library()
    if not path:
        logger.info("xcompute is not available: set %s or install the xcompute package", ENV)
        return None
    library = _Library(path)
    package, archs = _python_package(library), target_archs(library)
    if package is None or not archs:
        logger.info("xcompute is not available: %s has no Python package of its build with the unified entries, or "
                    "targets no architecture", library.path)
        return None
    return package, library, archs, f"unified xcompute.so sha256:{library.sha256[:12]}"


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


def _python_package(library):
    """The xcompute Python package of ``library``'s build with the entries in ``UNIFIED``, or None.

    None also when ``library`` lacks a function in ``LIBRARY_ENTRIES``: the package then comes from another build.
    """
    os.environ.setdefault("XCOMPUTE_LIBRARY_PATH", library.path)
    os.environ.setdefault("XCOMPUTE_SKIP_AUTO_BUILD", "1")
    try:
        import xcompute
    except ImportError:
        return None
    if (Path(xcompute.KERNEL_LIB_PATH).resolve() != Path(library.path)
            or not all(hasattr(xcompute, name) for name in UNIFIED)
            or not all(library.has_function(name) for name in LIBRARY_ENTRIES)):
        return None
    return xcompute


def target_archs(library) -> tuple[str, ...]:
    """The arch names (``"sm120"``) the library's build targets."""
    return tuple(f"sm{arch}" for arch in str(library.function("build_target_archs")()).split(",") if arch.isdigit())


xc, library, archs, version = _import() or (None, None, (), None)
# ``register`` arguments of every operator that calls xcompute
XCOMPUTE = dict(requires=xc, platforms=archs, version=version)


def xcompute_with(*entries, functions=()):
    """Register a kernel only when this xcompute build exports the additional entries it uses."""
    available = (xc is not None and all(hasattr(xc, name) for name in entries)
                 and (not functions or library is not None and all(library.has_function(name) for name in functions)))
    return dict(XCOMPUTE, requires=xc if available else None)
