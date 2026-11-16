"""The external xcompute build of the Ascend backend: ``xc`` is its Python package, or None.

Set ``XINFER_XCOMPUTE_LIBRARY`` to the path of an Ascend xcompute shared library or install an xcompute Python
package with its matching library. The operators registered with ``ASCEND_XCOMPUTE`` need the Python package of the
same build; without it (or with ``XINFER_XCOMPUTE=0``) they are not registered and run on their torch references.
Ascend builds report their backend and bind entries individually. The CUDA backend builds its kernel library from
this repository (``cuda/_native.py``) and does not come through here.
"""

from __future__ import annotations

import hashlib
import importlib.util
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

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


def is_ascend(library) -> bool:
    """Whether ``library`` is an Ascend build.

    Two shapes exist. The older Ascend builds export ``build_backend()``, which answers ``"ascend"``. The delivery
    builds drop that export and identify themselves only by reporting no CUDA arch from ``build_target_archs()``:
    their work is AOT device images, so there is no SM family to name. Treating that as Ascend keeps such a build
    from being classified as a CUDA library, which would leave every Ascend operator on the reference.
    """
    if not hasattr(library, "has_function"):
        return False
    if library.has_function("build_backend"):
        return str(library.function("build_backend")()) == "ascend"
    if not library.has_function("build_target_archs"):
        return False
    reported = str(library.function("build_target_archs")())
    return not any(part.isdigit() for part in reported.split(","))


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
    if not is_ascend(library):  # a CUDA kernel library: cuda/_native.py loads it
        return None
    package = _python_package(library)
    if package is None:
        logger.info("xcompute is not available: %s has no Python package of its build", library.path)
        return None
    return package, library, target_archs(library), f"unified xcompute.so sha256:{library.sha256[:12]}"


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
    """The xcompute Python package of ``library``'s build, or None."""
    os.environ.setdefault("XCOMPUTE_LIBRARY_PATH", library.path)
    os.environ.setdefault("XCOMPUTE_SKIP_AUTO_BUILD", "1")
    try:
        import xcompute
    except ImportError:
        return None
    if Path(xcompute.KERNEL_LIB_PATH).resolve() != Path(library.path):
        return None
    return xcompute


def target_archs(library) -> tuple[str, ...]:
    """The device family the library's build targets: ``("ascend",)``, or nothing for another build."""
    return ("ascend",) if is_ascend(library) else ()


xc, library, archs, version = _import() or (None, None, (), None)
# ``register`` arguments of every Ascend operator that calls xcompute
ASCEND_XCOMPUTE = dict(requires=xc if "ascend" in archs else None, version=version)
