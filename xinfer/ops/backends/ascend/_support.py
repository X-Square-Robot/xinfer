"""Ascend entry discovery and cached partial-support adapters.

The shared xcompute loader loads and pairs the package and library once. These helpers discover usable
exports and optional bridges, and share layout-based fallback across GEMM, attention and normalization.
The registry selects a backend at model build time; unsupported call layouts remain the backend's concern.
"""

import inspect
from functools import wraps
from importlib import import_module
from pathlib import Path

import torch

from xinfer.ops.backends._xcompute import ASCEND_XCOMPUTE

xc = ASCEND_XCOMPUTE["requires"]
VERSION = ASCEND_XCOMPUTE["version"]


def canonical(name, reference):
    """An explicit Python export with the operator's call signature, or None.

    Inspect the package namespace directly: a module-level __getattr__ may expose raw, output-first FFI
    functions, whose names alone do not establish the XInfer operator contract.
    The registry checks our wrappers' signatures, so the wrapped provider also needs this check.
    """
    function = vars(xc).get(name) if xc is not None else None
    if not callable(function):
        return None
    expected = inspect.signature(reference)
    try:
        actual = inspect.signature(function)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"xcompute.{name} must expose the Python operator signature {expected}") from exc
    actual_parameters = tuple((p.name, p.kind) for p in actual.parameters.values())
    expected_parameters = tuple((p.name, p.kind) for p in expected.parameters.values())
    if actual_parameters != expected_parameters:
        raise TypeError(f"xcompute.{name} has signature {actual}; expected {expected}")
    return function


def has_functions(*names):
    """Whether the paired library exports every requested function."""
    return xc is not None and all(callable(getattr(xc.xcompute_lib, name, None)) for name in names)


def entry(*names):
    """The paired package when every Python entry is available, otherwise None for ``requires``."""
    return xc if xc is not None and all(callable(getattr(xc, name, None)) for name in names) else None


def bridge(name):
    """An optional bridge from the paired Python package; missing dependencies still raise."""
    if xc is None:
        return None
    module_name = f"xcompute.kernel.{name}"
    try:
        module = import_module(module_name)
    except ModuleNotFoundError as exc:
        missing = exc.name or ""
        if missing == module_name or module_name.startswith(missing + "."):
            return None
        raise
    path = getattr(module, "__file__", None)
    if not path or not Path(path).resolve().is_relative_to(Path(xc.__file__).resolve().parent):
        raise RuntimeError(f"{module_name} must come from the paired xcompute package")
    if getattr(module, "xcompute_lib", xc.xcompute_lib) is not xc.xcompute_lib:
        raise RuntimeError(f"{module_name} must use the paired xcompute library")
    return module


def _key(value):
    """Layout key of one argument. ``data_ptr() % 16`` records the alignment class, not the address, so the
    support cache stays bounded by (device, dtype, shape, stride, alignment) combinations instead of growing with
    every new allocation."""
    if torch.is_tensor(value):
        return (value.device, value.dtype, tuple(value.shape), value.stride(), value.data_ptr() % 16)
    if isinstance(value, (tuple, list)):
        return tuple(_key(item) for item in value)
    return value


def call_key(args, kwargs):
    """Layout and scalar options of an operator call, without reading device tensor values."""
    return (_key(args), tuple((name, _key(value)) for name, value in sorted(kwargs.items())))


def partial(reference, *, check=None, key=None):
    """Cache whether the first eager call of each layout is supported.

    Entries reject unsupported layouts with ValueError before launching work. Once a layout succeeds, errors
    propagate: a later execution failure must not silently switch its numerical implementation.
    """
    def decorate(function):
        supported = {}

        @wraps(function)
        def dispatch(*args, **kwargs):
            layout = key(*args, **kwargs) if key is not None else call_key(args, kwargs)
            admitted = supported.get(layout)
            if admitted is False:
                return reference(*args, **kwargs)
            if admitted is True:
                return function(*args, **kwargs)
            if check is not None and not check(*args, **kwargs):
                supported[layout] = False
                return reference(*args, **kwargs)
            try:
                result = function(*args, **kwargs)
            except ValueError:
                supported[layout] = False
                return reference(*args, **kwargs)
            supported[layout] = True
            return result

        return dispatch

    return decorate
