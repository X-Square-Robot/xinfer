"""Ascend entry discovery and cached partial-support adapters.

The unified branch's own provider (``_provider``) loads the kernel libraries;
these helpers discover usable exports and optional bridges, and share
layout-based fallback across GEMM, attention and normalization. The registry
selects a backend at model build time; unsupported call layouts remain the
backend's concern.

The semantics are the former paired-loader helpers, re-pointed at the
in-repo glue namespace: ``canonical()`` still inspects a package namespace and
compares signatures, ``bridge()`` still imports a glue module and checks it
comes from this tree, ``partial()`` still caches per-layout support. The
provider namespace object (``xc``, below) exposes the class-style operator
entries from ``glue.core`` so the binding modules' shape is unchanged.
"""

import inspect
from functools import wraps
from importlib import import_module
from pathlib import Path

import torch

from xinfer.ops.backends.ascend import _provider

# The glue namespace object: a module-like surface over glue/core.py's
# class-style entries. It is None when the provider is disabled or the core
# library is absent (no library, no registrations), which keeps every
# `requires=` None and the backend on the reference compositions.
#
# The namespace answers to the registry's library name ``xinfer_ascend``:
# ``register(..., requires=xc)`` spells the selection report's library (and
# the ``--op-backends <op>=ascend:xinfer_ascend`` pin, ARCH-DESIGN §4.3) from
# ``requires.__name__``. The retired paired loader's module spelled it
# ``xcompute``; forwarding attribute access to the glue core keeps the
# absent-entry AttributeError semantics (getattr(xc, name, None) → None) the
# class glues' fail-safes depend on.
class _XinferAscendNamespace:
    __name__ = "xinfer_ascend"

    def __getattr__(self, name):
        return getattr(_glue_core, name)

    def __repr__(self):  # pragma: no cover - diagnostics only
        return f"<xinfer_ascend namespace over {_glue_core!r}>"


try:
    from xinfer.ops.backends.ascend.glue import core as _glue_core

    xc = (_XinferAscendNamespace()
          if _provider.enabled() and _provider.core_library() is not None else None)
except Exception:  # noqa: BLE001 - the glue import never fails in-tree; stay safe for odd installs
    xc = None

VERSION = "unified-1" if xc is not None else None


def canonical(name, reference):
    """An explicit glue export with the operator's call signature, or None.

    Inspect the glue namespace directly: a module-level __getattr__ may expose
    raw, output-first launcher functions, whose names alone do not establish
    the XInfer operator contract. The registry checks our wrappers'
    signatures, so the wrapped provider also needs this check.
    """
    function = getattr(xc, name, None) if xc is not None else None
    if not callable(function):
        return None
    expected = inspect.signature(reference)
    try:
        actual = inspect.signature(function)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"glue.{name} must expose the Python operator signature {expected}") from exc
    actual_parameters = tuple((p.name, p.kind) for p in actual.parameters.values())
    expected_parameters = tuple((p.name, p.kind) for p in expected.parameters.values())
    if actual_parameters != expected_parameters:
        raise TypeError(f"glue.{name} has signature {actual}; expected {expected}")
    return function


def has_functions(*names):
    """Whether the core library exports every requested launcher.

    The former tvm_ffi probe over the paired v22 library; here the glue core
    owns the entries, so probe its namespace (entries are absent exactly when
    the kernels are not built or the envelope is not carried).
    """
    return xc is not None and all(callable(getattr(xc, name, None)) for name in names)


def entry(*names):
    """The glue namespace when every Python entry is available, otherwise None for ``requires``."""
    return xc if xc is not None and all(callable(getattr(xc, name, None)) for name in names) else None


def bridge(name):
    """A glue module by name; missing modules still raise, absent families return None."""
    if xc is None:
        return None
    module_name = f"xinfer.ops.backends.ascend.glue.{name}"
    try:
        module = import_module(module_name)
    except ModuleNotFoundError as exc:
        missing = exc.name or ""
        if missing == module_name or module_name.startswith(missing + "."):
            return None
        raise
    path = getattr(module, "__file__", None)
    if not path or not Path(path).resolve().is_relative_to(Path(__file__).resolve().parent):
        raise RuntimeError(f"{module_name} must come from the in-tree glue package")
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
