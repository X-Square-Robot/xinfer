# Copyright 2026 X-Square Robot
# SPDX-License-Identifier: Apache-2.0
"""RPU AOT bindings and host-only operation planning.

Importing this backend does not load a native library, initialize the SDK or
compile code. Runtime context construction requires external REF assets.
"""

from types import SimpleNamespace

__all__ = [
    "RpuBuffer", "RpuContext", "RpuExecution", "RpuWeight",
    "RpuTpContext", "RpuTpExecution", "RpuTpOperation", "RpuTpTensor",
    "bind_ops", "create_backend", "load_backend", "resolve",
]


def bind_ops(table):
    """Capture the selected prepared callables for a platform operation factory.

    Selection wrappers are retained. A partial table need not contain families
    that the model never calls. Missing selections never bypass the table.
    """
    selected = {row["op"][len("prepare_"):]: table.selection(row["op"]).function
                for row in table.report() if row["op"].startswith("prepare_")}
    if not selected:
        raise RuntimeError(f"no prepared operators are selected for {table.platform_name}")

    def prepare(context, op, *, tensors, integers=None, scalars=None, **selectors):
        try:
            function = selected[op]
        except KeyError:
            raise RuntimeError(f"prepared operator 'prepare_{op}' is not selected for {table.platform_name}") from None
        return function(context, tensors=tensors, integers=integers, scalars=scalars, **selectors)

    return prepare


def load_backend():
    """Load the explicit AOT library and return its factory and provenance.

    A host-only module can be inspected here; ``create_backend`` rejects it.
    This function does not allocate device resources or execute a warmup.
    """
    from ._native import provenance

    return SimpleNamespace(backend_factory=create_backend, provenance=provenance())


def create_backend(*, allow_candidate=False, retained_segments=128):
    """Construct a native TP8 context after runtime and external-asset validation."""
    from .rpu_tp import RpuTpContext

    return RpuTpContext(allow_candidate=allow_candidate, retained_segments=retained_segments)


def resolve(op, tensors, integers=None, **selectors):
    """Resolve an operation family without a device; tensor descriptions need shape/layout/distribution."""
    from .rpu_tp import resolve as resolve_family

    return resolve_family(op, tensors, integers, **selectors)


def __getattr__(name):
    if name in ("RpuBuffer", "RpuContext", "RpuExecution", "RpuWeight"):
        from . import rpu

        return getattr(rpu, name)
    if name in ("RpuTpContext", "RpuTpExecution", "RpuTpOperation", "RpuTpTensor"):
        from . import rpu_tp

        return getattr(rpu_tp, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
