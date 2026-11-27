# Copyright 2026 X-Square Robot
# SPDX-License-Identifier: Apache-2.0
"""Prepared RPU operations selected through the shared operator registry.

Registration describes Python callables only: it loads no native library or
SDK. Numerical qualification is pending the opt-in board tests.
"""

from xinfer.ops.registry import registry
from xinfer.ops.spec import Contract, OpSpec

from .rpu_tp import FAMILIES

_CONTRACT = Contract(numerics="unqualified", deterministic=False, graph_safe=False)


def _implementation(op):
    def prepare(context, *, tensors, integers=None, scalars=None, **selectors):
        return context.prepare(op, tensors=tensors, integers=integers, scalars=scalars, **selectors)

    prepare.__name__ = prepare.__qualname__ = f"prepare_{op}"
    return prepare


for _op in (*FAMILIES, "copy_f16"):
    _name = f"prepare_{_op}"
    if _name not in registry.names():
        registry.define(OpSpec(
            _name,
            f"{_name}(context, *, tensors, integers=None, scalars=None, **selectors) -> RpuTpOperation",
        ))
    registry.add(_name, _implementation(_op), backend="rpu", platforms=("rpu",),
                 library="kernels", contract=_CONTRACT)
