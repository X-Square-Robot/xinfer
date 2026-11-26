# Copyright 2026 X-Square Robot
# SPDX-License-Identifier: Apache-2.0
"""Prepared table selection dispatches the selected callable, without device execution."""

import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from xinfer import ops
from xinfer.ops import Contract, OpRegistry, OpSpec, OpTable
from xinfer.ops.backends import rpu
from xinfer.ops.backends.rpu.rpu_tp import FAMILIES
from xinfer.platforms.interface import Platform, PlatformCapabilities


def platform(device_type="rpu"):
    return Platform(device_type, f"{device_type}:0", PlatformCapabilities(
        device_type, None, False, False, False, None, None, None))


def test_registration_and_table_build_do_not_load_native_or_sdk():
    script = """
import importlib.abc
import subprocess
import sys
from xinfer import ops
from xinfer.platforms.interface import Platform, PlatformCapabilities
class BlockNative(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith('tvm_ffi'):
            raise AssertionError('registration requested native package ' + fullname)
sys.meta_path.insert(0, BlockNative())
def unexpected(*args, **kwargs):
    raise AssertionError('registration started a process')
subprocess.run = subprocess.Popen = unexpected
import xinfer.ops.backends.rpu.registration
p = Platform('rpu', 'rpu:0', PlatformCapabilities('rpu', None, False, False, False, None, None, None))
table = ops.registry.build_table(p)
assert table.selection('prepare_gemm').name == 'rpu:kernels'
assert table.selection('prepare_copy_f16').name == 'rpu:kernels'
assert 'tvm_ffi' not in sys.modules
"""
    environment = {**os.environ, "XINFER_KERNELS": "0", "XINFER_FLASH_ATTN": "0"}
    result = subprocess.run([sys.executable, "-c", script], env=environment, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("family", (*FAMILIES, "copy_f16"))
def test_registered_prepared_callable_forwards_to_native_without_executing(family):
    table = ops.registry.build_table(platform())
    calls = []
    result = object()

    def prepare(op, **options):
        calls.append((op, options))
        return result

    native = SimpleNamespace(prepare=prepare)
    tensors = {"out": object()}
    integers, scalars = {"value": 7}, {"epsilon": 1e-5}
    bound = rpu.bind_ops(table)
    assert bound(native, family, tensors=tensors, integers=integers, scalars=scalars,
                 selector="explicit") is result
    assert calls == [(family, {"tensors": tensors, "integers": integers,
                               "scalars": scalars, "selector": "explicit"})]
    row = next(row for row in table.report() if row["op"] == f"prepare_{family}")
    assert row["backend"] == "rpu" and row["library"] == "kernels"
    assert row["numerics"] == "unqualified"
    assert not row["deterministic"] and not row["graph_safe"]


def test_registered_prepared_ops_have_no_torch_reference_fallback():
    for name in (*FAMILIES, "copy_f16"):
        assert ops.registry.spec(f"prepare_{name}").reference is None
    table = ops.registry.build_table(platform())
    assert {row["op"] for row in table.report()} == {f"prepare_{name}" for name in (*FAMILIES, "copy_f16")}
    with pytest.raises(RuntimeError, match="no reference implementation"):
        ops.registry.build_table(platform(), reference_only=True)
    with pytest.raises(RuntimeError, match="does not support"):
        ops.registry.build_table(platform(), backends={"prepare_gemm": "reference"})


def test_pinned_alternative_and_wrapped_table_are_the_actual_dispatch():
    registry = OpRegistry()
    registry.define(OpSpec("prepare_gemm", "prepare_gemm(context, *, tensors, **selectors)"))
    calls = []
    default_result, selected_result = object(), object()

    def default(context, **options):
        calls.append(("default", context, options))
        return default_result

    def alternative(context, **options):
        calls.append(("selected", context, options))
        return selected_result

    contract = Contract(numerics="unqualified", deterministic=False, graph_safe=False)
    registry.add("prepare_gemm", default, backend="rpu", platforms=("rpu",), library="kernels", contract=contract)
    registry.add("prepare_gemm", alternative, backend="rpu", platforms=("rpu",), library="alternate", contract=contract)

    def instrument(implementation):
        def wrapped(*args, **kwargs):
            calls.append(("wrapper", implementation.name))
            return implementation.function(*args, **kwargs)
        return wrapped

    table = registry.build_table(platform(), backends={"prepare_gemm": "rpu:alternate"}).wrapped(instrument)
    native = object()
    assert rpu.bind_ops(table)(native, "gemm", tensors={"x": 3}) is selected_result
    assert calls == [("wrapper", "rpu:alternate"),
                     ("selected", native, {"tensors": {"x": 3}, "integers": None, "scalars": None})]
    assert table.report()[0]["library"] == "alternate"


def test_partial_table_accepts_model_subset_but_never_bypasses_missing_selection():
    registry = OpRegistry()
    registry.define(OpSpec("prepare_gemm", "prepare_gemm(context, *, tensors, **selectors)"))
    registry.define(OpSpec("prepare_rmsnorm", "prepare_rmsnorm(context, *, tensors, **selectors)"))
    sentinel = object()
    registry.add("prepare_gemm", lambda *args, **kwargs: sentinel, backend="rpu", platforms=("rpu",),
                 contract=Contract(numerics="unqualified"))
    table = registry.build_table(platform())
    bound = rpu.bind_ops(table)
    assert bound(object(), "gemm", tensors={}) is sentinel
    with pytest.raises(RuntimeError, match="prepare_rmsnorm.*not selected"):
        bound(object(), "rmsnorm", tensors={})
    with pytest.raises(RuntimeError, match="no prepared operators"):
        rpu.bind_ops(OpTable("empty", {}))
