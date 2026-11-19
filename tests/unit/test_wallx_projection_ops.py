"""Split projection contracts and the build-time layer binding used by Wall-X."""

from functools import wraps

import pytest
import torch

from xinfer import ops
from xinfer.ops import Contract, OpRegistry, OpSpec
from xinfer.ops.reference import gemm as reference
from xinfer.platforms import resolve_platform


def test_split_projections_keep_the_existing_rounding_and_values():
    generator = torch.Generator().manual_seed(42)
    x = torch.randn(5, 16, generator=generator).bfloat16()
    weight = torch.randn(48, 16, generator=generator).bfloat16()
    bias = torch.randn(48, generator=generator).bfloat16()
    angles = torch.randn(5, 4, generator=generator)
    q, k, v = reference.qkv_rope_split(x, weight, bias, angles.cos(), angles.sin(), 2, 8)
    packed = reference.qkv_rope_epilogue(x, weight, bias, angles.cos(), angles.sin(), 2, 8)
    torch.testing.assert_close(torch.stack((q, k, v), 1).reshape(5, -1), packed, rtol=0, atol=0)
    row_scale = torch.tensor([1.0, 0.0, 0.5, -0.25, 1.0])
    left, right = reference.gemm_split(x, weight, 32, row_scale=row_scale)
    expected = reference.gemm_epilogue(x, weight, row_scale=row_scale)
    torch.testing.assert_close(torch.cat((left, right), 1), expected, rtol=0, atol=0)
    assert left.shape == (5, 32) and right.shape == (5, 16)


def test_cuda_split_adapters_reuse_the_existing_entries(monkeypatch):
    from xinfer.ops.backends.cuda import gemm

    x, weight, bias = torch.ones(5, 16).bfloat16(), torch.ones(48, 16).bfloat16(), torch.arange(48).bfloat16()
    cos, sin = torch.ones(5, 4), torch.zeros(5, 4)
    packed = reference.qkv_rope_epilogue(x, weight, bias, cos, sin, 2, 8)
    monkeypatch.setattr(gemm, "qkv_proj_rope", lambda *args, **kwargs: packed)
    monkeypatch.setattr(gemm, "gemm_epilogue", reference.gemm_epilogue)
    assert gemm.qkv_rope_epilogue(x, weight, bias, cos, sin, 2, 8) is packed
    q, k, v = gemm.qkv_rope_split(x, weight, bias, cos, sin, 2, 8)
    assert q.stride() == (48, 8, 1)  # the existing CUDA path's views, without a packing copy
    torch.testing.assert_close(torch.stack((q, k, v), 1).reshape_as(packed), packed, rtol=0, atol=0)
    left, right = gemm.gemm_split(x, weight, 32)
    torch.testing.assert_close(torch.cat((left, right), 1), reference.gemm_epilogue(x, weight), rtol=0, atol=0)


def test_split_calls_a_plugin_that_only_registers_packed_projections():
    registry = OpRegistry()
    names = ("gemm_epilogue", "qkv_rope_epilogue")
    splits = ("gemm_split", "qkv_rope_split")
    for name in (*names, *splits):
        registry.define(ops.registry.spec(name))
    calls = {}

    def legacy(name):
        ref = registry.spec(name).reference

        @wraps(ref)
        def project(*args, **kwargs):
            out = ref(*args, **kwargs)
            calls[name] = (kwargs, out)
            return out

        return project

    contract = Contract(numerics="tolerance", deterministic=False)
    for name in names:
        registry.add(name, legacy(name), backend="custom", library="old", version="1", contract=contract)
    platform = resolve_platform("cpu")
    pins = dict.fromkeys(names, "custom:old")
    table = registry.build_table(platform, backends=pins, reference_only=True)
    x, weight, bias = torch.ones(5, 16).bfloat16(), torch.ones(48, 16).bfloat16(), torch.arange(48).bfloat16()
    rows = torch.arange(5).float()
    cos, sin = torch.ones(5, 4), torch.zeros(5, 4)
    left, right = table.gemm_split(x, weight, 32, row_scale=rows)
    q, k, v = table.qkv_rope_split(x, weight, bias, cos, sin, 2, 8, qk_order="any")
    kwargs, packed = calls["gemm_epilogue"]
    assert kwargs["row_scale"] is rows
    assert torch.equal(torch.cat((left, right), 1), packed)
    assert left.shape == (5, 32) and right.shape == (5, 16)
    for view in (left, right):
        assert view.untyped_storage().data_ptr() == packed.untyped_storage().data_ptr()
    kwargs, packed = calls["qkv_rope_epilogue"]
    assert kwargs["qk_order"] == "any"
    assert all(view.shape == (5, 2, 8) and view.untyped_storage().data_ptr() == packed.untyped_storage().data_ptr()
               for view in (q, k, v))
    assert torch.equal(torch.stack((q, k, v), 1).reshape_as(packed), packed)
    for name in splits:
        selected = table.selection(name)
        assert (selected.name, selected.version, selected.contract) == ("custom:old", "1", contract)
        assert registry.build_table(platform).selection(name).name == "custom:old"
        assert registry.build_table(platform, reference_only=True).selection(name).name == "reference"
        assert registry.build_table(platform, backends={name: "custom:old"}).selection(name).name == "custom:old"
    with pytest.raises(RuntimeError, match="does not support"):
        registry.build_table(platform, backends={"gemm_split": "missing"})

    # A native split belonging to a different library wins by default. A legacy pin keeps the chosen library.
    native_split = registry.spec("gemm_split").reference
    registry.add("gemm_split", native_split, backend="custom", library="other", contract=contract)
    assert registry.build_table(platform).gemm_split is native_split
    assert registry.build_table(platform, backends=pins).selection("gemm_split").name == "custom:old"
    assert registry.build_table(platform, backends={**pins, "gemm_split": "custom:other"}).gemm_split is native_split
    assert registry.build_table(platform, backends={**pins, "gemm_split": "reference"}).selection(
        "gemm_split").name == "reference"
    registry.add("gemm_split", native_split, backend="custom", library="old", contract=contract)
    assert registry.build_table(platform, backends=pins).gemm_split is native_split


def _registry_with_binder(bound):
    registry = OpRegistry()
    registry.define(OpSpec("double", "double(x)", lambda x: x * 2))

    def double(x):
        return x * 2

    def bind(*, layer_index):
        bound.append(layer_index)

        @wraps(double)
        def selected(x):
            if x < 0:
                raise RuntimeError("backend execution failed")
            return double(x)

        return selected

    double.bind = bind
    registry.add("double", double, backend="test", contract=Contract(numerics="bitwise"),
                 platforms=("cpu",), priority=10)
    return registry


def test_binding_happens_at_build_time_and_preserves_errors_and_reference_pins():
    bound = []
    registry = _registry_with_binder(bound)
    platform = resolve_platform("cpu")
    table = registry.build_table(platform)
    function = table.bind("double", layer_index=8)
    assert function(2) == 4 and function(3) == 6 and bound == [8]
    with pytest.raises(RuntimeError, match="backend execution failed"):
        function(-1)
    references = registry.build_table(platform, reference_only=True)
    assert references.bind("double", layer_index=8) is references.double
    assert references.double(-1) == -2 and bound == [8]


def test_instrumented_tables_still_measure_bound_calls():
    bound, calls = [], []
    table = _registry_with_binder(bound).build_table(resolve_platform("cpu"))

    def instrument(implementation):
        @wraps(implementation.function)
        def run(x):
            calls.append(implementation.op)
            return implementation.function(x)
        return run

    wrapped = table.wrapped(instrument)
    function = wrapped.bind("double", layer_index=4)
    assert function(2) == 4 and calls == ["double"] and bound == [4]
    assert wrapped.report() == table.report()
