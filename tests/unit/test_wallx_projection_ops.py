"""Split projection contracts and the build-time layer binding used by Wall-X."""

from functools import wraps
from types import SimpleNamespace

import pytest
import torch

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

    monkeypatch.setattr(gemm, "qkv_rope_epilogue", reference.qkv_rope_epilogue)
    monkeypatch.setattr(gemm, "gemm_epilogue", reference.gemm_epilogue)
    x, weight, bias = torch.ones(5, 16).bfloat16(), torch.ones(48, 16).bfloat16(), torch.zeros(48).bfloat16()
    cos, sin = torch.ones(5, 4), torch.zeros(5, 4)
    q, k, v = gemm.qkv_rope_split(x, weight, bias, cos, sin, 2, 8)
    assert q.stride() == (48, 8, 1)  # the existing CUDA path's views, without a packing copy
    assert torch.equal(q, k) and torch.equal(k, v)
    left, right = gemm.gemm_split(x, weight, 32)
    torch.testing.assert_close(torch.cat((left, right), 1), reference.gemm_epilogue(x, weight), rtol=0, atol=0)


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


def test_expanded_rotary_tables_are_shared_only_while_sources_are_valid():
    from xinfer.ops.backends.ascend._projection_layout import _expanded_rope

    cos, sin = torch.ones(5, 4), torch.zeros(5, 4)
    first = _expanded_rope(cos, sin)
    assert _expanded_rope(cos, sin)[0] is first[0]
    cos.mul_(0.5)
    second = _expanded_rope(cos, sin)
    assert second[0] is not first[0]
    torch.testing.assert_close(second[0], torch.full((5, 8), 0.5), rtol=0, atol=0)


def test_fused_projection_execution_errors_do_not_fall_back(monkeypatch):
    from xinfer.ops.backends.ascend import _projection_layout as layout

    monkeypatch.setattr(layout.finishbf16, "enabled", lambda family: True)

    def fail(*args):
        raise RuntimeError("launch failed")

    monkeypatch.setattr(layout.finishbf16, "zbasplit_bf16", fail)
    with pytest.raises(RuntimeError, match="launch failed"):
        layout.gemm_split(torch.ones(5, 16), torch.ones(48, 16), 32, torch.ones(5), None)


def test_ascend_norm_binding_preserves_the_layer_threshold(monkeypatch):
    from xinfer.ops.backends.ascend import norm, soupunlock

    monkeypatch.setattr(soupunlock, "enabled", lambda family: family == "rmsfull")
    monkeypatch.setattr(soupunlock, "from_layer", lambda: 8)
    inputs = torch.ones(1, 3, 4), torch.ones(1, 3, 4), torch.zeros(4), 1e-6
    marker = torch.full((1, 3, 4), 7.0)
    monkeypatch.setattr(soupunlock, "rmsnorm_full", lambda *args: (marker, marker))
    lower = norm.add_zero_centered_rms_norm.bind(layer_index=7)
    upper = norm.add_zero_centered_rms_norm.bind(layer_index=8)
    assert lower is norm.add_zero_centered_rms_norm
    assert upper(*inputs)[0] is marker
    assert torch.equal(lower(*inputs)[0], torch.full((1, 3, 4), 2.0))

    def fail(*args):
        raise RuntimeError("norm launch failed")

    monkeypatch.setattr(soupunlock, "rmsnorm_full", fail)
    with pytest.raises(RuntimeError, match="norm launch failed"):
        upper(*inputs)


def test_fused_projection_uses_reference_norm_when_that_family_is_absent(monkeypatch):
    from xinfer.ops.backends.ascend import gemm

    monkeypatch.setattr(gemm, "cann_projection", None)
    monkeypatch.setattr(gemm, "norm_library", None)
    monkeypatch.setattr(gemm, "gemm_epilogue", reference.gemm_epilogue)
    x, projection = torch.ones(3, 4).bfloat16(), torch.ones(8, 4).bfloat16()
    residual, weight = torch.ones(1, 3, 8).bfloat16(), torch.ones(8)
    actual = gemm.gemm_add_rms_norm(x, projection, residual, weight, 1e-6)
    expected = reference.gemm_add_rms_norm(x, projection, residual, weight, 1e-6)
    for value, reference_value in zip(actual, expected):
        torch.testing.assert_close(value, reference_value, rtol=0, atol=0)


def test_composite_projection_does_not_activate_previously_unselected_fusions(monkeypatch):
    from xinfer.ops.backends.ascend import gemm, rmsfuse_r4

    def unselected(*args, **kwargs):
        pytest.fail("the composite operator activated an unselected fusion")

    monkeypatch.setattr(rmsfuse_r4, "enabled", lambda family: True)
    monkeypatch.setattr(rmsfuse_r4, "gemm_add_rms_r4", unselected)
    monkeypatch.setattr(gemm, "cann_projection", SimpleNamespace(
        _stable_contiguous=lambda weight: weight.contiguous(), gemm_add_rms_norm=unselected))
    monkeypatch.setattr(gemm, "norm_library", None)
    monkeypatch.setattr(gemm, "gemm_epilogue", reference.gemm_epilogue)
    x, projection = torch.ones(3, 4).bfloat16(), torch.ones(4, 8).bfloat16().t()
    residual, weight = torch.ones(1, 3, 8).bfloat16(), torch.ones(8)
    for actual, expected in zip(gemm.gemm_add_rms_norm(x, projection, residual, weight, 1e-6),
                               reference.gemm_add_rms_norm(x, projection, residual, weight, 1e-6)):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
