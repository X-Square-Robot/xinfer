"""Wall-OSS projection and prefill operators on Thor and their portable compositions."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from xinfer import ops
from xinfer.ops.backends.cuda import thor
from xinfer.ops.reference import gemm, norm


@pytest.mark.parametrize("platforms, exports, enabled", [
    (("sm110", "sm120"), True, True), (("sm120",), True, False), (("sm110",), False, False),
])
def test_thor_registration_needs_the_target_and_kernel(monkeypatch, platforms, exports, enabled):
    package = object()
    library = SimpleNamespace(has_function=lambda name: exports)
    from xinfer.ops.backends import _xcompute

    monkeypatch.setattr(_xcompute, "XCOMPUTE", dict(requires=package, platforms=platforms, version="test"))
    monkeypatch.setattr(_xcompute, "xc", package)
    monkeypatch.setattr(_xcompute, "library", library)
    options = thor.sm110_with(functions=("bf16_qkv_rope_sm110",))
    assert options["platforms"] == ("sm110",)
    assert options["requires"] is (package if enabled else None)


@pytest.mark.parametrize("use_mlp", [False, True])
def test_prefill_reference_preserves_both_residual_outputs(use_mlp):
    gen = torch.Generator().manual_seed(40)
    x = torch.randn(5, 16, generator=gen).bfloat16()
    residual = torch.randn(1, 5, 8, generator=gen).bfloat16()
    gamma = torch.rand(8, generator=gen) + 0.5
    if use_mlp:
        gate_up = torch.randn(24, 16, generator=gen).bfloat16()
        down = torch.randn(8, 12, generator=gen).bfloat16()
        projected = gemm.gemm_epilogue(gemm.gemm_glu_epilogue(x, gate_up), down)
        actual = ops.registry.spec("mlp_add_rms_norm").reference(x, gate_up, down, residual, gamma, 1e-6)
    else:
        weight = torch.randn(8, 16, generator=gen).bfloat16()
        bias = torch.randn(8, generator=gen).bfloat16()
        projected = gemm.gemm_epilogue(x, weight, bias)
        actual = ops.registry.spec("gemm_add_rms_norm").reference(x, weight, residual, gamma, 1e-6, bias=bias)
    expected = norm.add_rms_norm(projected.view_as(residual), residual, gamma, 1e-6)
    for value, want in zip(actual, expected, strict=True):
        torch.testing.assert_close(value, want, rtol=0, atol=0)


@pytest.mark.parametrize("listed", [True, False])
def test_vision_requests_paired_channels_only_for_a_listed_kernel(monkeypatch, listed):
    packed = SimpleNamespace(layout="nk_rope_pairs")
    package = Mock()
    package.pack_rope_table.return_value = object()
    package.prepack_weight.return_value = packed
    package.selected_backend.return_value = "bf16_sm110" if listed else "generic"
    monkeypatch.setattr(thor, "xc", package)
    x, w, bias = torch.zeros(5, 16), torch.zeros(48, 16), torch.zeros(48)
    cos, sin = torch.ones(5, 4), torch.zeros(5, 4)
    table_cache = {}
    for _ in range(2):
        thor.qkv_rope_for_attention(x, w, bias, cos, sin, 2, 8, table_cache=table_cache)
    package.pack_rope_table.assert_called_once_with(cos, sin)
    package.prepack_weight.assert_called_once()
    args, options = package.qkv_proj_rope.call_args
    if listed:
        assert args[1] is packed and options["out_layout"] == "rope_pairs"
    else:
        assert args[1] is w and args[2] is bias and "out_layout" not in options


@pytest.mark.parametrize("layout, listed", [
    ("nk", False), ("nk_glu_blocked128_k64", False), ("nk_glu_blocked128_k64", True),
])
def test_thor_mlp_checks_both_packing_and_shape_before_fusion(monkeypatch, layout, listed):
    x, gate_up, down = torch.zeros(4, 8), torch.zeros(128, 8), torch.zeros(512, 64)
    residual, gamma = torch.ones(1, 4, 512), torch.ones(512)
    package = Mock()
    package.prepack_weight.side_effect = [SimpleNamespace(layout=layout), object()]
    package.selected_backend.return_value = "mcast_bf16_sm110" if listed else "generic"
    package.rmsnorm.side_effect = lambda x, w, eps, inference: norm.rms_norm(x, w, eps)
    monkeypatch.setattr(thor, "xc", package)
    monkeypatch.setattr(thor, "gemm_epilogue", gemm.gemm_epilogue)
    monkeypatch.setattr(thor, "gemm_glu_epilogue", gemm.gemm_glu_epilogue)
    total, normalized = thor.mlp_add_rms_norm(x, gate_up, down, residual, gamma, 1e-6)
    assert torch.equal(total, residual) and torch.equal(normalized, norm.rms_norm(residual, gamma, 1e-6))
    assert package.selected_backend.called == (layout != "nk")
    if listed:
        options = package.fused_mlp.call_args.kwargs
        assert options["activation"] == "silu" and options["out"].data_ptr() == residual.data_ptr()
        assert options["residual"].data_ptr() == residual.data_ptr()
    else:
        package.fused_mlp.assert_not_called()


def test_partial_rotary_uses_the_standard_qkv_path(monkeypatch):
    package = Mock()
    monkeypatch.setattr(thor, "xc", package)
    x, weight, bias = torch.zeros(5, 16), torch.zeros(48, 16), torch.zeros(48)
    cos, sin = torch.ones(5, 2), torch.zeros(5, 2)
    thor.qkv_rope_for_attention(x, weight, bias, cos, sin, 2, 8)
    package.prepack_weight.assert_not_called()
    package.pack_rope_table.assert_not_called()
    package.qkv_proj_rope.assert_called_once_with(x, weight, bias, cos=cos, sin=sin, rotary_dim=4,
                                                   num_heads=2, num_kv_heads=2, head_dim=8, backend="auto")


def test_mlp_rejects_unsupported_down_shape_before_packing(monkeypatch):
    package = Mock()
    package.rmsnorm.side_effect = lambda total, gamma, eps, inference: norm.rms_norm(total, gamma, eps)
    monkeypatch.setattr(thor, "xc", package)
    monkeypatch.setattr(thor, "gemm_epilogue", gemm.gemm_epilogue)
    monkeypatch.setattr(thor, "gemm_glu_epilogue", gemm.gemm_glu_epilogue)
    x, gate_up, down = torch.zeros(4, 8), torch.zeros(12, 8), torch.zeros(8, 6)
    residual, gamma = torch.ones(1, 4, 8), torch.ones(8)
    total, normed = thor.mlp_add_rms_norm(x, gate_up, down, residual, gamma, 1e-6)
    package.prepack_weight.assert_not_called()
    assert torch.equal(total, residual) and torch.equal(normed, norm.rms_norm(residual, gamma, 1e-6))
