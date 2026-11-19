"""Ascend projection layout, launch support and composed operators."""
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from xinfer.ops.backends.ascend import _native, cann_projection, core
from xinfer.ops.reference import gemm as reference


def test_glu_finish_dispatches_the_actual_built_width(monkeypatch):
    monkeypatch.setattr(_native, "core_library", lambda: object())
    monkeypatch.setattr(_native, "has_symbol", lambda lib, name: name.endswith("split_2048_call"))
    calls = []
    monkeypatch.setattr(core, "_launch", lambda lib, name, tensors, scalars: calls.append((name, tensors, scalars)))
    out, accumulator = torch.empty(64, 2048, dtype=torch.bfloat16), torch.zeros(64, 4096)
    core.gemm_epilogue_glu(out, accumulator, 0, 2048)
    assert calls[0][0] == "gemm_glu_fp32_finish_split_2048"
    assert calls[0][1][0] is out and calls[0][1][1] is accumulator
    assert calls[0][2] == (64, 0)


def test_unbuilt_glu_finish_declines_before_launch(monkeypatch):
    monkeypatch.setattr(_native, "core_library", lambda: object())
    monkeypatch.setattr(_native, "has_symbol", lambda *args: False)
    monkeypatch.setattr(core, "_launch", lambda *args: pytest.fail("unbuilt finish launched"))
    with pytest.raises(ValueError, match="not built"):
        core.gemm_epilogue_glu(torch.empty(64, 2048, dtype=torch.bfloat16), torch.zeros(64, 4096), 0, 2048)


def test_missing_finish_reuses_the_projection_without_late_rejection(monkeypatch):
    monkeypatch.setattr(cann_projection, "_geometry", lambda *args: (64, 4096, 1024))
    monkeypatch.setattr(cann_projection, "_has_glu_finish", lambda width: False)
    projection = cann_projection.CannProjection.__new__(cann_projection.CannProjection)
    projection._glu_fp32_deferred = None
    projection._glu_split_fp32_finish = lambda *args: pytest.fail("unbuilt finish was selected")
    calls = []

    @contextmanager
    def project(*args):
        calls.append("projection")
        yield torch.zeros(64, 4096)

    projection._project = project
    output = projection.gemm_glu_epilogue(torch.zeros(64, 1024, dtype=torch.bfloat16),
                                        torch.zeros(4096, 1024, dtype=torch.bfloat16))
    assert calls == ["projection"]
    assert output.shape == (64, 2048) and output.dtype == torch.bfloat16 and output.eq(0).all()


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


def test_composite_projection_does_not_activate_previously_unselected_fusions(monkeypatch):
    from xinfer.ops.backends.ascend import gemm, fused_rmsnorm

    def unselected(*args, **kwargs):
        pytest.fail("the composite operator activated an unselected fusion")

    monkeypatch.setattr(fused_rmsnorm, "enabled", lambda family: True)
    monkeypatch.setattr(fused_rmsnorm, "gemm_add_rmsnorm", unselected)
    monkeypatch.setattr(gemm, "cann_projection", SimpleNamespace(
        _stable_contiguous=lambda weight: weight.contiguous(), gemm_add_rms_norm=unselected))
    monkeypatch.setattr(gemm, "norm_library", None)
    monkeypatch.setattr(gemm, "gemm_epilogue", reference.gemm_epilogue)
    x, projection = torch.ones(3, 4).bfloat16(), torch.ones(4, 8).bfloat16().t()
    residual, weight = torch.ones(1, 3, 8).bfloat16(), torch.ones(8)
    for actual, expected in zip(gemm.gemm_add_rms_norm(x, projection, residual, weight, 1e-6),
                               reference.gemm_add_rms_norm(x, projection, residual, weight, 1e-6)):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_untuned_qkv_layout_uses_reference():
    from xinfer.ops.backends.ascend import gemm

    packed = torch.randn(1, 3, 12)
    q_weight, k_weight = torch.randn(2), torch.randn(2)
    expected = reference.qkv_norm_split(packed, q_weight, k_weight, 2, 1, 2, 1e-6)
    actual = gemm.qkv_norm_split(packed, q_weight, k_weight, 2, 1, 2, 1e-6)
    for result, reference_value in zip(actual, expected):
        torch.testing.assert_close(result, reference_value, rtol=0, atol=0)


@pytest.mark.parametrize("fused_first", [False, True], ids=["packed-first", "head-major-first"])
def test_m16_qkv_native_and_composed_finishes_both_return_packed_rows(monkeypatch, fused_first):
    from xinfer.ops.backends.ascend import cann_action, gemm

    rows, heads, kv_heads, head_dim, start = 16, 16, 2, 128, 768
    q_width, kv_width = heads * head_dim, kv_heads * head_dim
    width = q_width + 2 * kv_width
    projected = torch.arange(rows, dtype=torch.float32)[:, None] + torch.arange(width) / 32
    expected_q, expected_k, expected_v = projected.bfloat16().split((q_width, kv_width, kv_width), dim=1)
    expected_head_major_q = expected_q.view(rows, heads, head_dim).transpose(0, 1)
    expected_k = expected_k.view(rows, kv_heads, head_dim).transpose(0, 1)
    expected_v = expected_v.view(rows, kv_heads, head_dim).transpose(0, 1)
    calls = []

    @contextmanager
    def accumulate(x, weight):
        calls.append("project")
        yield projected

    def launch(lib, name, tensors, scalars):
        calls.append(name)
        assert scalars == (start + rows, start)
        q, keys, values, *_ = tensors
        q.view(heads, rows, head_dim).copy_(expected_head_major_q)
        keys[:, start:].copy_(expected_k)
        values[:, start:].copy_(expected_v)

    action = cann_action.CannAction.__new__(cann_action.CannAction)
    action._accumulator = accumulate
    action._rotary = None
    action._use_qk_rms = False
    action._qkv_bias_rope_prefill = None
    action._qkv_bias_rope_m16 = None
    monkeypatch.setattr(action, "_check_qkv", lambda *args, **kwargs: (rows, width))
    monkeypatch.setattr(gemm, "native_cache", action.qkv_rope_kvcache)
    monkeypatch.setattr(_native, "core_library", lambda: object())
    monkeypatch.setattr(core, "_launch", launch)

    x = torch.zeros(rows, 1024, dtype=torch.bfloat16)
    weight = torch.zeros(width, 1024, dtype=torch.bfloat16)
    cos, sin = torch.ones(rows, head_dim // 2), torch.zeros(rows, head_dim // 2)
    for fused in (fused_first, not fused_first):
        bias = torch.zeros(width, dtype=torch.bfloat16 if fused else torch.float32)
        keys = torch.zeros(kv_heads, start + rows, head_dim, dtype=torch.bfloat16)
        values = torch.zeros_like(keys)
        q = gemm.qkv_rope_kvcache(x, weight, bias, cos, sin, keys, values, start, heads, kv_heads, head_dim)
        torch.testing.assert_close(q, expected_q, rtol=0, atol=0)
        torch.testing.assert_close(keys[:, start:], expected_k, rtol=0, atol=0)
        torch.testing.assert_close(values[:, start:], expected_v, rtol=0, atol=0)

    fused_calls = ["project", "gemm_qkv_cache_bias_rope_m16"]
    packed_calls = ["project"]
    assert calls == (fused_calls + packed_calls if fused_first else packed_calls + fused_calls)
