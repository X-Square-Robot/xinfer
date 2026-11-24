"""Warmup routing on compiled KV-cache shapes outside the measured ranges."""

from contextlib import nullcontext

import pytest
import torch

from xinfer.ops.backends.cuda import _kv_tuning as tuning
from xinfer.ops.backends.cuda import attn


@pytest.mark.parametrize(("setting", "expected"), [(None, "kernels"), ("1", "kernels"), ("0", "kernels")])
def test_op_tuning_controls_unmeasured_prefix_dispatch(monkeypatch, setting, expected):
    if setting is None:
        monkeypatch.delenv("XINFER_OP_TUNING", raising=False)
    else:
        monkeypatch.setenv("XINFER_OP_TUNING", setting)
    monkeypatch.setattr(attn, "arch", lambda device: 120)
    monkeypatch.setattr(attn, "BUILD_ARCHS", (120,))
    monkeypatch.setattr(attn.libraries, "FLEX", True)
    def choose(*_):
        assert setting != "0"
        return "kernels"

    monkeypatch.setattr(tuning, "choose", choose)
    monkeypatch.setattr(attn, "attention_kvcache", lambda *args, **kwargs: "kernels")
    monkeypatch.setattr(attn, "flex_prefix", lambda *args, **kwargs: "static")
    q = torch.empty(1, 8, 16, 128, dtype=torch.bfloat16)
    cache = torch.empty(1, 48, 2, 128, dtype=q.dtype)
    assert attn.prefix_kvcache_attention(q, cache, cache, torch.tensor([16], dtype=torch.int32), 40) == expected


def test_warmup_choices_are_scoped_to_layout_options_and_alignment(monkeypatch):
    calls = []
    monkeypatch.setattr(tuning, "_winners", {})
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(tuning, "capturing", lambda device: True)
    monkeypatch.setattr(tuning, "_compare", lambda *args: calls.append(True) or "kernels")
    t = torch.zeros(64)
    a, b = t[1:17], t[2:18]
    assert tuning.choose("op", (a, None), None, None) is None
    assert not calls
    monkeypatch.setattr(tuning, "capturing", lambda device: False)
    for args in ((a, None), (a, None), (b, None), (a, 4)):
        assert tuning.choose("op", args, None, None) == "kernels"
    assert len(calls) == 3
    monkeypatch.setattr(tuning, "capturing", lambda device: True)
    assert tuning.choose("op", (a, None), None, None) == "kernels" and len(calls) == 3


def test_a_compiled_configuration_can_be_outside_the_measured_range(monkeypatch):
    monkeypatch.setattr(attn, "arch", lambda device: 120)
    monkeypatch.setattr(attn, "BUILD_ARCHS", (120,))
    q = torch.empty(1, 8, 16, 128, dtype=torch.bfloat16)
    k = torch.empty(1, 48, 2, 128, dtype=torch.bfloat16)
    assert attn._compiled_kvcache_takes(q, k) and not attn.attention_kvcache_takes(q, k)
    assert not attn._compiled_kvcache_takes(q.float(), k)
    assert not attn._compiled_kvcache_takes(q, k[:, :4])
    assert not attn._compiled_kvcache_takes(q[:, :, :3], k)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_a_numerically_ineligible_candidate_is_never_timed(monkeypatch):
    monkeypatch.setattr(tuning, "_time", lambda *_: pytest.fail("timed an inaccurate candidate"))
    x = torch.ones(20, device="cuda", dtype=torch.bfloat16)
    assert tuning._compare(x.device, lambda: x * 2, x.clone) == "flex_attention"
    assert tuning._compare(x.device, lambda: x * float("nan"), x.clone) == "flex_attention"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("gated", [False, True])
def test_an_unmeasured_native_kv_shape_uses_the_warmup_choice_in_capture(gated, monkeypatch):
    from xinfer.ops.backends.cuda import _attention_libraries as libraries
    from xinfer.ops.reference import attn as reference
    from xinfer.ops.registry import observe_reference_calls

    device = torch.device("cuda", torch.cuda.current_device())
    if attn.arch(device) not in attn.BUILD_ARCHS or not libraries.FLEX:
        pytest.skip("needs a native CUDA kernel build and FlexAttention")
    monkeypatch.delenv("XINFER_OP_TUNING", raising=False)
    monkeypatch.setattr(tuning, "_winners", {})
    generator = torch.Generator(device=device).manual_seed(7)

    def tensor(*shape):
        return torch.randn(shape, device=device, dtype=torch.bfloat16, generator=generator)

    q, k, v = tensor(1, 8, 16, 128), tensor(1, 2, 48, 128).transpose(1, 2), tensor(1, 2, 48, 128).transpose(1, 2)
    args = (q, k, v, torch.tensor([16], device=device, dtype=torch.int32), 40)
    op = "gated_kvcache_attention" if gated else "prefix_kvcache_attention"
    if gated:
        args += (tensor(1, 8, 16 * 128),)
    assert attn._compiled_kvcache_takes(q, k) and not attn.attention_kvcache_takes(q, k)
    before = (k.clone(), v.clone())
    selected = getattr(attn, op)
    reference_output = getattr(reference, op)(*args)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with monkeypatch.context() as config:
        config.setenv("XINFER_OP_TUNING", "0")
        static_routes = []
        with (torch.cuda.stream(stream),
              observe_reference_calls(lambda _: pytest.fail("fallback"), static_routes.append)):
            selected(*args)
        assert static_routes == ["kernels"] and not tuning._winners
    routes = []
    with torch.cuda.stream(stream), observe_reference_calls(lambda _: pytest.fail("fallback"), routes.append):
        result = selected(*args)
    assert routes == list(tuning._winners.values()) and routes[0] in ("kernels", "flex_attention")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        replay = selected(*args)
    graph.replay()
    assert torch.equal(replay, result) and torch.equal(k, before[0]) and torch.equal(v, before[1])
    err = torch.linalg.vector_norm(result.float() - reference_output.float())
    assert err < 0.01 * torch.linalg.vector_norm(reference_output.float())

@pytest.mark.parametrize("gated", [False, True])
def test_measured_shape_with_strided_v_reaches_reference(monkeypatch, gated):
    from xinfer.ops.registry import observe_reference_calls

    monkeypatch.setattr(attn, "arch", lambda device: 120)
    monkeypatch.setattr(attn, "BUILD_ARCHS", (120,))
    monkeypatch.setattr(attn.libraries, "FLEX", None)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    q = torch.zeros((1, 1, 16, 128), dtype=torch.bfloat16, device=device)
    k = q.new_zeros((1, 64, 2, 128))
    v = q.new_ones((1, 64, 2, 256))[..., ::2]
    prefix = torch.tensor([12], dtype=torch.int32, device=device)
    fn = attn.gated_kvcache_attention if gated else attn.prefix_kvcache_attention
    args = (q, k, v, prefix, 12, q.flatten(2)) if gated else (q, k, v, prefix, 12)
    calls = []
    with observe_reference_calls(calls.append):
        result = fn(*args)
    assert calls == [fn.__name__]
    torch.testing.assert_close(result, torch.full_like(result, 0.5 if gated else 1.0))
