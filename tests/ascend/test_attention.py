"""Attention adapters preserve layout and native error boundaries."""
from inspect import unwrap
from threading import Lock
from types import SimpleNamespace

import pytest
import torch

from xinfer.ops.backends.ascend import (
    _gated_pack,
    _native,
    attn,
    cann_attention,
    cann_gated_delta_net,
    conv_gemm,
    decodeattn_r4,
    fia_fuse_route,
    soup2,
)


@pytest.mark.parametrize(("new_pos", "row_reduce"), [(768, True), (767, False)])
def test_cache_attention_returns_packed_rows_from_decode_or_fia(monkeypatch, new_pos, row_reduce):
    monkeypatch.setenv("XINFER_DECODEATTN_R4_ROW", "1")
    monkeypatch.setattr(decodeattn_r4, "applies", lambda q, k, v, prefix, pos, mask, scale: pos == 768
                        and mask is not None)
    monkeypatch.setattr(decodeattn_r4, "_SCAN_VARIANT", "b")
    monkeypatch.setattr(decodeattn_r4, "_SEEN", set())
    monkeypatch.setattr(decodeattn_r4, "_entry", lambda *args: (None, None))
    monkeypatch.setattr(torch, "npu", SimpleNamespace(is_current_stream_capturing=lambda: False,
                        current_stream=lambda device: SimpleNamespace(npu_stream=0)), raising=False)
    monkeypatch.setattr(cann_attention, "_check_qkv", lambda *args: None)
    launches = []

    def launch(name, tensors, *args):
        launches.append((name, tensors))
        if name == "decodeattn_r4_reduce_row":
            tensors[1].copy_(q[0])
        return True

    def launch_fia(out, workspace, q, k, v, drop_mask, *args):
        assert drop_mask is not None
        out.copy_(q)
        launches.append(("fia", (q, k, v, out)))

    native = cann_attention.CannAttention.__new__(cann_attention.CannAttention)
    native._lock, native._buffers = Lock(), {}
    native._bnsd_query = lambda *args: 0
    native._bnsd_launch = launch_fia
    monkeypatch.setattr(attn, "native_cache", native.prefix_kvcache_attention)
    monkeypatch.setattr(attn, "native_cache_bnsd", native.prefix_kvcache_attention)
    monkeypatch.setattr(_native, "launch_kernel", launch)
    q = torch.arange(16 * 16 * 128, dtype=torch.float32).to(torch.bfloat16).view(1, 16, 16, 128)
    k, v = (torch.empty(1, 2, 784, 128, dtype=torch.bfloat16).transpose(1, 2) for _ in range(2))
    prefix = torch.tensor([new_pos], dtype=torch.int32)
    mask = torch.zeros(1, 1, 784, 2, dtype=torch.int32)
    result = unwrap(attn.prefix_kvcache_attention)(q, k, v, prefix, new_pos, flash_mask=mask)
    names = ["decodeattn_r4_scan_b", "decodeattn_r4_reduce_row"] if row_reduce else ["fia"]
    assert [name for name, _ in launches] == names
    query, keys, values = launches[0][1][:3]
    torch.testing.assert_close(query.reshape(1, 16, 16, 128), q.transpose(1, 2), atol=0, rtol=0)
    assert keys.data_ptr() == k.data_ptr() and values.data_ptr() == v.data_ptr()
    torch.testing.assert_close(result, q.flatten(2), atol=0, rtol=0)
    assert result.is_contiguous()


def test_admitted_fia_launch_failure_propagates(monkeypatch):
    monkeypatch.setattr(fia_fuse_route, "applies", lambda *args: True)
    monkeypatch.setattr(fia_fuse_route, "_SEEN", {("warm",)})
    monkeypatch.setattr(fia_fuse_route, "_STATS", {"errors": []})
    monkeypatch.setattr(fia_fuse_route, "_entry", lambda *args: (lambda *args: 0, lambda: b""))
    monkeypatch.setattr(torch, "npu", SimpleNamespace(current_stream=lambda device: SimpleNamespace(npu_stream=0)),
                        raising=False)

    def fail(*args):
        raise ValueError("native execution failed")

    monkeypatch.setattr(fia_fuse_route._native, "launch_kernel", fail)
    q = torch.zeros(1, 8, 16, 256, dtype=torch.bfloat16)
    cache = torch.zeros(1, 2, 432, 256, dtype=torch.bfloat16).transpose(1, 2)
    with pytest.raises(RuntimeError, match="native execution failed"):
        fia_fuse_route.try_route(q, cache, cache, torch.tensor([416], dtype=torch.int32), 416,
                                torch.zeros(1, 16, 2048, dtype=torch.bfloat16),
                                torch.zeros(1, 1, 432, 2, dtype=torch.int32), 256 ** -0.5)
    assert fia_fuse_route._STATS["errors"]


def test_gated_attention_does_not_retry_a_failed_fused_route(monkeypatch):
    def fail(*args):
        raise ImportError("fused route failed")

    monkeypatch.setattr(fia_fuse_route, "try_route", fail)
    launch = unwrap(attn.gated_kvcache_attention)
    with pytest.raises(ImportError, match="fused route failed"):
        launch(None, None, None, None, 0, None)


def test_fia_uses_the_native_query_storage_from_a_canonical_view(monkeypatch):
    native = torch.zeros(1, 8, 16, 256, dtype=torch.bfloat16)

    def decline(q, *args):
        assert q.shape == native.shape and q.stride() == native.stride()
        assert q.data_ptr() == native.data_ptr()
        return False

    monkeypatch.setattr(fia_fuse_route, "applies", decline)
    assert fia_fuse_route.try_route(native.transpose(1, 2), None, None, None, 0,
                                   torch.zeros(1, 16, 2048, dtype=torch.bfloat16), None, None) is None


@pytest.mark.parametrize("batch", [1, 2, 4])
def test_gated_attention_materializes_only_the_final_packed_output(monkeypatch, batch):
    from xinfer.ops.backends.ascend import attn

    monkeypatch.setattr(fia_fuse_route, "try_route", lambda *args: None)
    native_query = torch.arange(batch * 8 * 16 * 256, dtype=torch.float32).to(torch.bfloat16).view(batch, 8, 16, 256)
    native_output = torch.linspace(-2, 2, native_query.numel()).to(torch.bfloat16).reshape_as(native_query)
    gate = torch.linspace(-1, 1, batch * 16 * 2048).to(torch.bfloat16).view(batch, 16, 2048)

    def prefix(q, *args, **kwargs):
        assert q.shape == (batch, 16, 8, 256) and q.data_ptr() == native_query.data_ptr()
        assert q.transpose(1, 2).is_contiguous()
        return native_output

    monkeypatch.setattr(attn, "native_cache_bnsd", prefix)
    output = attn.gated_kvcache_attention.__wrapped__(native_query.transpose(1, 2), None, None, None, 0, gate)
    expected = native_output.transpose(1, 2).reshape(batch, 16, 2048) * torch.sigmoid(gate)
    assert output.shape == expected.shape and output.is_contiguous()
    assert torch.equal(output, expected)


def test_gated_pack_preserves_the_existing_launcher_arguments(monkeypatch):
    monkeypatch.setattr(_gated_pack, "_applies", lambda *args: True)
    calls = []
    monkeypatch.setattr(_gated_pack._native, "launch_kernel",
                        lambda name, tensors, scalars: calls.append((name, tensors, scalars)) or True)
    out = torch.empty(4, 8, 16, 256, dtype=torch.bfloat16)
    gate = torch.empty(4, 16, 4096, dtype=torch.bfloat16)[..., :2048]
    packed = _gated_pack.gated_pack(out, gate)
    assert packed.shape == (4, 16, 2048) and packed.is_contiguous()
    assert calls[0][0] == "gated_attn_bnsd_pack_pow2"
    assert calls[0][1] == (packed, out, gate)
    assert calls[0][2] == (131072, 260096, 2048, 16, 256, 4096, 0)


def test_pack_marshalling_error_cannot_restart_the_attention_operator(monkeypatch):
    monkeypatch.setattr(_gated_pack, "_applies", lambda *args: True)

    def fail(*args):
        raise ValueError("invalid native argument")

    monkeypatch.setattr(_gated_pack._native, "launch_kernel", fail)
    with pytest.raises(RuntimeError, match="invalid native argument"):
        _gated_pack.gated_pack(torch.empty(1, 8, 16, 256, dtype=torch.bfloat16),
                              torch.empty(1, 16, 2048, dtype=torch.bfloat16))


def test_cann_convolution_status_cannot_select_a_fallback(monkeypatch):
    monkeypatch.setattr(conv_gemm, "_envelope", lambda *args: True)
    monkeypatch.setattr(conv_gemm, "_weight_rows", lambda weight: weight)
    monkeypatch.setattr(conv_gemm, "_pad", lambda *args: torch.zeros(1))
    monkeypatch.setattr(conv_gemm, "_library", lambda: (lambda *args: 1, lambda: b"CANN call failed"))
    monkeypatch.setattr(_native, "launch_kernel", lambda *args: False)
    monkeypatch.setattr(torch, "npu", SimpleNamespace(
        current_stream=lambda *args: SimpleNamespace(npu_stream=0)), raising=False)
    with pytest.raises(RuntimeError, match="CANN call failed"):
        conv_gemm.conv_mixed_tokenmajor(torch.zeros(1, 1, 6144), torch.zeros(6144, 4))


def test_nested_gdn_postprocessing_preserves_the_launcher_error(monkeypatch):
    def fail(*args):
        raise RuntimeError("qksq launch failed")

    monkeypatch.setattr(soup2, "qksq", fail)
    with pytest.raises(RuntimeError, match="qksq launch failed"):
        cann_gated_delta_net._soup2_qksq(torch.zeros(1), 1, 2, 128)


def test_finish_failure_does_not_reclassify_the_entry_as_unavailable(monkeypatch):
    monkeypatch.setattr(cann_gated_delta_net, "_finish2_entry", lambda: (lambda *args: 1, lambda: b"finish failed"))
    operator = cann_gated_delta_net.CannGatedDeltaNet.__new__(cann_gated_delta_net.CannGatedDeltaNet)
    operator._finish2_ok = True
    value = torch.zeros(1, 16, 128)
    with pytest.raises(RuntimeError, match="finish failed"):
        operator._finish2(value, value, value, value, 1, 16, 1e-6, 0)
    assert operator._finish2_ok
