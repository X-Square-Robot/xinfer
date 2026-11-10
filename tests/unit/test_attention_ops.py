"""Attention operators: reference semantics (CPU) and the flash-attn backend (GPU)."""

import pytest
import torch

from xinfer import ops
from xinfer.ops import reference
from xinfer.platforms import resolve_platform


def _qkv(b, n, heads=4, kv=2, d=32, dtype=torch.float32, device="cpu", seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    q = torch.randn(b, n, heads, d, generator=g, device=device).to(dtype)
    k = torch.randn(b, n, kv, d, generator=g, device=device).to(dtype)
    v = torch.randn(b, n, kv, d, generator=g, device=device).to(dtype)
    return q, k, v


def _dense(q, k, v):
    """Unmasked attention on sequence-major tensors (GQA by repetition), FP64."""
    rep = q.shape[2] // k.shape[2]
    k, v = k.repeat_interleave(rep, 2).double(), v.repeat_interleave(rep, 2).double()
    s = torch.einsum("bqhd,bkhd->bhqk", q.double(), k) / q.shape[-1] ** 0.5
    return torch.einsum("bhqk,bkhd->bqhd", s.softmax(-1), v)


def test_causal_attention_reference_is_the_masked_prefill():
    q, k, v = _qkv(2, 7)
    out = reference.causal_attention(q, k, v)
    assert out.shape == q.shape
    for i in range(7):  # row i sees keys 0..i
        torch.testing.assert_close(out[:, i:i + 1].double(), _dense(q[:, i:i + 1], k[:, :i + 1], v[:, :i + 1]),
                                   rtol=1e-5, atol=1e-5)


def test_action_attention_reference_pins_a_closed_group():
    q, k, v = _qkv(3, 6, seed=1)
    _, kp, vp = _qkv(3, 5, seed=2)
    kp, vp = kp.transpose(1, 2).contiguous(), vp.transpose(1, 2).contiguous()  # [B, KV, P, D]
    pinned = torch.tensor([0, 2, 6])
    out = reference.action_attention(q, k, v, kp, vp, pinned)
    pre_k, pre_v = kp.transpose(1, 2), vp.transpose(1, 2)
    for b in range(3):
        for i in range(6):
            own = int(pinned[b]) if i < pinned[b] else 6
            keys = torch.cat([pre_k[b:b + 1], k[b:b + 1, :own]], 1)
            vals = torch.cat([pre_v[b:b + 1], v[b:b + 1, :own]], 1)
            torch.testing.assert_close(out[b:b + 1, i:i + 1].double(), _dense(q[b:b + 1, i:i + 1], keys, vals),
                                       rtol=1e-5, atol=1e-5)


def test_prefix_attention_reference_sees_only_valid_keys():
    q, k, v = _qkv(2, 9, seed=5)
    valid = torch.tensor([6, 9])
    out = reference.prefix_attention(q, k, v, valid)
    for b in range(2):
        n = int(valid[b])
        torch.testing.assert_close(out[b:b + 1].double(), _dense(q[b:b + 1], k[b:b + 1, :n], v[b:b + 1, :n]),
                                   rtol=1e-5, atol=1e-5)
    garbage_k, garbage_v = k.clone(), v.clone()
    garbage_k[0, 6:], garbage_v[0, 6:] = 50.0, -50.0  # padded positions never act as keys
    assert torch.equal(reference.prefix_attention(q, garbage_k, garbage_v, valid), out)


def test_kvcache_attention_reference_appends_after_the_cached_keys():
    q, k, v = _qkv(2, 4, seed=6)
    _, kc, vc = _qkv(2, 11, seed=7)  # [B, S = 7 + 4, KV, D]
    start, valid = torch.tensor([2, 0]), torch.tensor([3, 6])  # row 0: cached columns 2..4
    k_cache, v_cache = kc.clone(), vc.clone()
    out = reference.kvcache_attention(q, k, v, k_cache, v_cache, start, valid)
    for b, (s0, n) in enumerate([(2, 3), (0, 6)]):
        keys = torch.cat([kc[b:b + 1, s0:s0 + n], k[b:b + 1]], 1)
        vals = torch.cat([vc[b:b + 1, s0:s0 + n], v[b:b + 1]], 1)
        torch.testing.assert_close(out[b:b + 1].double(), _dense(q[b:b + 1], keys, vals), rtol=1e-5, atol=1e-5)
        assert torch.equal(k_cache[b, s0 + n:s0 + n + 4], k[b]) and torch.equal(v_cache[b, s0 + n:s0 + n + 4], v[b])


def test_rtc_pinned_mask():
    m = reference.rtc_pinned_mask(torch.tensor([2]), 4)[0]
    assert m.tolist() == [[True, True, False, False], [True, True, False, False],
                          [True, True, True, True], [True, True, True, True]]


# ------------------------------------------------------------------ flash-attn (GPU)
def _flash_table():
    from xinfer.ops.backends.cuda import attn

    if not (torch.cuda.is_available() and attn.flash_attn is not None):
        pytest.skip("flash-attn + CUDA required")
    names = ("varlen_attention", "causal_attention", "action_attention", "prefix_attention", "kvcache_attention")
    table = ops.registry.build_table(resolve_platform("cuda"))
    assert all(table.selection(n).name == "cuda:flash_attn" for n in names)
    return table


@pytest.fixture(scope="module")
def flash():
    return _flash_table()


def _close(got, want):
    err = (got.float() - want.float()).abs()
    assert float(err.max()) <= 4 * 2 ** -8 * max(1.0, float(want.float().abs().max())), float(err.max())


@pytest.mark.gpu
def test_flash_attn_matches_reference(flash):
    bf16 = dict(dtype=torch.bfloat16, device="cuda")
    q, k, v = _qkv(1, 300, heads=16, kv=2, d=128, seed=3, **bf16)  # Wall-X prompt shape class
    _close(flash.causal_attention(q, k, v), reference.causal_attention(q, k, v))
    qa, ka, va = _qkv(3, 16, heads=16, kv=2, d=128, seed=4, **bf16)
    _, kp, vp = _qkv(3, 300, heads=16, kv=2, d=128, seed=5, **bf16)
    kp, vp = kp.transpose(1, 2).contiguous(), vp.transpose(1, 2).contiguous()
    pinned = torch.tensor([0, 5, 8], device="cuda")
    _close(flash.action_attention(qa, ka, va, kp, vp, pinned), reference.action_attention(qa, ka, va, kp, vp, pinned))
    t, h, d = 400, 16, 80  # ViT: 16 heads of 80
    g = torch.Generator(device="cuda").manual_seed(6)
    qv, kv_, vv = (torch.randn(t, h, d, generator=g, device="cuda").bfloat16() for _ in range(3))
    cu_seqlens = [0, 64, 128, 400]
    _close(flash.varlen_attention(qv, kv_, vv, cu_seqlens), reference.varlen_attention(qv, kv_, vv, cu_seqlens))


@pytest.mark.gpu
def test_flash_causal_attention_gives_the_bits_of_the_appending_call(flash):
    """Keys used as the cache give the bits of the service's call that appended them to empty caches."""
    import flash_attn as flash_attn_package

    bf16 = dict(dtype=torch.bfloat16, device="cuda")
    for b, n, heads, kv, d in ((1, 768, 16, 2, 128), (2, 129, 16, 4, 256), (3, 64, 8, 1, 256)):
        q, k, v = _qkv(b, n, heads=heads, kv=kv, d=d, seed=n, **bf16)
        k, v = (x.transpose(1, 2).contiguous().transpose(1, 2) for x in (k, v))  # views of [B, KV, L, D] caches
        keys, values = k.clone(), v.clone()
        empty = torch.zeros(b, dtype=torch.int32, device="cuda")
        appended = flash_attn_package.flash_attn_with_kvcache(q, torch.empty_like(k), torch.empty_like(v), k, v,
                                                              cache_seqlens=empty, causal=True, num_splits=1)
        assert torch.equal(flash.causal_attention(q, k, v), appended), (b, n, heads, kv, d)
        assert torch.equal(k, keys) and torch.equal(v, values)


@pytest.mark.gpu
def test_flash_attn_captures_in_cuda_graphs(flash):
    bf16 = dict(dtype=torch.bfloat16, device="cuda")
    q, k, v = _qkv(2, 16, heads=16, kv=2, d=128, seed=11, **bf16)
    _, kp, vp = _qkv(2, 100, heads=16, kv=2, d=128, seed=12, **bf16)
    kp, vp = kp.transpose(1, 2).contiguous(), vp.transpose(1, 2).contiguous()
    pinned = torch.tensor([3, 0], device="cuda")
    g = torch.Generator(device="cuda").manual_seed(13)
    qv, kv_, vv = (torch.randn(200, 16, 80, generator=g, device="cuda").bfloat16() for _ in range(3))

    def run():
        return (flash.causal_attention(q, k, v), flash.action_attention(q, k, v, kp, vp, pinned),
                flash.varlen_attention(qv, kv_, vv, [0, 50, 200]))

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream), torch.inference_mode():
        eager = run()  # creates the cached segment tensor
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured = run()
        pinned.fill_(5)  # a device value: the captured graph follows it
        graph.replay()
        moved = flash.action_attention(q, k, v, kp, vp, pinned)
    stream.synchronize()
    assert torch.equal(captured[0], eager[0]) and torch.equal(captured[2], eager[2])
    assert torch.equal(captured[1], moved) and not torch.equal(moved, eager[1])


def test_action_attention_reference_ignores_left_padding():
    q, k, v = _qkv(2, 6, seed=3)
    _, kp, vp = _qkv(2, 5, seed=4)
    kp, vp = kp.transpose(1, 2).contiguous(), vp.transpose(1, 2).contiguous()  # [B, KV, 5, D]
    pinned = torch.tensor([2, 0])
    lengths = torch.tensor([3, 5])  # row 0 uses only its last 3 prompt columns
    padded = reference.action_attention(q, k, v, kp, vp, pinned, lengths)
    alone = reference.action_attention(q[:1], k[:1], v[:1], kp[:1, :, 2:], vp[:1, :, 2:], pinned[:1])
    torch.testing.assert_close(padded[:1], alone, rtol=1e-5, atol=1e-6)
    garbage_k, garbage_v = kp.clone(), vp.clone()
    garbage_k[0, :, :2], garbage_v[0, :, :2] = 50.0, -50.0
    torch.testing.assert_close(reference.action_attention(q, k, v, garbage_k, garbage_v, pinned, lengths)[:1],
                               padded[:1], rtol=0, atol=0)


@pytest.mark.gpu
def test_flash_attn_left_padded_rows_equal_their_unpadded_selves(flash):
    bf16 = dict(dtype=torch.bfloat16, device="cuda")
    prompts, n = [756, 761, 700], 16
    rows = []
    for i, p in enumerate(prompts):
        q, k, v = _qkv(1, n, heads=16, kv=2, d=128, seed=20 + i, **bf16)
        _, kp, vp = _qkv(1, p, heads=16, kv=2, d=128, seed=30 + i, **bf16)
        rows.append((q, k, v, kp.transpose(1, 2).contiguous(), vp.transpose(1, 2).contiguous()))
    pinned = torch.tensor([5, 0, 8], device="cuda")
    pmax = max(prompts)
    kp_pad = torch.full((3, 2, pmax, 128), 7.0, **bf16)  # padding content must not matter
    vp_pad = torch.full((3, 2, pmax, 128), -7.0, **bf16)
    for i, (row, p) in enumerate(zip(rows, prompts)):
        kp_pad[i, :, pmax - p:], vp_pad[i, :, pmax - p:] = row[3][0], row[4][0]
    q, k, v = (torch.cat([r[j] for r in rows]) for j in range(3))
    lengths = torch.tensor(prompts, device="cuda", dtype=torch.int32)
    batched = flash.action_attention(q, k, v, kp_pad, vp_pad, pinned, lengths)
    for i, row in enumerate(rows):
        alone = flash.action_attention(*row, pinned[i:i + 1])
        _close(batched[i:i + 1], alone)
    _close(batched, reference.action_attention(q, k, v, kp_pad, vp_pad, pinned, lengths))


@pytest.mark.gpu
def test_flash_attn_prefix_and_kvcache_forms_match_reference(flash):
    """8 heads, 1 K/V head, head_dim 256; right-padded prefixes inside left-padded batches."""
    bf16 = dict(dtype=torch.bfloat16, device="cuda")
    q, k, v = _qkv(1, 848, heads=8, kv=1, d=256, seed=40, **bf16)
    valid = torch.tensor([835], device="cuda")
    prefix = flash.prefix_attention(q, k, v, valid)
    _close(prefix[:, :835], reference.prefix_attention(q, k, v, valid)[:, :835])
    widths, valids, n = [848, 864, 968], [835, 850, 840], 30
    rows = []
    for i, (p, m) in enumerate(zip(widths, valids)):
        qa, ka, va = _qkv(1, n, heads=8, kv=1, d=256, seed=50 + i, **bf16)
        _, kc, vc = _qkv(1, p + n, heads=8, kv=1, d=256, seed=60 + i, **bf16)
        kc[:, m:], vc[:, m:] = 9.0, -9.0  # right padding inside the prefix and the free columns: never keys
        rows.append((qa, ka, va, kc, vc, m))
    smax = max(widths) + n
    kc_pad = torch.full((3, smax, 1, 256), 7.0, **bf16)
    vc_pad = torch.full((3, smax, 1, 256), -7.0, **bf16)
    for i, (row, p) in enumerate(zip(rows, widths)):
        kc_pad[i, smax - p - n:], vc_pad[i, smax - p - n:] = row[3][0], row[4][0]
    q, k, v = (torch.cat([r[j] for r in rows]) for j in range(3))
    start = torch.tensor([smax - p - n for p in widths], device="cuda", dtype=torch.int32)
    valid = torch.tensor(valids, device="cuda", dtype=torch.int32)
    batched = flash.kvcache_attention(q, k, v, kc_pad.clone(), vc_pad.clone(), start, valid)
    for i, (qa, ka, va, kc, vc, m) in enumerate(rows):
        alone = flash.kvcache_attention(qa, ka, va, kc.clone(), vc.clone(), torch.zeros_like(valid[:1]), valid[i:i + 1])
        _close(batched[i:i + 1], alone)
    _close(batched, reference.kvcache_attention(q, k, v, kc_pad.clone(), vc_pad.clone(), start, valid))
    # Same bits as the per-row action_attention on the same keys.
    qa, ka, va, kc, vc, m = rows[0]
    zero = torch.zeros(1, dtype=torch.int64, device="cuda")
    per_row = flash.action_attention(qa, ka, va, kc[:, :m].transpose(1, 2).contiguous(),
                                     vc[:, :m].transpose(1, 2).contiguous(), zero)
    assert torch.equal(flash.kvcache_attention(qa, ka, va, kc.clone(), vc.clone(), torch.zeros_like(valid[:1]),
                                               valid[:1]), per_row)
