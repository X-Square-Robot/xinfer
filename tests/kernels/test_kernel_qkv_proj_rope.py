"""qkv_proj_rope against FP32 torch: projection, rotate-half RoPE on Q and K, KV-cache writes."""
import pytest
import torch
import torch.nn.functional as F

from xinfer.ops.backends.cuda import gemm
from xinfer.ops.layout import mxfp8 as layout
from conftest import DEVICE, close


def angles(m, rotary_dim):
    position = torch.arange(m, dtype=torch.float32)[:, None]
    frequency = 1.0 / (10000 ** (torch.arange(rotary_dim // 2, dtype=torch.float32) / (rotary_dim // 2)))
    return torch.cos(position * frequency).to(DEVICE), torch.sin(position * frequency).to(DEVICE)


def reference(x, weight, bias, cos, sin, heads, kv_heads, d):
    y = F.linear(x.float(), weight.float(), bias.float())
    m, half = y.shape[0], cos.shape[1]
    qk = y[:, : (heads + kv_heads) * d].view(m, heads + kv_heads, d).clone()
    x1, x2 = qk[..., :half].clone(), qk[..., half : 2 * half].clone()
    qk[..., :half] = x1 * cos[:, None] - x2 * sin[:, None]
    qk[..., half : 2 * half] = x2 * cos[:, None] + x1 * sin[:, None]
    return torch.cat([qk.view(m, -1), y[:, (heads + kv_heads) * d :]], dim=1)


@pytest.mark.parametrize("m,k,heads,kv_heads,d", [(5, 64, 2, 2, 16), (300, 1280, 16, 16, 80), (16, 1024, 16, 2, 128)])
@pytest.mark.parametrize("table", [False, True])
def test_standard_output(m, k, heads, kv_heads, d, table, rnd):
    n = (heads + 2 * kv_heads) * d
    x, weight, bias = rnd(m, k), rnd(n, k, scale=0.05), rnd(n)
    cos, sin = angles(m, d)
    tables = dict(rope_table=layout.rope_table(cos, sin)) if table else dict(cos=cos, sin=sin)
    out = gemm.qkv_proj_rope(x, weight, bias, num_heads=heads, num_kv_heads=kv_heads, head_dim=d, **tables)
    assert close(out, reference(x, weight, bias, cos, sin, heads, kv_heads, d))


@pytest.mark.parametrize("head_major", [True, False])
@pytest.mark.parametrize("m,start", [(16, 40), (300, 0)])
def test_kv_cache(m, start, head_major, rnd):
    k, heads, kv_heads, d, rows = 1024, 16, 2, 128, 400
    n = (heads + 2 * kv_heads) * d
    x, weight, bias = rnd(m, k), rnd(n, k, scale=0.05), rnd(n)
    cos, sin = angles(m, d)
    shape = (kv_heads, rows, d) if head_major else (rows, kv_heads * d)
    k_cache = torch.full(shape, 7.0, dtype=torch.bfloat16, device=DEVICE)
    v_cache = torch.full(shape, 7.0, dtype=torch.bfloat16, device=DEVICE)
    q = gemm.qkv_proj_rope(x, weight, bias, cos=cos, sin=sin, num_heads=heads, num_kv_heads=kv_heads, head_dim=d,
                         kv_cache=(k_cache, v_cache, start))
    full = gemm.qkv_proj_rope(x, weight, bias, cos=cos, sin=sin, num_heads=heads, num_kv_heads=kv_heads, head_dim=d)
    assert torch.equal(q, full[:, : heads * d])
    for cache, columns in ((k_cache, full[:, heads * d : (heads + kv_heads) * d]),
                           (v_cache, full[:, (heads + kv_heads) * d :])):
        written = cache[:, start : start + m].transpose(0, 1).reshape(m, -1) if head_major else cache[start : start + m]
        assert torch.equal(written, columns)
        untouched = torch.cat([cache[:, :start], cache[:, start + m :]], 1) if head_major else \
            torch.cat([cache[:start], cache[start + m :]])
        assert bool((untouched == 7.0).all())


@pytest.mark.parametrize("zero_centered", [False, True])
@pytest.mark.parametrize("cache", [False, True])
def test_qk_norm_partial_rope_and_extra_rows(zero_centered, cache, rnd):
    """Heads of 256 rotated on their first 64 dims after a per-head RMSNorm of Q and K, with gate rows passed on."""
    m, k, heads, kv_heads, d, rotary, extra, rows = 16, 1024, 8, 2, 256, 64, 2048, 480
    n = (heads + 2 * kv_heads) * d
    x, weight = rnd(m, k), rnd(n + extra, k, scale=0.05)
    q_weight, k_weight = rnd(d, scale=0.2), rnd(d, scale=0.2)
    cos, sin = angles(m, rotary)
    y = F.linear(x.float(), weight.float())
    qk = y[:, : (heads + kv_heads) * d].view(m, heads + kv_heads, d)
    w = torch.cat([q_weight.float().expand(heads, d), k_weight.float().expand(kv_heads, d)])
    qk = qk * torch.rsqrt(qk.pow(2).mean(-1, keepdim=True) + 1e-6) * (w + 1 if zero_centered else w)
    half = rotary // 2
    x1, x2 = qk[..., :half], qk[..., half:rotary]
    qk = torch.cat([x1 * cos[:, None] - x2 * sin[:, None], x2 * cos[:, None] + x1 * sin[:, None], qk[..., rotary:]], -1)
    expected = torch.cat([qk.reshape(m, -1), y[:, (heads + kv_heads) * d :]], dim=1)
    options = dict(cos=cos, sin=sin, rotary_dim=rotary, num_heads=heads, num_kv_heads=kv_heads, head_dim=d,
                   qk_norm=gemm.QkNorm(q_weight, k_weight, 1e-6, zero_centered))
    if not cache:
        assert close(gemm.qkv_proj_rope(x, weight, **options), expected)
        return
    k_cache = torch.zeros(kv_heads, rows, d, dtype=torch.bfloat16, device=DEVICE)
    v_cache = torch.zeros_like(k_cache)
    out = gemm.qkv_proj_rope(x, weight, kv_cache=(k_cache, v_cache, rows - m), **options)
    assert tuple(out.shape) == (m, heads * d + extra)
    assert close(out, torch.cat([expected[:, : heads * d], expected[:, n:]], dim=1))
    assert close(k_cache[:, rows - m :].transpose(0, 1).reshape(m, -1), expected[:, heads * d : (heads + kv_heads) * d])
    assert close(v_cache[:, rows - m :].transpose(0, 1).reshape(m, -1), expected[:, (heads + kv_heads) * d : n])


def test_errors(rnd):
    x, weight, bias = rnd(4, 64), rnd(96, 64), rnd(96)
    cos, sin = angles(4, 16)
    with pytest.raises(ValueError):
        gemm.qkv_proj_rope(x, weight, bias, num_heads=2, head_dim=16)  # no RoPE tables
    with pytest.raises(ValueError):
        gemm.qkv_proj_rope(x, weight, bias, cos=cos, sin=sin, num_heads=3, head_dim=16)  # rows do not match
