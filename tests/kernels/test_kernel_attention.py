"""attention_kvcache and attention_varlen against masked softmax attention in FP32 torch."""
import math

import pytest
import torch

from xinfer.ops.backends.cuda import attn
from conftest import DEVICE, close, sm


def reference(q, k_cache, v_cache, prefix_len, new_pos, flash_mask):
    b, m, heads, d = q.shape
    rows, kv_heads = k_cache.shape[1], k_cache.shape[2]
    group = heads // kv_heads
    k = k_cache.float().repeat_interleave(group, dim=2).permute(0, 2, 1, 3)  # [B, H, L, D]
    v = v_cache.float().repeat_interleave(group, dim=2).permute(0, 2, 1, 3)
    scores = q.float().permute(0, 2, 1, 3) @ k.transpose(-1, -2) / math.sqrt(d)  # [B, H, M, L]
    column, t = torch.arange(rows, device=DEVICE), torch.arange(m, device=DEVICE)
    keep = torch.zeros(b, 1, m, rows, dtype=torch.bool, device=DEVICE)
    for i in range(b):
        start = int(prefix_len[i]) if new_pos is None else new_pos
        keep[i, 0] = ((column < int(prefix_len[i])) | ((column >= start) & (column < start + m)))[None]
    if flash_mask is not None:  # query row t skips key j when t >= LTS[j] or t < UTE[j]
        lts, ute = flash_mask[..., 0], flash_mask[..., 1]  # [B, 1, L]
        keep = keep & (t[None, None, :, None] < lts[:, :, None, :]) & (t[None, None, :, None] >= ute[:, :, None, :])
    out = torch.softmax(scores.masked_fill(~keep, float("-inf")), dim=-1) @ v
    return out.permute(0, 2, 1, 3).reshape(b, m, heads * d)


@sm(120)
@pytest.mark.parametrize("b,m,rows,prefix", [(1, 16, 784, 768), (1, 1, 64, 40), (2, 32, 600, 500), (1, 16, 1984, 1900)])
@pytest.mark.parametrize("heads,d", [(16, 128), (8, 256)])
@pytest.mark.parametrize("masked", [False, True])
def test_matches_reference(b, m, rows, prefix, heads, d, masked, rnd):
    check(b, m, rows, prefix, heads, d, masked, rnd)


@sm(110)
@pytest.mark.parametrize("m,rows,prefix,heads,d", [
    (16, 784, 768, 16, 128), (1, 192, 170, 16, 128), (32, 448, 400, 16, 128), (8, 2048, 2000, 16, 128),
    (32, 256, 200, 8, 256), (16, 480, 464, 8, 256), (9, 1280, 1200, 8, 256), (4, 2048, 2000, 8, 256)])
@pytest.mark.parametrize("masked", [False, True])
def test_matches_reference_sm110(m, rows, prefix, heads, d, masked, rnd):
    check(1, m, rows, prefix, heads, d, masked, rnd)


@sm(110)
def test_sm110_takes_the_measured_boxes(rnd):
    """8 query / 2 KV heads of 256: fewer query rows the longer the cache; one request at a time."""
    assert attn.attention_kvcache_takes(rnd(1, 16, 8, 256), rnd(1, 832, 2, 256))
    assert not attn.attention_kvcache_takes(rnd(1, 17, 8, 256), rnd(1, 832, 2, 256))
    assert not attn.attention_kvcache_takes(rnd(1, 1, 8, 256), rnd(1, 2048, 2, 256))
    assert not attn.attention_kvcache_takes(rnd(2, 16, 16, 128), rnd(2, 784, 2, 128))


def check(b, m, rows, prefix, heads, d, masked, rnd):
    q, k_cache, v_cache = rnd(b, m, heads, d), rnd(b, rows, 2, d), rnd(b, rows, 2, d)
    prefix_len = torch.full((b,), prefix, dtype=torch.int32, device=DEVICE)
    new_pos = rows - m
    flash_mask = None
    if masked:
        column = torch.arange(rows, device=DEVICE)
        lts = torch.where((column < prefix - 7) | (column >= new_pos), m, 0)  # the last 7 prefix rows are padding
        ute = torch.where(column >= new_pos + 3, min(3, m - 1), 0)  # late new keys are hidden from the first rows
        flash_mask = torch.stack((lts, ute), -1).to(torch.int32)[None, None].expand(b, 1, rows, 2).contiguous()
    assert attn.attention_kvcache_takes(q, k_cache)
    out = attn.attention_kvcache(q, k_cache, v_cache, prefix_len=prefix_len, new_pos=new_pos, flash_mask=flash_mask)
    expected = reference(q, k_cache, v_cache, prefix_len, new_pos, flash_mask)
    assert close(out, expected)
    gate = rnd(b, m, heads * d)
    gated = attn.attention_kvcache(q, k_cache, v_cache, prefix_len=prefix_len, new_pos=new_pos, flash_mask=flash_mask,
                                 gate=gate)
    assert torch.equal(gated, (out.float() * torch.sigmoid(gate.float()).to(torch.bfloat16).float())
                       .to(torch.bfloat16))


@sm(120)
@pytest.mark.parametrize("lengths", [(308, 280, 280), (1024,), (64,) * 9, (5, 700, 131, 60), (37,)])
def test_varlen_matches_reference(lengths, rnd):
    total, heads, d = sum(lengths), 16, 64
    q, k, v = rnd(total, heads, d), rnd(total, heads, d), rnd(total, heads, d)
    bounds = [0]
    for n in lengths:
        bounds.append(bounds[-1] + n)
    assert attn.attention_varlen_takes(q, bounds)
    out = attn.attention_varlen(q, k, v, bounds)
    expected = torch.empty(total, heads * d, device=DEVICE)
    for start, end in zip(bounds[:-1], bounds[1:]):
        scores = q[start:end].float().transpose(0, 1) @ k[start:end].float().permute(1, 2, 0) / math.sqrt(d)
        expected[start:end] = (torch.softmax(scores, -1) @ v[start:end].float().transpose(0, 1)).transpose(0, 1) \
            .reshape(end - start, -1)
    assert close(out, expected)
    assert torch.equal(attn.attention_varlen(q, k, v, bounds), out)  # the same bits on a second call
    stream, graph = torch.cuda.Stream(), torch.cuda.CUDAGraph()
    replayed = torch.empty_like(out)
    with torch.cuda.stream(stream):
        attn.attention_varlen(q, k, v, bounds, out=replayed)  # the scratch memory of the capture stream
    with torch.cuda.graph(graph, stream=stream):
        attn.attention_varlen(q, k, v, bounds, out=replayed)
    replayed.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(replayed, out)


def test_varlen_takes_only_listed_calls(rnd):
    assert not attn.attention_varlen_takes(rnd(64, 12, 64), [0, 64])  # heads no kernel is tuned for
    assert not attn.attention_varlen_takes(rnd(64, 16, 64), [0, 32])  # bounds that do not end at T
    with pytest.raises(ValueError):
        attn.attention_varlen(rnd(64, 12, 64), rnd(64, 12, 64), rnd(64, 12, 64), [0, 64])


def test_takes_only_listed_calls(rnd):
    k_cache = rnd(1, 784, 2, 128)
    assert not attn.attention_kvcache_takes(rnd(1, 16, 6, 96), rnd(1, 784, 2, 96))  # heads no kernel is tuned for
    assert not attn.attention_kvcache_takes(rnd(1, 64, 16, 128), k_cache)  # more query rows than the kernel takes
    with pytest.raises(ValueError):
        attn.attention_kvcache(rnd(1, 64, 16, 128), k_cache, k_cache,
                             prefix_len=torch.zeros(1, dtype=torch.int32, device=DEVICE))
