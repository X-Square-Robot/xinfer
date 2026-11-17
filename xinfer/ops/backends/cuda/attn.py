"""Attention operators on CUDA.

flash-attn 2 (flash-attn 2.8.3.post1), registered when the ``flash_attn`` package and its ``flash_attn_2_cuda``
extension import (``XINFER_FLASH_ATTN=0``: never):

| operator | flash-attn call |
|---|---|
| ``varlen_attention`` | ``flash_attn_2_cuda.varlen_fwd`` (the ``flash_attn_varlen_func`` kernel), segments from ``cu_seqlens``: the Wall-X 4B service's ViT call |
| ``causal_attention`` | ``flash_attn_with_kvcache`` with the keys as the cache (``cache_seqlens = L``, ``cache_batch_idx``), ``causal=True``, ``num_splits=1``: the kernel of the service's Stage-1 call |
| ``action_attention`` | ``flash_attn_with_kvcache`` over ``[prompt | own]`` keys, one decode-style row per query (``cache_batch_idx``, ``seqlen_q = 1``, per-row ``cache_seqlens``), ``num_splits=4`` |
| ``prefix_attention`` | ``flash_attn_with_kvcache`` with the keys as the cache, ``cache_seqlens`` = valid length, non-causal, ``num_splits=1`` |
| ``kvcache_attention`` | ``flash_attn_with_kvcache`` appending ``k``/``v`` at ``cache_seqlens`` in the kernel, one sequence per batch row (``cache_leftpad``), non-causal, ``num_splits=4`` |

The first two reproduce the service's attention bit for bit;
``flash_attn_func`` uses another kernel configuration and would not.
For ``action_attention`` the service made two ``num_splits=1`` calls (all
keys, and ``P + pinned`` keys for the pinned rows) with only 16 CTAs on the
Wall-X shapes, which is slower than the torch reference. The per-row form
with a fixed split count is about 4x faster (22 vs 92 µs per call at B1).
Left-padded prompts (``prompt_lengths``) use ``cache_leftpad``.

Contracts against the torch references: tolerance (different accumulation
order), deterministic, graph safe. ``varlen_attention``
takes ``cu_seqlens`` as a host sequence; its device copy is created on the
first (eager) call for each layout and reused, also inside captured graphs.

The kernels of this backend (``_native``), each on the shapes it is tuned and tested on (``*_takes``); the other
calls run on the reference:

| operator | function here |
|---|---|
| ``prefix_kvcache_attention`` | ``attention_kvcache`` |
| ``segment_attention`` | ``attention_varlen`` |
| ``gated_kvcache_attention`` | ``attention_kvcache`` (``gate``) |
| ``gated_delta_net`` | ``gated_delta_rule_prefill`` (one sequence) |

``attention_kvcache``: a few new tokens over a KV cache. The keys of request ``b`` are the cache rows
``[0, prefix_len[b])`` and ``[p, p + M)``, with ``p = new_pos`` or ``prefix_len[b]`` when ``new_pos`` is None;
the K and V of the new tokens are already in the cache. The kernel splits the keys into chunks, one CTA per
(head group, chunk) computing a partial softmax, and a second kernel merges the partials; the chunk size depends
on the cache length only.

``attention_varlen``: bidirectional attention inside each segment of a packed sequence (the images of a vision
tower). One persistent CTA per SM with two teams of warps; the (segment, head, 64 query rows) units are spread
over the SMs by their key count on the host, and a unit cut between two SMs is merged by whichever half finishes
last, so no CTA waits for another.

``gated_delta_rule_prefill``: the gated delta rule (a linear-attention mixer) over one sequence, from a zero
state, from the mixer's input projections to its normalized output: a causal depthwise convolution (4 taps) + SiLU
on ``qkv``, L2-normalized q and k, the delta-rule recurrence in chunks of 32 rows, and a gated RMSNorm with ``z``.
Three kernels chained on the stream; every element is summed in an order that depends on the shape only, so results
repeat bit for bit.

Contracts against the torch references: tolerance.
"""

from __future__ import annotations

import collections
import math
import os

import torch

from xinfer.ops import reference
from xinfer.ops.backends.cuda._native import BUILD_ARCHS, NATIVE, arch, capturing, lib, scratch
from xinfer.ops.reference import attn as reference_attn
from xinfer.ops.registry import register

flash_attn = flash_attn_2_cuda = None
if os.environ.get("XINFER_FLASH_ATTN", "1") != "0":
    try:
        import flash_attn
        import flash_attn_2_cuda
    except Exception:  # noqa: BLE001 - not installed, or no binary for this CUDA: the references run
        flash_attn = flash_attn_2_cuda = None

ACTION_SPLITS = 4  # fixed split-KV count of action_attention

_segments = {}  # (host boundaries, device) -> (device int32 boundaries, longest segment)


def _device_segments(cu_seqlens, device):
    """Device ``int32`` boundaries and the longest segment for a host layout (cached per layout).

    The device tensor is created on the first call for a layout, which must be eager (not under capture).

    Args:
        cu_seqlens: segment boundaries as a host sequence or tensor.
        device: device of the boundary tensor.

    Returns:
        ``(cu_seqlens_device, longest)``: int32 ``[S + 1]`` boundaries on ``device`` and the longest segment length.
    """
    bounds = tuple(int(b) for b in (cu_seqlens.tolist() if torch.is_tensor(cu_seqlens) else cu_seqlens))
    key = (bounds, device)
    if key not in _segments:
        _segments[key] = (torch.tensor(bounds, dtype=torch.int32, device=device),
                          max(e - s for s, e in zip(bounds[:-1], bounds[1:])))
    return _segments[key]


@register("varlen_attention", backend="cuda", requires=flash_attn)
def varlen_attention(q, k, v, cu_seqlens, *, scale=None):
    """``varlen_attention`` via ``flash_attn_2_cuda.varlen_fwd`` (the Wall-X 4B service's ViT call).

    Non-causal attention within each segment, no dropout, no window, no softcap.

    Args:
        q: ``[T, H, D]`` packed queries (FP16/BF16).
        k: ``[T, KV, D]`` packed keys.
        v: ``[T, KV, D]`` packed values.
        cu_seqlens: segment boundaries as a host sequence (or tensor); see ``_device_segments``.
        scale: softmax scale; None for ``1/sqrt(D)``.

    Returns:
        ``[T, H, D]`` in ``q.dtype``.
    """
    cu_seqlens_device, longest = _device_segments(cu_seqlens, q.device)
    scale = q.shape[-1] ** -0.5 if scale is None else scale
    # (q, k, v, out, cu_q, cu_k, seqused_k, leftpad_k, block_table, alibi, max_q, max_k, dropout,
    #  scale, zero_tensors, causal, window_left, window_right, softcap, return_softmax, generator)
    return flash_attn_2_cuda.varlen_fwd(q, k, v, None, cu_seqlens_device, cu_seqlens_device, None, None, None,
                                        None, longest, longest, 0.0, scale, False, False, -1, -1, 0.0, False,
                                        None)[0]


@register("causal_attention", backend="cuda", requires=flash_attn)
def causal_attention(q, k, v, *, scale=None, return_bnsd=False):
    """``causal_attention`` via ``flash_attn_with_kvcache`` (the Wall-X 4B service's Stage-1 kernel).

    The service appended ``k``/``v`` to empty caches with ``num_splits=1``. Here ``k``/``v`` are the
    caches (``cache_seqlens = L``), so no key is written; ``cache_batch_idx`` keeps flash-attn on
    the same split-KV kernel, which gives the service's bits without its per-CTA rewrite of the keys
    (68 -> 28 µs at L = 768, 16/2/128 heads, RTX 5090). Without ``cache_batch_idx`` flash-attn runs
    another kernel configuration and the bits differ.

    Args:
        q: ``[B, L, H, D]`` queries (FP16/BF16).
        k: ``[B, L, KV, D]`` keys.
        v: ``[B, L, KV, D]`` values.
        scale: softmax scale; None for ``1/sqrt(D)``.
        return_bnsd: Return the head-major ``[B, H, L, D]`` form instead of the transposed layout.

    Returns:
        ``[B, L, H, D]`` in ``q.dtype``, or ``[B, H, L, D]`` with ``return_bnsd``.
    """
    batch_size, length = q.shape[:2]
    rows = torch.arange(batch_size, dtype=torch.int32, device=q.device)
    out = flash_attn.flash_attn_with_kvcache(q, k, v, cache_seqlens=length, cache_batch_idx=rows,
                                             softmax_scale=scale, causal=True, num_splits=1)
    return out.transpose(1, 2) if return_bnsd else out


@register("prefix_attention", backend="cuda", requires=flash_attn)
def prefix_attention(q, k, v, valid, *, scale=None):
    """``prefix_attention`` via ``flash_attn_with_kvcache`` with the keys as the cache.

    ``cache_seqlens = valid`` limits every query row to its row's valid keys (non-causal,
    ``num_splits=1``); padded query rows attend to the same keys.

    Args:
        q: ``[B, S, H, D]`` queries (FP16/BF16).
        k: ``[B, S, KV, D]`` keys.
        v: ``[B, S, KV, D]`` values.
        valid: ``[B]`` integer tensor, the number of leading valid positions per row.
        scale: softmax scale; None for ``1/sqrt(D)``.

    Returns:
        ``[B, S, H, D]`` in ``q.dtype``.
    """
    return flash_attn.flash_attn_with_kvcache(q, k, v, cache_seqlens=valid.reshape(-1).to(torch.int32),
                                              softmax_scale=scale, causal=False, num_splits=1)


@register("kvcache_attention", backend="cuda", requires=flash_attn)
def kvcache_attention(q, k, v, k_cache, v_cache, cache_start, cache_valid, *, scale=None):
    """``kvcache_attention`` as one ``flash_attn_with_kvcache`` call that appends ``k``/``v`` in the kernel.

    Each batch row is one sequence: ``cache_leftpad = cache_start``, ``cache_seqlens = start + valid`` (the
    append position; the kernel writes ``k``/``v`` there and attends to ``[start, start + valid + L)``),
    non-causal, ``num_splits = ACTION_SPLITS``. A row's split-KV boundaries follow its own key range.

    Args:
        q: ``[B, L, H, D]`` queries (FP16/BF16).
        k: ``[B, L, KV, D]`` own keys.
        v: ``[B, L, KV, D]`` own values.
        k_cache: ``[B, S, KV, D]`` key cache, last dim contiguous; modified in place.
        v_cache: ``[B, S, KV, D]`` value cache; modified in place.
        cache_start: ``[B]`` integer tensor, each row's first cached column.
        cache_valid: ``[B]`` integer tensor, each row's number of cached keys.
        scale: softmax scale; None for ``1/sqrt(D)``.

    Returns:
        ``[B, L, H, D]`` in ``q.dtype``.
    """
    start = cache_start.reshape(-1).to(torch.int32)
    return flash_attn.flash_attn_with_kvcache(q, k_cache, v_cache, k=k, v=v,
                                              cache_seqlens=start + cache_valid.reshape(-1).to(torch.int32),
                                              cache_leftpad=start, softmax_scale=scale, causal=False,
                                              num_splits=ACTION_SPLITS)


@register("action_attention", backend="cuda", requires=flash_attn)
def action_attention(q, k, v, k_prompt, v_prompt, num_pinned, prompt_lengths=None, *, scale=None):
    """``action_attention`` as one decode-style ``flash_attn_with_kvcache`` call per query row.

    Builds per-batch caches ``[prompt | own]`` (``[B, P + L, KV, D]``) and runs the ``B * L`` query rows as
    a batch of single-token queries: ``cache_batch_idx`` maps each to its batch row, ``cache_seqlens`` ends
    at ``P + num_pinned[b]`` for pinned rows and ``P + L`` otherwise, and ``cache_leftpad`` skips the left
    padding of short prompts. ``num_splits`` is the constant ``ACTION_SPLITS``, so a row's split-KV
    boundaries depend on its own key range only.

    Args:
        q: ``[B, L, H, D]`` queries of the own tokens (FP16/BF16).
        k: ``[B, L, KV, D]`` own keys.
        v: ``[B, L, KV, D]`` own values.
        k_prompt: ``[B, KV, P, D]`` prompt keys (head-major).
        v_prompt: ``[B, KV, P, D]`` prompt values.
        num_pinned: ``[B]`` integer tensor, the number of pinned rows per batch row.
        prompt_lengths: optional ``[B]`` integer tensor; row ``b``'s prompt is the last ``prompt_lengths[b]``
            of the ``P`` columns.
        scale: softmax scale; None for ``1/sqrt(D)``.

    Returns:
        ``[B, L, H, D]`` in ``q.dtype``.
    """
    b, n, heads, d = q.shape
    p = k_prompt.shape[2]
    k_cache = torch.cat([k_prompt.transpose(1, 2), k], dim=1)  # [B, P + L, KV, D]
    v_cache = torch.cat([v_prompt.transpose(1, 2), v], dim=1)
    rows = torch.arange(n, device=q.device)[None]
    num_pinned = num_pinned.reshape(b, 1)
    own = torch.where(rows < num_pinned, num_pinned, torch.full_like(rows, n))  # own keys each row sees
    ends = (own + p).to(torch.int32).reshape(-1)  # cache_seqlens: absolute end index
    batch_of_row = torch.arange(b, dtype=torch.int32, device=q.device).repeat_interleave(n)
    leftpad = None
    if prompt_lengths is not None:  # left-padded prompts start at P - length
        leftpad = (p - prompt_lengths.reshape(b).to(torch.int32)).repeat_interleave(n)
    out = flash_attn.flash_attn_with_kvcache(q.reshape(b * n, 1, heads, d), k_cache, v_cache,
                                             cache_seqlens=ends, cache_batch_idx=batch_of_row,
                                             cache_leftpad=leftpad, softmax_scale=scale, causal=False,
                                             num_splits=ACTION_SPLITS)
    return out.reshape(b, n, heads, d)


# ---------------------------------------------------------------------------
# Kernels of this backend
# ---------------------------------------------------------------------------
# arch -> (query heads, KV heads, head dim) -> inclusive ranges of batch, M (query rows) and L (cache rows); a list of
# such boxes where the measured ranges are not one box
KVCACHE_SHAPES = {
    120: {(16, 2, 128): [dict(batch=(1, 8), m=(1, 32), l=(64, 1984))],
          (8, 2, 256): [dict(batch=(1, 2), m=(1, 32), l=(64, 1984))]},
    110: {(16, 2, 128): [dict(batch=(1, 1), m=(1, 32), l=(192, 2048))],
          (8, 2, 256): [dict(batch=(1, 1), m=(1, 32), l=(128, 256)), dict(batch=(1, 1), m=(1, 16), l=(257, 832)),
                        dict(batch=(1, 1), m=(1, 9), l=(833, 1280)), dict(batch=(1, 1), m=(2, 4), l=(1281, 2048))]},
}
# arch -> (query heads, KV heads, head dim) -> (L, keys per chunk up to that L, keys per chunk beyond): shorter caches
# use smaller chunks so that every SM has work
KVCACHE_CHUNKS = {
    120: {(16, 2, 128): (576, 32, 64), (8, 2, 256): (576, 32, 64)},
    110: {(16, 2, 128): (448, 64, 128), (8, 2, 256): (576, 32, 64)},
}


def _chunk(device_arch, heads, cache_rows):
    limit, short, long = KVCACHE_CHUNKS[device_arch][heads]
    return short if cache_rows <= limit else long


def attention_kvcache_takes(q, k_cache) -> bool:
    """Whether :func:`attention_kvcache` runs ``q`` ``[B, M, H, D]`` over ``k_cache`` ``[B, L, H_kv, D]`` here."""
    if q.dim() != 4 or k_cache.dim() != 4 or q.dtype != torch.bfloat16:
        return False
    device_arch = arch(q.device)
    b, m, heads, d = q.shape
    boxes = KVCACHE_SHAPES.get(device_arch, {}).get((heads, k_cache.shape[2], d)) if device_arch in BUILD_ARCHS else None
    return any(all(lo <= v <= hi for v, (lo, hi) in ((b, box["batch"]), (m, box["m"]), (k_cache.shape[1], box["l"])))
               for box in boxes or ())


def attention_kvcache(q, k_cache, v_cache, *, prefix_len, new_pos=None, flash_mask=None, softmax_scale=None,
                      gate=None, out=None):
    """Attention of ``q`` ``[B, M, H, D]`` over ``k_cache`` / ``v_cache`` ``[B, L, H_kv, D]`` (BF16, the last
    dimension dense); BF16 ``[B, M, H * D]``.

    ``prefix_len``: int32 ``[B]`` on the device. ``new_pos``: an int, the same for every request, or None.
    ``flash_mask``: int32 ``[B, 1 | H_kv, L, 2]`` holding (LTS, UTE) per cache row: query row ``t`` skips the key
    when ``t >= LTS`` or ``t < UTE``. ``softmax_scale`` defaults to ``1 / sqrt(D)``. ``gate``: BF16
    ``[B, M, H * D]`` (the last dimension dense); the result is multiplied by ``bf16(sigmoid(gate))`` in a second
    kernel.
    """
    if not attention_kvcache_takes(q, k_cache):
        raise ValueError(f"attention_kvcache: no kernel for q {tuple(q.shape)} over a cache {tuple(k_cache.shape)} "
                         f"on SM{arch(q.device)} (see attention_kvcache_takes)")
    b, m, heads, d = q.shape
    cache_rows, kv_heads = k_cache.shape[1], k_cache.shape[2]
    if tuple(v_cache.shape) != tuple(k_cache.shape) or k_cache.shape[0] != b or k_cache.shape[3] != d:
        raise ValueError("attention_kvcache: k_cache and v_cache must be [B, L, H_kv, D]")
    for t in (q, k_cache, v_cache):
        if t.dtype != torch.bfloat16 or t.stride(-1) != 1:
            raise ValueError("attention_kvcache: q and the caches must be BF16 with a dense last dimension")
    if prefix_len.dtype != torch.int32 or tuple(prefix_len.shape) != (b,):
        raise ValueError("attention_kvcache: prefix_len must be an int32 device tensor [B]")
    if new_pos is not None and not 0 <= int(new_pos) <= cache_rows - m:
        raise ValueError(f"attention_kvcache: new_pos={new_pos} needs 0 <= new_pos and new_pos + M <= L")
    if out is None:
        out = torch.empty(b, m, heads * d, dtype=torch.bfloat16, device=q.device)
    chunk = _chunk(arch(q.device), (heads, kv_heads, d), cache_rows)
    need = int(lib.attention_kvcache_splitkv_workspace_size(b, m, heads, kv_heads, d, cache_rows, chunk))
    lib.attention_kvcache_splitkv(
        out, None, k_cache, v_cache, scratch("attention_kvcache", need, q.device), q, prefix_len, None, None,
        flash_mask, None, None, -1 if new_pos is None else int(new_pos), False,
        1.0 / math.sqrt(d) if softmax_scale is None else float(softmax_scale), chunk, 0, False)
    if gate is not None:
        lib.sigmoid_gate_mul_inplace(out, gate, True)
    return out


# arch -> (heads, head dim) -> inclusive ranges of the segment count and of T (total rows)
VARLEN_SHAPES = {
    120: {(16, 64): dict(segments=(1, 64), t=(1, 4096))},
}

_plans = collections.OrderedDict()  # (device, heads, segment lengths) -> (pieces, team ranges) on the device
_plans_in_graph = set()  # plans a captured graph points at: never dropped
_PLANS_MAX = 256
_varlen_scratch = {}  # (device, stream) -> (FP32 partial sums, zeroed int32 pair counters the kernel leaves zeroed)


def _lengths(segments):
    bounds = [int(b) for b in (segments.tolist() if torch.is_tensor(segments) else segments)]
    return tuple(b - a for a, b in zip(bounds[:-1], bounds[1:])), bounds


def attention_varlen_takes(q, segments) -> bool:
    """Whether :func:`attention_varlen` runs ``q`` ``[T, H, D]`` with these segment bounds here."""
    if q.dim() != 3 or q.dtype != torch.bfloat16:
        return False
    device_arch = arch(q.device)
    ranges = VARLEN_SHAPES.get(device_arch, {}).get(tuple(q.shape[1:])) if device_arch in BUILD_ARCHS else None
    if ranges is None:
        return False
    lengths, bounds = _lengths(segments)
    if not lengths or bounds[0] != 0 or bounds[-1] != q.shape[0] or min(lengths) < 1:
        return False
    return all(lo <= v <= hi for v, (lo, hi) in ((len(lengths), ranges["segments"]), (q.shape[0], ranges["t"])))


def _plan(lengths, heads, sms):
    """Pieces (first row of the segment, its length, head, first query row, first and last key block, part,
    pair id) and the piece ranges of the ``2 * sms`` teams.

    A unit is 64 query rows of one (segment, head) and costs its key blocks (64 keys each). The units, largest
    first, are cut into ``sms`` equal runs of key blocks; a unit crossing a boundary becomes a front and a back
    piece (never more than two), and the pieces of an SM go to its two teams by longest-processing-time-first."""
    starts, total = [], 0
    for n in lengths:
        starts.append(total)
        total += n
    units = [[starts[s], n, h, row, 0, -(-n // 64), -1, 0]
             for h in range(heads) for s, n in enumerate(lengths) for row in range(0, n, 64)]
    units.sort(key=lambda u: -u[5])  # stable
    blocks = sum(u[5] for u in units)
    per_sm = [[] for _ in range(sms)]
    done = unit = offset = pairs = 0
    for sm in range(sms):
        end = (sm + 1) * blocks // sms
        while done < end and unit < len(units):
            u = units[unit]
            # a unit is cut at most once: its back piece takes all that is left, even past this SM's share
            take = u[5] - offset if offset > 0 else min(u[5], end - done)
            piece = list(u)
            piece[4], piece[5] = offset, offset + take
            if take < u[5] or offset > 0:
                piece[6] = 0 if offset == 0 else 1
                piece[7] = pairs if offset == 0 else pairs - 1
                pairs += offset == 0
            per_sm[sm].append(piece)
            done += take
            offset += take
            if offset == u[5]:
                offset, unit = 0, unit + 1
    pieces, team_start = [], [0]
    for mine in per_sm:
        teams, load = ([], []), [0, 0]
        for piece in sorted(mine, key=lambda p: -(p[5] - p[4])):
            team = 0 if load[0] <= load[1] else 1
            teams[team].append(piece)
            load[team] += piece[5] - piece[4]
        for team in teams:
            pieces += team
            team_start.append(len(pieces))
    return pieces or [[0] * 8], team_start


def attention_varlen(q, k, v, segments, *, softmax_scale=None, out=None):
    """Attention of every row of ``q`` ``[T, H, D]`` over the ``k`` / ``v`` ``[T, H, D]`` rows of its own
    segment (BF16, the last dimension dense); BF16 ``[T, H * D]``.

    ``segments``: the ``n + 1`` row bounds of the ``n`` segments, ``0 .. T``, as a host sequence. The work plan
    of a segment layout and the scratch memory of a stream are made on their first call, which must not be inside
    a CUDA graph capture.
    """
    if not attention_varlen_takes(q, segments):
        raise ValueError(f"attention_varlen: no kernel for q {tuple(q.shape)} with segments {list(segments)} on "
                         f"SM{arch(q.device)} (see attention_varlen_takes)")
    if tuple(k.shape) != tuple(q.shape) or tuple(v.shape) != tuple(q.shape):
        raise ValueError("attention_varlen: q, k and v must have the same shape [T, H, D]")
    device = q.device
    total, heads, d = q.shape
    lengths, _ = _lengths(segments)
    in_capture = capturing(device)
    key = (device.index, heads, lengths)
    plan = _plans.get(key)
    if plan is None:
        if in_capture:
            raise RuntimeError("attention_varlen plans a segment layout on its first call, which must not "
                               "be inside a CUDA graph capture")
        pieces, team_start = _plan(lengths, heads, torch.cuda.get_device_properties(device).multi_processor_count)
        plan = _plans[key] = (torch.tensor(pieces, dtype=torch.int32).to(device),
                              torch.tensor(team_start, dtype=torch.int32).to(device))
        for old in [p for p in _plans if p not in _plans_in_graph][: max(0, len(_plans) - _PLANS_MAX)]:
            del _plans[old]
    else:
        _plans.move_to_end(key)
    if in_capture:
        _plans_in_graph.add(key)
    stream_key = (device.index, torch.cuda.current_stream(device).cuda_stream)
    buffers = _varlen_scratch.get(stream_key)
    if buffers is None:
        if in_capture:
            raise RuntimeError("attention_varlen needs scratch memory on a stream that is capturing a CUDA "
                               "graph; run it once before capture")
        sms = torch.cuda.get_device_properties(device).multi_processor_count
        buffers = _varlen_scratch[stream_key] = (torch.empty(sms * 2 * 128 * 36, dtype=torch.float32, device=device),
                                                 torch.zeros(sms, dtype=torch.int32, device=device))
    if out is None:
        out = torch.empty(total, heads * d, dtype=torch.bfloat16, device=device)
    lib.attention_varlen_lpt_bf16_sm120(out, q, k, v, plan[0], plan[1], buffers[0], buffers[1],
                                        1.0 / math.sqrt(d) if softmax_scale is None else float(softmax_scale))
    return out


HEAD_DIM = 128
# arch -> (key heads, value heads) -> inclusive range of the sequence length
DELTA_RULE_SHAPES = {
    120: {(16, 16): (1, 1024)},
    110: {(16, 16): (1, 1024)},
}


def _heads(qkv, zba, num_k_heads):
    value_heads, rest = divmod(zba.shape[-1], HEAD_DIM + 2)
    if rest or qkv.shape[-1] != (2 * num_k_heads + value_heads) * HEAD_DIM:
        return None
    return int(num_k_heads), value_heads


def gated_delta_rule_prefill_takes(qkv, zba, *, num_k_heads) -> bool:
    """Whether :func:`gated_delta_rule_prefill` runs ``qkv`` ``[S, ...]`` and ``zba`` ``[S, ...]`` here."""
    if qkv.dim() != 2 or zba.dim() != 2 or qkv.dtype != torch.bfloat16 or qkv.shape[0] != zba.shape[0]:
        return False
    device_arch = arch(qkv.device)
    lo, hi = DELTA_RULE_SHAPES.get(device_arch if device_arch in BUILD_ARCHS else None, {}).get(_heads(qkv, zba, num_k_heads),
                                                                                     (1, 0))
    return lo <= qkv.shape[0] <= hi


def gated_delta_rule_prefill(qkv, zba, conv_weight, A_log, dt_bias, norm_weight, *, num_k_heads, eps=1e-6, out=None):
    """``qkv`` BF16 ``[S, (2 H_k + H_v) * 128]``: the q, k and v inputs of the convolution. ``zba`` BF16
    ``[S, H_v * 130]``: per value head the gate input ``z`` (128), then the logits of the write strength and of
    the decay. ``conv_weight`` BF16 ``[(2 H_k + H_v) * 128, 4]``; ``A_log``, ``dt_bias`` FP32 ``[H_v]``
    (``decay = exp(-exp(A_log) * softplus(logit + dt_bias))``); ``norm_weight`` BF16 ``[128]``.

    Returns BF16 ``[S, H_v * 128]``.
    """
    if not gated_delta_rule_prefill_takes(qkv, zba, num_k_heads=num_k_heads):
        raise ValueError(f"gated_delta_rule_prefill: no kernel for qkv {tuple(qkv.shape)}, zba {tuple(zba.shape)}, "
                         f"num_k_heads={num_k_heads} on SM{arch(qkv.device)} (see gated_delta_rule_prefill_takes)")
    key_heads, value_heads = _heads(qkv, zba, num_k_heads)
    rows = qkv.shape[0]
    if out is None:
        out = torch.empty(rows, value_heads * HEAD_DIM, dtype=torch.bfloat16, device=qkv.device)
    need = int(lib.gated_delta_rule_prefill_workspace_size(0, rows, key_heads, value_heads))
    lib.gated_delta_rule_prefill(out, scratch("gated_delta_rule", need, qkv.device), qkv, zba, conv_weight, A_log,
                                 dt_bias, norm_weight, None, key_heads, 0, float(eps), False)
    return out


# ---------------------------------------------------------------------------
# Operators on them
# ---------------------------------------------------------------------------
@register("prefix_kvcache_attention", backend="cuda", **NATIVE)
def prefix_kvcache_attention(q, k_cache, v_cache, prefix_len, new_pos, *, flash_mask=None, scale=None,
                             all_keep=False, return_bnsd=False):
    if not attention_kvcache_takes(q, k_cache):
        return reference_attn.prefix_kvcache_attention(q, k_cache, v_cache, prefix_len, new_pos,
                                                       flash_mask=flash_mask, scale=scale,
                                                       all_keep=all_keep, return_bnsd=return_bnsd)
    out = attention_kvcache(q, k_cache, v_cache, prefix_len=prefix_len, new_pos=int(new_pos),
                                flash_mask=flash_mask, softmax_scale=scale)
    if return_bnsd:
        b, m, h, d = q.shape[:2] + q.shape[2:]
        return out.view(b, m, h, d).transpose(1, 2)
    return out


@register("segment_attention", backend="cuda", **NATIVE)
def segment_attention(q, k, v, cu_seqlens, *, scale=None):
    bounds = tuple(cu_seqlens.tolist() if torch.is_tensor(cu_seqlens) else cu_seqlens)
    if not attention_varlen_takes(q, bounds):
        return reference.varlen_attention(q, k, v, bounds, scale=scale)
    return attention_varlen(q, k, v, bounds, softmax_scale=scale).view(q.shape)


@register("gated_kvcache_attention", backend="cuda", **NATIVE)
def gated_kvcache_attention(q, k_cache, v_cache, prefix_len, new_pos, gate, *, flash_mask=None, scale=None):
    if not attention_kvcache_takes(q, k_cache):
        return reference_attn.gated_kvcache_attention(q, k_cache, v_cache, prefix_len, new_pos, gate,
                                                      flash_mask=flash_mask, scale=scale)
    return attention_kvcache(q, k_cache, v_cache, prefix_len=prefix_len, new_pos=int(new_pos),
                                flash_mask=flash_mask, softmax_scale=scale, gate=gate)


@register("gated_delta_net", backend="cuda", **NATIVE)
def gated_delta_net(qkv, zba, conv_weight, A_log, dt_bias, norm_weight, num_k_heads, num_v_heads, head_k_dim,
                    head_v_dim, eps):
    b, n, _ = qkv.shape
    rows = (qkv.reshape(b * n, -1), zba.reshape(b * n, -1))
    if b != 1 or head_k_dim != head_v_dim or not gated_delta_rule_prefill_takes(*rows, num_k_heads=num_k_heads):
        return reference_attn.gated_delta_net(qkv, zba, conv_weight, A_log, dt_bias, norm_weight, num_k_heads,
                                              num_v_heads, head_k_dim, head_v_dim, eps)
    out = gated_delta_rule_prefill(*rows, conv_weight, A_log, dt_bias, norm_weight, num_k_heads=num_k_heads,
                                      eps=float(eps))
    return out.view(1, n, -1)
