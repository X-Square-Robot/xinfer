"""Torch reference implementations of the attention operators.

The operators of this module correspond one-to-one with the attention registrations of the hardware backends
(``xinfer/ops/backends/<backend>/attn.py``). The implementations are architecture-agnostic: they run on every
device, and every backend is validated against them the same way.

Private composition helpers share layouts, masks and cache updates with backends that need a different SDPA primitive.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from xinfer.ops.reference.conv import causal_conv1d
from xinfer.ops.reference.norm import gated_rms_norm, l2_normalize


def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: float | None = None,
    mask: torch.Tensor | None = None,
    causal: bool = False,
) -> torch.Tensor:
    """Scaled dot-product attention on ``[B, H, L, D]`` tensors.

    ``k``/``v`` may have fewer heads than ``q`` (grouped-query attention); they
    are repeated to match. ``mask`` is boolean (True = attend) or additive and
    broadcasts to ``[B, H, Lq, Lk]``. ``scale`` defaults to ``1/sqrt(D)``.

    Args:
        q: ``[B, H, Lq, D]`` queries.
        k: ``[B, KV, Lk, D]`` keys; ``H`` must be a multiple of ``KV``.
        v: ``[B, KV, Lk, D]`` values.
        scale: softmax scale; None for ``1/sqrt(D)``.
        mask: optional boolean or additive mask broadcastable to ``[B, H, Lq, Lk]``.
        causal: apply a causal mask (``torch.nn.functional.scaled_dot_product_attention`` ``is_causal``).

    Returns:
        ``[B, H, Lq, D]`` in ``q.dtype``.
    """
    return _attention(F.scaled_dot_product_attention, q, k, v, scale=scale, mask=mask, causal=causal)


def _attention(sdpa, q, k, v, *, scale=None, mask=None, causal=False):
    if k.shape[1] != q.shape[1]:
        groups = q.shape[1] // k.shape[1]
        k = k.repeat_interleave(groups, dim=1)
        v = v.repeat_interleave(groups, dim=1)
    return sdpa(q, k, v, attn_mask=mask, is_causal=causal, scale=scale)


def varlen_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens,
    *,
    scale: float | None = None,
) -> torch.Tensor:
    """Bidirectional attention within each segment of packed ``[T, H, D]`` tensors.

    ``cu_seqlens`` holds segment boundaries (``[0, l0, l0 + l1, ...]``) as a
    host sequence or tensor. Tokens attend only within their segment, as
    flash-attention ``varlen`` does.

    Args:
        q: ``[T, H, D]`` packed queries of all segments.
        k: packed keys, same layout as ``q`` (no grouped-query repetition).
        v: packed values, same layout as ``q``.
        cu_seqlens: segment boundaries ``[0, l0, l0 + l1, ..., T]``; a host sequence keeps the call capturable,
            a tensor is read back with ``tolist()``.
        scale: softmax scale; None for ``1/sqrt(D)``.

    Returns:
        ``[T, H, D]`` in ``q.dtype``; tokens outside ``[cu_seqlens[0], cu_seqlens[-1])`` are left uninitialized.
    """
    return _varlen_attention(F.scaled_dot_product_attention, q, k, v, cu_seqlens, scale=scale)


def _varlen_attention(sdpa, q, k, v, cu_seqlens, *, scale=None):
    bounds = cu_seqlens.tolist() if torch.is_tensor(cu_seqlens) else list(cu_seqlens)
    out = torch.empty_like(q)
    for start, end in zip(bounds[:-1], bounds[1:]):
        if end == start:
            continue
        q_segment = q[start:end].transpose(0, 1).unsqueeze(0)
        k_segment = k[start:end].transpose(0, 1).unsqueeze(0)
        v_segment = v[start:end].transpose(0, 1).unsqueeze(0)
        o = sdpa(q_segment, k_segment, v_segment, scale=scale)
        out[start:end] = o.squeeze(0).transpose(0, 1)
    return out


def causal_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, scale: float | None = None) -> torch.Tensor:
    """Causal self-attention on sequence-major tensors.

    ``q [B, L, H, D]``, ``k``/``v [B, L, KV, D]`` → ``[B, L, H, D]``.
    Computed as ``attention`` on head-major copies with an explicit lower-triangular
    boolean mask (the Wall-X eager prefill call).

    Args:
        q: ``[B, L, H, D]`` queries.
        k: ``[B, L, KV, D]`` keys; ``H`` must be a multiple of ``KV``.
        v: ``[B, L, KV, D]`` values.
        scale: softmax scale; None for ``1/sqrt(D)``.

    Returns:
        ``[B, L, H, D]`` in ``q.dtype`` (a transposed, non-contiguous view).
    """
    return _causal_attention(F.scaled_dot_product_attention, q, k, v, scale=scale)


def causal_attention_bnsd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *,
                           scale: float | None = None) -> torch.Tensor:
    """Causal attention on sequence-major inputs with head-major ``[B, H, L, D]`` output."""
    return _causal_attention(F.scaled_dot_product_attention, q, k, v, scale=scale, return_bnsd=True)


def _causal_attention(sdpa, q, k, v, *, scale=None, return_bnsd=False):
    n = q.shape[1]
    mask = torch.ones(n, n, dtype=torch.bool, device=q.device).tril()[None, None]
    q, k, v = (t.transpose(1, 2).contiguous() for t in (q, k, v))
    out = _attention(sdpa, q, k, v, scale=scale, mask=mask)
    return out if return_bnsd else out.transpose(1, 2)


def rtc_pinned_mask(num_pinned: torch.Tensor, rows: int) -> torch.Tensor:
    """``[B, L, L]`` visibility among ``L`` own tokens: rows ``< num_pinned[b]`` see only keys ``< num_pinned[b]``.

    Args:
        num_pinned: ``[B]`` integer tensor, the number of pinned (RTC) rows per batch row.
        rows: ``L``, the number of own tokens.

    Returns:
        Boolean ``[B, L, L]`` on ``num_pinned.device``, True where query row ``i`` may attend to key ``j``.
    """
    pin = torch.arange(rows, device=num_pinned.device)[None] < num_pinned.reshape(-1, 1)
    return ~(pin[:, :, None] & ~pin[:, None, :])


def action_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_prompt: torch.Tensor,
    v_prompt: torch.Tensor,
    num_pinned: torch.Tensor,
    prompt_lengths: torch.Tensor | None = None,
    *,
    scale: float | None = None,
) -> torch.Tensor:
    """Queries of ``L`` own tokens attend to a prompt and to the own tokens (RTC joint attention).

    ``q [B, L, H, D]``, own ``k``/``v [B, L, KV, D]``, prompt ``k_prompt``/`v_prompt
    [B, KV, P, D]`` (head-major, as stage handoffs store it), ``pinned [B]`` int.
    Every row sees its whole prompt. Own rows ``i < num_pinned[b]`` form a closed
    group that sees only own keys ``< num_pinned[b]``; the other rows see all own
    keys (``num_pinned = 0``: plain bidirectional attention over prompt + own).
    ``prompt_lengths [B]`` marks left-padded prompts: row ``b``'s prompt is
    the last ``prompt_lengths[b]`` of the ``P`` columns (default: all ``P``).
    Returns ``[B, L, H, D]``.

    Args:
        q: ``[B, L, H, D]`` queries of the ``L`` own (action) tokens.
        k: ``[B, L, KV, D]`` keys of the own tokens.
        v: ``[B, L, KV, D]`` values of the own tokens.
        k_prompt: ``[B, KV, P, D]`` prompt keys (head-major).
        v_prompt: ``[B, KV, P, D]`` prompt values.
        num_pinned: ``[B]`` integer tensor, the number of pinned rows per batch row.
        prompt_lengths: optional ``[B]`` integer tensor of valid prompt lengths (left padding).
        scale: softmax scale; None for ``1/sqrt(D)``.

    Returns:
        ``[B, L, H, D]`` in ``q.dtype`` (a transposed, non-contiguous view).
    """
    return _action_attention(
        F.scaled_dot_product_attention, q, k, v, k_prompt, v_prompt, num_pinned, prompt_lengths, scale=scale)


def _action_attention(sdpa, q, k, v, k_prompt, v_prompt, num_pinned, prompt_lengths=None, *, scale=None):
    b, n = q.shape[:2]
    p = k_prompt.shape[2]
    if prompt_lengths is None:
        prompt_cols = torch.ones(b, n, p, dtype=torch.bool, device=q.device)
    else:
        start = p - prompt_lengths.reshape(b, 1).to(torch.int64)
        prompt_cols = (torch.arange(p, device=q.device)[None] >= start)[:, None, :].expand(b, n, p)
    mask = torch.cat([prompt_cols, rtc_pinned_mask(num_pinned, n)], dim=-1)[:, None]  # [B, 1, L, P + L]
    k_all = torch.cat([k_prompt, k.transpose(1, 2)], dim=2)
    v_all = torch.cat([v_prompt, v.transpose(1, 2)], dim=2)
    return _attention(sdpa, q.transpose(1, 2).contiguous(), k_all, v_all, scale=scale, mask=mask).transpose(1, 2)


def masked_joint_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_prompt: torch.Tensor,
    v_prompt: torch.Tensor,
    mask: torch.Tensor,
    *,
    scale: float | None = None,
) -> torch.Tensor:
    """Queries of ``L`` own tokens attend to a prompt and to the own tokens under an explicit mask.

    The keys are laid out as ``[prompt columns | own tokens]``, the order of a single forward over the prompt
    followed by the own tokens, and ``mask`` gives the visibility of every key to every own query.

    Args:
        q: ``[B, L, H, D]`` queries of the own tokens.
        k: ``[B, L, KV, D]`` own keys.
        v: ``[B, L, KV, D]`` own values.
        k_prompt: ``[B, KV, P, D]`` prompt keys (head-major).
        v_prompt: ``[B, KV, P, D]`` prompt values.
        mask: ``[B, 1, L, P + L]`` boolean, True where a query may attend to a key; every row needs one True.
        scale: softmax scale; None for ``1/sqrt(D)``.

    Returns:
        ``[B, L, H, D]`` in ``q.dtype`` (a transposed, non-contiguous view).
    """
    return _masked_joint_attention(F.scaled_dot_product_attention, q, k, v, k_prompt, v_prompt, mask, scale=scale)


def _masked_joint_attention(sdpa, q, k, v, k_prompt, v_prompt, mask, *, scale=None):
    k_all = torch.cat([k_prompt, k.transpose(1, 2)], dim=2)
    v_all = torch.cat([v_prompt, v.transpose(1, 2)], dim=2)
    return _attention(sdpa, q.transpose(1, 2).contiguous(), k_all, v_all, scale=scale, mask=mask).transpose(1, 2)


def prefix_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    valid: torch.Tensor,
    *,
    scale: float | None = None,
) -> torch.Tensor:
    """Bidirectional attention over right-padded rows (a bidirectional prefix).

    Row ``b``'s keys are its first ``valid[b]`` positions; every query, including the padded ones, attends
    to those keys, so padded rows get finite values (they are never keys of valid rows). Computed as
    ``attention`` on head-major views with a boolean key mask, which gives the same bits as an additive
    mask on its valid rows.

    Args:
        q: ``[B, S, H, D]`` queries.
        k: ``[B, S, KV, D]`` keys; ``H`` must be a multiple of ``KV``.
        v: ``[B, S, KV, D]`` values.
        valid: ``[B]`` integer tensor, the number of leading valid positions per row.
        scale: softmax scale; None for ``1/sqrt(D)``.

    Returns:
        ``[B, S, H, D]`` in ``q.dtype`` (a transposed, non-contiguous view).
    """
    return _prefix_attention(F.scaled_dot_product_attention, q, k, v, valid, scale=scale)


def _prefix_attention(sdpa, q, k, v, valid, *, scale=None):
    b, s = q.shape[:2]
    keep = (torch.arange(s, device=q.device)[None] < valid.reshape(b, 1).to(torch.int64))[:, None, None, :]
    return _attention(sdpa, q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                      scale=scale, mask=keep).transpose(1, 2)


def kvcache_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cache_start: torch.Tensor,
    cache_valid: torch.Tensor,
    *,
    scale: float | None = None,
) -> torch.Tensor:
    """Append ``L`` own tokens' K/V to a per-row cache and attend over the row's cached and own keys.

    Row ``b`` of ``k_cache``/`v_cache [B, S, KV, D]`` holds its cached keys in columns
    ``[cache_start[b], cache_start[b] + cache_valid[b])``; the own keys are written (in place) right after them,
    and every own query attends, without a causal mask, to those cached keys and all ``L`` own keys. The last
    ``L`` columns are room for the append: ``cache_start + cache_valid <= S - L``.

    The keys are laid out as ``[cache columns 0 .. S - L | own]`` with a boolean mask, the layout of a joint
    ``[prefix | suffix]`` attention, so a row with ``cache_start = 0`` gives the same bits as that form.

    Args:
        q: ``[B, L, H, D]`` queries of the own tokens.
        k: ``[B, L, KV, D]`` own keys.
        v: ``[B, L, KV, D]`` own values.
        k_cache: ``[B, S, KV, D]`` key cache (sequence-major); modified: columns ``[start + valid, start + valid + L)``
            receive ``k``.
        v_cache: ``[B, S, KV, D]`` value cache, modified likewise.
        cache_start: ``[B]`` integer tensor, each row's first cached column.
        cache_valid: ``[B]`` integer tensor, each row's number of cached keys.
        scale: softmax scale; None for ``1/sqrt(D)``.

    Returns:
        ``[B, L, H, D]`` in ``q.dtype`` (a transposed, non-contiguous view).
    """
    return _kvcache_attention(
        F.scaled_dot_product_attention, q, k, v, k_cache, v_cache, cache_start, cache_valid, scale=scale)


def _kvcache_attention(sdpa, q, k, v, k_cache, v_cache, cache_start, cache_valid, *, scale=None):
    b, n = q.shape[:2]
    p = k_cache.shape[1] - n
    start = cache_start.reshape(b, 1).to(torch.int64)
    end = start + cache_valid.reshape(b, 1).to(torch.int64)
    index = (end + torch.arange(n, device=q.device)[None])[:, :, None, None].expand_as(k)
    k_cache.scatter_(1, index, k)
    v_cache.scatter_(1, index, v)
    columns = torch.arange(p, device=q.device)[None]
    cached = ((columns >= start) & (columns < end))[:, None, :].expand(b, n, p)
    mask = torch.cat([cached, torch.ones(b, n, n, dtype=torch.bool, device=q.device)], dim=-1)[:, None]
    k_all = torch.cat([k_cache[:, :p].transpose(1, 2), k.transpose(1, 2)], dim=2)
    v_all = torch.cat([v_cache[:, :p].transpose(1, 2), v.transpose(1, 2)], dim=2)
    return _attention(sdpa, q.transpose(1, 2).contiguous(), k_all, v_all, scale=scale, mask=mask).transpose(1, 2)


def prefix_kvcache_attention(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor,
                             prefix_len: torch.Tensor, new_pos: int, *, flash_mask: torch.Tensor | None = None,
                             scale: float | None = None) -> torch.Tensor:
    """Attention of ``M`` new tokens over a cache prefix and their own keys (the CUDA backend's ``attention_kvcache``;
    the Ascend backend's cache-attention kernels serve it through the same contract).

    Each query can attend to the valid prefix and the new keys. An optional column mask supplies the valid query
    row range for each key.

    Args:
        q: ``[B, M, H, D]`` BF16.
        k_cache: ``[B, L, KV, D]`` BF16.
        v_cache: ``[B, L, KV, D]`` BF16.
        prefix_len: int32 ``[B]``.
        new_pos: First cache row of the new tokens' keys (``new_pos + M <= L``).
        flash_mask: Column mask or None.
        scale: Softmax scale; None for ``1/sqrt(D)``.

    Returns:
        ``[B, M, H·D]`` BF16.
    """
    return _prefix_kvcache_attention(
        F.scaled_dot_product_attention, q, k_cache, v_cache, prefix_len, new_pos, flash_mask=flash_mask, scale=scale)


def prefix_kvcache_attention_bnsd(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor,
                                  prefix_len: torch.Tensor, new_pos: int, *,
                                  flash_mask: torch.Tensor | None = None, scale: float | None = None,
                                  all_keep: bool = False) -> torch.Tensor:
    """Cache attention with head-major ``[B, H, M, D]`` output.

    ``all_keep`` guarantees every cache row is visible (``prefix_len == new_pos == L - M`` and no flash mask).
    """
    return _prefix_kvcache_attention(
        F.scaled_dot_product_attention, q, k_cache, v_cache, prefix_len, new_pos, flash_mask=flash_mask, scale=scale,
        all_keep=all_keep, return_bnsd=True)


def _prefix_kvcache_attention(sdpa, q, k_cache, v_cache, prefix_len, new_pos, *, flash_mask=None, scale=None,
                              all_keep=False, return_bnsd=False):
    b, m = q.shape[:2]
    if all_keep:
        if flash_mask is not None:
            raise ValueError("all_keep cannot be combined with a flash mask")
        keep = None
    else:
        rows = torch.arange(k_cache.shape[1], device=q.device)
        keep = (rows[None] < prefix_len.reshape(-1, 1)) | ((rows >= new_pos) & (rows < new_pos + m))[None]
        keep = keep[:, None, :].expand(b, m, -1)
        if flash_mask is not None:
            t = torch.arange(m, device=q.device)[None, :, None]
            keep = keep & (t < flash_mask[:, 0, None, :, 0]) & (t >= flash_mask[:, 0, None, :, 1])
        keep = keep[:, None]
    o = _attention(sdpa, q.transpose(1, 2), k_cache.transpose(1, 2), v_cache.transpose(1, 2), scale=scale,
                   mask=keep)
    if return_bnsd:
        return o
    return o.transpose(1, 2).reshape(b, m, -1)


def chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    chunk_size: int = 64,
) -> torch.Tensor:
    """Gated delta rule over a whole sequence from a zero state, in chunks (Qwen3.5 linear attention).

    The recurrence per head is ``S_t = exp(g_t) S_{t-1} + beta_t k_t (v_t - exp(g_t) S_{t-1}^T k_t)^T`` and
    ``o_t = S_t^T q_t / sqrt(Dk)`` with ``q`` and ``k`` L2-normalized. It is evaluated chunk by chunk in the order
    of the Hugging Face ``torch_chunk_gated_delta_rule`` (the sequence is zero-padded to a multiple of
    ``chunk_size``), under autocast to the input dtype as the Wall-X model runs it: for BF16 inputs the matrix
    products run in BF16 and reductions, ``exp`` and ``rsqrt`` in FP32; FP32 inputs stay in FP32.

    Args:
        q: ``[B, L, H, Dk]`` queries.
        k: ``[B, L, H, Dk]`` keys.
        v: ``[B, L, H, Dv]`` values.
        g: ``[B, L, H]`` FP32 log decay (``<= 0``).
        beta: ``[B, L, H]`` write strength.
        chunk_size: tokens per chunk.

    Returns:
        ``[B, L, H, Dv]`` in ``q.dtype``.
    """
    dtype = q.dtype
    with torch.autocast(q.device.type, dtype=dtype, enabled=dtype in (torch.bfloat16, torch.float16)):
        return _chunk_gated_delta_rule(q, k, v, g, beta, chunk_size).to(dtype)


def _chunk_gated_delta_rule(q, k, v, g, beta, chunk_size):
    q, k = l2_normalize(q), l2_normalize(k)
    q, k, v, beta, g = (t.transpose(1, 2).contiguous().to(torch.float32) for t in (q, k, v, beta, g))
    b, h, length, dk = k.shape
    dv = v.shape[-1]
    pad = (chunk_size - length % chunk_size) % chunk_size
    q, k, v = (F.pad(t, (0, 0, 0, pad)) for t in (q, k, v))
    beta, g = F.pad(beta, (0, pad)), F.pad(g, (0, pad))
    q = q * (1 / (dk ** 0.5))
    v_beta = v * beta.unsqueeze(-1)
    k_beta = k * beta.unsqueeze(-1)
    q, k, v, k_beta, v_beta = (t.reshape(b, h, -1, chunk_size, t.shape[-1]) for t in (q, k, v, k_beta, v_beta))
    g = g.reshape(b, h, -1, chunk_size).cumsum(dim=-1)
    upper = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), diagonal=0)
    decay = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ k.transpose(-1, -2)) * decay).masked_fill(upper, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    v = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    state = torch.zeros(b, h, dk, dv, dtype=v.dtype, device=v.device)
    out = torch.zeros_like(v)
    strict_upper = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), diagonal=1)
    for i in range(q.shape[2]):
        q_i, k_i, v_i = q[:, :, i], k[:, :, i], v[:, :, i]
        attn = (q_i @ k_i.transpose(-1, -2) * decay[:, :, i]).masked_fill_(strict_upper, 0)
        v_new = v_i - k_cumdecay[:, :, i] @ state
        out[:, :, i] = (q_i * g[:, :, i, :, None].exp()) @ state + attn @ v_new
        state = (state * g[:, :, i, -1, None, None].exp()
                 + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new)
    return out.reshape(b, h, -1, dv)[:, :, :length].transpose(1, 2).contiguous()


# ---------------------------------------------------------------- Wall-X 2B fused operators

def gate_output(o: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """``bf16(bf16(o) · bf16(sigmoid(gate)))``: the attention output times its gate, as two BF16 operations."""
    return o.to(gate.dtype) * torch.sigmoid(gate)

def segment_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, cu_seqlens, *,
                      scale: float | None = None) -> torch.Tensor:
    """``varlen_attention`` in the form of ``attention_varlen``: bidirectional attention within each segment.

    The same function as ``varlen_attention``, as a separate operator so that only the compositions validated
    with it run the unified entry (the others keep ``varlen_attention``'s backend). Arguments and result as
    ``reference.varlen_attention``.
    """
    return varlen_attention(q, k, v, cu_seqlens, scale=scale)

def gated_causal_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, gate: torch.Tensor, *,
                           scale: float | None = None) -> torch.Tensor:
    """Causal attention over one sequence, output times ``sigmoid(gate)`` (the CUDA backend's ``attention_varlen``).

    Args:
        q: ``[T, H, D]`` BF16.
        k: ``[T, KV, D]`` BF16 (grouped-query heads).
        v: ``[T, KV, D]`` BF16.
        gate: ``[T, H·D]`` BF16 output gate before the sigmoid.
        scale: Softmax scale; None for ``1/sqrt(D)``.

    Returns:
        ``[T, H·D]`` BF16.
    """
    t = q.shape[0]
    o = attention(q.transpose(0, 1)[None], k.transpose(0, 1)[None], v.transpose(0, 1)[None],
                            scale=scale, causal=True)
    return gate_output(o[0].transpose(0, 1).reshape(t, -1), gate)

def gated_kvcache_attention(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor,
                            prefix_len: torch.Tensor, new_pos: int, gate: torch.Tensor, *,
                            flash_mask: torch.Tensor | None = None, scale: float | None = None) -> torch.Tensor:
    """Attention of ``M`` new tokens over a cache prefix and their own keys, output times ``sigmoid(gate)``.

    The CUDA backend's ``attention_kvcache`` with ``new_pos``: request ``b`` sees cache rows ``[0, prefix_len[b])`` and
    ``[new_pos, new_pos + M)`` (the new tokens' keys, already in the cache). ``flash_mask`` (int32
    ``[B, 1, L, 2]``, ``(LTS, UTE)`` per cache row) removes row ``c`` for query ``t`` when ``t >= LTS`` or
    ``t < UTE``.

    Args:
        q: ``[B, M, H, D]`` BF16.
        k_cache: ``[B, L, KV, D]`` BF16.
        v_cache: ``[B, L, KV, D]`` BF16.
        prefix_len: int32 ``[B]``.
        new_pos: First cache row of the new tokens' keys (``new_pos + M <= L``).
        gate: ``[B, M, H·D]`` BF16 output gate before the sigmoid.
        flash_mask: Column mask or None.
        scale: Softmax scale; None for ``1/sqrt(D)``.

    Returns:
        ``[B, M, H·D]`` BF16.
    """
    o = prefix_kvcache_attention(q, k_cache, v_cache, prefix_len, new_pos, flash_mask=flash_mask, scale=scale)
    return gate_output(o, gate)

def gated_delta_net(qkv: torch.Tensor, zba: torch.Tensor, conv_weight: torch.Tensor, A_log: torch.Tensor,
                    dt_bias: torch.Tensor, norm_weight: torch.Tensor, num_k_heads: int, num_v_heads: int,
                    head_k_dim: int, head_v_dim: int, eps: float) -> torch.Tensor:
    """The Gated DeltaNet mixer between its input and output projections, from a zero state.

    ``mixed = silu(causal_conv1d(qkv))`` split into q, k, v heads (q and k repeated to the value heads);
    ``beta = sigmoid(b)``, ``g = -exp(A_log) · softplus(a + dt_bias)`` in FP32; ``chunk_gated_delta_rule``
    (q and k L2-normalized); ``gated_rms_norm`` with ``z``. The same arithmetic as ``GatedDeltaNet`` on the
    reference operators. Rows are a single sequence; padding rows must be zeroed by the caller.

    Args:
        qkv: ``[B, L, 2·Hk·Dk + Hv·Dv]`` input projection.
        zba: ``[B, L, Hv·Dv + 2·Hv]`` ``[z | b | a]`` projections.
        conv_weight: ``[C, Kc]`` depthwise filters.
        A_log: ``[Hv]`` log decay rates.
        dt_bias: ``[Hv]`` decay bias.
        norm_weight: ``[Dv]`` output norm weight, ``qkv.dtype``.
        num_k_heads: ``Hk``.
        num_v_heads: ``Hv``.
        head_k_dim: ``Dk``.
        head_v_dim: ``Dv``.
        eps: Output norm epsilon.

    Returns:
        ``[B, L, Hv·Dv]`` in ``qkv.dtype``.
    """
    b, n, _ = qkv.shape
    key_dim, value_dim = num_k_heads * head_k_dim, num_v_heads * head_v_dim
    mixed = causal_conv1d(qkv.transpose(1, 2), conv_weight[:, None, :]).transpose(1, 2)
    q, k, v = torch.split(mixed, [key_dim, key_dim, value_dim], dim=-1)
    repeat = num_v_heads // num_k_heads
    q = q.reshape(b, n, num_k_heads, head_k_dim).repeat_interleave(repeat, dim=2)
    k = k.reshape(b, n, num_k_heads, head_k_dim).repeat_interleave(repeat, dim=2)
    v = v.reshape(b, n, num_v_heads, head_v_dim)
    z, beta_logit, a = torch.split(zba, [value_dim, num_v_heads, num_v_heads], dim=-1)
    beta = beta_logit.sigmoid()
    g = -A_log.float().exp() * F.softplus(a.float() + dt_bias)
    o = chunk_gated_delta_rule(q, k, v, g, beta)
    o = gated_rms_norm(o.reshape(-1, head_v_dim), z.reshape(-1, head_v_dim), norm_weight, eps)
    return o.reshape(b, n, value_dim)
