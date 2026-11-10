"""Torch reference implementations of the operators.

These define the semantics every backend is checked against. They follow the
numerics of the Hugging Face / Wall-X modules they replace (for example,
RMSNorm normalizes and applies the weight in FP32 and rounds once, as Wall-X
does) so that
a model built on reference operators can be compared with its source model.
Backends with the same semantics register through ``xinfer.ops.registry``.

The reference runs on every platform, next to the hardware backends in
``xinfer/ops/backends/``. This package holds the basic operators; the fused BF16
operators are in ``fused``, the low-precision ones in ``mxfp8`` and ``nvfp4``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
    """``x @ weight.T + bias`` in the input dtype (``torch.nn.functional.linear``).

    Args:
        x: ``[..., K]`` input.
        weight: ``[N, K]`` weight, same dtype as ``x``.
        bias: optional ``[N]`` bias.

    Returns:
        ``[..., N]`` in ``x.dtype``.
    """
    return F.linear(x, weight, bias)


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm with one rounding: normalize and scale in FP32, then cast.

    ``(weight * (x32 * rsqrt(mean(x32²) + eps))).to(x.dtype)``; a BF16 weight is
    promoted to FP32 for the product. This is the Wall-X / Gemma convention;
    Hugging Face Qwen2 instead rounds before scaling (two roundings).

    Args:
        x: ``[..., K]`` input, normalized over the last dim.
        weight: ``[K]`` scale.
        eps: added to the mean square before the reciprocal square root.

    Returns:
        ``[..., K]`` in ``x.dtype``.
    """
    dtype = x.dtype
    h = x.to(torch.float32)
    variance = h.pow(2).mean(-1, keepdim=True)
    h = h * torch.rsqrt(variance + eps)
    return (weight * h).to(dtype)


def silu_and_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Gated activation ``silu(gate) * up``, each rounded to the input dtype.

    Args:
        gate: gate half, any shape.
        up: up half, same shape and dtype as ``gate``.

    Returns:
        Tensor of the same shape and dtype.
    """
    return F.silu(gate) * up


def silu(x: torch.Tensor) -> torch.Tensor:
    """Elementwise ``x * sigmoid(x)`` (``torch.nn.functional.silu``), same shape and dtype as ``x``."""
    return F.silu(x)


def gelu(x: torch.Tensor) -> torch.Tensor:
    """Exact (erf) GELU, elementwise; same shape and dtype as ``x``."""
    return F.gelu(x)


def gelu_tanh(x: torch.Tensor) -> torch.Tensor:
    """Tanh-approximate GELU (``F.gelu(x, approximate="tanh")``), elementwise; same shape and dtype as ``x``."""
    return F.gelu(x, approximate="tanh")


def gelu_tanh_and_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Gated activation ``gelu_tanh(gate) * up`` (Gemma MLP), each rounded to the input dtype.

    Args:
        gate: gate half, any shape.
        up: up half, same shape and dtype as ``gate``.

    Returns:
        Tensor of the same shape and dtype.
    """
    return F.gelu(gate, approximate="tanh") * up


def gated_residual(x: torch.Tensor, y: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """``x + y * gate`` (an action expert's gated residual), the product rounded to the input dtype before the add.

    Args:
        x: ``[B, L, K]`` residual.
        y: ``[B, L, K]`` branch output, same dtype.
        gate: ``[B, 1, K]`` (or broadcastable) gate, same dtype.

    Returns:
        ``[B, L, K]`` in ``x.dtype``.
    """
    return x + y * gate


def layer_norm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float) -> torch.Tensor:
    """LayerNorm over the last dim in FP32, as autocast runs it: input, weight and bias cast to FP32.

    Args:
        x: ``[..., K]`` input of any floating dtype.
        weight: ``[K]`` scale.
        bias: ``[K]`` shift.
        eps: added to the variance.

    Returns:
        ``[..., K]`` FP32.
    """
    return F.layer_norm(x.float(), (x.shape[-1],), weight.float(), bias.float(), eps)


def ada_rms_norm(x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor, eps: float) -> torch.Tensor:
    """Adaptive RMSNorm (action-expert time conditioning): ``x · rsqrt(mean(x²) + eps) · (1 + scale) + shift``.

    The normalization and modulation run in FP32 as separate roundings, in the order of the reference
    action expert (``x * rsqrt(mean(x32²) + eps)`` with ``x`` promoted, then ``· (1 + scale32)``, then ``+ shift32``),
    and the result is rounded once to ``x.dtype``. There is no learned weight.

    Args:
        x: ``[B, L, K]`` input.
        scale: ``[B, 1, K]`` (or broadcastable) modulation scale.
        shift: ``[B, 1, K]`` (or broadcastable) modulation shift.
        eps: added to the mean square.

    Returns:
        ``[B, L, K]`` in ``x.dtype``.
    """
    variance = torch.mean(torch.square(x.float()), dim=-1, keepdim=True)
    normed = x * torch.rsqrt(variance + eps)
    return (normed * (1 + scale.to(torch.float32)) + shift.to(torch.float32)).to(x.dtype)


def conv2d(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None, stride: int) -> torch.Tensor:
    """``torch.nn.functional.conv2d`` without padding (patch embeddings).

    Args:
        x: ``[B, C, H, W]`` input, same dtype as ``weight``.
        weight: ``[O, C, kH, kW]`` filters.
        bias: ``[O]`` or None.
        stride: stride in both spatial dims.

    Returns:
        ``[B, O, H', W']`` in ``x.dtype``.
    """
    return F.conv2d(x, weight, bias, stride=stride)


def causal_conv1d(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Depthwise causal convolution followed by SiLU (the Qwen3.5 Gated DeltaNet short convolution).

    ``silu(conv1d(x, weight, padding=K - 1, groups=C)[..., :L])``: output ``t`` sees inputs ``t - K + 1 .. t``,
    zero before the first. Computed in the input dtype.

    Args:
        x: ``[B, C, L]`` channels-first input.
        weight: ``[C, 1, K]`` per-channel filters, same dtype as ``x``.

    Returns:
        ``[B, C, L]`` in ``x.dtype``.
    """
    length = x.shape[-1]
    return F.silu(F.conv1d(x, weight, None, padding=weight.shape[-1] - 1, groups=x.shape[1])[..., :length])


def l2_normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """``x · rsqrt(sum(x²) + eps)`` over the last dim, in the input dtype (the FLA ``l2norm`` convention)."""
    return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + eps)


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


def gated_rms_norm(x: torch.Tensor, gate: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """``rms_norm(x) · weight · silu(gate)`` with the roundings of the Qwen3.5 Gated DeltaNet output norm.

    The normalization runs in FP32 and is rounded to ``x.dtype``; the weight product stays in that dtype; the
    gate is applied in FP32 and the result rounded once more.

    Args:
        x: ``[..., D]`` input.
        gate: ``[..., D]`` gate, same shape.
        weight: ``[D]`` scale in ``x.dtype``.
        eps: added to the mean square.

    Returns:
        ``[..., D]`` in ``x.dtype``.
    """
    dtype = x.dtype
    h = x.to(torch.float32)
    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps)
    h = weight * h.to(dtype)
    return (h * F.silu(gate.to(torch.float32))).to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """``[-x2, x1]`` for ``x = [x1, x2]`` split in halves along the last dim; same shape and dtype."""
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """``x * cos + rotate_half(x) * sin``; ``cos``/``sin`` broadcast to ``x``.

    Computed in the promoted dtype of the inputs, each product and the sum rounded separately.

    Args:
        x: ``[..., D]`` tensor to rotate (``D`` even).
        cos: cosines broadcastable to ``x``.
        sin: sines broadcastable to ``x``.

    Returns:
        The rotated tensor in the promoted dtype of ``x``, ``cos`` and ``sin``.
    """
    return (x * cos) + (rotate_half(x) * sin)


def rotary_embedding(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate ``x [..., H, D]`` in FP32 and round back to ``x.dtype``.

    ``cos``/``sin`` are FP32 ``[..., 1, D]`` (broadcast over the heads):
    ``(x32 * cos + rotate_half(x32) * sin).to(x.dtype)``, each product and the
    sum rounded separately (no fused multiply-add).

    Args:
        x: ``[..., H, D]`` queries or keys (``D`` even).
        cos: FP32 ``[..., 1, D]`` cosines, broadcast over the heads.
        sin: FP32 ``[..., 1, D]`` sines.

    Returns:
        ``[..., H, D]`` in ``x.dtype``.
    """
    return apply_rotary(x.float(), cos, sin).to(x.dtype)


def rotary_embedding_rounded(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate ``x [..., H, D]`` in its own dtype, rounding each product and the sum (Hugging Face Gemma).

    ``cos``/``sin`` have ``x``'s dtype and shape ``[..., 1, D]``:
    ``x * cos + rotate_half(x) * sin`` with ``x * cos`` and ``rotate_half(x) * sin`` each rounded to
    ``x.dtype`` before the add.

    Args:
        x: ``[..., H, D]`` queries or keys (``D`` even).
        cos: ``[..., 1, D]`` cosines in ``x.dtype``, broadcast over the heads.
        sin: ``[..., 1, D]`` sines in ``x.dtype``.

    Returns:
        ``[..., H, D]`` in ``x.dtype``.
    """
    return apply_rotary(x, cos, sin)


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
    if k.shape[1] != q.shape[1]:
        groups = q.shape[1] // k.shape[1]
        k = k.repeat_interleave(groups, dim=1)
        v = v.repeat_interleave(groups, dim=1)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=causal, scale=scale)


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
    bounds = cu_seqlens.tolist() if torch.is_tensor(cu_seqlens) else list(cu_seqlens)
    out = torch.empty_like(q)
    for start, end in zip(bounds[:-1], bounds[1:]):
        if end == start:
            continue
        q_segment = q[start:end].transpose(0, 1).unsqueeze(0)
        k_segment = k[start:end].transpose(0, 1).unsqueeze(0)
        v_segment = v[start:end].transpose(0, 1).unsqueeze(0)
        o = F.scaled_dot_product_attention(q_segment, k_segment, v_segment, scale=scale)
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
    n = q.shape[1]
    mask = torch.ones(n, n, dtype=torch.bool, device=q.device).tril()[None, None]
    q, k, v = (t.transpose(1, 2).contiguous() for t in (q, k, v))
    return attention(q, k, v, scale=scale, mask=mask).transpose(1, 2)


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

    ``q [B, L, H, D]``, own ``k``/``v [B, L, KV, D]``, prompt ``k_prompt``/``v_prompt
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
    return attention(q.transpose(1, 2).contiguous(), k_all, v_all, scale=scale, mask=mask).transpose(1, 2)


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
    k_all = torch.cat([k_prompt, k.transpose(1, 2)], dim=2)
    v_all = torch.cat([v_prompt, v.transpose(1, 2)], dim=2)
    return attention(q.transpose(1, 2).contiguous(), k_all, v_all, scale=scale, mask=mask).transpose(1, 2)


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
    b, s = q.shape[:2]
    keep = (torch.arange(s, device=q.device)[None] < valid.reshape(b, 1).to(torch.int64))[:, None, None, :]
    return attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), scale=scale, mask=keep).transpose(1, 2)


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

    Row ``b`` of ``k_cache``/``v_cache [B, S, KV, D]`` holds its cached keys in columns
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
    return attention(q.transpose(1, 2).contiguous(), k_all, v_all, scale=scale, mask=mask).transpose(1, 2)
