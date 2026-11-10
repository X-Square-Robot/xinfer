"""Torch reference implementations of the fused BF16 operations used by Wall-OSS."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from xinfer.ops import reference


ACTIVATIONS = {"silu": F.silu, "gelu": F.gelu, "gelu_tanh": lambda y: F.gelu(y, approximate="tanh")}


def _adarms(x: torch.Tensor, modulation: torch.Tensor, row_class: torch.Tensor, eps: float) -> torch.Tensor:
    """FP32 ``x · rsqrt(mean(x²) + eps) · (1 + scale) + shift`` with each token's modulation row (not rounded)."""
    k = x.shape[-1]
    m = modulation[row_class].float()
    x = x.float()
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * (1 + m[:, :k]) + m[:, k:2 * k]


def gemm_epilogue(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None, *,
                  activation: str | None = None, residual: torch.Tensor | None = None,
                  residual_gate: torch.Tensor | None = None, row_class: torch.Tensor | None = None,
                  row_scale: torch.Tensor | None = None) -> torch.Tensor:
    """``x · weightᵀ`` with xcompute ``gemm``'s epilogue.

    ``y = act(row_scale · (x·Wᵀ + bias))`` in FP32. Without ``residual`` the result is ``bf16(y)``; with it
    ``bf16(residual + y)``; with a ``residual_gate`` ``[R, N]`` and ``row_class`` the gated branch is rounded first:
    ``bf16(residual + bf16(y · residual_gate[row_class]))``.

    Args:
        x: ``[M, K]`` BF16.
        weight: ``[N, K]`` BF16.
        bias: ``[N]`` or None.
        activation: ``"silu"``, ``"gelu"`` (erf), ``"gelu_tanh"`` or None.
        residual: ``[M, N]`` or None.
        residual_gate: ``[R, N]`` gate rows (column slices of a modulation table may be strided) or None.
        row_class: int ``[M]`` gate row of each token (with ``residual_gate``).
        row_scale: FP32 ``[M]`` per-row factor (for example a 0/1 token mask) or None.

    Returns:
        ``[M, N]`` in ``x.dtype``.
    """
    y = x.float() @ weight.float().t()
    if bias is not None:
        y = y + bias.float()
    if row_scale is not None:
        y = y * row_scale.float()[:, None]
    if activation is not None:
        y = ACTIVATIONS[activation](y)
    if residual is None:
        return y.to(x.dtype)
    if residual_gate is not None:
        y = (y * residual_gate[row_class].float()).to(x.dtype).float()
    return (residual.float() + y).to(x.dtype)


def gemm_glu_epilogue(x: torch.Tensor, weight: torch.Tensor, *, bias: torch.Tensor | None = None,
                      modulation: torch.Tensor | None = None, row_class: torch.Tensor | None = None,
                      eps: float = 1e-6) -> torch.Tensor:
    """``silu(x̂·W_gateᵀ) · (x̂·W_upᵀ)`` (xcompute ``gemm_glu``), with an optional AdaRMS input norm.

    ``x̂ = x`` or, with a ``modulation`` table, ``x · rsqrt(mean(x²) + eps) · (1 + scale) + shift`` of each token's
    row in FP32, not rounded before the GEMM.

    Args:
        x: ``[M, K]`` BF16.
        weight: ``[2N, K]`` BF16, the gate rows then the up rows.
        bias: ``[2N]`` added to the FP32 products, or None.
        modulation: ``[R, ≥2K]`` ``[scale | shift | ...]`` rows, or None.
        row_class: int ``[M]`` modulation row of each token (with ``modulation``).
        eps: Added to the mean square.

    Returns:
        ``[M, N]`` in ``x.dtype``.
    """
    h = x.float() if modulation is None else _adarms(x, modulation, row_class, eps)
    y = h @ weight.float().t()
    if bias is not None:
        y = y + bias.float()
    n = weight.shape[0] // 2
    return (F.silu(y[:, :n]) * y[:, n:]).to(x.dtype)


def qkv_rope_epilogue(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, cos: torch.Tensor,
                      sin: torch.Tensor, num_heads: int, head_dim: int) -> torch.Tensor:
    """``[Q | K | V] = x·Wᵀ + bias`` with rotate-half RoPE on every Q and K head (xcompute ``qkv_proj_rope``).

    Equal numbers of Q, K and V heads (a ViT); the rotation covers ``2·cos.shape[-1]`` channels of each head and
    is applied to the FP32 accumulator, so the output rounds once.

    Args:
        x: ``[M, K]`` BF16.
        weight: ``[3·H·D, K]`` BF16.
        bias: ``[3·H·D]``.
        cos: FP32 ``[M, R/2]`` cosines of the rotated pairs ``(j, j + R/2)``.
        sin: FP32 ``[M, R/2]``.
        num_heads: Heads ``H``.
        head_dim: Head dimension ``D``.

    Returns:
        ``[M, 3·H·D]`` in ``x.dtype``.
    """
    m, half = x.shape[0], cos.shape[-1]
    y = (x.float() @ weight.float().t() + bias.float()).view(m, 3, num_heads, head_dim)
    c, s = cos[:, None, None, :], sin[:, None, None, :]
    qk = y[:, :2]
    a, b = qk[..., :half], qk[..., half:2 * half]
    rotated = torch.cat((a * c - b * s, b * c + a * s, qk[..., 2 * half:]), dim=-1)
    return torch.cat((rotated, y[:, 2:]), dim=1).reshape(m, -1).to(x.dtype)


def _qk_norm_rope_kvcache(y, dtype, q_weight, k_weight, cos, sin, k_cache, v_cache, start, num_heads, num_kv_heads,
                          head_dim, eps):
    """The q/k norms (none when ``q_weight`` is None), RoPE and K/V append of FP32 projections ``y``
    ``[M, (H + 2·KV)·D + G]``."""
    m, half = y.shape[0], cos.shape[-1]
    qd, kvd = num_heads * head_dim, num_kv_heads * head_dim
    q, k, v, gate = y.split((qd, kvd, kvd, y.shape[1] - qd - 2 * kvd), dim=1)

    def norm_rope(t, w, heads):
        t = t.view(m, heads, head_dim)
        if w is not None:
            t = t * torch.rsqrt(t.pow(2).mean(-1, keepdim=True) + eps) * (1 + w.float())
        c, s = cos[:, None, :], sin[:, None, :]
        a, b = t[..., :half], t[..., half:2 * half]
        return torch.cat((a * c - b * s, b * c + a * s, t[..., 2 * half:]), dim=-1).to(dtype)

    k_cache[:, start:start + m].copy_(norm_rope(k, k_weight, num_kv_heads).transpose(0, 1))
    v_cache[:, start:start + m].copy_(v.reshape(m, num_kv_heads, head_dim).transpose(0, 1).to(dtype))
    return torch.cat((norm_rope(q, q_weight, num_heads).reshape(m, qd), gate.to(dtype)), dim=1)


def prefix_kvcache_attention(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor,
                             prefix_len: torch.Tensor, new_pos: int, *, flash_mask: torch.Tensor | None = None,
                             scale: float | None = None) -> torch.Tensor:
    """Attention of ``M`` new tokens over a cache prefix and their own keys (xcompute ``attention_kvcache``).

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
    b, m = q.shape[:2]
    rows = torch.arange(k_cache.shape[1], device=q.device)
    keep = (rows[None] < prefix_len.reshape(-1, 1)) | ((rows >= new_pos) & (rows < new_pos + m))[None]
    keep = keep[:, None, :].expand(b, m, -1)
    if flash_mask is not None:
        t = torch.arange(m, device=q.device)[None, :, None]
        keep = keep & (t < flash_mask[:, 0, None, :, 0]) & (t >= flash_mask[:, 0, None, :, 1])
    o = reference.attention(q.transpose(1, 2), k_cache.transpose(1, 2), v_cache.transpose(1, 2), scale=scale,
                            mask=keep[:, None])
    return o.transpose(1, 2).reshape(b, m, -1)


def qkv_rope_kvcache(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None, cos: torch.Tensor,
                     sin: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, start: int, num_heads: int,
                     num_kv_heads: int, head_dim: int) -> torch.Tensor:
    """``[Q | K | V] = x·Wᵀ + bias`` in FP32, rotate-half RoPE on the Q and K heads, K and V written to head-major
    caches (xcompute ``qkv_proj_rope`` with ``kv_cache``): grouped-query attention without q/k norms.

    Args:
        x: ``[M, K]`` BF16.
        weight: ``[(H + 2·KV)·D, K]`` BF16, rows ``[q | k | v]``.
        bias: ``[(H + 2·KV)·D]``, or None.
        cos: FP32 ``[M, R/2]`` cosines of the rotated pairs ``(j, j + R/2)``.
        sin: FP32 ``[M, R/2]``.
        k_cache: ``[KV, rows, D]`` BF16 head-major cache (a strided view), written in place.
        v_cache: ``[KV, rows, D]`` BF16, written in place.
        start: First cache row of the new tokens.
        num_heads: Query heads ``H``.
        num_kv_heads: K/V heads ``KV``.
        head_dim: Head dimension ``D``.

    Returns:
        ``[M, H·D]`` BF16: ``Q``, rotated.
    """
    y = x.float() @ weight.float().t()
    if bias is not None:
        y = y + bias.float()
    return _qk_norm_rope_kvcache(y, x.dtype, None, None, cos, sin, k_cache, v_cache, start, num_heads, num_kv_heads,
                                 head_dim, 0.0)


def add_rms_norm(x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor,
                 eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    """``s = residual + x`` (one BF16 rounding), then ``reference.rms_norm(s)`` (xcompute ``add_rmsnorm``).

    Args:
        x: ``[..., K]`` BF16 branch output.
        residual: ``[..., K]`` BF16 residual stream.
        weight: ``[K]`` FP32 or BF16.
        eps: Added to the mean square.

    Returns:
        ``(s, normed)`` in ``x.dtype``.
    """
    s = residual + x
    return s, reference.rms_norm(s, weight, eps)


def flow_velocity_euler(x: torch.Tensor, h: torch.Tensor, weight: torch.Tensor, dt: float,
                        dof_mask: torch.Tensor | None = None, padding_velocity: torch.Tensor | None = None,
                        pinned: torch.Tensor | None = None, pinned_values: torch.Tensor | None = None) -> torch.Tensor:
    """One Euler step of action rows from their normed stream output (xcompute ``denoise_output_euler``, velocity
    ``flow``).

    ``p = h · weightᵀ`` in FP32; the velocity is ``padding_velocity · (1 − dof_mask) + p · dof_mask`` (``p``
    without a mask), and ``x + dt · velocity`` the new values; the first ``pinned[b]`` rows take ``pinned_values``
    (``x`` without).

    Args:
        x: FP32 ``[B, M, A]`` current values.
        h: ``[B, M, K]`` BF16 normed stream output.
        weight: ``[A, K]`` FP32 or BF16 readout.
        dt: Step size.
        dof_mask: ``[B, M, A]`` 0/1 commanded values, or None.
        padding_velocity: FP32 ``[B, M, A]`` velocity of the uncommanded values (with ``dof_mask``).
        pinned: int ``[B]`` pinned leading rows, or None.
        pinned_values: FP32 ``[B, M, A]`` values of the pinned rows, or None to keep ``x``.

    Returns:
        FP32 ``[B, M, A]``.
    """
    p = h.float() @ weight.float().t()
    if dof_mask is not None:
        m = dof_mask.float()
        p = padding_velocity * (1 - m) + p * m
    out = x + dt * p
    if pinned is not None:
        rows = torch.arange(x.shape[1], device=x.device)[None, :, None] < pinned.reshape(-1, 1, 1)
        out = torch.where(rows, x if pinned_values is None else pinned_values, out)
    return out
