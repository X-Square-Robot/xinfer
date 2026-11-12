"""Torch reference implementations of the projection (GEMM) operators.

The operators of this module correspond one-to-one with the GEMM registrations of the hardware backends
(``xinfer/ops/backends/<backend>/gemm.py``): the plain projection, the GEMMs with fused epilogues and the
QKV projections with RoPE and optional K/V cache appends. The implementations are architecture-agnostic and
run on every device.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from xinfer.ops.reference.norm import _adarms, add_rms_norm

ACTIVATIONS = {"silu": F.silu, "gelu": F.gelu, "gelu_tanh": lambda y: F.gelu(y, approximate="tanh")}


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


def gemm_epilogue(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None, *,
                  activation: str | None = None, residual: torch.Tensor | None = None,
                  residual_gate: torch.Tensor | None = None, row_class: torch.Tensor | None = None,
                  row_scale: torch.Tensor | None = None) -> torch.Tensor:
    """``x · weightᵀ`` with xcompute ``gemm``'s epilogue.

    ``y = act(row_scale · (x·Wᵀ + bias))`` in FP32. Without ``residual`` the result is ``bf16(y)``; with it
    ``bf16(residual + y)``; with a ``residual_gate`` ``[R, N]`` and ``row_class`` the gated branch is rounded first:
    ``bf16(residual + bf16(y · residual_gate[row_class]))``.

    Args:
        x: ``[M, K]`` BF16, or the head-major attention output ``[H, M, D]`` (the native form the attention
            kernels return; equivalent to its packed ``[M, H·D]`` flatten).
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
    if x.ndim == 3:
        x = x.transpose(0, 1).reshape(x.shape[1], -1)
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
                      sin: torch.Tensor, num_heads: int, head_dim: int, *, qk_order: str = "standard") -> torch.Tensor:
    """``[Q | K | V] = x·Wᵀ + bias`` with rotate-half RoPE on every Q and K head (xcompute ``qkv_proj_rope``).

    Equal numbers of Q, K and V heads (a ViT); the rotation covers ``2·cos.shape[-1]`` channels of each head and
    is applied to the FP32 accumulator, so the output rounds once.

    ``qk_order="any"``: the caller only feeds Q and K to attention, which does not change when Q and K reorder their
    head channels the same way; an implementation may then return the rotated channels of each Q and K head in pair
    order (``x_0, x_{R/2}, x_1, x_{R/2+1}, ...``). V keeps its order. The reference returns the standard order.

    Args:
        x: ``[M, K]`` BF16.
        weight: ``[3·H·D, K]`` BF16.
        bias: ``[3·H·D]``.
        cos: FP32 ``[M, R/2]`` cosines of the rotated pairs ``(j, j + R/2)``.
        sin: FP32 ``[M, R/2]``.
        num_heads: Heads ``H``.
        head_dim: Head dimension ``D``.
        qk_order: ``"standard"``, or ``"any"`` (see above).

    Returns:
        ``[M, 3·H·D]`` in ``x.dtype``.
    """
    if qk_order not in ("standard", "any"):
        raise ValueError(f"qkv_rope_epilogue: qk_order must be 'standard' or 'any', got {qk_order!r}")
    m, half = x.shape[0], cos.shape[-1]
    y = (x.float() @ weight.float().t() + bias.float()).view(m, 3, num_heads, head_dim)
    c, s = cos[:, None, None, :], sin[:, None, None, :]
    qk = y[:, :2]
    a, b = qk[..., :half], qk[..., half:2 * half]
    rotated = torch.cat((a * c - b * s, b * c + a * s, qk[..., 2 * half:]), dim=-1)
    return torch.cat((rotated, y[:, 2:]), dim=1).reshape(m, -1).to(x.dtype)



def gemm_add_rms_norm(x: torch.Tensor, projection: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor,
                      eps: float, bias: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """A projection with the residual add and the next RMSNorm: ``s = residual + (x·projectionᵀ + bias)``, then
    ``rms_norm(s)``: ``gemm_epilogue`` and ``add_rms_norm``.

    Args:
        x: ``[M, K]`` BF16.
        projection: ``[N, K]`` BF16.
        residual: ``[..., M, N]`` BF16 residual stream.
        weight: ``[N]`` of the RMSNorm.
        eps: Added to the mean square.
        bias: ``[N]`` BF16, or None.

    Returns:
        ``(s, normed)`` shaped as ``residual``.
    """
    return add_rms_norm(gemm_epilogue(x, projection, bias).view(residual.shape), residual, weight, eps)


def mlp_add_rms_norm(x: torch.Tensor, gate_up: torch.Tensor, down: torch.Tensor, residual: torch.Tensor,
                     weight: torch.Tensor, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    """A decoder MLP with the residual add and the next RMSNorm: ``s = residual + (silu(x·W_gateᵀ)·(x·W_upᵀ))·W_downᵀ``,
    then ``rms_norm(s)``: ``gemm_glu_epilogue``, ``gemm_epilogue`` and ``add_rms_norm``. A backend may write ``s`` into
    ``residual``'s memory (the caller does not read ``residual`` afterwards); this reference does not.

    Args:
        x: ``[M, K]`` BF16, the normed input.
        gate_up: ``[2I, K]`` BF16, gate rows first.
        down: ``[N, I]`` BF16.
        residual: ``[..., M, N]`` BF16 residual stream.
        weight: ``[N]`` of the RMSNorm.
        eps: Added to the mean square.

    Returns:
        ``(s, normed)`` shaped as ``residual``.
    """
    y = gemm_epilogue(gemm_glu_epilogue(x, gate_up), down)
    return add_rms_norm(y.view(residual.shape), residual, weight, eps)


def prefetch_weights(weights) -> None:
    """A hint that ``weights`` (a sequence of tensors) are read soon: implementations may start moving them closer to
    the compute units (for example into L2) without delaying the caller. No effect on results; the reference does
    nothing.

    Args:
        weights: Tensors about to be read.
    """


def join_prefetch() -> None:
    """The caller's stream waits for the work ``prefetch_weights`` started, which must happen before a CUDA graph
    capture ends; the reference does nothing."""


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


# ---------------------------------------------------------------- Wall-X 2B fused operators

def _zero_centered_heads(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """FP32 ``x · (1 + w) · rsqrt(mean(x²) + eps)`` per head, rounded once."""
    x32 = x.float()
    return (x32 * (weight.float() + 1) * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype)

def qkv_norm_split(packed: torch.Tensor, q_weight: torch.Tensor, k_weight: torch.Tensor, num_heads: int,
                   num_kv_heads: int, head_dim: int,
                   eps: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split a Qwen3.5 gated-attention projection and apply the zero-centered q and k norms.

    ``packed`` holds, per token, the query projection with each head's ``[q_h | gate_h]`` side by side (the
    checkpoint's layout), then K, then V.

    Args:
        packed: ``[B, L, 2·H·D + 2·KV·D]``.
        q_weight: ``[D]`` raw zero-centered q-norm weight.
        k_weight: ``[D]`` raw zero-centered k-norm weight.
        num_heads: Query heads ``H``.
        num_kv_heads: K/V heads ``KV``.
        head_dim: Head dimension ``D``.
        eps: Norm epsilon.

    Returns:
        ``(q [B, L, H, D], k [B, L, KV, D], v [B, L, KV, D], gate [B, L, H·D])``, contiguous; q and k normed in
        FP32 and rounded once, v and the gate copied.
    """
    b, n = packed.shape[:2]
    q_end, k_end = 2 * num_heads * head_dim, (2 * num_heads + num_kv_heads) * head_dim
    query = packed[..., :q_end].reshape(b, n, num_heads, 2, head_dim)
    k = packed[..., q_end:k_end].reshape(b, n, num_kv_heads, head_dim)
    v = packed[..., k_end:k_end + num_kv_heads * head_dim].reshape(b, n, num_kv_heads, head_dim)
    return (_zero_centered_heads(query[..., 0, :], q_weight, eps).contiguous(),
            _zero_centered_heads(k, k_weight, eps).contiguous(), v.contiguous(),
            query[..., 1, :].reshape(b, n, num_heads * head_dim))

def qkv_norm_rope_kvcache(x: torch.Tensor, weight: torch.Tensor, q_weight: torch.Tensor, k_weight: torch.Tensor,
                          cos: torch.Tensor, sin: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor,
                          start: int, num_heads: int, num_kv_heads: int, head_dim: int, eps: float) -> torch.Tensor:
    """Project normalized rows, normalize and rotate Q/K, append K/V to the head-major cache, and return Q and gate."""
    return _qk_norm_rope_kvcache(x.float() @ weight.float().t(), x.dtype, q_weight, k_weight, cos, sin, k_cache,
                                 v_cache, start, num_heads, num_kv_heads, head_dim, eps)
