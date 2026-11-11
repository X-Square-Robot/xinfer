"""Fused BF16 operators (Qwen3.5 and the Wall-X flow head), torch only.

Each operator is one xcompute call that replaces a sequence of reference operators: norms with their residual add,
the Gated DeltaNet mixer, the ViT position table and RoPE angles, the q/k split and RoPE of a full-attention
projection, GEMMs with their epilogues and input norms, gated attentions, and the flow head's embedding and Euler
step. The references below compute the same values with torch and fix where the fused kernel rounds (products
accumulate in FP32 and round once at the output unless stated; they follow the production kernels). Kernels evaluate
in their own order (reduction trees, FMA contraction), so backends match these references within a tolerance unless
their registration says ``"bitwise"``.

Conventions: GEMM weights are ``[N, K]`` (the checkpoint's orientation). Zero-centered RMSNorm weights are the
checkpoint's raw ``w`` (the norm scales by ``1 + w``). Modulation tables are ``[R, 3K]`` ``[scale | shift | gate]``
per row class (for example a token's flow role); ``row_class`` (int ``[M]``) picks the row of each token. RoPE tables
are "compact": FP32 ``[..., R / 2]`` cosines and sines of the rotated half-split pairs. Image grids are host sequences
of ``(t, h, w)`` patch counts; patches are in the merge-block order of the Qwen2-VL image processor.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from xinfer.ops import reference


def _zero_centered(x32: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """FP32 ``x · rsqrt(mean(x²) + eps) · (1 + w)``."""
    return x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps) * (1 + weight.float())


def layer_norm_cast(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float) -> torch.Tensor:
    """LayerNorm computed in FP32 and rounded to ``x.dtype`` (``F.layer_norm`` on BF16 input and weights).

    Args:
        x: ``[..., K]`` input.
        weight: ``[K]`` scale, ``x.dtype``.
        bias: ``[K]`` shift, ``x.dtype``.
        eps: Variance epsilon.

    Returns:
        ``[..., K]`` in ``x.dtype``.
    """
    return F.layer_norm(x, (x.shape[-1],), weight, bias, eps)


def zero_centered_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """``x · rsqrt(mean(x²) + eps) · (1 + w)`` in FP32, rounded once (Qwen3.5 RMSNorm with the raw weight).

    Args:
        x: ``[..., K]`` input.
        weight: ``[K]`` raw weight ``w`` (BF16 or FP32).
        eps: Added to the mean square.

    Returns:
        ``[..., K]`` in ``x.dtype``.
    """
    return _zero_centered(x.float(), weight, eps).to(x.dtype)


def add_zero_centered_rms_norm(x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor,
                               eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    """``s = residual + x`` (rounded), then ``zero_centered_rms_norm(s)``.

    Args:
        x: ``[..., K]`` branch output.
        residual: ``[..., K]`` residual stream, same dtype.
        weight: ``[K]`` raw weight ``w``.
        eps: Added to the mean square.

    Returns:
        ``(s, normed)`` in ``x.dtype``.
    """
    s = residual + x
    return s, zero_centered_rms_norm(s, weight, eps)


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
    mixed = reference.causal_conv1d(qkv.transpose(1, 2), conv_weight[:, None, :]).transpose(1, 2)
    q, k, v = torch.split(mixed, [key_dim, key_dim, value_dim], dim=-1)
    repeat = num_v_heads // num_k_heads
    q = q.reshape(b, n, num_k_heads, head_k_dim).repeat_interleave(repeat, dim=2)
    k = k.reshape(b, n, num_k_heads, head_k_dim).repeat_interleave(repeat, dim=2)
    v = v.reshape(b, n, num_v_heads, head_v_dim)
    z, beta_logit, a = torch.split(zba, [value_dim, num_v_heads, num_v_heads], dim=-1)
    beta = beta_logit.sigmoid()
    g = -A_log.float().exp() * F.softplus(a.float() + dt_bias)
    o = reference.chunk_gated_delta_rule(q, k, v, g, beta)
    o = reference.gated_rms_norm(o.reshape(-1, head_v_dim), z.reshape(-1, head_v_dim), norm_weight, eps)
    return o.reshape(b, n, value_dim)


def _merge_order(h: int, w: int, merge: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Patch rows and columns of an ``h × w`` grid in merge-block order."""
    shape = (h // merge, w // merge, merge, merge)
    rows = torch.arange(h, device=device).view(h // merge, 1, merge, 1).expand(shape).reshape(-1)
    cols = torch.arange(w, device=device).view(1, w // merge, 1, merge).expand(shape).reshape(-1)
    return rows, cols


def pos_embed_interp(weight: torch.Tensor, grids, merge_size: int) -> torch.Tensor:
    """Bilinear interpolation of a square position table to each image grid.

    Sample points are ``linspace(0, G - 1, h)`` (rows) and ``linspace(0, G - 1, w)`` (columns); the four corner
    rows are weighted in FP32 (``(1 - dh)(1 - dw)``, ``(1 - dh) dw``, ``dh (1 - dw)``, ``dh dw``, summed in that
    order) and rounded once. Every frame of a grid repeats its first frame.

    Args:
        weight: ``[G², C]`` position table.
        grids: Host sequence of ``(t, h, w)``; ``h`` and ``w`` divisible by ``merge_size``.
        merge_size: Patches merged per side.

    Returns:
        ``[Σ t·h·w, C]`` in ``weight.dtype``.
    """
    side = math.isqrt(weight.shape[0])
    out = []
    for t, h, w in grids:
        points = [torch.linspace(0, side - 1, n, device=weight.device) for n in (h, w)]
        floor = [p.int() for p in points]
        ceil = [(f + 1).clip(max=side - 1) for f in floor]
        dh, dw = (p - f for p, f in zip(points, floor))
        rows, cols = _merge_order(h, w, merge_size, weight.device)
        dh, dw = dh[rows, None], dw[cols, None]
        corner = lambda r, c: weight[(r[rows] * side + c[cols]).long()].float()  # noqa: E731
        value = (1 - dh) * (1 - dw) * corner(floor[0], floor[1])
        value = value + (1 - dh) * dw * corner(floor[0], ceil[1])
        value = value + dh * (1 - dw) * corner(ceil[0], floor[1])
        value = value + dh * dw * corner(ceil[0], ceil[1])
        out.append(value.to(weight.dtype).repeat(t, 1))
    return torch.cat(out)


def rot_pos(inv_freq: torch.Tensor, grids, merge_size: int) -> torch.Tensor:
    """2-D rotary angles of the patches: ``[row · inv_freq | col · inv_freq]`` in FP32.

    Args:
        inv_freq: ``[F]`` FP32 inverse frequencies.
        grids: Host sequence of ``(t, h, w)``.
        merge_size: Patches merged per side.

    Returns:
        ``[Σ t·h·w, 2F]`` FP32.
    """
    out = []
    for t, h, w in grids:
        rows, cols = _merge_order(h, w, merge_size, inv_freq.device)
        angles = torch.cat((rows[:, None].float() * inv_freq, cols[:, None].float() * inv_freq), dim=-1)
        out.append(angles.repeat(t, 1))
    return torch.cat(out)


def _rotate_half_split(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate the first ``2·cos.shape[-1]`` channels of ``x [..., heads, D]`` in FP32, rounded once."""
    half = cos.shape[-1]
    a, b = x[..., :half].float(), x[..., half:2 * half].float()
    rotated = torch.cat((a * cos - b * sin, b * cos + a * sin), dim=-1).to(x.dtype)
    return torch.cat((rotated, x[..., 2 * half:]), dim=-1)


def rope_full(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
              rotary_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Partial half-split RoPE of q and k into new tensors; channels past ``rotary_dim`` are copied.

    Args:
        q: ``[B, L, H, D]`` queries.
        k: ``[B, L, KV, D]`` keys.
        cos: FP32 ``[B or 1, L, rotary_dim / 2]`` compact cosines.
        sin: FP32 ``[B or 1, L, rotary_dim / 2]`` compact sines.
        rotary_dim: Rotated channels per head.

    Returns:
        ``(q, k)`` rotated, contiguous, in their input dtypes.
    """
    half = rotary_dim // 2
    c, s = cos[..., None, :half], sin[..., None, :half]
    return _rotate_half_split(q, c, s).contiguous(), _rotate_half_split(k, c, s).contiguous()


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


def segment_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, cu_seqlens, *,
                      scale: float | None = None) -> torch.Tensor:
    """``varlen_attention`` in the form of ``attention_varlen``: bidirectional attention within each segment.

    The same function as ``varlen_attention``, as a separate operator so that only the compositions validated
    with it run the unified entry (the others keep ``varlen_attention``'s backend). Arguments and result as
    ``reference.varlen_attention``.
    """
    return reference.varlen_attention(q, k, v, cu_seqlens, scale=scale)


def gate_output(o: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """``bf16(bf16(o) · bf16(sigmoid(gate)))``: the attention output times its gate, as two BF16 operations."""
    return o.to(gate.dtype) * torch.sigmoid(gate)


def gated_causal_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, gate: torch.Tensor, *,
                           scale: float | None = None) -> torch.Tensor:
    """Causal attention over one sequence, output times ``sigmoid(gate)`` (xcompute ``attention_varlen``).

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
    o = reference.attention(q.transpose(0, 1)[None], k.transpose(0, 1)[None], v.transpose(0, 1)[None],
                            scale=scale, causal=True)
    return gate_output(o[0].transpose(0, 1).reshape(t, -1), gate)


def gated_kvcache_attention(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor,
                            prefix_len: torch.Tensor, new_pos: int, gate: torch.Tensor, *,
                            flash_mask: torch.Tensor | None = None, scale: float | None = None) -> torch.Tensor:
    """Attention of ``M`` new tokens over a cache prefix and their own keys, output times ``sigmoid(gate)``.

    xcompute ``attention_kvcache`` with ``new_pos``: request ``b`` sees cache rows ``[0, prefix_len[b])`` and
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


def prefix_kvcache_attention(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor,
                             prefix_len: torch.Tensor, new_pos: int, *, flash_mask: torch.Tensor | None = None,
                             scale: float | None = None) -> torch.Tensor:
    """Attention of ``M`` new tokens over a cache prefix and their own keys (xcompute ``attention_kvcache``).

    The visible rows and ``flash_mask`` are those of ``gated_kvcache_attention``, which is this attention times its
    output gate.

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


def flow_input_embed(x: torch.Tensor, weight: torch.Tensor, head_index: torch.Tensor | None = None,
                     extra: torch.Tensor | None = None, input_mask: torch.Tensor | None = None,
                     addends: tuple[torch.Tensor, ...] = (), override: torch.Tensor | None = None,
                     override_mask: torch.Tensor | None = None) -> torch.Tensor:
    """Token inputs of flow rows (xcompute ``denoise_input_embed``).

    ``p = [x | extra] · weight[head]ᵀ`` in FP32 (zero where ``input_mask`` is 0), then the addends in order, or
    ``override`` where ``override_mask`` is set; one rounding to BF16.

    Args:
        x: FP32 ``[B, M, A]`` values.
        weight: FP32 ``[heads, N, A]``, or ``[heads, N, 2A]`` with ``extra`` (``heads <= 2``).
        head_index: int ``[B, M]`` head of each token; None with one head.
        extra: FP32 ``[B, M, A]`` second input half (for example the validity mask), or None.
        input_mask: bool ``[B, M]``, tokens whose projection counts; None for all.
        addends: FP32 ``[B, M, N]`` terms added in order (at most 3).
        override: FP32 ``[B, M, N]`` rows taken as they are where ``override_mask`` is set, or None.
        override_mask: bool ``[B, M]``, or None.

    Returns:
        ``[B, M, N]`` BF16.
    """
    w = weight[0] if head_index is None else weight[head_index.long()]
    inputs = x if extra is None else torch.cat((x, extra), dim=-1)
    p = (inputs[..., None, :] * w).sum(-1)
    if input_mask is not None:
        p = p * input_mask[..., None]
    for term in addends:
        p = p + term
    if override is not None:
        p = torch.where(override_mask[..., None], override, p)
    return p.to(torch.bfloat16)


def _row_modulation(x: torch.Tensor, modulation: torch.Tensor, pinned_modulation, pinned_rows) -> torch.Tensor:
    """``[B, 1 or L, 3K]``: the shared ``[B, 3K]`` modulation, rows ``< pinned_rows[b]`` taking the pinned one."""
    shared = modulation.reshape(modulation.shape[0], 1, -1)
    if pinned_rows is None:
        return shared
    rows = torch.arange(x.shape[1], device=x.device)[None, :, None] < pinned_rows.reshape(-1, 1, 1)
    return torch.where(rows, pinned_modulation.reshape(-1, 1, shared.shape[-1]), shared)


def modulated_rms_norm(x: torch.Tensor, modulation: torch.Tensor, eps: float,
                       pinned_modulation: torch.Tensor | None = None,
                       pinned_rows: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Adaptive RMSNorm with a packed ``[scale | shift | gate]`` modulation (xcompute ``adarmsnorm``).

    ``normed = bf16(x · rsqrt(mean(x²) + eps) · (1 + scale) + shift)`` in FP32 (``reference.ada_rms_norm``). With
    ``pinned_rows`` the first ``pinned_rows[b]`` rows of batch row ``b`` take ``pinned_modulation`` (RTC pinned rows)
    and the gate is per row.

    Args:
        x: ``[B, L, K]`` BF16.
        modulation: ``[B, 3K]`` (FP32 or BF16).
        eps: Added to the mean square.
        pinned_modulation: ``[B, 3K]`` of the pinned rows, or None.
        pinned_rows: int32 ``[B]``, or None (given with ``pinned_modulation``).

    Returns:
        ``(normed [B, L, K], gate [B, 1 or L, K])`` in ``x.dtype``.
    """
    scale, shift, gate = _row_modulation(x, modulation, pinned_modulation, pinned_rows).chunk(3, dim=-1)
    return reference.ada_rms_norm(x, scale, shift, eps), gate.to(x.dtype)


def gated_residual_modulated_rms_norm(residual: torch.Tensor, branch: torch.Tensor, branch_gate: torch.Tensor,
                                      modulation: torch.Tensor, eps: float,
                                      pinned_modulation: torch.Tensor | None = None,
                                      pinned_rows: torch.Tensor | None = None
                                      ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``s = residual + bf16(branch · branch_gate)``, then ``modulated_rms_norm(s)`` (xcompute
    ``gated_residual_adarmsnorm``).

    Args:
        residual: ``[B, L, K]`` BF16 residual stream.
        branch: ``[B, L, K]`` BF16 branch output.
        branch_gate: ``[B, 1 or L, K]`` BF16 gate of the branch.
        modulation: ``[B, 3K]`` of the next norm.
        eps: Added to the mean square.
        pinned_modulation: As in ``modulated_rms_norm``.
        pinned_rows: As in ``modulated_rms_norm``.

    Returns:
        ``(normed, gate, s)`` in ``residual.dtype``.
    """
    s = reference.gated_residual(residual, branch, branch_gate)
    normed, gate = modulated_rms_norm(s, modulation, eps, pinned_modulation, pinned_rows)
    return normed, gate, s


def identity_layer_norms_(residual: torch.Tensor, branch: torch.Tensor, branch_gate: torch.Tensor,
                          input_modulation: torch.Tensor, post_modulation: torch.Tensor, eps: float,
                          pinned_input_modulation: torch.Tensor | None = None,
                          pinned_post_modulation: torch.Tensor | None = None,
                          pinned_rows: torch.Tensor | None = None) -> torch.Tensor:
    """The norms of a layer whose mixer is the identity, in place (xcompute ``double_adarmsnorm``).

    Two ``gated_residual_modulated_rms_norm`` transitions: the previous branch into the residual and the input norm,
    then the identity mixer's gated output into the residual and the post norm. ``residual`` becomes the new
    residual and ``branch`` the post norm's output.

    Args:
        residual: ``[B, L, K]`` BF16 residual stream; overwritten.
        branch: ``[B, L, K]`` BF16 previous branch output; overwritten with the normed rows.
        branch_gate: ``[B, 1 or L, K]`` gate of ``branch``.
        input_modulation: ``[B, 3K]`` of the input norm.
        post_modulation: ``[B, 3K]`` of the post norm.
        eps: Added to the mean square.
        pinned_input_modulation, pinned_post_modulation: ``[B, 3K]`` of the pinned rows, or None.
        pinned_rows: int32 ``[B]``, or None.

    Returns:
        The post norm's gate ``[B, 1 or L, K]``.
    """
    normed, gate, s = gated_residual_modulated_rms_norm(residual, branch, branch_gate, input_modulation, eps,
                                                        pinned_input_modulation, pinned_rows)
    normed, gate, s = gated_residual_modulated_rms_norm(s, normed, gate, post_modulation, eps,
                                                        pinned_post_modulation, pinned_rows)
    residual.copy_(s)
    branch.copy_(normed)
    return gate


def qkv_norm_rope_kvcache(x: torch.Tensor, weight: torch.Tensor, q_weight: torch.Tensor, k_weight: torch.Tensor,
                          cos: torch.Tensor, sin: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor,
                          start: int, num_heads: int, num_kv_heads: int, head_dim: int, eps: float) -> torch.Tensor:
    """Project normalized rows, normalize and rotate Q/K, append K/V to the head-major cache, and return Q and gate."""
    return _qk_norm_rope_kvcache(x.float() @ weight.float().t(), x.dtype, q_weight, k_weight, cos, sin, k_cache,
                                 v_cache, start, num_heads, num_kv_heads, head_dim, eps)


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
