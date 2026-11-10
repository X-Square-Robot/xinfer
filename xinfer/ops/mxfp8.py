"""MXFP8 reference: E4M3 elements with E8M0 block scales (OCP MX), torch only.

Format used by the PyTorch reference operators:

- elements are ``float8_e4m3fn``; scales are E8M0 bytes, one per 32 columns
  of K. With ``share_scale_128`` one scale covers 128 columns (the max over
  the 128) and is replicated into its four 32-column slots ("1x128 blocks").
- scale = 2^(e - 127) with ``e = exponent(amax) - 8 + (mantissa(amax) > 0.75)``
  clamped to [0, 254] (255 for inf/NaN amax): the smallest power of two that
  maps amax into the E4M3 range (448).
- an element is ``x * 2^(127 - e)`` in FP32, clamped to ±448, rounded to
  nearest even.
- scales are stored in the SWIZZLE_32_4_4 layout used by block-scaled GEMMs:
  rows padded to 128, 32-column blocks padded to 4.

``quantize`` preserves NaN inputs as E4M3 NaNs. ``gemm`` dequantizes both operands
and multiplies in FP32. Tensor-core kernels may accumulate in a different order.

Fused forms (the Wall-X MXFP8 service recipe). Each skips a BF16 rounding
that the unfused composition has:

- ``rmsnorm_quantize``: RMSNorm computed in FP32 and quantized directly;
- ``silu_mul_quantize``: ``silu(gate) * up`` computed in FP32 and quantized directly;
- ``gemm_rope``: QKV GEMM + bias + rotary embedding on the FP32 accumulator;
- ``gemm_silu_mul``: gate/up GEMM + ``silu(gate) * up`` on the FP32 accumulator;
- ``gemm_residual``: GEMM + residual on the FP32 accumulator.

``gemm_rope`` and ``gemm_silu_mul`` take weights whose rows were reordered
for the epilogue (``rope_pair_permutation``, ``gate_up_interleave``).
"""

from __future__ import annotations

import torch

E4M3_MAX = 448.0
BLOCK = 32
SHARED_BLOCK = 128


def scale_elements(rows: int, k: int, *, block: int = BLOCK) -> int:
    """Bytes of a SWIZZLE_32_4_4 scale tensor for a logical [rows, k] matrix.

    Args:
        rows: logical rows ``M`` (padded to a multiple of 128).
        k: logical columns ``K`` (``ceil(K / block)`` scale blocks, padded to a multiple of 4).
        block: columns per scale (32 for MXFP8, 16 for NVFP4).

    Returns:
        The number of scale bytes, including padding.
    """
    padded_rows = (rows + 127) // 128 * 128
    blocks = (k + block - 1) // block
    return padded_rows * ((blocks + 3) // 4 * 4)


def scale_offsets(rows: int, k: int, device=None, *, block: int = BLOCK) -> torch.Tensor:
    """Physical offset of scale[row, block] for row < rows, block < padded block count.

    SWIZZLE_32_4_4: each 128-row x 4-block tile occupies 512 consecutive bytes, tiles in row-tile-major
    order; within a tile, ``r = row % 128`` and ``b = block % 4`` sit at ``(r % 32) * 16 + (r // 32) * 4 + b``.

    Args:
        rows: logical rows ``M``.
        k: logical columns ``K``.
        device: device of the returned tensor.
        block: columns per scale (32 for MXFP8, 16 for NVFP4).

    Returns:
        int64 ``[rows, padded block count]`` offsets into the flat scale tensor.
    """
    blocks = (k + block - 1) // block
    columns = (blocks + 3) // 4 * 4
    col_groups = columns // 4
    row = torch.arange(rows, device=device, dtype=torch.int64)[:, None]
    block = torch.arange(columns, device=device, dtype=torch.int64)[None, :]
    in_block = row & 127
    tile = (row // 128) * col_groups + block // 4
    return tile * 512 + (in_block & 31) * 16 + (in_block // 32) * 4 + (block & 3)


def e8m0_ceil(amax: torch.Tensor) -> torch.Tensor:
    """Biased E8M0 scale exponent of each block amax, from the FP32 bit pattern.

    ``e = exponent(amax) - 8 + (mantissa(amax) > 0.75)`` clamped to ``[0, 254]``, and 255 for inf/NaN: the
    smallest power of two ``2^(e - 127)`` that maps ``amax`` into ±448.

    Args:
        amax: non-negative block maxima, any shape (converted to FP32).

    Returns:
        int32 tensor of the same shape.
    """
    bits = amax.to(torch.float32).contiguous().view(torch.int32)
    exponent = (bits >> 23) & 0xFF
    mantissa = bits & 0x7FFFFF
    biased = (exponent - 8 + (mantissa > 0x600000).to(torch.int32)).clamp(0, 254)
    return torch.where(exponent == 0xFF, torch.full_like(biased, 255), biased)


def quantize(x: torch.Tensor, *, share_scale_128: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """BF16 ``[M, K]`` → (E4M3 ``[M, K]``, E8M0 scales uint8 ``[scale_elements(M, K)]``).

    The reference of the ``mxfp8_quantize`` operator. Each row's elements and scales depend only on that row.

    Args:
        x: BF16 ``[M, K]``, ``K`` a multiple of 32.
        share_scale_128: one scale per 128 columns (replicated into its four 32-column slots) instead of one
            per 32 columns.

    Returns:
        ``(q, scales)``: contiguous ``float8_e4m3fn [M, K]`` and flat uint8 ``[scale_elements(M, K)]`` in the
        SWIZZLE_32_4_4 layout, padding bytes zero.

    Raises:
        ValueError: ``x`` is not a 2D BF16 tensor, or ``K`` is not a multiple of 32.
    """
    if x.dtype != torch.bfloat16 or x.ndim != 2:
        raise ValueError("mxfp8 quantize expects a 2D BF16 tensor")
    return quantize_values(x.to(torch.float32), share_scale_128=share_scale_128)


def quantize_values(xf: torch.Tensor, *, share_scale_128: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize FP32 values ``[M, K]`` (the fused forms quantize FP32 intermediates).

    Args:
        xf: ``[M, K]`` values, converted to FP32; ``K`` a multiple of 32.
        share_scale_128: one scale per 128 columns instead of one per 32 columns.

    Returns:
        ``(q, scales)`` as in ``quantize``.

    Raises:
        ValueError: ``K`` is not a multiple of 32.
    """
    rows, k = xf.shape
    if k % BLOCK:
        raise ValueError(f"K={k} must be divisible by {BLOCK}")
    x = xf.to(torch.float32)
    blocks = (k // BLOCK + 3) // 4 * 4  # scale columns, padded to whole 128 tiles
    padded = torch.zeros(rows, blocks * BLOCK, dtype=torch.float32, device=x.device)
    padded[:, :k] = x
    if share_scale_128:
        amax = padded.view(rows, blocks // 4, SHARED_BLOCK).abs().amax(-1)
        e = e8m0_ceil(amax).repeat_interleave(4, dim=1)
    else:
        e = e8m0_ceil(padded.view(rows, blocks, BLOCK).abs().amax(-1))
    inverse = torch.ldexp(torch.ones_like(e, dtype=torch.float32), (127 - e))
    scaled = padded.view(rows, blocks, BLOCK) * inverse[..., None]
    q = scaled.clamp(-E4M3_MAX, E4M3_MAX).view(rows, -1)[:, :k].to(torch.float8_e4m3fn)
    scales = torch.zeros(scale_elements(rows, k), dtype=torch.uint8, device=x.device)
    scales[scale_offsets(rows, k, x.device).reshape(-1)] = e.to(torch.uint8).reshape(-1)
    return q.contiguous(), scales


def dequantize(q: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """E4M3 ``[M, K]`` + swizzled E8M0 scales → FP32 ``[M, K]``.

    Args:
        q: ``float8_e4m3fn [M, K]``, ``K`` a multiple of 32.
        scales: flat uint8 scales in the SWIZZLE_32_4_4 layout of ``quantize``.

    Returns:
        FP32 ``[M, K]``: each element times ``2^(e - 127)`` of its 32-column block.
    """
    rows, k = q.shape
    offsets = scale_offsets(rows, k, q.device)[:, : k // BLOCK]
    e = scales[offsets.reshape(-1)].reshape(rows, k // BLOCK).to(torch.int32)
    scale = torch.ldexp(torch.ones(e.shape, dtype=torch.float32, device=q.device), e - 127)
    return (q.to(torch.float32).view(rows, k // BLOCK, BLOCK) * scale[..., None]).view(rows, k)


def gemm(a_q, a_scales, w_q, w_scales, bias=None, *, fast_accum: bool = False) -> torch.Tensor:
    """``dequant(a) @ dequant(w).T (+ bias)`` in FP32, rounded to BF16: ``[M, N]``.

    ``fast_accum`` selects the kernels' faster, less precise FP8 accumulation;
    the reference ignores it and always accumulates in FP32.

    Args:
        a_q: ``float8_e4m3fn [M, K]`` activations.
        a_scales: swizzled E8M0 scales of ``a_q``.
        w_q: ``float8_e4m3fn [N, K]`` weight.
        w_scales: swizzled E8M0 scales of ``w_q``.
        bias: optional ``[N]``, added in FP32.
        fast_accum: backend accumulation mode; no effect on the reference.

    Returns:
        BF16 ``[M, N]``.
    """
    out = dequantize(a_q, a_scales) @ dequantize(w_q, w_scales).T
    if bias is not None:
        out = out + bias.to(torch.float32)
    return out.to(torch.bfloat16)


# ----------------------------------------------------------------- fused forms
NORM_THREADS = 128  # one CTA of 128 threads per row in the fused RMSNorm kernel


def _sum_of_squares(x32: torch.Tensor, threads: int = NORM_THREADS) -> torch.Tensor:
    """Row sums of squares in the fused RMSNorm kernels' order.

    Thread ``t`` accumulates columns ``t, t + threads, ...`` in sequence; the
    partial sums are then added in a binary tree (stride ``threads / 2``, ..., 1).
    For BF16 inputs every square is exact in FP32, so only the order matters.

    Args:
        x32: FP32 ``[M, K]``.
        threads: threads per row of the kernel (128 for MXFP8, 256 for NVFP4), a power of two.

    Returns:
        FP32 ``[M]`` row sums of squares.
    """
    rows, k = x32.shape
    sq = x32 * x32
    pad = (-k) % threads
    if pad:
        sq = torch.nn.functional.pad(sq, (0, pad))
    chunks = sq.view(rows, -1, threads)
    acc = chunks[:, 0]
    for c in range(1, chunks.shape[1]):
        acc = acc + chunks[:, c]
    stride = threads // 2
    while stride:
        acc = acc[:, :stride] + acc[:, stride: 2 * stride]
        stride //= 2
    return acc[:, 0]


def rmsnorm_quantize(x: torch.Tensor, weight: torch.Tensor, eps: float, *,
                     share_scale_128: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """MXFP8 of ``(x * rsqrt(mean(x²) + eps)) * weight`` computed in FP32, without a BF16 rounding.

    ``x`` BF16 ``[M, K]``; ``weight`` ``[K]`` (the Wall-X service used BF16 weights).

    The mean square uses ``_sum_of_squares`` (the fused kernel's summation order) divided by ``K``.

    Args:
        x: BF16 ``[M, K]``, ``K`` a multiple of 32.
        weight: ``[K]`` norm weight, converted to FP32.
        eps: added to the mean square.
        share_scale_128: as in ``quantize``.

    Returns:
        ``(q, scales)`` as in ``quantize``: ``float8_e4m3fn [M, K]`` and the swizzled scales.

    Raises:
        ValueError: ``x`` is not a 2D BF16 tensor, or ``K`` is not a multiple of 32.
    """
    if x.dtype != torch.bfloat16 or x.ndim != 2:
        raise ValueError("mxfp8 rmsnorm_quantize expects a 2D BF16 tensor")
    x32 = x.to(torch.float32)
    inv_rms = torch.rsqrt(_sum_of_squares(x32) / x.shape[1] + eps)
    return quantize_values((x32 * inv_rms[:, None]) * weight.to(torch.float32),
                           share_scale_128=share_scale_128)


def _silu_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """``(gate / (1 + exp(-gate))) * up`` in the inputs' dtype (FP32 in the fused forms)."""
    return (gate / (1 + torch.exp(-gate))) * up


def silu_mul_quantize(gate_up: torch.Tensor, *, share_scale_128: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """MXFP8 of ``silu(gate) * up`` computed in FP32 from BF16 ``[M, 2K] = [gate | up]``.

    Args:
        gate_up: BF16 ``[M, 2K]``, the gate half first; ``K`` a multiple of 32.
        share_scale_128: as in ``quantize``.

    Returns:
        ``(q, scales)``: ``float8_e4m3fn [M, K]`` and the swizzled scales of an ``[M, K]`` matrix.

    Raises:
        ValueError: ``gate_up`` is not a 2D BF16 tensor with an even width, or ``K`` is not a multiple of 32.
    """
    if gate_up.dtype != torch.bfloat16 or gate_up.ndim != 2 or gate_up.shape[1] % 2:
        raise ValueError("mxfp8 silu_mul_quantize expects a 2D BF16 [M, 2K] tensor")
    g, u = gate_up.to(torch.float32).chunk(2, dim=1)
    return quantize_values(_silu_mul(g, u), share_scale_128=share_scale_128)


def rope_pair_permutation(q_dim: int, k_dim: int, head_dim: int, n: int) -> torch.Tensor:
    """Row order of a QKV weight for ``gemm_rope``.

    Within each q/k head, rows ``j`` and ``j + head_dim/2`` become adjacent (``2j``, ``2j+1``); v rows keep
    their place.

    Args:
        q_dim: width of q (``num_heads * head_dim``).
        k_dim: width of k (``num_kv_heads * head_dim``).
        head_dim: head dimension, a multiple of 4.
        n: total output rows of the QKV weight.

    Returns:
        int64 ``[n]`` permutation; ``weight[perm]`` (and ``bias[perm]``) is the stored order.

    Raises:
        ValueError: ``head_dim`` is not a multiple of 4, ``q_dim``/``k_dim`` are not whole heads, or
            ``q_dim + k_dim > n``.
    """
    if head_dim % 4 or q_dim % head_dim or k_dim % head_dim or q_dim + k_dim > n:
        raise ValueError("rope pair permutation needs head_dim % 4 == 0 and whole q/k heads")
    perm = torch.arange(n)
    half = head_dim // 2
    j = torch.arange(half)
    for base in range(0, q_dim + k_dim, head_dim):
        perm[base + 2 * j] = base + j
        perm[base + 2 * j + 1] = base + half + j
    return perm


def gate_up_interleave(intermediate: int) -> torch.Tensor:
    """Row order of a ``[gate | up]`` weight for ``gemm_silu_mul``: ``g0, u0, g1, u1, ...``.

    Args:
        intermediate: width of each half.

    Returns:
        int64 ``[2 * intermediate]`` permutation; ``weight[perm]`` (and ``bias[perm]``) is the stored order.
    """
    return torch.arange(2 * intermediate).view(2, intermediate).t().reshape(-1)


def rope_table(cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """FP32 ``cos``/``sin`` ``[M, D/2]`` → the ``gemm_rope`` table ``[D/4, M, 4]``.

    Entry ``[i, m]`` is ``(cos[m, 2i], sin[m, 2i], cos[m, 2i+1], sin[m, 2i+1])``,
    read by the kernels as one float4.

    Args:
        cos: ``[M, D/2]`` cosines of the distinct frequencies (``D`` the head dim, a multiple of 4).
        sin: ``[M, D/2]`` sines.

    Returns:
        Contiguous FP32 ``[D/4, M, 4]``.
    """
    c, s = cos.to(torch.float32).t(), sin.to(torch.float32).t()
    return torch.stack((c[0::2], s[0::2], c[1::2], s[1::2]), dim=-1).contiguous()


def rope_table_cos_sin(table: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Inverse of ``rope_table``: ``[D/4, M, 4]`` → ``cos``, ``sin`` ``[M, D/2]``.

    Args:
        table: ``[D/4, M, 4]`` as built by ``rope_table``.

    Returns:
        ``(cos, sin)``, each ``[M, D/2]`` in ``table.dtype`` (non-contiguous views).
    """
    m = table.shape[1]
    pairs = table.permute(1, 0, 2).reshape(m, -1, 2)
    return pairs[..., 0], pairs[..., 1]


def gemm_rope(a_q, a_scales, w_q, w_scales, bias, table, q_dim: int, k_dim: int, head_dim: int) -> torch.Tensor:
    """QKV GEMM with bias and the rotary embedding applied to the FP32 accumulator.

    ``w_q`` rows are in ``rope_pair_permutation`` order (and ``bias`` with them);
    the output ``[M, N]`` BF16 is in the standard layout ``[q | k | v]``, q and k
    rotated (``x * cos + rotate_half(x) * sin``), each value rounded once.

    Args:
        a_q: ``float8_e4m3fn [M, K]`` activations.
        a_scales: swizzled E8M0 scales of ``a_q``.
        w_q: ``float8_e4m3fn [N, K]`` QKV weight, rows in ``rope_pair_permutation`` order.
        w_scales: swizzled E8M0 scales of ``w_q``.
        bias: ``[N]`` in the same row order, or None.
        table: FP32 ``[head_dim/4, M, 4]`` from ``rope_table`` (one row per activation row).
        q_dim: width of q.
        k_dim: width of k; columns from ``q_dim + k_dim`` on are v and are not rotated.
        head_dim: head dimension.

    Returns:
        BF16 ``[M, N]`` in the standard ``[q | k | v]`` layout.
    """
    acc = dequantize(a_q, a_scales) @ dequantize(w_q, w_scales).T
    if bias is not None:
        acc = acc + bias.to(torch.float32)
    m, n = acc.shape
    rot, half = q_dim + k_dim, head_dim // 2
    cos, sin = rope_table_cos_sin(table)
    c, s = cos[:, None, :], sin[:, None, :]
    pairs = acc[:, :rot].reshape(m, rot // head_dim, half, 2)
    a, b = pairs[..., 0], pairs[..., 1]
    roped = torch.cat((a * c - b * s, b * c + a * s), dim=-1).reshape(m, rot)
    return torch.cat((roped, acc[:, rot:]), dim=1).to(torch.bfloat16)


def gemm_silu_mul(a_q, a_scales, w_q, w_scales, bias=None) -> torch.Tensor:
    """Gate/up GEMM (rows in ``gate_up_interleave`` order) → ``silu(gate) * up`` ``[M, N/2]`` BF16.

    Computed on the FP32 accumulator and rounded once.

    Args:
        a_q: ``float8_e4m3fn [M, K]`` activations.
        a_scales: swizzled E8M0 scales of ``a_q``.
        w_q: ``float8_e4m3fn [N, K]`` gate/up weight, rows in ``gate_up_interleave`` order.
        w_scales: swizzled E8M0 scales of ``w_q``.
        bias: optional ``[N]`` in the same row order.

    Returns:
        BF16 ``[M, N/2]``.
    """
    acc = dequantize(a_q, a_scales) @ dequantize(w_q, w_scales).T
    if bias is not None:
        acc = acc + bias.to(torch.float32)
    return _silu_mul(acc[:, 0::2], acc[:, 1::2]).to(torch.bfloat16)


def gemm_residual(a_q, a_scales, w_q, w_scales, residual, bias=None) -> torch.Tensor:
    """``a @ w.T (+ bias) + residual`` added to the FP32 accumulator, rounded once to BF16.

    Args:
        a_q: ``float8_e4m3fn [M, K]`` activations.
        a_scales: swizzled E8M0 scales of ``a_q``.
        w_q: ``float8_e4m3fn [N, K]`` weight.
        w_scales: swizzled E8M0 scales of ``w_q``.
        residual: ``[M, N]`` (BF16 in the models), converted to FP32.
        bias: optional ``[N]``.

    Returns:
        BF16 ``[M, N]``.
    """
    acc = dequantize(a_q, a_scales) @ dequantize(w_q, w_scales).T
    if bias is not None:
        acc = acc + bias.to(torch.float32)
    return (acc + residual.to(torch.float32)).to(torch.bfloat16)
