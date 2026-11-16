"""NVFP4 reference: E2M1 elements with E4M3 block scales, torch only.

Format, following the Wall-X NVFP4 kernels of the production service and the
GEMMs they feed:

- elements are E2M1 (0, 0.5, 1, 1.5, 2, 3, 4, 6 and their negatives), packed
  two per byte as uint8 ``[M, K/2]``: column ``2j`` in the low nibble,
  ``2j + 1`` in the high nibble; bit 3 of a nibble is the sign.
- one E4M3 (``float8_e4m3fn``) scale per 16 columns of K, stored in the
  SWIZZLE_32_4_4 layout (``xinfer.ops.layout.nvfp4``; padding bytes zero).
- a weight may carry a per-tensor FP32 global scale ``g``; a value is
  ``e2m1 · scale · g``. Activations have ``g = 1``.
- block scale, ``mse=False``: ``e4m3(clamp(amax / (6g), 2^-6, 448))``, rounded
  to nearest even.
- block scale, ``mse=True`` (weights): of the 126 positive finite E4M3 values
  ``v``, the first whose reconstruction with ``v · g`` has the smallest
  squared error over the block, summed in the kernel's order (``_block_sum``).
- an element is ``x / (scale · g)`` in FP32, rounded to the nearest E2M1 value
  (ties to the even code) and saturated to ±6; a negative value that rounds
  to zero keeps its sign bit.

The weight global scale is ``amax / (448 · 6)`` of the whole weight, rounded to FP32
(``xinfer.models.layers.quantization.nvfp4.global_scale``),
the Wall-X ``nvfp4_mse_1d1d_16bs`` recipe. ``gemm`` multiplies the dequantized
operands in FP32 and applies ``alpha`` (the weight's global scale) and the
bias to the FP32 accumulator, rounding once to BF16.

Fused forms (the Wall-X NVFP4 service):

- ``rmsnorm_quantize``: RMSNorm in FP32 in the kernel's summation order; the
  normalized value is rounded to BF16 before it is quantized, as the kernel does;
- ``silu_mul_quantize``: ``silu(gate) * up`` computed in FP32 and quantized
  without a BF16 rounding.

The reference divides exactly. The Wall-X kernels are built with fast-math
(approximate division, ``rsqrtf``, ``__expf``), so where an exact quotient is a
rounding tie, or within a few ulps of one, a kernel may pick the neighbouring
scale or element.
"""

from __future__ import annotations

import torch

from xinfer.ops.reference import mxfp8
from xinfer.ops.reference.gemm import ACTIVATIONS, rope_qk_heads
from xinfer.ops.reference.norm import add_zero_centered_rms_norm as bf16_add_zero_centered_rms_norm
from xinfer.ops.reference.norm import layer_norm_cast
from xinfer.ops.layout.nvfp4 import BLOCK, scale_elements, scale_offsets

E2M1_MAX = 6.0
E4M3_MAX = 448.0
MIN_SCALE = 2.0 ** -6  # smallest normal E4M3: the floor of an amax block scale
NORM_THREADS = 256  # one CTA of 256 threads per row in the fused RMSNorm kernel
_FLT_MAX = torch.finfo(torch.float32).max
_E2M1_MIDPOINTS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)


def e2m1_codes(v: torch.Tensor) -> torch.Tensor:
    """E2M1 codes of FP32 values: nearest value, ties to the even code, saturated to ±6.

    Args:
        v: FP32 values (already divided by their scale), any shape.

    Returns:
        uint8 codes ``0..15`` of the same shape: magnitude index in bits 0-2 (into 0, 0.5, ..., 6), sign in bit 3
        (set for negative values, including those that round to zero).
    """
    a = v.abs().clamp(max=E2M1_MAX)
    # comparisons with host scalars, so that the reference runs inside a CUDA graph capture
    down = sum((a > m).to(torch.int64) for m in _E2M1_MIDPOINTS)  # midpoints below a: the code when ties round down
    tie = sum((a == m).to(torch.int64) for m in _E2M1_MIDPOINTS) > 0
    code = torch.where(tie, down + (down & 1), down)
    return (code | (torch.signbit(v).to(code.dtype) << 3)).to(torch.uint8)


def e2m1_values(codes: torch.Tensor) -> torch.Tensor:
    """FP32 values of E2M1 codes (``e2m1_codes``), any shape."""
    c = codes.to(torch.int64)
    exponent, mantissa = (c >> 1) & 3, (c & 1).to(torch.float32)
    # 0.5·m for the subnormals (exponent 0), else 2^(exponent - 1)·(1 + 0.5·m); exact in FP32
    magnitude = torch.where(exponent == 0, 0.5 * mantissa,
                            torch.exp2((exponent - 1).to(torch.float32)) * (1.0 + 0.5 * mantissa))
    return torch.where((c & 8) > 0, -magnitude, magnitude)


def _block_sum(e: torch.Tensor) -> torch.Tensor:
    """Sum the last dim (16) as the kernel's warp reduction does: halves added pairwise, 8 + 8, 4 + 4, ..."""
    while e.shape[-1] > 1:
        half = e.shape[-1] // 2
        e = e[..., :half] + e[..., half:]
    return e[..., 0]


def _mse_scales(blocks: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """E4M3 scale per block of FP32 ``[rows, n, 16]`` minimizing the reconstruction error under global scale ``g``.

    Candidates are the E4M3 codes 1..126 in increasing order; a later candidate replaces the best only with a
    strictly smaller error, so ties keep the smaller scale.
    """
    candidates = torch.arange(1, 127, dtype=torch.uint8, device=blocks.device)
    values = candidates.view(torch.float8_e4m3fn).float()
    best = torch.full(blocks.shape[:2], _FLT_MAX, device=blocks.device)
    best_code = torch.ones(blocks.shape[:2], dtype=torch.uint8, device=blocks.device)
    for code, value in zip(candidates, values):
        decode = value * g
        delta = blocks - e2m1_values(e2m1_codes(blocks / decode)) * decode
        error = _block_sum(delta * delta)
        better = error < best
        best = torch.where(better, error, best)
        best_code = torch.where(better, code, best_code)
    return best_code.view(torch.float8_e4m3fn)


def quantize_values(xf: torch.Tensor, *, global_scale: float = 1.0,
                    mse: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize FP32 values ``[M, K]`` (the fused forms quantize FP32 intermediates).

    Args:
        xf: ``[M, K]`` values, converted to FP32; ``K`` a multiple of 16.
        global_scale: per-tensor scale ``g`` (FP32-representable); 1.0 for activations.
        mse: choose block scales by the MSE search instead of the block amax.

    Returns:
        ``(q, scales)``: packed E2M1 uint8 ``[M, K/2]`` and flat ``float8_e4m3fn [scale_elements(M, K)]`` in the
        SWIZZLE_32_4_4 layout, padding bytes zero.

    Raises:
        ValueError: ``K`` is not a multiple of 16.
    """
    rows, k = xf.shape
    if k % BLOCK:
        raise ValueError(f"K={k} must be divisible by {BLOCK}")
    x = xf.to(torch.float32).reshape(rows, k // BLOCK, BLOCK)
    g = torch.full((), global_scale, device=x.device)  # on the device: CUDA divides by a host scalar via a reciprocal
    if mse:
        scale = _mse_scales(x, g)
    else:
        scale = (x.abs().amax(-1) / (g * E2M1_MAX)).clamp(MIN_SCALE, E4M3_MAX).to(torch.float8_e4m3fn)
    codes = e2m1_codes(x / (scale.float() * g)[..., None]).reshape(rows, k)
    q = codes[:, 0::2] | (codes[:, 1::2] << 4)
    scales = torch.zeros(scale_elements(rows, k), dtype=torch.uint8, device=x.device)
    scales[scale_offsets(rows, k, x.device)[:, : k // BLOCK].reshape(-1)] = scale.view(torch.uint8).reshape(-1)
    return q.contiguous(), scales.view(torch.float8_e4m3fn)


def quantize(x: torch.Tensor, *, global_scale: float = 1.0, mse: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """BF16 ``[M, K]`` → (packed E2M1 uint8 ``[M, K/2]``, swizzled E4M3 scales).

    The reference of the ``nvfp4_quantize`` operator. Each row's elements and scales depend only on that row.

    Args:
        x: BF16 ``[M, K]``, ``K`` a multiple of 16.
        global_scale: per-tensor scale ``g``; 1.0 for activations and ``nvfp4`` weights.
        mse: MSE block scales (``nvfp4_mse`` weights) instead of amax scales.

    Returns:
        ``(q, scales)`` as in ``quantize_values``.

    Raises:
        ValueError: ``x`` is not a 2D BF16 tensor, or ``K`` is not a multiple of 16.
    """
    if x.dtype != torch.bfloat16 or x.ndim != 2:
        raise ValueError("nvfp4 quantize expects a 2D BF16 tensor")
    return quantize_values(x, global_scale=global_scale, mse=mse)


def dequantize(q: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Packed E2M1 ``[M, K/2]`` + swizzled E4M3 scales → FP32 ``[M, K]`` (without the global scale).

    Args:
        q: uint8 ``[M, K/2]`` as returned by ``quantize``.
        scales: flat ``float8_e4m3fn`` scales in the SWIZZLE_32_4_4 layout.

    Returns:
        FP32 ``[M, K]``: each element times its 16-column block's scale.
    """
    rows, k = q.shape[0], 2 * q.shape[1]
    values = e2m1_values(torch.stack((q & 15, q >> 4), dim=-1).reshape(rows, k))
    offsets = scale_offsets(rows, k, q.device)[:, : k // BLOCK].reshape(-1)
    scale = scales.view(torch.uint8)[offsets].view(torch.float8_e4m3fn).float().reshape(rows, k // BLOCK)
    return (values.view(rows, k // BLOCK, BLOCK) * scale[..., None]).view(rows, k)


def gemm(a_q, a_scales, w_q, w_scales, bias=None, *, alpha: float = 1.0) -> torch.Tensor:
    """``alpha · dequant(a) @ dequant(w).T (+ bias)`` in FP32, rounded once to BF16: ``[M, N]``.

    Args:
        a_q: packed E2M1 ``[M, K/2]`` activations.
        a_scales: their E4M3 scales.
        w_q: packed E2M1 ``[N, K/2]`` weight.
        w_scales: its E4M3 scales.
        bias: optional ``[N]``, added to the FP32 accumulator after ``alpha``.
        alpha: the weight's global scale.

    Returns:
        BF16 ``[M, N]``.
    """
    out = (dequantize(a_q, a_scales) @ dequantize(w_q, w_scales).T) * alpha
    if bias is not None:
        out = out + bias.to(torch.float32)
    return out.to(torch.bfloat16)


def rmsnorm_quantize(x: torch.Tensor, weight: torch.Tensor, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    """NVFP4 of ``bf16(x · (rsqrt(mean(x²) + eps) · weight))``, the Wall-X fused RMSNorm quantizer.

    The mean square sums in the kernel's order (256 threads per row, ``mxfp8._sum_of_squares``). The weight is
    rounded to BF16 first (the kernel's input type), and the normalized value is rounded to BF16 before it is
    quantized, so this equals an RMSNorm with that arithmetic followed by ``quantize``.

    Args:
        x: BF16 ``[M, K]``, ``K`` a multiple of 16.
        weight: ``[K]`` norm weight.
        eps: added to the mean square.

    Returns:
        ``(q, scales)`` as in ``quantize``.

    Raises:
        ValueError: ``x`` is not a 2D BF16 tensor, or ``K`` is not a multiple of 16.
    """
    if x.dtype != torch.bfloat16 or x.ndim != 2:
        raise ValueError("nvfp4 rmsnorm_quantize expects a 2D BF16 tensor")
    x32 = x.to(torch.float32)
    mean = mxfp8._sum_of_squares(x32, NORM_THREADS) / x32.new_tensor(x.shape[1])
    inv_rms = torch.rsqrt(mean + eps)
    w32 = weight.to(torch.bfloat16).to(torch.float32)
    return quantize_values((x32 * (inv_rms[:, None] * w32)).to(torch.bfloat16))


def silu_mul_quantize(gate_up: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """NVFP4 of ``silu(gate) * up`` computed in FP32 from BF16 ``[M, 2K] = [gate | up]``.

    Args:
        gate_up: BF16 ``[M, 2K]``, the gate half first; ``K`` a multiple of 16.

    Returns:
        ``(q, scales)``: packed E2M1 ``[M, K/2]`` and the swizzled scales of an ``[M, K]`` matrix.

    Raises:
        ValueError: ``gate_up`` is not a 2D BF16 tensor with an even width, or ``K`` is not a multiple of 16.
    """
    if gate_up.dtype != torch.bfloat16 or gate_up.ndim != 2 or gate_up.shape[1] % 2:
        raise ValueError("nvfp4 silu_mul_quantize expects a 2D BF16 [M, 2K] tensor")
    g, u = gate_up.to(torch.float32).chunk(2, dim=1)
    return quantize_values(mxfp8._silu_mul(g, u))


# Forms of xcompute's GEMM family with NVFP4 inputs: activations ``(q [M, K/2], scales)``, weights
# ``(q [N, K/2], scales, global_scale)``. The GEMM rounds ``g · x̂·Ŵᵀ + bias`` once to BF16 (``gemm``); the epilogue
# then works on that BF16 tensor in FP32 and rounds again, and a quantized output is ``quantize`` of the final BF16
# tensor, so every quantized activation is a BF16 tensor quantized with ``quantize`` (the Wall-X NVFP4 recipe).

def _act(name: str | None, y: torch.Tensor) -> torch.Tensor:
    """``act(y)`` of a BF16 tensor computed in FP32 and rounded to BF16 (``"silu"``, ``"gelu"``, ``"gelu_tanh"``)."""
    return y if name is None else ACTIVATIONS[name](y.float()).to(torch.bfloat16)


def gemm_epilogue(x, weight, bias=None, *, activation: str | None = None, residual: torch.Tensor | None = None,
                  out_quant: bool = False):
    """NVFP4 ``x · weightᵀ`` with xcompute ``gemm``'s epilogue.

    ``y = bf16(g · x̂·Ŵᵀ + bias)`` (``gemm``), then ``bf16(act(y))``; with ``residual`` ``bf16(y + residual)``; with
    ``out_quant`` the NVFP4 ``quantize`` of the result.

    Args:
        x: ``(q [M, K/2], scales)``.
        weight: ``(q [N, K/2], scales, global_scale)``.
        bias: ``[N]`` or None.
        activation: ``"silu"``, ``"gelu"`` (erf), ``"gelu_tanh"`` or None.
        residual: BF16 ``[M, N]`` or None (not with ``out_quant``).
        out_quant: return ``(q [M, N/2], scales)`` instead of BF16.

    Returns:
        BF16 ``[M, N]``, or ``(q, scales)`` with ``out_quant``.

    Raises:
        ValueError: ``residual`` with ``out_quant``.
    """
    if residual is not None and out_quant:
        raise ValueError("nvfp4 gemm_epilogue: residual and out_quant exclude each other")
    w_q, w_scales, g = weight
    y = _act(activation, gemm(*x, w_q, w_scales, bias, alpha=g))
    if out_quant:
        return quantize(y)
    if residual is not None:
        y = (y.float() + residual.float()).to(torch.bfloat16)
    return y


def gemm_glu_quantize(x, weight) -> tuple[torch.Tensor, torch.Tensor]:
    """NVFP4 of ``bf16(silu(y_gate) · y_up)``, ``y = bf16(g · x̂·Ŵᵀ)`` (xcompute ``gemm_glu`` with ``out_quant``).

    Args:
        x: ``(q [M, K/2], scales)``.
        weight: ``(q [2N, K/2], scales, global_scale)``, the gate rows then the up rows.

    Returns:
        ``(q [M, N/2], scales)``.
    """
    w_q, w_scales, g = weight
    y = gemm(*x, w_q, w_scales, alpha=g).float()
    n = y.shape[1] // 2
    return quantize((torch.nn.functional.silu(y[:, :n]) * y[:, n:]).to(torch.bfloat16))


def qkv_rope_epilogue(x, weight, bias: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, num_heads: int,
                      head_dim: int) -> torch.Tensor:
    """NVFP4 ``[Q | K | V]``: ``y = bf16(g · x̂·Ŵᵀ + bias)``, then rotate-half RoPE on every Q and K head of ``y`` in
    FP32, rounded to BF16 (xcompute ``qkv_proj_rope``; equal Q, K and V heads, the rotation covers ``2·cos.shape[-1]``
    channels of a head).

    Args:
        x: ``(q [M, K/2], scales)``.
        weight: ``(q [3·H·D, K/2], scales, global_scale)``.
        bias: ``[3·H·D]``.
        cos: FP32 ``[M, R/2]``.
        sin: FP32 ``[M, R/2]``.
        num_heads: Heads ``H``.
        head_dim: Head dimension ``D``.

    Returns:
        BF16 ``[M, 3·H·D]``.
    """
    w_q, w_scales, g = weight
    y = gemm(*x, w_q, w_scales, bias, alpha=g).float()
    return rope_qk_heads(y, cos, sin, num_heads, head_dim).to(torch.bfloat16)


def layer_norm_quantize(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor,
                        eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    """NVFP4 of the BF16 LayerNorm (``layer_norm_cast``): ``(q [M, K/2], scales)``, ``M`` the leading dims."""
    return quantize(layer_norm_cast(x, weight, bias, eps).reshape(-1, x.shape[-1]))


def add_zero_centered_rms_norm_quantize(x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float):
    """``add_zero_centered_rms_norm`` with its BF16 norm quantized.

    Returns:
        ``(s, (q [M, K/2], scales))``: ``s = residual + x`` as ``add_zero_centered_rms_norm`` gives it, ``M`` the
        leading dims.
    """
    s, normed = bf16_add_zero_centered_rms_norm(x, residual, weight, eps)
    return s, quantize(normed.reshape(-1, x.shape[-1]))
