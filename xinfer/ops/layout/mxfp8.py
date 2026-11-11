"""MXFP8 storage layouts shared by the operators, their kernels and the quantization methods.

- E8M0 scales, one per ``BLOCK`` (32) columns of K, or one per ``SHARED_BLOCK`` (128) columns replicated into its
  four 32-column slots, stored in the SWIZZLE_32_4_4 layout of block-scaled GEMMs: rows padded to 128, scale blocks
  padded to 4 (``scale_offsets``; NVFP4 uses the same layout with 16-column blocks);
- the weight row orders the fused epilogues read (``rope_pair_permutation`` for ``mxfp8_gemm_rope``,
  ``gate_up_interleave`` for ``mxfp8_gemm_silu_mul``) and the ``mxfp8_gemm_rope`` cos/sin table (``rope_table``).
"""

from __future__ import annotations

import torch

from xinfer.ops.layout import _swizzle_32_4_4

BLOCK = 32
SHARED_BLOCK = 128


def scale_elements(rows: int, k: int, *, block: int = BLOCK) -> int:
    """Bytes of swizzled E8M0 scales for a logical ``[rows, k]`` matrix, including padding."""
    return _swizzle_32_4_4.scale_elements(rows, k, block=block)


def scale_offsets(rows: int, k: int, device=None, *, block: int = BLOCK) -> torch.Tensor:
    """int64 ``[rows, padded block count]`` offsets of the E8M0 scales, one per ``block`` columns."""
    return _swizzle_32_4_4.scale_offsets(rows, k, device, block=block)


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
