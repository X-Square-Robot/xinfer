"""MXFP8 storage layouts shared by the operators, their kernels and the quantization methods.

- E8M0 scales, one per ``BLOCK`` (32) columns of K, or one per ``SHARED_BLOCK`` (128) columns replicated into its
  four 32-column slots, stored in the SWIZZLE_32_4_4 layout of block-scaled GEMMs: rows padded to 128, scale blocks
  padded to 4 (``scale_offsets``; NVFP4 uses the same layout with 16-column blocks);
- the weight row orders the fused epilogues read (``rope_pair_permutation`` for ``mxfp8_gemm_rope``,
  ``gate_up_interleave`` for ``mxfp8_gemm_silu_mul``) and the ``mxfp8_gemm_rope`` cos/sin table (``rope_table``).
"""

from __future__ import annotations

import torch

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
