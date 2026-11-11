"""SWIZZLE_32_4_4 storage for block-scaled GEMM scales."""

from __future__ import annotations

import torch


def _padded_blocks(k: int, block: int) -> int:
    blocks = (k + block - 1) // block
    return (blocks + 3) // 4 * 4


def scale_elements(rows: int, k: int, *, block: int) -> int:
    """Bytes of a SWIZZLE_32_4_4 scale tensor for a logical [rows, k] matrix.

    Args:
        rows: logical rows ``M`` (padded to a multiple of 128).
        k: logical columns ``K`` (``ceil(K / block)`` scale blocks, padded to a multiple of 4).
        block: number of logical columns covered by one scale.

    Returns:
        The number of scale bytes, including padding.
    """
    padded_rows = (rows + 127) // 128 * 128
    return padded_rows * _padded_blocks(k, block)


def scale_offsets(rows: int, k: int, device=None, *, block: int) -> torch.Tensor:
    """Physical offset of scale[row, block] for row < rows, block < padded block count.

    SWIZZLE_32_4_4: each 128-row x 4-block tile occupies 512 consecutive bytes, tiles in row-tile-major
    order; within a tile, ``r = row % 128`` and ``b = block % 4`` sit at ``(r % 32) * 16 + (r // 32) * 4 + b``.

    Args:
        rows: logical rows ``M``.
        k: logical columns ``K``.
        device: device of the returned tensor.
        block: number of logical columns covered by one scale.

    Returns:
        int64 ``[rows, padded block count]`` offsets into the flat scale tensor.
    """
    columns = _padded_blocks(k, block)
    col_groups = columns // 4
    row = torch.arange(rows, device=device, dtype=torch.int64)[:, None]
    col = torch.arange(columns, device=device, dtype=torch.int64)[None, :]
    in_block = row & 127
    tile = (row // 128) * col_groups + col // 4
    return tile * 512 + (in_block & 31) * 16 + (in_block // 32) * 4 + (col & 3)
