"""NVFP4 storage layouts: packed E2M1 elements ``[M, K/2]`` and E4M3 scales per ``BLOCK`` (16) columns of K, in the
SWIZZLE_32_4_4 layout (rows padded to 128, 16-column blocks padded to 4).
"""

from __future__ import annotations

import torch

from xinfer.ops.layout import swizzle_32_4_4

BLOCK = 16


def scale_elements(rows: int, k: int) -> int:
    """Bytes of the swizzled E4M3 scales of a logical ``[rows, k]`` matrix, padding included."""
    return swizzle_32_4_4.scale_elements(rows, k, block=BLOCK)


def scale_offsets(rows: int, k: int, device=None) -> torch.Tensor:
    """int64 ``[rows, padded block count]`` offsets of each 16-column block's scale."""
    return swizzle_32_4_4.scale_offsets(rows, k, device, block=BLOCK)
