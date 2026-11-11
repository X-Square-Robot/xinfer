"""Position-encoding operators on CUDA, through xcompute library functions (``_xcompute``).

| operator | xcompute library function | contract vs the torch reference |
|---|---|---|
| ``rope_full`` | ``rope_full`` | tolerance: FMA in the rotation |
| ``rot_pos`` | ``get_token_counts``, ``rot_pos`` | bitwise: one FP32 product per angle |
| ``pos_embed_interp`` | ``pos_embed_interp`` | tolerance: FMA in the corner sum |

The device copies of an image grid are created on the first call for that grid, which must therefore be eager
(before a capture).
"""

from __future__ import annotations

import math

import torch

from xinfer.ops.backends._xcompute import library, xcompute_with
from xinfer.ops.registry import register

_grids = {}  # (host grids, merge size, device) -> (grid, out_cumsum, hw_cumsum, rot_cumsum, total)


def _grid_tensors(grids, merge_size, device):
    """``(grid, out_cumsum, hw_cumsum, rot_cumsum, total)`` of a host grid list, cached."""
    key = (tuple(tuple(int(v) for v in g) for g in grids), int(merge_size), device)
    if key not in _grids:
        grid = torch.tensor(key[0], dtype=torch.int32, device=device)
        hw_cumsum = torch.zeros(len(grids) + 1, dtype=torch.int32, device=device)
        hw_cumsum[1:] = (grid[:, 1] * grid[:, 2]).cumsum(0)
        out_cumsum = torch.zeros_like(hw_cumsum)
        out_cumsum[1:] = (grid[:, 0] * grid[:, 1] * grid[:, 2]).cumsum(0)
        counts = torch.empty(len(grids), dtype=torch.int32, device=device)
        library.call("get_token_counts", grid, counts, int(merge_size))
        rot_cumsum = torch.zeros_like(hw_cumsum)
        rot_cumsum[1:] = counts.cumsum(0)
        _grids[key] = (grid, out_cumsum, hw_cumsum, rot_cumsum, sum(t * h * w for t, h, w in key[0]))
    return _grids[key]


@register("rope_full", backend="cuda", **xcompute_with(functions=("rope_full",)))
def rope_full(q, k, cos, sin, rotary_dim):
    q_out, k_out = torch.empty_like(q), torch.empty_like(k)
    library.call("rope_full", q, k, q_out, k_out, cos, sin, False, int(rotary_dim))
    return q_out, k_out


@register("rot_pos", backend="cuda", numerics="bitwise", **xcompute_with(functions=("rot_pos", "get_token_counts")))
def rot_pos(inv_freq, grids, merge_size):
    grid, _, _, rot_cumsum, total = _grid_tensors(grids, merge_size, inv_freq.device)
    out = torch.empty((total, 2 * inv_freq.numel()), dtype=torch.float32, device=inv_freq.device)
    library.call("rot_pos", inv_freq, grid, out, rot_cumsum, int(merge_size))
    return out


@register("pos_embed_interp", backend="cuda", **xcompute_with(functions=("pos_embed_interp", "get_token_counts")))
def pos_embed_interp(weight, grids, merge_size):
    grid, out_cumsum, hw_cumsum, _, total = _grid_tensors(grids, merge_size, weight.device)
    out = torch.empty((total, weight.shape[1]), dtype=weight.dtype, device=weight.device)
    library.call("pos_embed_interp", weight, grid, out_cumsum, hw_cumsum, out, math.isqrt(weight.shape[0]),
                 int(merge_size))
    return out
