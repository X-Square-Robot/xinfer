"""Position-encoding operators on CUDA.

| operator | contract vs the torch reference |
|---|---|
| ``rope_full`` | tolerance: FMA in the rotation |
| ``rot_pos`` | bitwise: one FP32 product per angle |
| ``pos_embed_interp`` | tolerance: FMA in the corner sum |

``grids`` is a host list of ``(frames, height, width)`` in patches, one per image; the patches of an image are
ordered in ``merge_size x merge_size`` blocks, as a vision tower with a patch merger stores them. The device
copies of a grid list are created on its first call, which must therefore be eager (before a capture).
"""

from __future__ import annotations

import math

import torch

from xinfer.ops.backends.cuda._native import NATIVE, lib
from xinfer.ops.registry import register

_grids = {}  # (grids, merge size, device) -> (grid, output offsets, plane offsets, angle offsets, total rows)


def _grid_tensors(grids, merge_size, device):
    key = (tuple(tuple(int(v) for v in g) for g in grids), int(merge_size), device)
    if key not in _grids:
        grid = torch.tensor(key[0], dtype=torch.int32, device=device)
        planes = torch.zeros(len(grids) + 1, dtype=torch.int32, device=device)
        planes[1:] = (grid[:, 1] * grid[:, 2]).cumsum(0)
        outputs = torch.zeros_like(planes)
        outputs[1:] = (grid[:, 0] * grid[:, 1] * grid[:, 2]).cumsum(0)
        counts = torch.empty(len(grids), dtype=torch.int32, device=device)
        lib.get_token_counts(grid, counts, int(merge_size))
        angles = torch.zeros_like(planes)
        angles[1:] = counts.cumsum(0)
        _grids[key] = (grid, outputs, planes, angles, sum(t * h * w for t, h, w in key[0]))
    return _grids[key]


@register("rope_full", backend="cuda", **NATIVE)
def rope_full(q, k, cos, sin, rotary_dim):
    """Rotate-half RoPE: ``q`` ``[B, M, H, D]`` and ``k`` ``[B, M, H_kv, D]`` with ``cos`` / ``sin``
    ``[B, M, rotary_dim / 2]``; the dims past ``rotary_dim`` are copied. Returns new ``(q, k)``. Each product and
    sum is rounded as torch does (no FMA)."""
    q_out, k_out = torch.empty_like(q), torch.empty_like(k)
    lib.rope_full(q, k, q_out, k_out, cos, sin, False, int(rotary_dim))
    return q_out, k_out


@register("rot_pos", backend="cuda", numerics="bitwise", **NATIVE)
def rot_pos(inv_freq, grids, merge_size):
    """FP32 ``[patches, 2 * F]``: for each patch ``[row * inv_freq | column * inv_freq]`` with ``inv_freq`` FP32
    ``[F]``; bit for bit the FP32 products."""
    grid, _, _, angles, total = _grid_tensors(grids, merge_size, inv_freq.device)
    out = torch.empty((total, 2 * inv_freq.numel()), dtype=torch.float32, device=inv_freq.device)
    lib.rot_pos(inv_freq, grid, out, angles, int(merge_size))
    return out


@register("pos_embed_interp", backend="cuda", **NATIVE)
def pos_embed_interp(weight, grids, merge_size):
    """``[patches, K]``: the ``[S * S, K]`` table ``weight`` resized to each image's ``height x width`` (bilinear,
    corners aligned) and repeated over its frames."""
    grid, outputs, planes, _, total = _grid_tensors(grids, merge_size, weight.device)
    out = torch.empty((total, weight.shape[1]), dtype=weight.dtype, device=weight.device)
    lib.pos_embed_interp(weight, grid, outputs, planes, out, math.isqrt(weight.shape[0]), int(merge_size))
    return out
