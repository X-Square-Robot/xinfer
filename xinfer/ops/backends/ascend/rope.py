"""Ascend rotary position embedding and the vision position operators for Wall-X 2B.

``rope_full`` rotates the rotary prefix of every head and passes the rest through, which is what the entry's
``Rope.full`` does with ``interleave=False`` (the half-split layout the reference uses). ``rot_pos`` and
``pos_embed_interp`` are the Qwen3.5 ViT's 2-D rotary angles and its interpolated position table, through the
library functions of the same names; a host grid list becomes the int32 device tensors those functions take,
cached per (grids, merge, device) so the first — eager — call of a layout has them before any capture.
"""

import math

import torch

from xinfer.ops.backends.ascend._support import VERSION, has_functions, partial, xc
from xinfer.ops.reference.rope import pos_embed_interp as reference_pos_embed_interp
from xinfer.ops.reference.rope import rope_full as reference_rope_full
from xinfer.ops.reference.rope import rot_pos as reference_rot_pos
from xinfer.ops.registry import register

_rope = getattr(xc, "Rope", None) if xc is not None else None
_full = _rope().full if callable(_rope) else None


@register("rope_full", backend="ascend", requires=xc if callable(_full) else None,
          version=VERSION)
@partial(reference_rope_full,
         check=lambda q, k, cos, sin, rotary_dim:
         q.device.type == "npu" and q.ndim == 4 and k.ndim == 4
         and all(t.dtype == torch.bfloat16 and t.device == q.device for t in (q, k))
         and cos.dtype == torch.float32 and sin.dtype == torch.float32
         and cos.device == q.device and sin.device == q.device
         and 0 < rotary_dim <= q.shape[-1] and cos.shape[-1] * 2 >= rotary_dim)
def rope_full(q, k, cos, sin, rotary_dim):
    return _full(q, k, cos, sin, False, rotary_dim)


# --------------------------------------------------------------- Wall-X 2B vision positions
_position_functions = (has_functions("pos_embed_interp", "get_token_counts")
                       and has_functions("rot_pos"))
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
        xc.xcompute_lib.get_token_counts(grid, counts, int(merge_size))
        rot_cumsum = torch.zeros_like(hw_cumsum)
        rot_cumsum[1:] = counts.cumsum(0)
        _grids[key] = (grid, out_cumsum, hw_cumsum, rot_cumsum, sum(t * h * w for t, h, w in key[0]))
    return _grids[key]


@register("rot_pos", backend="ascend", numerics="bitwise",
          requires=xc if _position_functions else None, version=VERSION)
@partial(reference_rot_pos)
def rot_pos(inv_freq, grids, merge_size):
    grid, _, _, rot_cumsum, total = _grid_tensors(grids, merge_size, inv_freq.device)
    out = torch.empty((total, 2 * inv_freq.numel()), dtype=torch.float32, device=inv_freq.device)
    xc.xcompute_lib.rot_pos(inv_freq, grid, out, rot_cumsum, int(merge_size))
    return out


@register("pos_embed_interp", backend="ascend", requires=xc if _position_functions else None,
          version=VERSION)
@partial(reference_pos_embed_interp,
         check=lambda weight, grids, merge_size:
         weight.device.type == "npu" and weight.dim() == 2 and weight.shape[0] > 0
         and math.isqrt(weight.shape[0]) ** 2 == weight.shape[0])
def pos_embed_interp(weight, grids, merge_size):
    grid, out_cumsum, hw_cumsum, _, total = _grid_tensors(grids, merge_size, weight.device)
    out = torch.empty((total, weight.shape[1]), dtype=weight.dtype, device=weight.device)
    xc.xcompute_lib.pos_embed_interp(weight, grid, out_cumsum, hw_cumsum, out, math.isqrt(weight.shape[0]),
                                     int(merge_size))
    return out
