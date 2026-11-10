"""Flow-matching operators on CUDA, through xcompute (``_xcompute``)."""

from __future__ import annotations

import torch

from xinfer.ops.backends._xcompute import XCOMPUTE, xc
from xinfer.ops.registry import register


@register("flow_velocity_euler", backend="cuda", **XCOMPUTE)
def flow_velocity_euler(x, h, weight, dt, dof_mask=None, padding_velocity=None, pinned=None, pinned_values=None):
    return xc.denoise_output_euler(x, h, weight, dt=float(dt), dof_mask=dof_mask,
                                   padding_velocity=padding_velocity,
                                   pinned=None if pinned is None else pinned.to(torch.int32),
                                   pinned_values=pinned_values, out=torch.empty_like(x))
