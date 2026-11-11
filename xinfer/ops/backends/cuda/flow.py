"""Flow-matching operators on CUDA, through xcompute (``_xcompute``).

| operator | xcompute entry |
|---|---|
| ``flow_input_embed`` | ``denoise_input_embed`` |
| ``flow_velocity_euler`` | ``denoise_output_euler`` |

Contracts against the torch references: tolerance.
"""

from __future__ import annotations

import torch

from xinfer.ops.backends._xcompute import XCOMPUTE, xc, xcompute_with
from xinfer.ops.registry import register


@register("flow_input_embed", backend="cuda", **xcompute_with("denoise_input_embed"))
def flow_input_embed(x, weight, head_index=None, extra=None, input_mask=None, addends=(), override=None,
                     override_mask=None):
    return xc.denoise_input_embed(x, weight, head_index=head_index, extra=extra, input_mask=input_mask,
                                  addends=tuple(addends), override=override, override_mask=override_mask)


@register("flow_velocity_euler", backend="cuda", **XCOMPUTE)
def flow_velocity_euler(x, h, weight, dt, dof_mask=None, padding_velocity=None, pinned=None, pinned_values=None):
    return xc.denoise_output_euler(x, h, weight, dt=float(dt), dof_mask=dof_mask,
                                   padding_velocity=padding_velocity,
                                   pinned=None if pinned is None else pinned.to(torch.int32),
                                   pinned_values=pinned_values, out=torch.empty_like(x))
