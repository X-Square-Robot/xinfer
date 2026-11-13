"""Flow-matching operators on CUDA, through xcompute (``_xcompute``).

| operator | xcompute entry |
|---|---|
| ``flow_input_embed`` | ``denoise_input_embed`` |
| ``flow_velocity_euler`` | ``denoise_output_euler`` |

Contracts against the torch references: tolerance.
"""

from __future__ import annotations

import torch

from xinfer.ops.backends._xcompute import xc, xcompute_with
from xinfer.ops.backends.cuda._support import takes
from xinfer.ops.reference import flow as reference_flow
from xinfer.ops.registry import register


@register("flow_input_embed", backend="cuda", **xcompute_with("denoise_input_embed"))
def flow_input_embed(x, weight, head_index=None, extra=None, input_mask=None, addends=(), override=None,
                     override_mask=None):
    options = dict(head_index=head_index, extra=extra, input_mask=input_mask, addends=tuple(addends),
                   override=override, override_mask=override_mask)
    probe = getattr(xc, "denoise_input_embed_backend", None)
    if probe is None:
        # Older packages select before launching but expose no separate query.
        try:
            return xc.denoise_input_embed(x, weight, **options)
        except ValueError as exc:
            if not str(exc).startswith("denoise_input_embed: no white-list entry accepts "):
                raise
    else:
        key = (x.shape, weight.shape, extra is not None, head_index is not None, len(options["addends"]))
        if takes(probe, key, x, weight, **options):
            return xc.denoise_input_embed(x, weight, **options)
    return reference_flow.flow_input_embed(x, weight, **options)


@register("flow_velocity_euler", backend="cuda", **xcompute_with("denoise_output_euler_backend"))
def flow_velocity_euler(x, h, weight, dt, dof_mask=None, padding_velocity=None, pinned=None, pinned_values=None):
    options = dict(dt=float(dt), dof_mask=dof_mask, padding_velocity=padding_velocity,
                   pinned=None if pinned is None else pinned.to(torch.int32), pinned_values=pinned_values)
    key = (x.shape, x.dtype, h.shape, h.dtype, weight.shape, weight.dtype, dof_mask is not None,
           padding_velocity is not None, pinned is not None, pinned_values is not None)
    if takes(xc.denoise_output_euler_backend, key, x, h, weight, **options):
        return xc.denoise_output_euler(x, h, weight, **options, out=torch.empty_like(x))
    return reference_flow.flow_velocity_euler(x, h, weight, dt, dof_mask, padding_velocity, pinned, pinned_values)
