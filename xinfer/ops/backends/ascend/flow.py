"""Ascend flow-head operators (Wall-X 2B action stage): the input embedding and the Euler step.

Both entries take the operator's own parameters in order; everything the models pass is forwarded unchanged, so
the OpSpec signatures stay the single description of the call. The delivery library's auto dispatch admits the
measured Ascend envelopes only (for the Euler step, A = 32); the same TileLang kernels cover the wider named
domains (A 26..32, and the embed's two-head form), so a call the auto table refuses falls back to the named
``ascend_tilelang`` backend before giving up — the white-list rejection is host-side, before any kernel launch.
"""

import torch

from xinfer.ops.backends.ascend._support import VERSION, partial, xc
from xinfer.ops.reference.flow import flow_input_embed as reference_flow_input_embed
from xinfer.ops.reference.flow import flow_velocity_euler as reference_flow_velocity_euler
from xinfer.ops.registry import register

_input_embed = getattr(xc, "denoise_input_embed", None) if xc is not None else None
_output_euler = getattr(xc, "denoise_output_euler", None) if xc is not None else None


@register("flow_input_embed", backend="ascend", requires=xc if callable(_input_embed) else None,
          version=VERSION)
@partial(reference_flow_input_embed,
         check=lambda x, weight, head_index=None, extra=None, input_mask=None, addends=(), override=None,
         override_mask=None:
         x.device.type == "npu" and weight.device == x.device
         and weight.shape[-1] == x.shape[-1] * (2 if extra is not None else 1))
def flow_input_embed(x, weight, head_index=None, extra=None, input_mask=None, addends=(), override=None,
                     override_mask=None):
    try:
        return _input_embed(x, weight, head_index=head_index, extra=extra, input_mask=input_mask, addends=addends,
                            override=override, override_mask=override_mask)
    except ValueError:  # the auto table has no admitted entry for this layout; the named domain may take it
        return _input_embed(x, weight, head_index=head_index, extra=extra, input_mask=input_mask, addends=addends,
                            override=override, override_mask=override_mask, backend="ascend_tilelang")


@register("flow_velocity_euler", backend="ascend", requires=xc if callable(_output_euler) else None,
          version=VERSION)
@partial(reference_flow_velocity_euler,
         check=lambda x, h, weight, dt, dof_mask=None, padding_velocity=None, pinned=None, pinned_values=None:
         x.device.type == "npu" and x.dtype == torch.float32 and h.dtype == torch.bfloat16
         and h.device == x.device and weight.device == x.device and weight.shape[-1] == h.shape[-1])
def flow_velocity_euler(x, h, weight, dt, dof_mask=None, padding_velocity=None, pinned=None, pinned_values=None):
    try:
        return _output_euler(x, h, weight, dt=dt, dof_mask=dof_mask, padding_velocity=padding_velocity,
                             pinned=pinned, pinned_values=pinned_values)
    except ValueError:  # the auto table's admitted envelope is narrower (A = 32) than the named A 26..32 domain
        return _output_euler(x, h, weight, dt=dt, dof_mask=dof_mask, padding_velocity=padding_velocity,
                             pinned=pinned, pinned_values=pinned_values, backend="ascend_tilelang")
