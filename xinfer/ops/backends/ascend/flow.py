"""Ascend flow-head operators (Wall-X 2B action stage): the input embedding and the Euler step.

Both entries take the operator's own parameters in order; everything the models pass is forwarded unchanged, so
the OpSpec signatures stay the single description of the call. The unified core carries the measured serving envelopes (the fp32-head Euler tail A 17..32 and the
embed's two-head form); an envelope miss raises before any kernel launch and the registered reference
composition serves the call.
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
    return _input_embed(x, weight, head_index=head_index, extra=extra, input_mask=input_mask, addends=addends,
                        override=override, override_mask=override_mask)


@register("flow_velocity_euler", backend="ascend", requires=xc if callable(_output_euler) else None,
          version=VERSION)
@partial(reference_flow_velocity_euler,
         check=lambda x, h, weight, dt, dof_mask=None, padding_velocity=None, pinned=None, pinned_values=None:
         x.device.type == "npu" and x.dtype == torch.float32 and h.dtype == torch.bfloat16
         and h.device == x.device and weight.device == x.device and weight.shape[-1] == h.shape[-1])
def flow_velocity_euler(x, h, weight, dt, dof_mask=None, padding_velocity=None, pinned=None, pinned_values=None):
    return _output_euler(x, h, weight, dt=dt, dof_mask=dof_mask, padding_velocity=padding_velocity,
                         pinned=pinned, pinned_values=pinned_values)
