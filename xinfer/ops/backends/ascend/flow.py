"""Ascend WALL-OSS-1.0-Nano flow-head operators backed by the in-tree launch adapters.

| Operator | Entry |
|---|---|
| ``flow_input_embed`` | ``denoise_input_embed`` |
| ``flow_velocity_euler`` | ``denoise_output_euler`` |

The core validates native layouts before launch; unsupported layouts use the
reference composition. The public Euler operator preserves its input tensor.
"""

import torch

from xinfer.ops.backends.ascend._support import PLATFORMS, PRIORITY, VERSION, partial, xc
from xinfer.ops.reference.flow import flow_input_embed as reference_flow_input_embed
from xinfer.ops.reference.flow import flow_velocity_euler as reference_flow_velocity_euler
from xinfer.ops.registry import register

_input_embed = getattr(xc, "denoise_input_embed", None) if xc is not None else None
_output_euler = getattr(xc, "denoise_output_euler", None) if xc is not None else None

# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------
@register("flow_input_embed", backend="ascend", priority=PRIORITY, requires=xc if callable(_input_embed) else None,
          version=VERSION, platforms=PLATFORMS)
@partial(reference_flow_input_embed,
         check=lambda x, weight, head_index=None, extra=None, input_mask=None, addends=(), override=None,
         override_mask=None:
         x.device.type == "npu" and weight.device == x.device
         and weight.shape[-1] == x.shape[-1] * (2 if extra is not None else 1))
def flow_input_embed(x, weight, head_index=None, extra=None, input_mask=None, addends=(), override=None,
                     override_mask=None):
    return _input_embed(x, weight, head_index=None if head_index is None else head_index.to(torch.int32),
                        extra=extra, input_mask=input_mask,
                        addends=addends, override=override, override_mask=override_mask)


@register("flow_velocity_euler", backend="ascend", priority=PRIORITY, requires=xc if callable(_output_euler) else None,
          version=VERSION, platforms=PLATFORMS)
@partial(reference_flow_velocity_euler,
         check=lambda x, h, weight, dt, dof_mask=None, padding_velocity=None, pinned=None, pinned_values=None:
         x.device.type == "npu" and x.dtype == torch.float32 and h.dtype == torch.bfloat16
         and h.device == x.device and weight.device == x.device and weight.shape[-1] == h.shape[-1])
def flow_velocity_euler(x, h, weight, dt, dof_mask=None, padding_velocity=None, pinned=None, pinned_values=None):
    return _output_euler(x, h, weight, dt=dt, dof_mask=dof_mask, padding_velocity=padding_velocity,
                         pinned=None if pinned is None else pinned.to(torch.int32),
                         pinned_values=pinned_values, out=torch.empty_like(x))
