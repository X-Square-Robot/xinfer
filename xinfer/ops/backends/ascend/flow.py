"""Flow-matching operators on Ascend: the two ends of a denoising step, one kernel each.

| operator | function here |
|---|---|
| ``flow_input_embed`` | ``denoise_input_embed`` |
| ``flow_velocity_euler`` | ``denoise_output_euler`` |

    denoise_input_embed(x, weight, ...)             [x | extra] @ weight^T plus up to three addends, to BF16
    denoise_output_euler(x, h, weight, dt=, ...)    x + dt * (h @ weight^T), blended and pinned per row

``h`` is the decoder's normalized last hidden state; the fp32-head kernel pair projects it to the
action width, takes the Euler step and applies the per-element and per-row overrides, so no
intermediate velocity tensor is written. The kernels' serving envelopes are the composition's -
the glue core validates each call's layout (shapes, dtypes, flags) and a layout it does not take
raises before any launch, so the registered reference composition serves it, never a silent
numerics switch; ``SHAPES`` and ``EMBED_SHAPES`` below are the *measured* serving ranges of the
2026-11-16 production audit (the 2B flow head's doses, 10 per request each), the tables the
``*_takes`` predicates answer with - observability, not a second gate.

Both kernels launch through the backend's unified boundary - the single library's TVM-FFI face
when it is loaded, the family library's ctypes launcher otherwise - with the flag packing and
the scratch discipline on the python side, as the CUDA backend's lanes normalize their complex
parameters host-side.

Contracts against the torch references: tolerance.
"""

import re

import torch

from xinfer.ops.backends.ascend._support import PRIORITY, VERSION, partial, xc
from xinfer.ops.reference.flow import flow_input_embed as reference_flow_input_embed
from xinfer.ops.reference.flow import flow_velocity_euler as reference_flow_velocity_euler
from xinfer.ops.registry import register

_input_embed = getattr(xc, "denoise_input_embed", None) if xc is not None else None
_output_euler = getattr(xc, "denoise_output_euler", None) if xc is not None else None

# SoC -> K (hidden width) -> inclusive ranges of batch, M (rows) and A (action width) the
# fp32-head Euler tail served in the production audit (the prep kernel widens the BF16 hidden
# state to FP32 and the cube kernel rounds the product to BF16 once).
SHAPES = {
    "ascend950dt9582": {1024: dict(batch=(1, 4), m=(1, 32), a=(17, 32))},
}

# SoC -> inclusive ranges of batch, M (rows), A (action width) and N (output width) the input
# embedding's two-head form served in the production audit.
EMBED_SHAPES = {
    "ascend950dt9582": dict(batch=(1, 1), m=(1, 80), a=(1, 32), n=(16, 4096)),
}

_soc_of = {}


def _soc(device) -> str:
    """The SoC of ``device`` (``Ascend950DT9582`` -> ``ascend950dt9582``), the platform's arch string."""
    index = torch.npu.current_device() if device.index is None else device.index
    if index not in _soc_of:
        name = str(getattr(torch.npu.get_device_properties(index), "name", "") or "Ascend")
        _soc_of[index] = re.sub(r"[^a-z0-9]", "", name.lower()) or "ascend"
    return _soc_of[index]


def denoise_output_euler_takes(x, h, weight) -> bool:
    """Whether the fp32-head Euler tail served x ``[B, M, A]``, h ``[B, M, K]``, weight ``[A, K]`` here.

    The measured serving envelope of the production audit, per SoC: the answer is observability
    (which shapes the serving stack exercised), not a dispatch gate - the kernel's own envelope
    is the glue core's validation, wider than these tables.
    """
    if x.dim() != 3 or h.dim() != 3 or weight.dim() != 2:
        return False
    if x.dtype != torch.float32 or (h.dtype, weight.dtype) != (torch.bfloat16, torch.bfloat16):
        return False
    ranges = SHAPES.get(_soc(h.device), {}).get(h.shape[2])
    if ranges is None:
        return False
    b, m, a = x.shape
    return all(lo <= v <= hi for v, (lo, hi) in ((b, ranges["batch"]), (m, ranges["m"]), (a, ranges["a"])))


def denoise_input_embed_takes(x, weight) -> bool:
    """Whether the input embedding served x ``[B, M, A]`` with weight ``[heads, N, A | 2 A]`` here.

    The measured serving envelope of the production audit, per SoC: the answer is observability
    (which shapes the serving stack exercised), not a dispatch gate - the kernel's own envelope
    is the glue core's validation, wider than these tables.
    """
    if x.dim() != 3 or weight.dim() != 3 or x.dtype != torch.float32 or weight.dtype != torch.float32:
        return False
    ranges = EMBED_SHAPES.get(_soc(x.device))
    if ranges is None or weight.shape[1] % 16:
        return False
    b, m, a = x.shape
    return all(lo <= v <= hi for v, (lo, hi) in ((b, ranges["batch"]), (m, ranges["m"]), (a, ranges["a"]),
                                                 (weight.shape[1], ranges["n"])))


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------
@register("flow_input_embed", backend="ascend", priority=PRIORITY, requires=xc if callable(_input_embed) else None,
          version=VERSION)
@partial(reference_flow_input_embed,
         check=lambda x, weight, head_index=None, extra=None, input_mask=None, addends=(), override=None,
         override_mask=None:
         x.device.type == "npu" and weight.device == x.device
         and weight.shape[-1] == x.shape[-1] * (2 if extra is not None else 1))
def flow_input_embed(x, weight, head_index=None, extra=None, input_mask=None, addends=(), override=None,
                     override_mask=None):
    # The kernel glue owns the envelope (the two-head serving form and its flag packing - the
    # audited 2B flow head); a layout it rejects raises before any launch and the partial cache
    # serves the reference composition, exactly as a build without the kernel would.
    return _input_embed(x, weight, head_index=head_index, extra=extra, input_mask=input_mask,
                        addends=addends, override=override, override_mask=override_mask)


@register("flow_velocity_euler", backend="ascend", priority=PRIORITY, requires=xc if callable(_output_euler) else None,
          version=VERSION)
@partial(reference_flow_velocity_euler,
         check=lambda x, h, weight, dt, dof_mask=None, padding_velocity=None, pinned=None, pinned_values=None:
         x.device.type == "npu" and x.dtype == torch.float32 and h.dtype == torch.bfloat16
         and h.device == x.device and weight.device == x.device and weight.shape[-1] == h.shape[-1])
def flow_velocity_euler(x, h, weight, dt, dof_mask=None, padding_velocity=None, pinned=None, pinned_values=None):
    # The fp32-head Euler tail (prep + cube, the audited serving pair): the call's own options
    # (masks, pinned rows, tensor dt) are the glue's parameters, and a layout it rejects falls
    # to the reference composition through the partial cache.
    return _output_euler(x, h, weight, dt=dt, dof_mask=dof_mask, padding_velocity=padding_velocity,
                         pinned=pinned, pinned_values=pinned_values)
