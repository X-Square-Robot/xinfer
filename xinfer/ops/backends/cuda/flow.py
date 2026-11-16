"""Flow-matching operators on CUDA: the two ends of a denoising step, one kernel each.

| operator | function here |
|---|---|
| ``flow_input_embed`` | ``denoise_input_embed`` |
| ``flow_velocity_euler`` | ``denoise_output_euler`` |

    denoise_input_embed(x, weight, ...)             [x | extra] @ weight^T plus up to three addends, to BF16
    denoise_output_euler(x, h, weight, dt=, ...)    x + dt * (h @ weight^T), blended and pinned per row

``h`` is the decoder's normalized last hidden state; the kernel projects it to the action width, takes the Euler
step and applies the per-element and per-row overrides, so no intermediate velocity tensor is written. One warp
per row. It takes the sizes in ``SHAPES``, the ones it is tuned and tested on. ``denoise_input_embed`` embeds the
step's actions for the decoder, 16 output features per CTA; its sizes are in ``EMBED_SHAPES``. The other calls run
on the reference.

Contracts against the torch references: tolerance.
"""

from __future__ import annotations

import torch

from xinfer.ops.backends.cuda._native import BUILD_ARCHS, NATIVE, arch, lib
from xinfer.ops.reference import flow as reference_flow
from xinfer.ops.registry import register

# arch -> K (hidden width) -> inclusive ranges of batch, M (rows) and A (action width), and of the batch with a BF16
# weight and a BF16 h where that is narrower
SHAPES = {
    120: {1024: dict(batch=(1, 4), m=(1, 32), a=(26, 32))},
    110: {1024: dict(batch=(1, 4), m=(1, 32), a=(26, 32), bf16_batch=(1, 2))},
}


def denoise_output_euler_takes(x, h, weight) -> bool:
    """Whether :func:`denoise_output_euler` runs x ``[B, M, A]``, h ``[B, M, K]``, weight ``[A, K]`` here."""
    if x.dim() != 3 or h.dim() != 3 or weight.dim() != 2:
        return False
    if x.dtype != torch.float32 or (h.dtype, weight.dtype) not in ((torch.bfloat16, torch.float32),
                                                                     (torch.float32, torch.float32),
                                                                     (torch.bfloat16, torch.bfloat16)):
        return False
    device_arch = arch(h.device)
    ranges = SHAPES.get(device_arch, {}).get(h.shape[2]) if device_arch in BUILD_ARCHS else None
    if ranges is None:
        return False
    b, m, a = x.shape
    batch = ranges.get("bf16_batch", ranges["batch"]) if weight.dtype == torch.bfloat16 else ranges["batch"]
    return all(lo <= v <= hi for v, (lo, hi) in ((b, batch), (m, ranges["m"]), (a, ranges["a"])))


def denoise_output_euler(x, h, weight, *, dt, dof_mask=None, padding_velocity=None, pinned=None,
                         pinned_values=None, out=None):
    """``x + dt * v`` with ``v = h @ weight^T``: x FP32 ``[B, M, A]``, h ``[B, M, K]``, weight ``[A, K]``.

    ``dt``: a float, or an FP32 tensor ``[B]``. ``dof_mask`` (FP32 ``[B, M, A]``) with ``padding_velocity`` (FP32
    ``[B, M, A]``): ``v`` becomes ``dof_mask * v + (1 - dof_mask) * padding_velocity``. ``pinned`` (int32 ``[B]``):
    the first ``pinned[b]`` rows of request ``b`` keep ``x``, or take ``pinned_values`` (FP32 ``[B, M, A]``) when
    given. Writes ``out`` (default: ``x`` in place) and returns it.
    """
    if not denoise_output_euler_takes(x, h, weight):
        raise ValueError(f"denoise_output_euler: no kernel for x {tuple(x.shape)}, h {tuple(h.shape)} {h.dtype}, "
                         f"weight {tuple(weight.shape)} {weight.dtype} on SM{arch(h.device)} "
                         "(see denoise_output_euler_takes)")
    if tuple(h.shape[:2]) != tuple(x.shape[:2]) or tuple(weight.shape) != (x.shape[2], h.shape[2]):
        raise ValueError("denoise_output_euler: x [B, M, A], h [B, M, K] and weight [A, K] do not match")
    if padding_velocity is not None and dof_mask is None:
        raise ValueError("denoise_output_euler: padding_velocity needs dof_mask")
    if pinned_values is not None and pinned is None:
        raise ValueError("denoise_output_euler: pinned_values needs pinned")
    out = x if out is None else out
    step = isinstance(dt, torch.Tensor)
    lib.denoise_output_euler(out, x, h, weight, None, 0, None, None, 1e-6, None, dof_mask, padding_velocity, pinned,
                             pinned_values, 0.0 if step else float(dt), dt if step else None, False)
    return out


# arch -> boxes of inclusive ranges of batch, M (rows), A (action width) and N (output width, a multiple of 16)
EMBED_SHAPES = {
    120: [dict(batch=(1, 1), m=(1, 80), a=(1, 32), n=(16, 4096))],
    110: [dict(batch=(1, 1), m=(1, 48), a=(1, 32), n=(16, 2048)),
          dict(batch=(1, 1), m=(49, 80), a=(1, 32), n=(16, 1024))],
}


def denoise_input_embed_takes(x, weight) -> bool:
    """Whether :func:`denoise_input_embed` runs x ``[B, M, A]`` with weight ``[heads, N, A | 2 A]`` here."""
    if x.dim() != 3 or weight.dim() != 3 or x.dtype != torch.float32 or weight.dtype != torch.float32:
        return False
    device_arch = arch(x.device)
    boxes = EMBED_SHAPES.get(device_arch) if device_arch in BUILD_ARCHS else None
    if boxes is None or weight.shape[1] % 16:
        return False
    b, m, a = x.shape
    return any(all(lo <= v <= hi for v, (lo, hi) in ((b, box["batch"]), (m, box["m"]), (a, box["a"]),
                                                     (weight.shape[1], box["n"]))) for box in boxes)


def denoise_input_embed(x, weight, *, head_index=None, extra=None, input_mask=None, addends=(), override=None,
                        override_mask=None, out=None):
    """``bf16(((p + addends[0]) + addends[1]) + addends[2])`` with ``p = [x | extra] @ weight[head]^T`` in FP32.

    x and ``extra`` FP32 ``[B, M, A]``; weight FP32 ``[heads, N, A]``, or ``[heads, N, 2 A]`` with ``extra``;
    ``head_index`` int32 ``[B, M]`` picks the head of each row (required with two heads). ``input_mask`` (bool or
    uint8 ``[B, M]``): ``p`` is zero where it is 0. ``addends``: up to three FP32 ``[B, M, N]`` tensors, added in
    order. ``override`` (FP32 ``[B, M, N]``) replaces the whole sum on the rows where ``override_mask`` is set.

    Returns BF16 ``[B, M, N]``.
    """
    addends = tuple(addends)
    if not denoise_input_embed_takes(x, weight):
        raise ValueError(f"denoise_input_embed: no kernel for x {tuple(x.shape)}, weight {tuple(weight.shape)} on "
                         f"SM{arch(x.device)} (see denoise_input_embed_takes)")
    if len(addends) > 3:
        raise ValueError("denoise_input_embed: at most three addends")
    if weight.shape[0] > 1 and head_index is None:
        raise ValueError("denoise_input_embed: a weight with two heads needs head_index")
    if out is None:
        out = torch.empty(*x.shape[:2], weight.shape[1], dtype=torch.bfloat16, device=x.device)

    def u8(mask):
        if mask is not None and mask.dtype not in (torch.bool, torch.uint8):
            raise ValueError("denoise_input_embed: masks must be bool or uint8")
        return mask if mask is None or mask.dtype == torch.uint8 else mask.view(torch.uint8)

    lib.denoise_input_embed(out, x, extra, weight, head_index, u8(input_mask), *(addends + (None,) * 3)[:3], override,
                            u8(override_mask), False)
    return out


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------
@register("flow_input_embed", backend="cuda", **NATIVE)
def flow_input_embed(x, weight, head_index=None, extra=None, input_mask=None, addends=(), override=None,
                     override_mask=None):
    options = dict(head_index=head_index, extra=extra, input_mask=input_mask, addends=tuple(addends),
                   override=override, override_mask=override_mask)
    if not denoise_input_embed_takes(x, weight):
        return reference_flow.flow_input_embed(x, weight, **options)
    return denoise_input_embed(x, weight, **options)


@register("flow_velocity_euler", backend="cuda", **NATIVE)
def flow_velocity_euler(x, h, weight, dt, dof_mask=None, padding_velocity=None, pinned=None, pinned_values=None):
    if not denoise_output_euler_takes(x, h, weight):
        return reference_flow.flow_velocity_euler(x, h, weight, dt, dof_mask, padding_velocity, pinned, pinned_values)
    return denoise_output_euler(x, h, weight, dt=float(dt), dof_mask=dof_mask, padding_velocity=padding_velocity,
                                   pinned=None if pinned is None else pinned.to(torch.int32),
                                   pinned_values=pinned_values, out=torch.empty_like(x))
