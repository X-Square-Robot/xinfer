"""Torch reference implementation of the flow (action denoising) operator.

The operator of this module corresponds one-to-one with the flow registrations of the hardware backends
(``xinfer/ops/backends/<backend>/flow.py``). The implementation is architecture-agnostic and runs on every
device.
"""

from __future__ import annotations

import torch


def flow_velocity_euler(x: torch.Tensor, h: torch.Tensor, weight: torch.Tensor, dt: float,
                        dof_mask: torch.Tensor | None = None, padding_velocity: torch.Tensor | None = None,
                        pinned: torch.Tensor | None = None, pinned_values: torch.Tensor | None = None) -> torch.Tensor:
    """One Euler step of action rows from their normed stream output (the CUDA backend's ``denoise_output_euler``,
    and the Ascend backend's kernel of the same name,
    velocity ``flow``).

    ``p = h · weightᵀ`` in FP32; the velocity is ``padding_velocity · (1 − dof_mask) + p · dof_mask`` (``p``
    without a mask), and ``x + dt · velocity`` the new values; the first ``pinned[b]`` rows take ``pinned_values``
    (``x`` without).

    Args:
        x: FP32 ``[B, M, A]`` current values.
        h: ``[B, M, K]`` BF16 normed stream output.
        weight: ``[A, K]`` FP32 or BF16 readout.
        dt: Step size.
        dof_mask: ``[B, M, A]`` 0/1 commanded values, or None.
        padding_velocity: FP32 ``[B, M, A]`` velocity of the uncommanded values (with ``dof_mask``).
        pinned: int ``[B]`` pinned leading rows, or None.
        pinned_values: FP32 ``[B, M, A]`` values of the pinned rows, or None to keep ``x``.

    Returns:
        FP32 ``[B, M, A]``.
    """
    p = h.float() @ weight.float().t()
    if dof_mask is not None:
        m = dof_mask.float()
        p = padding_velocity * (1 - m) + p * m
    out = x + dt * p
    if pinned is not None:
        rows = torch.arange(x.shape[1], device=x.device)[None, :, None] < pinned.reshape(-1, 1, 1)
        out = torch.where(rows, x if pinned_values is None else pinned_values, out)
    return out


# ---------------------------------------------------------------- WALL-OSS-1.0-Nano fused operators

def flow_input_embed(x: torch.Tensor, weight: torch.Tensor, head_index: torch.Tensor | None = None,
                     extra: torch.Tensor | None = None, input_mask: torch.Tensor | None = None,
                     addends: tuple[torch.Tensor, ...] = (), override: torch.Tensor | None = None,
                     override_mask: torch.Tensor | None = None) -> torch.Tensor:
    """Token inputs of flow rows (the CUDA backend's ``denoise_input_embed``, and the Ascend backend's).

    ``p = [x | extra] · weight[head]ᵀ`` in FP32 (zero where ``input_mask`` is 0), then the addends in order, or
    ``override`` where ``override_mask`` is set; one rounding to BF16.

    Args:
        x: FP32 ``[B, M, A]`` values.
        weight: FP32 ``[heads, N, A]``, or ``[heads, N, 2A]`` with ``extra`` (``heads <= 2``).
        head_index: int ``[B, M]`` head of each token; None with one head.
        extra: FP32 ``[B, M, A]`` second input half (for example the validity mask), or None.
        input_mask: bool ``[B, M]``, tokens whose projection counts; None for all.
        addends: FP32 ``[B, M, N]`` terms added in order (at most 3).
        override: FP32 ``[B, M, N]`` rows taken as they are where ``override_mask`` is set, or None.
        override_mask: bool ``[B, M]``, or None.

    Returns:
        ``[B, M, N]`` BF16.
    """
    w = weight[0] if head_index is None else weight[head_index.long()]
    inputs = x if extra is None else torch.cat((x, extra), dim=-1)
    p = (inputs[..., None, :] * w).sum(-1)
    if input_mask is not None:
        p = p * input_mask[..., None]
    for term in addends:
        p = p + term
    if override is not None:
        p = torch.where(override_mask[..., None], override, p)
    return p.to(torch.bfloat16)
