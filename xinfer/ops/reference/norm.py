"""Torch reference implementations of the normalization operators.

The operators of this module correspond one-to-one with the normalization registrations of the hardware
backends (``xinfer/ops/backends/<backend>/norm.py``). The implementations are architecture-agnostic and run on
every device.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from xinfer.ops.reference.elementwise import gated_residual


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm with one rounding: normalize and scale in FP32, then cast.

    ``(weight * (x32 * rsqrt(mean(x32²) + eps))).to(x.dtype)``; a BF16 weight is
    promoted to FP32 for the product. This is the Wall-X / Gemma convention;
    Hugging Face Qwen2 instead rounds before scaling (two roundings).

    Args:
        x: ``[..., K]`` input, normalized over the last dim.
        weight: ``[K]`` scale.
        eps: added to the mean square before the reciprocal square root.

    Returns:
        ``[..., K]`` in ``x.dtype``.
    """
    dtype = x.dtype
    h = x.to(torch.float32)
    variance = h.pow(2).mean(-1, keepdim=True)
    h = h * torch.rsqrt(variance + eps)
    return (weight * h).to(dtype)


def layer_norm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float) -> torch.Tensor:
    """LayerNorm over the last dim in FP32, as autocast runs it: input, weight and bias cast to FP32.

    Args:
        x: ``[..., K]`` input of any floating dtype.
        weight: ``[K]`` scale.
        bias: ``[K]`` shift.
        eps: added to the variance.

    Returns:
        ``[..., K]`` FP32.
    """
    return F.layer_norm(x.float(), (x.shape[-1],), weight.float(), bias.float(), eps)


def ada_rms_norm(x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor, eps: float) -> torch.Tensor:
    """Adaptive RMSNorm (action-expert time conditioning): ``x · rsqrt(mean(x²) + eps) · (1 + scale) + shift``.

    The normalization and modulation run in FP32 as separate roundings, in the order of the reference
    action expert (``x * rsqrt(mean(x32²) + eps)`` with ``x`` promoted, then ``· (1 + scale32)``, then ``+ shift32``),
    and the result is rounded once to ``x.dtype``. There is no learned weight.

    Args:
        x: ``[B, L, K]`` input.
        scale: ``[B, 1, K]`` (or broadcastable) modulation scale.
        shift: ``[B, 1, K]`` (or broadcastable) modulation shift.
        eps: added to the mean square.

    Returns:
        ``[B, L, K]`` in ``x.dtype``.
    """
    variance = torch.mean(torch.square(x.float()), dim=-1, keepdim=True)
    normed = x * torch.rsqrt(variance + eps)
    return (normed * (1 + scale.to(torch.float32)) + shift.to(torch.float32)).to(x.dtype)


def gated_rms_norm(x: torch.Tensor, gate: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """``rms_norm(x) · weight · silu(gate)`` with the roundings of the Qwen3.5 Gated DeltaNet output norm.

    The normalization runs in FP32 and is rounded to ``x.dtype``; the weight product stays in that dtype; the
    gate is applied in FP32 and the result rounded once more.

    Args:
        x: ``[..., D]`` input.
        gate: ``[..., D]`` gate, same shape.
        weight: ``[D]`` scale in ``x.dtype``.
        eps: added to the mean square.

    Returns:
        ``[..., D]`` in ``x.dtype``.
    """
    dtype = x.dtype
    h = x.to(torch.float32)
    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps)
    h = weight * h.to(dtype)
    return (h * F.silu(gate.to(torch.float32))).to(dtype)


def l2_normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """``x · rsqrt(sum(x²) + eps)`` over the last dim, in the input dtype (the FLA ``l2norm`` convention)."""
    return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + eps)


def _adarms(x: torch.Tensor, modulation: torch.Tensor, row_class: torch.Tensor, eps: float) -> torch.Tensor:
    """FP32 ``x · rsqrt(mean(x²) + eps) · (1 + scale) + shift`` with each token's modulation row (not rounded)."""
    k = x.shape[-1]
    m = modulation[row_class].float()
    x = x.float()
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * (1 + m[:, :k]) + m[:, k:2 * k]


def add_rms_norm(x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor,
                 eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    """``s = residual + x`` (one BF16 rounding), then ``rms_norm(s)`` (the CUDA backend's ``add_rmsnorm``).

    Args:
        x: ``[..., K]`` BF16 branch output.
        residual: ``[..., K]`` BF16 residual stream.
        weight: ``[K]`` FP32 or BF16.
        eps: Added to the mean square.

    Returns:
        ``(s, normed)`` in ``x.dtype``.
    """
    s = residual + x
    return s, rms_norm(s, weight, eps)


# ---------------------------------------------------------------- WALL-OSS-1.0-Nano fused operators

def _zero_centered(x32: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """FP32 ``x · rsqrt(mean(x²) + eps) · (1 + w)``."""
    return x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps) * (1 + weight.float())

def layer_norm_cast(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float) -> torch.Tensor:
    """LayerNorm computed in FP32 and rounded to ``x.dtype`` (``F.layer_norm`` on BF16 input and weights).

    Args:
        x: ``[..., K]`` input.
        weight: ``[K]`` scale, ``x.dtype``.
        bias: ``[K]`` shift, ``x.dtype``.
        eps: Variance epsilon.

    Returns:
        ``[..., K]`` in ``x.dtype``.
    """
    return F.layer_norm(x, (x.shape[-1],), weight, bias, eps)

def zero_centered_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """``x · rsqrt(mean(x²) + eps) · (1 + w)`` in FP32, rounded once (Qwen3.5 RMSNorm with the raw weight).

    Args:
        x: ``[..., K]`` input.
        weight: ``[K]`` raw weight ``w`` (BF16 or FP32).
        eps: Added to the mean square.

    Returns:
        ``[..., K]`` in ``x.dtype``.
    """
    return _zero_centered(x.float(), weight, eps).to(x.dtype)

def add_zero_centered_rms_norm(x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor,
                               eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    """``s = residual + x`` (rounded), then ``zero_centered_rms_norm(s)``.

    Args:
        x: ``[..., K]`` branch output.
        residual: ``[..., K]`` residual stream, same dtype.
        weight: ``[K]`` raw weight ``w``.
        eps: Added to the mean square.

    Returns:
        ``(s, normed)`` in ``x.dtype``.
    """
    s = residual + x
    return s, zero_centered_rms_norm(s, weight, eps)

def _row_modulation(x: torch.Tensor, modulation: torch.Tensor, pinned_modulation, pinned_rows) -> torch.Tensor:
    """``[B, 1 or L, 3K]``: the shared ``[B, 3K]`` modulation, rows ``< pinned_rows[b]`` taking the pinned one."""
    shared = modulation.reshape(modulation.shape[0], 1, -1)
    if pinned_rows is None:
        return shared
    rows = torch.arange(x.shape[1], device=x.device)[None, :, None] < pinned_rows.reshape(-1, 1, 1)
    return torch.where(rows, pinned_modulation.reshape(-1, 1, shared.shape[-1]), shared)

def modulated_rms_norm(x: torch.Tensor, modulation: torch.Tensor, eps: float,
                       pinned_modulation: torch.Tensor | None = None,
                       pinned_rows: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Adaptive RMSNorm with a packed ``[scale | shift | gate]`` modulation (the CUDA backend's ``adarmsnorm``).

    ``normed = bf16(x · rsqrt(mean(x²) + eps) · (1 + scale) + shift)`` in FP32 (``reference.ada_rms_norm``). With
    ``pinned_rows`` the first ``pinned_rows[b]`` rows of batch row ``b`` take ``pinned_modulation`` (RTC pinned rows)
    and the gate is per row.

    Args:
        x: ``[B, L, K]`` BF16.
        modulation: ``[B, 3K]`` (FP32 or BF16).
        eps: Added to the mean square.
        pinned_modulation: ``[B, 3K]`` of the pinned rows, or None.
        pinned_rows: int32 ``[B]``, or None (given with ``pinned_modulation``).

    Returns:
        ``(normed [B, L, K], gate [B, 1 or L, K])`` in ``x.dtype``.
    """
    scale, shift, gate = _row_modulation(x, modulation, pinned_modulation, pinned_rows).chunk(3, dim=-1)
    return ada_rms_norm(x, scale, shift, eps), gate.to(x.dtype)

def gated_residual_modulated_rms_norm(residual: torch.Tensor, branch: torch.Tensor, branch_gate: torch.Tensor,
                                      modulation: torch.Tensor, eps: float,
                                      pinned_modulation: torch.Tensor | None = None,
                                      pinned_rows: torch.Tensor | None = None
                                      ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``s = residual + bf16(branch · branch_gate)``, then ``modulated_rms_norm(s)`` (the CUDA backend's
    ``gated_residual_adarmsnorm``).

    Args:
        residual: ``[B, L, K]`` BF16 residual stream.
        branch: ``[B, L, K]`` BF16 branch output.
        branch_gate: ``[B, 1 or L, K]`` BF16 gate of the branch.
        modulation: ``[B, 3K]`` of the next norm.
        eps: Added to the mean square.
        pinned_modulation: As in ``modulated_rms_norm``.
        pinned_rows: As in ``modulated_rms_norm``.

    Returns:
        ``(normed, gate, s)`` in ``residual.dtype``.
    """
    s = gated_residual(residual, branch, branch_gate)
    normed, gate = modulated_rms_norm(s, modulation, eps, pinned_modulation, pinned_rows)
    return normed, gate, s

def identity_layer_norms_(residual: torch.Tensor, branch: torch.Tensor, branch_gate: torch.Tensor,
                          input_modulation: torch.Tensor, post_modulation: torch.Tensor, eps: float,
                          pinned_input_modulation: torch.Tensor | None = None,
                          pinned_post_modulation: torch.Tensor | None = None,
                          pinned_rows: torch.Tensor | None = None) -> torch.Tensor:
    """The norms of a layer whose mixer is the identity, in place (the CUDA backend's ``double_adarmsnorm``).

    Two ``gated_residual_modulated_rms_norm`` transitions: the previous branch into the residual and the input norm,
    then the identity mixer's gated output into the residual and the post norm. ``residual`` becomes the new
    residual and ``branch`` the post norm's output.

    Args:
        residual: ``[B, L, K]`` BF16 residual stream; overwritten.
        branch: ``[B, L, K]`` BF16 previous branch output; overwritten with the normed rows.
        branch_gate: ``[B, 1 or L, K]`` gate of ``branch``.
        input_modulation: ``[B, 3K]`` of the input norm.
        post_modulation: ``[B, 3K]`` of the post norm.
        eps: Added to the mean square.
        pinned_input_modulation, pinned_post_modulation: ``[B, 3K]`` of the pinned rows, or None.
        pinned_rows: int32 ``[B]``, or None.

    Returns:
        The post norm's gate ``[B, 1 or L, K]``.
    """
    normed, gate, s = gated_residual_modulated_rms_norm(residual, branch, branch_gate, input_modulation, eps,
                                                        pinned_input_modulation, pinned_rows)
    normed, gate, s = gated_residual_modulated_rms_norm(s, normed, gate, post_modulation, eps,
                                                        pinned_post_modulation, pinned_rows)
    residual.copy_(s)
    branch.copy_(normed)
    return gate
