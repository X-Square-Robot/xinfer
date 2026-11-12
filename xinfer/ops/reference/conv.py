"""Torch reference implementations of the convolution operators.

No hardware backend registers these yet; the implementations are architecture-agnostic and run on every
device.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def conv2d(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None, stride: int) -> torch.Tensor:
    """``torch.nn.functional.conv2d`` without padding (patch embeddings).

    Args:
        x: ``[B, C, H, W]`` input, same dtype as ``weight``.
        weight: ``[O, C, kH, kW]`` filters.
        bias: ``[O]`` or None.
        stride: stride in both spatial dims.

    Returns:
        ``[B, O, H', W']`` in ``x.dtype``.
    """
    return F.conv2d(x, weight, bias, stride=stride)


def causal_conv1d(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Depthwise causal convolution followed by SiLU (the Qwen3.5 Gated DeltaNet short convolution).

    ``silu(conv1d(x, weight, padding=K - 1, groups=C)[..., :L])``: output ``t`` sees inputs ``t - K + 1 .. t``,
    zero before the first. Computed in the input dtype.

    Args:
        x: ``[B, C, L]`` channels-first input.
        weight: ``[C, 1, K]`` per-channel filters, same dtype as ``x``.

    Returns:
        ``[B, C, L]`` in ``x.dtype``.
    """
    length = x.shape[-1]
    return F.silu(F.conv1d(x, weight, None, padding=weight.shape[-1] - 1, groups=x.shape[1])[..., :length])
