"""Torch reference implementations of the elementwise activation operators.

No hardware backend registers these yet; the implementations are architecture-agnostic and run on every
device.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def silu_and_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Gated activation ``silu(gate) * up``, each rounded to the input dtype.

    Args:
        gate: gate half, any shape.
        up: up half, same shape and dtype as ``gate``.

    Returns:
        Tensor of the same shape and dtype.
    """
    return F.silu(gate) * up


def silu(x: torch.Tensor) -> torch.Tensor:
    """Elementwise ``x * sigmoid(x)`` (``torch.nn.functional.silu``), same shape and dtype as ``x``."""
    return F.silu(x)


def gelu(x: torch.Tensor) -> torch.Tensor:
    """Exact (erf) GELU, elementwise; same shape and dtype as ``x``."""
    return F.gelu(x)


def gelu_tanh(x: torch.Tensor) -> torch.Tensor:
    """Tanh-approximate GELU (``F.gelu(x, approximate="tanh")``), elementwise; same shape and dtype as ``x``."""
    return F.gelu(x, approximate="tanh")


def gelu_tanh_and_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Gated activation ``gelu_tanh(gate) * up`` (Gemma MLP), each rounded to the input dtype.

    Args:
        gate: gate half, any shape.
        up: up half, same shape and dtype as ``gate``.

    Returns:
        Tensor of the same shape and dtype.
    """
    return F.gelu(gate, approximate="tanh") * up


def gated_residual(x: torch.Tensor, y: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """``x + y * gate`` (an action expert's gated residual), the product rounded to the input dtype before the add.

    Args:
        x: ``[B, L, K]`` residual.
        y: ``[B, L, K]`` branch output, same dtype.
        gate: ``[B, 1, K]`` (or broadcastable) gate, same dtype.

    Returns:
        ``[B, L, K]`` in ``x.dtype``.
    """
    return x + y * gate
