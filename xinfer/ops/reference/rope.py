"""Torch reference implementations of the rotary position embedding operators.

The backends fuse the rotation into their QKV projections; these standalone operators are the reference of
that rotation. The implementations are architecture-agnostic and run on every device.
"""

from __future__ import annotations

import torch
import math


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """``[-x2, x1]`` for ``x = [x1, x2]`` split in halves along the last dim; same shape and dtype."""
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """``x * cos + rotate_half(x) * sin``; ``cos``/``sin`` broadcast to ``x``.

    Computed in the promoted dtype of the inputs, each product and the sum rounded separately.

    Args:
        x: ``[..., D]`` tensor to rotate (``D`` even).
        cos: cosines broadcastable to ``x``.
        sin: sines broadcastable to ``x``.

    Returns:
        The rotated tensor in the promoted dtype of ``x``, ``cos`` and ``sin``.
    """
    return (x * cos) + (rotate_half(x) * sin)


def rotary_embedding(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate ``x [..., H, D]`` in FP32 and round back to ``x.dtype``.

    ``cos``/``sin`` are FP32 ``[..., 1, D]`` (broadcast over the heads):
    ``(x32 * cos + rotate_half(x32) * sin).to(x.dtype)``, each product and the
    sum rounded separately (no fused multiply-add).

    Args:
        x: ``[..., H, D]`` queries or keys (``D`` even).
        cos: FP32 ``[..., 1, D]`` cosines, broadcast over the heads.
        sin: FP32 ``[..., 1, D]`` sines.

    Returns:
        ``[..., H, D]`` in ``x.dtype``.
    """
    return apply_rotary(x.float(), cos, sin).to(x.dtype)


def rotary_embedding_rounded(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate ``x [..., H, D]`` in its own dtype, rounding each product and the sum (Hugging Face Gemma).

    ``cos``/``sin`` have ``x``'s dtype and shape ``[..., 1, D]``:
    ``x * cos + rotate_half(x) * sin`` with ``x * cos`` and ``rotate_half(x) * sin`` each rounded to
    ``x.dtype`` before the add.

    Args:
        x: ``[..., H, D]`` queries or keys (``D`` even).
        cos: ``[..., 1, D]`` cosines in ``x.dtype``, broadcast over the heads.
        sin: ``[..., 1, D]`` sines in ``x.dtype``.

    Returns:
        ``[..., H, D]`` in ``x.dtype``.
    """
    return apply_rotary(x, cos, sin)


# ---------------------------------------------------------------- WALL-OSS-1.0-Nano fused operators

def _merge_order(h: int, w: int, merge: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Patch rows and columns of an ``h × w`` grid in merge-block order."""
    shape = (h // merge, w // merge, merge, merge)
    rows = torch.arange(h, device=device).view(h // merge, 1, merge, 1).expand(shape).reshape(-1)
    cols = torch.arange(w, device=device).view(1, w // merge, 1, merge).expand(shape).reshape(-1)
    return rows, cols

def pos_embed_interp(weight: torch.Tensor, grids, merge_size: int) -> torch.Tensor:
    """Bilinear interpolation of a square position table to each image grid.

    Sample points are ``linspace(0, G - 1, h)`` (rows) and ``linspace(0, G - 1, w)`` (columns); the four corner
    rows are weighted in FP32 (``(1 - dh)(1 - dw)``, ``(1 - dh) dw``, ``dh (1 - dw)``, ``dh dw``, summed in that
    order) and rounded once. Every frame of a grid repeats its first frame.

    Args:
        weight: ``[G², C]`` position table.
        grids: Host sequence of ``(t, h, w)``; ``h`` and ``w`` divisible by ``merge_size``.
        merge_size: Patches merged per side.

    Returns:
        ``[Σ t·h·w, C]`` in ``weight.dtype``.
    """
    side = math.isqrt(weight.shape[0])
    out = []
    for t, h, w in grids:
        points = [torch.linspace(0, side - 1, n, device=weight.device) for n in (h, w)]
        floor = [p.int() for p in points]
        ceil = [(f + 1).clip(max=side - 1) for f in floor]
        dh, dw = (p - f for p, f in zip(points, floor))
        rows, cols = _merge_order(h, w, merge_size, weight.device)
        dh, dw = dh[rows, None], dw[cols, None]
        corner = lambda r, c: weight[(r[rows] * side + c[cols]).long()].float()  # noqa: E731
        value = (1 - dh) * (1 - dw) * corner(floor[0], floor[1])
        value = value + (1 - dh) * dw * corner(floor[0], ceil[1])
        value = value + dh * (1 - dw) * corner(ceil[0], floor[1])
        value = value + dh * dw * corner(ceil[0], ceil[1])
        out.append(value.to(weight.dtype).repeat(t, 1))
    return torch.cat(out)

def rot_pos(inv_freq: torch.Tensor, grids, merge_size: int) -> torch.Tensor:
    """2-D rotary angles of the patches: ``[row · inv_freq | col · inv_freq]`` in FP32.

    Args:
        inv_freq: ``[F]`` FP32 inverse frequencies.
        grids: Host sequence of ``(t, h, w)``.
        merge_size: Patches merged per side.

    Returns:
        ``[Σ t·h·w, 2F]`` FP32.
    """
    out = []
    for t, h, w in grids:
        rows, cols = _merge_order(h, w, merge_size, inv_freq.device)
        angles = torch.cat((rows[:, None].float() * inv_freq, cols[:, None].float() * inv_freq), dim=-1)
        out.append(angles.repeat(t, 1))
    return torch.cat(out)

def _rotate_half_split(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate the first ``2·cos.shape[-1]`` channels of ``x [..., heads, D]`` in FP32, rounded once."""
    half = cos.shape[-1]
    a, b = x[..., :half].float(), x[..., half:2 * half].float()
    rotated = torch.cat((a * cos - b * sin, b * cos + a * sin), dim=-1).to(x.dtype)
    return torch.cat((rotated, x[..., 2 * half:]), dim=-1)

def rope_full(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
              rotary_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Partial half-split RoPE of q and k into new tensors; channels past ``rotary_dim`` are copied.

    Args:
        q: ``[B, L, H, D]`` queries.
        k: ``[B, L, KV, D]`` keys.
        cos: FP32 ``[B or 1, L, rotary_dim / 2]`` compact cosines.
        sin: FP32 ``[B or 1, L, rotary_dim / 2]`` compact sines.
        rotary_dim: Rotated channels per head.

    Returns:
        ``(q, k)`` rotated, contiguous, in their input dtypes.
    """
    half = rotary_dim // 2
    c, s = cos[..., None, :half], sin[..., None, :half]
    return _rotate_half_split(q, c, s).contiguous(), _rotate_half_split(k, c, s).contiguous()
