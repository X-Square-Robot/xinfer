"""Normalization operators on CUDA, through xcompute (``_xcompute``)."""

from __future__ import annotations

from xinfer.ops.backends._xcompute import XCOMPUTE, xc
from xinfer.ops.registry import register


@register("rms_norm", backend="cuda", **XCOMPUTE)
def rms_norm(x, weight, eps):
    return xc.rmsnorm(x, weight, float(eps), True)


@register("add_rms_norm", backend="cuda", **XCOMPUTE)
def add_rms_norm(x, residual, weight, eps):
    return xc.add_rmsnorm(residual, x, weight, float(eps), inference=True)
