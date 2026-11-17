"""Existing TileLang JIT routes, without a second copy of their kernel definitions."""

from xinfer.ops.backends.ascend.src.gemm.g2112_tilelang import (
    G2112_FLAGS,
    accumulator_route,
    down_route,
    glu_route,
    mm_fp32_route,
    samu_route,
)

__all__ = ["G2112_FLAGS", "accumulator_route", "down_route", "glu_route", "mm_fp32_route", "samu_route"]
