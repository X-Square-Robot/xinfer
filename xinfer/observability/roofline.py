"""Roofline lower bound of one operator call, and the peaks it divides by (docs/profiling.md).

``PEAKS`` holds measured peaks per GPU name (``torch.cuda.get_device_name``): FLOP/s per compute pipe and DRAM
bandwidth in bytes/s. The RTX 5090 entry was measured on 2026-11-06 (driver 595.91.07, torch 2.9.1+cu130) at steady
state: each entry after about two seconds of continuous load, the median of 20 runs timed with CUDA events; the BF16
GEMM then ran at the 575 W power limit with the SM clock at 2617-2632 MHz (3090 MHz maximum). Methods:

- ``bf16``, ``fp32``: ``torch.mm`` of random 8192 x 8192 matrices (FP32: TF32 off), 2N³ / time;
- ``mxfp8``: the registered ``mxfp8_gemm`` (xcompute ``wallx_sm120``) on 8192³ operands with ``fast_accum=False``, the
  operator's default. The MXFP8 linear layers' ``fast_accum="auto"`` turns fast accumulation on for calls with at
  most 16 rows or at least 8192 outputs, which can run faster than this peak;
- ``nvfp4``: ``torch._scaled_mm`` of NVFP4 operands with E4M3 scales per 16 elements, 8192³;
- ``dram``: ``copy_`` of 1 GiB, bytes read and written / time.

Short bursts can run at higher clocks than this steady state, so a call can beat its bound by a few percent.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from xinfer.observability.op_cost import OpCost

PEAKS: dict[str, dict[str, float]] = {
    "NVIDIA GeForce RTX 5090": {"bf16": 220.1e12, "fp32": 66.2e12, "mxfp8": 574.3e12, "nvfp4": 1116.4e12,
                                "dram": 1522e9},
}


@dataclass(frozen=True)
class Bound:
    """The roofline lower bound of a call.

    Attributes:
        seconds: ``max(flops / peaks[pipe], bytes / peaks["dram"])``.
        limiter: ``"compute"`` when the compute term is the larger, else ``"memory"``.
    """

    seconds: float
    limiter: str


def roofline(cost: OpCost, peaks: Mapping[str, float]) -> Bound:
    """The lower bound of a call with ``cost`` on a device with ``peaks``.

    All bytes are charged at DRAM bandwidth: whether data is resident in L2 depends on the calls before, which a
    single call's cost does not tell, so there is no L2 limiter. Kernel launch latency is not included either: an
    operator may run several kernels.

    Args:
        cost: the call's ``OpCost``.
        peaks: FLOP/s per compute pipe (``"bf16"``, ``"mxfp8"``, ...) and ``"dram"`` bandwidth in bytes/s, as in
            ``PEAKS``.

    Returns:
        The ``Bound``. A ``"memory"`` pipe has no compute term.

    Raises:
        KeyError: ``peaks`` has no ``"dram"`` entry or none for the cost's pipe.
    """
    memory = cost.bytes / peaks["dram"]
    compute = 0.0 if cost.pipe == "memory" else cost.flops / peaks[cost.pipe]
    return Bound(max(compute, memory), "compute" if compute > memory else "memory")
