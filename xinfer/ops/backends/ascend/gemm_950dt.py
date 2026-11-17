"""GEMM kernels for the Ascend 950DT (``ascend950dt9582``): the calls each one takes, and how it is launched.

A kernel takes a call when the call has the epilogue the kernel implements and its shape is one the kernel
serves: ``SHAPES`` maps ``(N, K)`` to the inclusive ranges of ``M`` over which the kernel was deployed and
gate-validated on an Ascend 950DT. The envelopes come from the production audit of 2026-11-16, which measured
each kernel's route dose and numerical gate on the serving stacks (Wall-X 2B 20.62 ms, Wall-OSS 38.82 ms);
they are the served shapes - the decode batch is fixed at 16 rows, the prefill widths are the model's - not a
sweep over every step of 16 rows as the CUDA side measured its envelopes on an RTX 5090. Any other call runs
the generic path: the aclnn matmul operators with the audited direct routes (``finplain``, a single
aclnnMatmul bf16 over ``(k, n)`` in {(2048, 2048), (2048, 5120), (1024, 1536), (1024, 4096), (2048, 1024)}
with ``m`` in [1, 8192], bitwise 12/12 on the production shapes), and then the torch reference. The envelope
is static data here - what a kernel takes is declared next to it, never probed or cached at call time.
``Call.features`` names what a call asks for beyond the BF16 product, e.g. ``{"bias"}`` or ``{"silu"}``.

The ``run`` launches go through the backend's unified boundary (``_native.launch_kernel``: the single
library's TVM-FFI face, the family library's ctypes launcher as the fallback); the fused QKV finishes
delegate to the glue entries that own their serving envelopes (the cache bounds and gate-layout guards),
which launch through the same boundary. The dispatcher observability (``Call``, ``kernel_name``) follows the
CUDA backend's gemm module: a call is normalized once, the kernels of the device's SoC are asked in order,
and ``kernel_name`` answers which one takes it - "generic" when none does.
"""

from __future__ import annotations

import dataclasses
from typing import Optional

import torch

from xinfer.ops.backends.ascend import _native
from xinfer.ops.backends.ascend._native import kernel, launch_kernel


def _tuned(shapes, c):
    return any(lo <= c.m <= hi for lo, hi in shapes.get((c.n, c.k), ()))


@dataclasses.dataclass
class Call:
    """One GEMM call as the kernels see it."""

    x: torch.Tensor  # [M, K] BF16 rows (or the BNSD [H, M, D] attention output the o-projection routes read)
    weight: torch.Tensor  # [N, K] as the model stores it
    bias: Optional[torch.Tensor]
    activation: Optional[str]
    m: int
    n: int  # output features (gemm_glu: per half)
    k: int
    caches: tuple = ()  # qkv finishes: (k_cache, v_cache, start) the kernel appends to
    cos: Optional[torch.Tensor] = None  # qkv finishes: the rotation tables
    sin: Optional[torch.Tensor] = None
    factors: tuple = ()  # the per-head (1 + weight) norm factors the fused finishes read
    out_hint: tuple = ()  # the caller's output allocation the finish writes

    @property
    def device(self):
        return self.x.device

    @property
    def features(self):
        """What the call asks for beyond the BF16 product: a subset of {"bias", <activation>}."""
        parts = {name for name in ("bias",) if getattr(self, name) is not None}
        return frozenset(parts | ({self.activation} if self.activation else set()))


def kernel_name(op, x, weight, bias=None, activation=None, **finish):
    """Name of the kernel that runs ``op(x, weight, bias, ...)`` on this NPU, or ``"generic"``.

    The observability the CUDA backend's gemm module exposes: the SoC's kernels are asked in
    order, the first that takes the normalized call wins - never the registration order.
    """
    c = Call(x=x, weight=weight, bias=bias, activation=activation, m=x.shape[0],
             n=weight.shape[0], k=weight.shape[-1], **finish)
    for cls in _native.KERNELS[op]:
        if cls.takes(c):
            return cls.__name__
    return "generic"


@kernel("gemm", "ascend950dt9582")
class smallm_r3_down:
    """The m16 down / o_proj GEMM of the smallm-r3 knife (EPILOGUE-R2 knife A).

    One kernel serves both models' small-batch projections: Wall-X 2B's down and o_proj (300 calls per
    request, 240 down + 60 o_proj, at 4.6 us) and Wall-OSS's down + o_proj (360 calls). Gate: bitwise against
    the vendor route, action gate 0.0320437253.
    """

    SHAPES = {(1024, 2048): [(16, 16)]}  # 2B [16, 2048] x [1024, 2048]; walloss's down / o_proj shapes match
    FEATURES = frozenset()

    @classmethod
    def takes(cls, c):
        return c.features == cls.FEATURES and _tuned(cls.SHAPES, c)

    @staticmethod
    def run(c, out=None):
        if out is None:
            out = c.x.new_empty((c.x.shape[0], c.weight.shape[0]), dtype=torch.bfloat16)
        if not launch_kernel("smallm_r3_down", (c.x, c.weight, out)):
            from xinfer.ops.backends.ascend import smallm_r3

            return smallm_r3.down(c.x, c.weight)  # the family-library fallback owns the same launch
        return out


@kernel("gemm_glu", "ascend950dt9582")
class smallm_r3_gateup_glu:
    """The m16 gate/up GEMM with the SiLU * up epilogue in one launch (GLU-EPILOGUE knife A).

    The weight is read as the model stores it and the gate/up product, SiLU and multiply never leave the
    kernel. Serves 240 calls per Wall-X 2B request and 180 on Wall-OSS, at 7.5 us. Gate: 0.00489, bitwise
    level. The only two kernels the models share with ``smallm_r3_down``.
    """

    SHAPES = {(2048, 1024): [(16, 16)]}  # [16, 1024] x [4096, 1024], 4096 = 2 x 2048 (gate | up)
    FEATURES = frozenset({"silu"})

    @classmethod
    def takes(cls, c):
        return c.features == cls.FEATURES and _tuned(cls.SHAPES, c)

    @staticmethod
    def run(c, out):
        from xinfer.ops.backends.ascend import smallm_r3

        # the interleaved (gate | up) prepack is cached per weight by the module that owns the
        # serving envelope; the kernel launches through the unified boundary inside it
        return smallm_r3.gateup(c.x, c.weight)


@kernel("qkv_proj_rope", "ascend950dt9582")
class gemm_qkv_2b_norm_gate_cache_bnsdq:
    """Wall-X 2B's fused decode QKV projection: GEMM + QkNorm + gate pack + KV-cache append in one kernel.

    ``[16, 1024] x [5120, 1024]`` with the 5120/8/2/256 lineage (20 heads of 256: 8 dense Q in BNSD order,
    the packed gate, and the 2+2 KV append). Serves 60 calls per request at 8.3 us. Gate: the 2b-qkv wave's
    bitwise witness.
    """

    SHAPES = {(5120, 1024): [(16, 16)]}
    FEATURES = frozenset({"bias"})

    @classmethod
    def takes(cls, c):
        return c.features == cls.FEATURES and _tuned(cls.SHAPES, c)

    @staticmethod
    def run(c, out):
        # the serving envelope (the cache bounds, the gate layout, the (1 + w) factors and the
        # identity rotation tables of the prefill split form) is the action bridge's own entry -
        # it launches through the unified boundary
        raise NotImplementedError("the fused 2B QKV tail serves through the action bridge's entry")


@kernel("qkv_proj_rope", "ascend950dt9582")
class gemm_qkv_cache_bias_rope_m16:
    """Wall-OSS's decode QKV finish: bias + RoPE + KV-cache append over the vendor GEMM's output.

    The GEMM body stays aclnn (``16, 1024 -> 2560``); this kernel finishes it. Serves 180 calls per request
    at 4.6 us. Gate: the walloss-fw lineage.
    """

    SHAPES = {(2560, 1024): [(16, 16)]}
    FEATURES = frozenset({"bias"})

    @classmethod
    def takes(cls, c):
        return c.features == cls.FEATURES and _tuned(cls.SHAPES, c)

    @staticmethod
    def run(c, out):
        from xinfer.ops.backends.ascend import core

        # the M16 finish's serving envelope (the BNSD out forms, the cache bounds) is the glue's
        return core.gemm_qkv_cache_bias_rope_m16(out, c.caches[0], c.caches[1], c.x, c.bias, c.cos, c.sin,
                                                 c.caches[2])


@kernel("qkv_proj_rope", "ascend950dt9582")
class gemm_qkv_cache_bias_rope_prefill:
    """Wall-OSS's prefill QKV cache finish (the prefill arm of the same knife).

    Serves 36 calls per request at 15.2 us over the FP32 ``[rows, 2560]`` accumulator of the prefill QKV
    GEMM (``[rows, 2048] x [2560, 2048]``; the served prefill rows are the protocol's 736-768 family, the
    leaf battery's envelope). Gate: the walloss-fw lineage.
    """

    SHAPES = {(2560, 2048): [(736, 768)]}
    FEATURES = frozenset({"bias"})

    @classmethod
    def takes(cls, c):
        return c.features == cls.FEATURES and _tuned(cls.SHAPES, c)

    @staticmethod
    def run(c, out):
        from xinfer.ops.backends.ascend import core

        return core.gemm_qkv_cache_bias_rope_prefill(out, c.caches[0], c.caches[1], c.x, c.bias, c.cos,
                                                     c.sin, c.caches[2])


@kernel("gemm", "ascend950dt9582")
class bnsd_o_proj_m768:
    """Wall-OSS's o_proj GEMM of the prefill/vision segments at M 768 (32 blocks padded).

    The prefill attention output ``[16, 768, 128]`` over the ``[2048, 2048]`` weight (the model's expert-0
    width), with the head transpose folded into the Cube A-load and a BF16 ``[768, 2048]`` out. Serves 35
    calls per request at 24.2 us. Gate: the smallm-r3 lineage routing.
    """

    SHAPES = {(2048, 2048): [(768, 768)]}
    FEATURES = frozenset()

    @classmethod
    def takes(cls, c):
        return c.features == cls.FEATURES and _tuned(cls.SHAPES, c)

    @staticmethod
    def run(c, out=None):
        if out is None:
            out = torch.empty((c.x.shape[1], c.weight.shape[0]), dtype=torch.bfloat16,
                              device=c.x.device)  # the [768, 2048] rows of the BNSD [16, 768, 128] input
        if not launch_kernel("bnsd_o_proj_m768_bf16", (c.x, c.weight, out)):
            raise RuntimeError("the bnsd_o_proj_m768 kernel is not in this build")
        return out


@kernel("gemm", "ascend950dt9582")
class g2112_prefqkv:
    """Wall-OSS's prefill QKV GEMM of the g2112 family (AICORE).

    The vision segments' patch rows ``[2112, 1280]`` over the ``[3840, 1280]`` packed QKV weight (the ViT
    width); the finish that follows is ``gemm_qkv_vision_rope_finish``'s. Serves 36 calls per request at
    25.9 us; v0.79, cu91. Gate: the g2112 wave. The g2112 pair is delivered as TileLang JIT modules (the
    family map's documented exception - ``src/FAMILIES.md``); this class declares its envelope for the
    dispatcher, and the serving route stays the g2112 module's.
    """

    SHAPES = {(3840, 1280): [(2112, 2112)]}
    FEATURES = frozenset()

    @classmethod
    def takes(cls, c):
        return c.features == cls.FEATURES and _tuned(cls.SHAPES, c)

    @staticmethod
    def run(c, out):
        # the g2112 family is JIT (not an AOT object of the library): the serving route is the
        # module's own (XINFER_G2112_PREFQKV), outside the unified launch boundary
        raise NotImplementedError("the g2112 prefqkv GEMM serves through the g2112 module's JIT route")
