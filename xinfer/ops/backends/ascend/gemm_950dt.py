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

This module is the scaffold: IN-TREE-R1 brings the kernel sources (``src/gemm/``), fills the ``run``
launches, and records the ``(N, K)`` of the calls the audit lists by dose only. The backend's ``gemm``
dispatcher imports it at its tail, as the CUDA dispatcher imports ``gemm_sm120``.
"""

from __future__ import annotations

from xinfer.ops.backends.ascend._native import kernel


def _tuned(shapes, c):
    return any(lo <= c.m <= hi for lo, hi in shapes.get((c.n, c.k), ()))


@kernel("gemm", "ascend950dt9582")
class smallm_r3_down:
    """The m16 down / o_proj GEMM of the smallm-r3 knife (EPILOGUE-R2 knife A).

    One kernel serves both models' small-batch projections: Wall-X 2B's down and o_proj (300 calls per
    request, 240 down + 60 o_proj, at 4.6 us) and Wall-OSS's down + o_proj (360 calls). Gate: bitwise against
    the vendor route, action gate 0.0320437253.

    TODO (R1): the launch (``lib.<entry>`` over ``src/gemm/smallm_r3.py``) and the exact feature set the
    epilogue takes (bias).
    """

    SHAPES = {(1024, 2048): [(16, 16)]}  # 2B [16, 2048] x [1024, 2048]; walloss's down / o_proj shapes match
    FEATURES = frozenset()

    @classmethod
    def takes(cls, c):
        return c.features == cls.FEATURES and _tuned(cls.SHAPES, c)

    @staticmethod
    def run(c, out):
        raise NotImplementedError("IN-TREE-R1: launch smallm_r3_down from src/gemm/ (smallm_r3.py)")


@kernel("gemm_glu", "ascend950dt9582")
class smallm_r3_gateup_glu:
    """The m16 gate/up GEMM with the SiLU * up epilogue in one launch (GLU-EPILOGUE knife A).

    The weight is read as the model stores it and the gate/up product, SiLU and multiply never leave the
    kernel. Serves 240 calls per Wall-X 2B request and 180 on Wall-OSS, at 7.5 us. Gate: 0.00489, bitwise
    level. The only two kernels the models share with ``smallm_r3_down``.

    TODO (R1): the launch (``lib.<entry>`` over ``src/gemm/smallm_r3.py``).
    """

    SHAPES = {(2048, 1024): [(16, 16)]}  # [16, 1024] x [4096, 1024], 4096 = 2 x 2048 (gate | up)
    FEATURES = frozenset({"silu"})

    @classmethod
    def takes(cls, c):
        return c.features == cls.FEATURES and _tuned(cls.SHAPES, c)

    @staticmethod
    def run(c, out):
        raise NotImplementedError("IN-TREE-R1: launch smallm_r3_gateup_glu from src/gemm/ (smallm_r3.py)")


@kernel("qkv_proj_rope", "ascend950dt9582")
class gemm_qkv_2b_norm_gate_cache_bnsdq:
    """Wall-X 2B's fused decode QKV projection: GEMM + QkNorm + gate pack + KV-cache append in one kernel.

    ``[16, 1024] x [5120, 1024]`` with the 5120/8/2/256 lineage (20 heads of 256: 8 dense Q in BNSD order,
    the packed gate, and the 2+2 KV append). Serves 60 calls per request at 8.3 us. Gate: the 2b-qkv wave's
    bitwise witness. TODO (R1): the launch and the epilogue guards (the cache slice bounds and the gate
    layout the kernel asserts).
    """

    SHAPES = {(5120, 1024): [(16, 16)]}
    FEATURES = frozenset({"bias"})

    @classmethod
    def takes(cls, c):
        return c.features == cls.FEATURES and _tuned(cls.SHAPES, c)

    @staticmethod
    def run(c, out):
        raise NotImplementedError("IN-TREE-R1: launch gemm_qkv_2b_norm_gate_cache_bnsdq from src/gemm/ "
                                  "(qkv_bnsdq.py)")


@kernel("qkv_proj_rope", "ascend950dt9582")
class gemm_qkv_cache_bias_rope_m16:
    """Wall-OSS's decode QKV finish: bias + RoPE + KV-cache append over the vendor GEMM's output.

    The GEMM body stays aclnn (``16, 1024 -> 2560``); this kernel finishes it. Serves 180 calls per request
    at 4.6 us. Gate: the walloss-fw lineage. TODO (R1): the launch and the rope-table handling.
    """

    SHAPES = {(2560, 1024): [(16, 16)]}
    FEATURES = frozenset({"bias"})

    @classmethod
    def takes(cls, c):
        return c.features == cls.FEATURES and _tuned(cls.SHAPES, c)

    @staticmethod
    def run(c, out):
        raise NotImplementedError("IN-TREE-R1: launch gemm_qkv_cache_bias_rope_m16 from src/gemm/ "
                                  "(gemm_qkv_cache.py)")


@kernel("qkv_proj_rope", "ascend950dt9582")
class gemm_qkv_cache_bias_rope_prefill:
    """Wall-OSS's prefill QKV cache finish (the prefill arm of the same knife).

    Serves 36 calls per request at 15.2 us. Gate: the walloss-fw lineage. TODO (R1): the measured ``(N, K)``
    of the prefill widths (the audit lists the dose only) and the launch.
    """

    SHAPES = {}  # TODO (R1): the prefill (N, K) widths, then [(M of the served prefill batches)]
    FEATURES = frozenset({"bias"})

    @classmethod
    def takes(cls, c):
        return c.features == cls.FEATURES and _tuned(cls.SHAPES, c)

    @staticmethod
    def run(c, out):
        raise NotImplementedError("IN-TREE-R1: launch gemm_qkv_cache_bias_rope_prefill from src/gemm/ "
                                  "(gemm_qkv_cache.py)")


@kernel("gemm", "ascend950dt9582")
class bnsd_o_proj_m768:
    """Wall-OSS's o_proj GEMM of the prefill/vision segments at M 768 (32 blocks padded).

    Serves 35 calls per request at 24.2 us. Gate: the smallm-r3 lineage routing. TODO (R1): the measured
    ``(N, K)`` (the audit lists M and the dose only) and the launch (``src/gemm/gemm_smallm_action.py``).
    """

    SHAPES = {}  # TODO (R1): (N, K) of the o_proj weights, then [(768, 768)]
    FEATURES = frozenset()

    @classmethod
    def takes(cls, c):
        return c.features == cls.FEATURES and _tuned(cls.SHAPES, c)

    @staticmethod
    def run(c, out):
        raise NotImplementedError("IN-TREE-R1: launch bnsd_o_proj_m768 from src/gemm/ (gemm_smallm_action.py)")


@kernel("gemm", "ascend950dt9582")
class g2112_prefqkv:
    """Wall-OSS's prefill QKV GEMM of the g2112 family (AICORE).

    Serves 36 calls per request at 25.9 us; v0.79, cu91. Gate: the g2112 wave. TODO (R1): the measured
    ``(N, K)`` and the launch (``knives/g2112_tilelang.py``, prefqkv).
    """

    SHAPES = {}  # TODO (R1): the prefill qkv (N, K), then the M range of the served widths
    FEATURES = frozenset()

    @classmethod
    def takes(cls, c):
        return c.features == cls.FEATURES and _tuned(cls.SHAPES, c)

    @staticmethod
    def run(c, out):
        raise NotImplementedError("IN-TREE-R1: launch g2112_prefqkv from src/gemm/ (g2112_tilelang.py)")
