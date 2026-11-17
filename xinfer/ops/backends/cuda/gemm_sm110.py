"""GEMM kernels for SM110 (Jetson Thor): the calls each one takes, and how it is launched.

As on SM120 (``gemm_sm120``), a kernel takes a call when the call has the epilogue the kernel implements and its
shape is one the kernel was tuned on: ``SHAPES`` maps ``(N, K)`` to the inclusive ranges of ``M`` over which it was
measured at least 5% faster than the generic path on a Jetson AGX Thor at three or more consecutive scan points
(the selection table of xcompute ``main`` ``f2b553b`` for the shapes of this repository's models). Any other call
runs the generic path.

The kernels are persistent tcgen05 GEMMs on CTA pairs: ``tcgen05_*`` for the vision blocks, ``mcast_*`` for the
language model, where a 6-CTA cluster shares each weight tile by TMA multicast for 512 < M <= 768. The ``mcast``
down projection and fused MLP add their partial sums into the residual in arrival order (``deterministic=False``).
"""

from __future__ import annotations

import torch

from xinfer.ops.backends.cuda._native import derived, kernel, lib, scratch
from xinfer.ops.backends.cuda.gemm import ACTIVATIONS, weight_nk


def _tuned(shapes, c):
    return any(lo <= c.m <= hi for lo, hi in shapes.get((c.n, c.k), ()))


def _glu_blocked(weight, block=128):
    """A ``[gate; up]`` weight with its rows in blocks of ``block`` gate rows, then the matching up rows."""
    half = weight.shape[0] // 2
    c = torch.arange(half // block, device=weight.device)[:, None]
    j = torch.arange(block, device=weight.device)[None, :]
    return weight[torch.cat([c * block + j, half + c * block + j], dim=1).reshape(-1)]


def _k_blocked(weight):
    """``[N, K]`` -> K-blocked: groups of 512 rows, each stored as K / 64 consecutive ``[512, 64]`` boxes."""
    n, k = weight.shape
    return weight.reshape(n // 512, 512, k // 64, 64).permute(0, 2, 1, 3).reshape(n, k).contiguous()


def _pad_halves(weight, multiple):
    """A ``[gate; up]`` weight with each half zero-padded to a multiple of ``multiple`` rows."""
    half = weight.shape[0] // 2
    extra = -half % multiple
    if not extra:
        return weight
    zeros = weight.new_zeros(extra, weight.shape[1])
    return torch.cat([weight[:half], zeros, weight[half:], zeros])


# ---------------------------------------------------------------------------
# Vision blocks
# ---------------------------------------------------------------------------
@kernel("qkv_proj_rope", 110)
class tcgen05_qkv_rope_bf16_sm110:
    """The Q | K | V GEMM + bias with rotate-half RoPE over heads of 80, writing Q and K in rotary-pair order
    (``qk_order="any"``). Reads the weight and bias with the rows of each Q and K head in pair order."""

    SHAPES = {(3840, 1280): [(16, 4096)]}
    HEAD_DIM = 80

    @classmethod
    def takes(cls, c):
        heads, kv_heads, head_dim, rotary_dim, table = c.rope
        return (c.qk_order == "any" and c.features == {"bias"} and table is not None and head_dim == cls.HEAD_DIM
                and rotary_dim == head_dim and c.n == (heads + 2 * kv_heads) * head_dim and _tuned(cls.SHAPES, c))

    @staticmethod
    def _pairs(weight, bias, heads, head_dim):
        half = head_dim // 2
        within = torch.arange(head_dim, device=weight.device)
        within = torch.stack([within[:half], within[half:]], dim=1).reshape(-1)
        order = torch.arange(weight.shape[0], device=weight.device)
        order[: heads * head_dim] = (torch.arange(heads, device=weight.device)[:, None] * head_dim + within).reshape(-1)
        return weight[order].contiguous(), bias[order].contiguous()

    @classmethod
    def run(cls, c, out):
        heads, kv_heads, head_dim, _, table = c.rope
        weight, bias = derived(f"rope_pairs:{c.bias.data_ptr()}", c.weight,
                               lambda: cls._pairs(c.weight, c.bias, heads + kv_heads, head_dim))
        lib.bf16_qkv_rope_sm110(out, c.x, weight, bias, table, heads * head_dim, kv_heads * head_dim, head_dim)


@kernel("gemm_glu", 110)
class tcgen05_glu_bf16_sm110:
    """Gate/up GEMM (+ bias) + SiLU(gate) * up, the gate and up rows read as two operands."""

    SHAPES = {(3424, 1280): [(1, 4096)], (3456, 1280): [(1, 4096)]}

    @classmethod
    def takes(cls, c):
        return c.features <= {"silu", "bias"} and c.activation == "silu" and _tuned(cls.SHAPES, c)

    @staticmethod
    def run(c, out):
        weight = weight_nk(c.weight)
        bias = (None, None) if c.bias is None else (c.bias[: c.n], c.bias[c.n :])
        lib.bf16_gemm_silu_mul_sm110(out, c.x, weight[: c.n], weight[c.n :], *bias)


# ---------------------------------------------------------------------------
# Language model (hidden 2048)
# ---------------------------------------------------------------------------
@kernel("gemm_glu", 110)
class mcast_glu_bf16_sm110:
    """Gate/up GEMM + SiLU(gate) * up on the FP32 accumulators, the weight rows in blocks of 128 gate rows, then
    the matching up rows."""

    SHAPES = {(11008, 2048): [(1, 224), (320, 2048)], (6144, 2048): [(160, 224), (272, 2048)]}

    @classmethod
    def takes(cls, c):
        return c.features == {"silu"} and _tuned(cls.SHAPES, c)

    @staticmethod
    def run(c, out):
        lib.mcast_glu_sm110(out, c.x, derived("glu_blocked128", c.weight, lambda: _glu_blocked(c.weight).contiguous()),
                            ACTIVATIONS[c.activation])


@kernel("gemm", 110)
class mcast_out_bf16_sm110:
    """GEMM + ``act(row_scale * product + bias)`` stored, act none or tanh GELU. Takes weights stored ``[N, K]``."""

    ENTRIES = (
        (frozenset({"bias", "gelu_tanh"}), {(4096, 1024): [(9, 11), (14, 512), (592, 2048)]}),
        (frozenset({"row_scale"}), {(8224, 2048): [(9, 2048)]}),
    )
    SHAPES = {shape: ranges for _, shapes in ENTRIES for shape, ranges in shapes.items()}
    LAYOUT = "nk"
    ACT = {None: -1, "gelu_tanh": 1}

    @classmethod
    def takes(cls, c):
        return c.weight.is_contiguous() and any(c.features == f and _tuned(s, c) for f, s in cls.ENTRIES)

    @classmethod
    def run(cls, c, out):
        lib.mcast_gemm_out_sm110(out, c.x, c.weight, c.bias, c.row_scale, cls.ACT[c.activation])


@kernel("gemm", 110)
class mcast_residual_bf16_sm110:
    """GEMM added to the BF16 residual. K 2048: split-K reduced in a fixed order in FP32, the sum rounded once.
    The down projections (K the intermediate size) read the weight K-blocked and add their split-K partial sums to
    the residual in BF16, in arrival order. Takes weights stored ``[N, K]``."""

    SHAPES = {(2048, 2048): [(352, 480), (576, 768)], (2048, 11008): [(16, 2048)],
              (2048, 6144): [(336, 768), (816, 896), (928, 2048)]}
    LAYOUT = "nk"
    DETERMINISTIC = False
    TOLERANCE = 1e-2  # partial sums rounded to BF16 before they reach the residual

    @classmethod
    def takes(cls, c):
        return c.weight.is_contiguous() and c.features == {"residual"} and _tuned(cls.SHAPES, c)

    @staticmethod
    def run(c, out):
        if out.data_ptr() != c.residual.data_ptr():
            out.copy_(c.residual)
        blocked = c.k != 2048
        weight = derived("k_blocked", c.weight, lambda: _k_blocked(c.weight)) if blocked else c.weight
        workspace = scratch("mcast_sm110", int(lib.mcast_sm110_workspace_bytes(c.m)), c.device, zeroed=True)
        lib.mcast_gemm_residual_sm110(out, c.x, weight, workspace, blocked)


@kernel("fused_mlp", 110)
class mcast_mlp_bf16_sm110:
    """The MLP ``residual += (SiLU(gate) * up) @ down^T`` in one persistent kernel for 512 < M <= 768: the
    intermediate (zero-padded to a multiple of ``PAD``) goes through an L2-resident ring of chunks between three
    6-CTA clusters, and the down projection adds its partial sums into the residual in arrival order. Takes the
    gate/up call (``Call`` of ``gemm_glu``); ``run`` adds into ``residual`` in place."""

    SHAPES = {(11008, 2048): [(520, 768)]}
    PAD = 1024

    @classmethod
    def takes(cls, c):
        return c.features == {"silu"} and _tuned(cls.SHAPES, c)

    @classmethod
    def _weights(cls, gate_up, down):
        gate_up = _k_blocked(_glu_blocked(_pad_halves(gate_up, cls.PAD)))
        return gate_up, _k_blocked(torch.nn.functional.pad(down, (0, -down.shape[1] % cls.PAD)))

    @classmethod
    def run(cls, c, down, residual):
        gate_up, down = derived(f"fused_mlp:{down.data_ptr()}", c.weight, lambda: cls._weights(c.weight, down))
        inter = down.shape[1]
        hbuf = scratch("gemm", int(lib.mcast_fused_mlp_sm110_hbuf_bytes(c.m, inter)), c.device)
        flags = scratch("mcast_mlp_sm110", int(lib.mcast_fused_mlp_sm110_flags_bytes()), c.device, zeroed=True)
        lib.mcast_fused_mlp_sm110(residual, c.x, gate_up, down, hbuf, flags, ACTIVATIONS[c.activation])
