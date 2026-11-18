"""GEMM kernels for SM120 (RTX 50 series): the calls each one takes, and how it is launched.

A kernel takes a call when the call has the epilogue the kernel implements and its shape is one the kernel was
tuned on: ``SHAPES`` maps ``(N, K)`` to the inclusive ranges of ``M`` over which the kernel was measured against
the generic path at every step of 16 rows, on an RTX 5090, and was not slower (within 1%, the noise of the
measurement). Any other call runs the generic path. The ranges have gaps because the generic path's time grows in
steps of M (whole waves of its tiles over the SMs), a kernel's more evenly.

``Call.features`` names what a call asks for beyond the BF16 product, e.g. ``{"bias", "gelu"}`` or
``{"mxfp8", "bias", "residual"}``.
"""

from __future__ import annotations

import torch

from xinfer.ops.backends.cuda._native import derived, kernel, lib, scratch
from xinfer.ops.backends.cuda.gemm import weight_kn, weight_nk
from xinfer.ops.layout import mxfp8 as layout


def _tuned(shapes, c):
    return any(lo <= c.m <= hi for lo, hi in shapes.get((c.n, c.k), ()))


@kernel("gemm", 120)
class smallm_bf16_sm120:
    """Tensor-core split-K GEMM for a few rows: the CTAs of a cluster each take a slice of K and add their FP32
    sums in a fixed order through distributed shared memory. The split depends on (K, N) only, so a row's result
    does not depend on M. Reads the weight as ``[K, N]``."""

    SHAPES = {(1024, 2048): [(1, 32)]}

    @classmethod
    def takes(cls, c):
        return c.features <= {"bias"} and _tuned(cls.SHAPES, c)

    @staticmethod
    def run(c, out):
        lib.smallm_bf16_sm120(out, c.x, weight_kn(c.weight), c.bias)


@kernel("gemm", 120)
class streamk_bf16_sm120:
    """Stream-K GEMM over all SMs. A tile split between CTAs is finished by the one that owns it, which waits
    only on earlier CTAs; the barriers live in a zeroed buffer the kernel leaves zeroed."""

    SHAPES = {(2048, 2048): [(336, 512), (656, 1152), (1296, 2048)], (2048, 11008): [(648, 800)]}
    FEATURES = frozenset()

    @classmethod
    def takes(cls, c):
        return c.features == cls.FEATURES and _tuned(cls.SHAPES, c)

    @staticmethod
    def run(c, out):
        shape = (c.m, c.n, c.k)
        workspace = scratch("gemm", int(lib.gemm_streamk_bf16_sm120_workspace_size(*shape)), c.device)
        barriers = scratch("streamk_bf16_sm120", int(lib.gemm_streamk_bf16_sm120_barrier_size(*shape)), c.device,
                           zeroed=True)
        lib.gemm_streamk_bf16_sm120(out, workspace, barriers, c.x, weight_nk(c.weight), c.bias, c.residual)


@kernel("gemm", 120)
class streamk_residual_bf16_sm120(streamk_bf16_sm120):
    """The same kernel with the bias and a BF16 residual added in its epilogue (one rounding of the sum)."""

    SHAPES = {(1024, 4096): [(352, 1152), (1360, 2048)]}
    FEATURES = frozenset({"bias", "residual"})


@kernel("gemm", 120)
class interleaved_streamk_gelu_bf16_sm120:
    """GEMM + bias + tanh GELU, persistent stream-K on 112 x 128 tiles with the epilogue of a tile interleaved
    into the main loop of the next one. A tile spans at most two CTAs; its owner merges the other's FP32 partial
    sum after a bounded wait, with a wait-free fallback in the same summation order."""

    SHAPES = {(4096, 1024): [(656, 4096)]}

    @classmethod
    def takes(cls, c):
        return c.features == {"bias", "gelu_tanh"} and _tuned(cls.SHAPES, c)

    @staticmethod
    def run(c, out):
        shape = (c.m, c.n, c.k)
        workspace = scratch("gemm", int(lib.interleaved_streamk_bf16_sm120_workspace_bytes(*shape)), c.device)
        flags = scratch("interleaved_streamk_bf16_sm120", int(lib.interleaved_streamk_bf16_sm120_flag_bytes(*shape)),
                        c.device, zeroed=True)
        lib.interleaved_streamk_bf16_sm120_gelu_tanh(out, c.x, weight_nk(c.weight), c.bias, workspace, flags)


@kernel("gemm", 120)
class tma_tile_bf16_sm120:
    """GEMM + bias + exact GELU on 80 x 128 output tiles, one CTA per tile, operands loaded with TMA: no
    workspace and no synchronisation between CTAs."""

    SHAPES = {(5120, 5120): [(520, 640)]}

    @classmethod
    def takes(cls, c):
        return c.features == {"bias", "gelu"} and _tuned(cls.SHAPES, c)

    @staticmethod
    def run(c, out):
        lib.tma_tile_bf16_sm120(out, c.x, weight_nk(c.weight), c.bias)


@kernel("gemm_glu", 120)
class tma_streamk_glu_bf16_sm120:
    """Gate/up GEMM + SiLU(gate) * up, persistent CTAs: data-parallel rounds, then a stream-K tail whose split
    tiles are merged after a bounded wait (with a wait-free fallback in the same summation order, so the result
    is deterministic)."""

    SHAPES = {(11008, 2048): [(528, 2016)], (6144, 2048): [(464, 480), (528, 1760), (1808, 2032)]}

    @classmethod
    def takes(cls, c):
        return c.features == {"silu"} and _tuned(cls.SHAPES, c)

    @staticmethod
    def run(c, out):
        shape = (c.m, c.n, c.k, c.device.index)
        workspace = scratch("gemm", int(lib.tma_streamk_glu_bf16_sm120_workspace_bytes(*shape)), c.device)
        flags = scratch("tma_streamk_glu_bf16_sm120", int(lib.tma_streamk_glu_bf16_sm120_flag_bytes(*shape)),
                        c.device, zeroed=True)
        lib.tma_streamk_glu_bf16_sm120(out, c.x, weight_nk(c.weight), workspace, flags)


@kernel("gemm_glu", 120)
class gateup_silu:
    """CUTLASS gate/up GEMM with SiLU(gate) * up in the epilogue. Reads the weight as ``[K, 2 N]`` with the gate
    and up columns of one output adjacent."""

    SHAPES = {(2048, 1024): [(1, 256)]}

    @classmethod
    def takes(cls, c):
        return c.features == {"silu"} and _tuned(cls.SHAPES, c)

    @staticmethod
    def _paired(weight):
        half = weight.shape[0] // 2
        order = torch.stack([torch.arange(half), torch.arange(half, 2 * half)], dim=1).reshape(-1)
        return weight[order.to(weight.device)].t().contiguous()

    @classmethod
    def run(cls, c, out):
        lib.gateup_silu(out, c.x, derived("kn_glu_paired", c.weight, lambda: cls._paired(c.weight)), -1)


# ---------------------------------------------------------------------------
# MXFP8 operands
# ---------------------------------------------------------------------------
def _tiles(values, scales, rope_rows=0):
    """An MXFP8 ``[N, K]`` weight (N a multiple of 256, K of 128) as the tile kernels read it: per tile of 256
    rows and step of 64 columns, the tile's values (16-byte chunks XOR-swizzled by ``(row >> 1) & 3``), then 1024
    bytes of its scales as words ``[wn 4][g 8][n 8]`` = row ``128 (n >> 2) + 32 wn + 8 (n & 3) + g``.

    Rows are permuted inside every group of 128 so that a thread of the kernel owns what its epilogue needs:
    stored row ``32 wn + 8 n + 2 t + j`` is row ``32 wn + 8 t + 2 n + j`` (8 adjacent outputs per thread), or,
    in the first ``rope_rows`` rows (heads of 64 with rotate-half RoPE), row
    ``64 (wn >> 1) + 32 (n >> 1) + 16 (wn & 1) + 4 t + 2 (n & 1) + j`` (both halves of 4 rotary pairs)."""
    n, k = values.shape
    device = values.device
    logical = scales.view(torch.uint8).reshape(-1)[layout.scale_offsets(n, k, device)][:, : k // 32]
    r = torch.arange(128, device=device)
    wn, nf, t, j = r >> 5, (r >> 3) & 3, (r >> 1) & 3, r & 1
    plain = 32 * wn + 8 * t + 2 * nf + j
    rotary = 64 * (wn >> 1) + 32 * (nf >> 1) + 16 * (wn & 1) + 4 * t + 2 * (nf & 1) + j
    base = torch.arange(0, n, 128, device=device)[:, None]
    order = (base + torch.where(base < rope_rows, rotary[None, :], plain[None, :])).reshape(-1)
    tiles, steps = n // 256, k // 64
    data = values.view(torch.uint8)[order].view(tiles, 256, steps, 4, 16).permute(0, 2, 1, 3, 4)
    row = torch.arange(256, device=device)
    chunk = torch.arange(4, device=device)[None, :] ^ ((row[:, None] >> 1) & 3)
    data = torch.gather(data, 3, chunk[None, None, :, :, None].expand(tiles, steps, 256, 4, 16))
    data = data.reshape(tiles, steps, 256 * 64)
    wn, g, nf = torch.meshgrid(torch.arange(4, device=device), torch.arange(8, device=device),
                               torch.arange(8, device=device), indexing="ij")
    rows = (128 * (nf >> 2) + 32 * wn + 8 * (nf & 3) + g).reshape(-1)
    words = logical[order].view(tiles, 256, steps // 2, 4)[:, rows]  # [tiles, 256 words, K / 128, 4]
    words = words.permute(0, 2, 1, 3).reshape(tiles, steps // 2, 1024)
    return torch.cat([data, words.repeat_interleave(2, dim=1)], dim=2).contiguous()


class _tile_mxfp8_sm120:
    """Tensor-core MXFP8 GEMM on whole output tiles of up to 160 rows, one CTA per tile with the epilogue in it:
    no workspace and nothing shared between CTAs. It takes at most one tile per SM, which bounds M by the SM
    count (``tile_mxfp8_sm120_supports``)."""

    FEATURES = frozenset()
    RESIDUAL = False

    @classmethod
    def takes(cls, c):
        return (c.features == cls.FEATURES and _tuned(cls.SHAPES, c)
                and bool(lib.tile_mxfp8_sm120_supports(c.m, c.n, c.k, c.device.index or 0, cls.RESIDUAL)))

    @staticmethod
    def weight(c, rope_rows=0):
        return derived(f"tiles{rope_rows}", c.weight, lambda: _tiles(c.weight, c.weight_scale, rope_rows))


@kernel("gemm", 120)
class tile_gelu_quantize_mxfp8_sm120(_tile_mxfp8_sm120):
    """GEMM + bias + tanh GELU on 256-column tiles, the result written as MXFP8 with one scale per 128 columns
    (K up to 1024)."""

    SHAPES = {(4096, 1024): [(16, 1600)]}
    FEATURES = frozenset({"mxfp8", "bias", "gelu_tanh", "quantize"})

    @classmethod
    def run(cls, c, out):
        lib.tile_mxfp8_gelu_quant128_sm120(out[0], out[1], c.x, c.x_scale, cls.weight(c), c.bias)


@kernel("gemm", 120)
class tile_residual_mxfp8_sm120(_tile_mxfp8_sm120):
    """GEMM + bias + BF16 residual on 128-column tiles, BF16 result with the bits of the generic path
    (``bf16(bf16(product + bias) + residual)``)."""

    SHAPES = {(1024, 1024): [(16, 3360)], (1024, 4096): [(16, 3360)]}
    FEATURES = frozenset({"mxfp8", "bias", "residual"})
    RESIDUAL = True

    @classmethod
    def run(cls, c, out):
        lib.tile_mxfp8_residual_sm120(out, c.x, c.x_scale, cls.weight(c), c.bias, c.residual, 0)


@kernel("qkv_proj_rope", 120)
class tile_qkv_rope_mxfp8_sm120(_tile_mxfp8_sm120):
    """The Q | K | V GEMM + bias with rotate-half RoPE on Q and K in the epilogue, for heads of 64 rotated over
    all their dims with a packed table; BF16 result with the bits of the generic path."""

    SHAPES = {(3072, 1024): [(16, 2240)]}
    FEATURES = frozenset({"mxfp8", "bias"})

    @classmethod
    def takes(cls, c):
        heads, kv_heads, head_dim, rotary_dim, table = c.rope
        return (head_dim == 64 and rotary_dim == 64 and table is not None and (heads + kv_heads) % 2 == 0
                and c.n == (heads + 2 * kv_heads) * 64 and super().takes(c))

    @classmethod
    def run(cls, c, out):
        rope_columns = (c.rope[0] + c.rope[1]) * 64
        lib.tile_mxfp8_qkv_rope_sm120(out, c.x, c.x_scale, cls.weight(c, rope_columns), c.bias, c.rope[4],
                                      rope_columns)
