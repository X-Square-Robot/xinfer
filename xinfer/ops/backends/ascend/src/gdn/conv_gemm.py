# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""conv-gemm knife: depthwise 4-tap causal conv as a fused single-pass vector kernel.

CONV-GEMM (MSPROF-SCAN-R4 roadmap #4). The production call replaced:

    mixed = F.conv1d(qkv.transpose(1, 2), w[:, None, :], padding=3, groups=C)[..., :T]
    x_t = mixed.transpose(1, 2).contiguous()     # token-major copy
    F.silu(x_t, inplace=True)

decomposes on aclnn into aclnnConvolution_{Slice,Transpose} satellites plus
Conv2dWithFlag_Conv2DV2 (23.5 us mean, cube-busy cu91 / mac 0.07 -- the small-M
form; R4 ledger pool 422.9 us/request over 18 GDN layers). A depthwise K=4 conv
has no GEMM form (im2col would need a block-diagonal weight), but as a shifted
4-tap FMA it is pure bandwidth: one pass reads the strided qkv rows
(token-major, stride 8224, columns 0..6143), accumulates the taps in FP32 and
writes the token-major BF16 result. That removes the slice, both transposes and
the vendor conv in one kernel; silu stays on aclnnSilu because aclnnSilu's
internal approximation is not bit-reproducible from vexp/vdiv (soupkill knife-B
evidence, 3-5/2.75M residual elements).

Numerics (probe2_pairwise.py, card 2, CANN 9.1.1): the vendor conv is bitwise
equal to sequential ascending FP32 accumulation

    acc = x[t-3]*w0; acc += x[t-2]*w1; acc += x[t-1]*w2; acc += x[t]*w3;  rne_bf16(acc)

on every probed shape (P in {416,432,448,464} x 3 seeds, edges 1..8, dense sweep
400..512) -- BF16 products are exact in FP32, so only the add order matters.
The kernel keeps that exact order; missing taps at t<3 contribute exact zero.
"""

import tilelang.ascend.language as T

CORES = 64
LANES = 64
CHANNELS = 6144
COL_BLOCK = 1024
ROWS_PER_TILE = 16
NCB = CHANNELS // COL_BLOCK  # 6 col blocks


def dconv4_c6144(cols=CHANNELS, col_block=COL_BLOCK, rows_per_tile=ROWS_PER_TILE, cores=CORES):
    """out[t, c] = rne_bf16(((w0*x[t-3] + w1*x[t-2]) + w2*x[t-1]) + w3*x[t]) ascending adds.

    out: [rows, cols] BF16 contiguous token-major.
    qkv: flat BF16, row stride ``stride`` elements, columns 0..cols-1 valid.
    w:   [4, cols] BF16 contiguous, pre-transposed tap-major weight.
    pad: [3, cols] BF16 zeros (zero rows for out-of-range reads).
    """
    window = rows_per_tile + 3
    in_span = window * col_block
    out_span = rows_per_tile * col_block
    w_span = 4 * col_block

    @T.prim_func
    def main(
        oh: T.handle,
        qh: T.handle,
        wh: T.handle,
        ph: T.handle,
        rows: T.int64,
        stride: T.int64,
    ):
        out = T.match_buffer(oh, (rows * cols,), "bfloat16")
        qkv = T.match_buffer(qh, ((rows + 3) * stride,), "bfloat16")
        wbuf = T.match_buffer(wh, (4 * cols,), "bfloat16")
        pad = T.match_buffer(ph, (3 * cols,), "bfloat16")
        row_tiles = T.ceildiv(rows, rows_per_tile)
        tiles = row_tiles * NCB
        with T.Kernel(cores) as core:
            in_ub = T.alloc_shared((in_span,), "bfloat16")
            out_ub = T.alloc_shared((out_span,), "bfloat16")
            w_ub = T.alloc_shared((w_span,), "bfloat16")
            T.annotate_buffer_versions({in_ub: 2, out_ub: 2, w_ub: 2})
            for step in T.Pipelined(T.ceildiv(tiles, cores), num_stages=2):
                tile = step * cores + core
                if tile < tiles:
                    rt = tile // NCB
                    cb = tile % NCB
                    base = rt * rows_per_tile
                    cbase = cb * col_block
                    padbase = cb * col_block
                    # history rows: global input rows base-3 .. base-1; rows
                    # before the sequence start read the zero pad.
                    for j in T.serial(3):
                        grow = base + j - 3
                        if grow >= 0:
                            src = grow * stride + cbase
                            T.copy(qkv[src : src + col_block],
                                   in_ub[j * col_block : (j + 1) * col_block])
                        else:
                            T.copy(pad[padbase : padbase + col_block],
                                   in_ub[j * col_block : (j + 1) * col_block])
                    # in-tile rows: global input rows base .. base+rows_per_tile-1
                    for j in T.serial(rows_per_tile):
                        grow = base + j
                        if grow < rows:
                            src = grow * stride + cbase
                            T.copy(qkv[src : src + col_block],
                                   in_ub[(3 + j) * col_block : (4 + j) * col_block])
                        else:
                            T.copy(pad[padbase : padbase + col_block],
                                   in_ub[(3 + j) * col_block : (4 + j) * col_block])
                    # tap-major weight rows for this column block
                    for k in T.serial(4):
                        wsrc = k * cols + cbase
                        T.copy(wbuf[wsrc : wsrc + col_block],
                               w_ub[k * col_block : (k + 1) * col_block])
                    with T.SimdVF():
                        m = T.simd.pset(32)
                        for r in T.serial(rows_per_tile):
                            for c in range(col_block // LANES):
                                acc = T.simd.vmul(
                                    T.simd.vcvt(
                                        T.simd.vld(in_ub[r * col_block + c * LANES], dist="UNPK_B16"),
                                        "float32", part=0),
                                    T.simd.vcvt(
                                        T.simd.vld(w_ub[c * LANES], dist="UNPK_B16"),
                                        "float32", part=0),
                                    m)
                                acc = T.simd.vadd(acc, T.simd.vmul(
                                    T.simd.vcvt(
                                        T.simd.vld(in_ub[(r + 1) * col_block + c * LANES], dist="UNPK_B16"),
                                        "float32", part=0),
                                    T.simd.vcvt(
                                        T.simd.vld(w_ub[col_block + c * LANES], dist="UNPK_B16"),
                                        "float32", part=0), m), m)
                                acc = T.simd.vadd(acc, T.simd.vmul(
                                    T.simd.vcvt(
                                        T.simd.vld(in_ub[(r + 2) * col_block + c * LANES], dist="UNPK_B16"),
                                        "float32", part=0),
                                    T.simd.vcvt(
                                        T.simd.vld(w_ub[2 * col_block + c * LANES], dist="UNPK_B16"),
                                        "float32", part=0), m), m)
                                acc = T.simd.vadd(acc, T.simd.vmul(
                                    T.simd.vcvt(
                                        T.simd.vld(in_ub[(r + 3) * col_block + c * LANES], dist="UNPK_B16"),
                                        "float32", part=0),
                                    T.simd.vcvt(
                                        T.simd.vld(w_ub[3 * col_block + c * LANES], dist="UNPK_B16"),
                                        "float32", part=0), m), m)
                                T.simd.vsts(
                                    out_ub[r * col_block + c * LANES],
                                    T.simd.vcvt(acc, "bfloat16", round="ROUND_R"),
                                    dist="PK_B32")
                    for r in T.serial(rows_per_tile):
                        if base + r < rows:
                            dst = (base + r) * cols + cbase
                            T.copy(out_ub[r * col_block : (r + 1) * col_block],
                                   out[dst : dst + col_block])

    return main


def build_programs():
    yield "dconv4_c6144", dconv4_c6144()
