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

"""Elementwise square-and-cast kernels for Gated DeltaNet statistics.

The Q/K kernel squares strided BF16 values, rounds to BF16 and stores the
widened values as FP32. The output-statistics kernel widens BF16 to FP32
before squaring. The following reduction and reciprocal square root remain
in the caller, preserving their original reduction order."""

import tilelang.ascend.language as T

CORES = 64
LANES = 64


def qksq_c6144(width=6144, qcols=4096, cores=CORES):
    """sq[t, h, :] = f32(RNE_bf16(qk * qk)) over the strided qk window of x_t.

    x_t is [1, T, width] bf16 contiguous; the qk window is rows[:, :qcols]
    viewed [1, T, qcols//dk, dk]. One row per tile (row base = t * width).
    """
    span = qcols

    @T.prim_func
    def main(sh: T.handle, xh: T.handle, rows: T.int64):
        sq = T.match_buffer(sh, (rows * span,), "float32")
        x = T.match_buffer(xh, (rows * width,), "bfloat16")
        with T.Kernel(cores) as core:
            x_ub = T.alloc_shared((span,), "bfloat16")
            sq_ub = T.alloc_shared((span,), "float32")
            T.annotate_buffer_versions({x_ub: 2, sq_ub: 2})
            for step in T.Pipelined(T.ceildiv(rows, cores), num_stages=2):
                row = step * cores + core
                if row < rows:
                    xbase = T.cast(row * width, "int32")
                    sbase = T.cast(row * span, "int32")
                    T.copy(x[xbase : xbase + span], x_ub)
                    with T.SimdVF():
                        m = T.simd.pset(32)
                        for i in range(span // LANES):
                            a = T.simd.vcvt(
                                T.simd.vld(x_ub[i * LANES], dist="UNPK_B16"),
                                "float32", part=0)
                            p = T.simd.vmul(a, a, m)
                            r = T.simd.vcvt(p, "bfloat16", round="ROUND_R")
                            back = T.simd.vcvt(r, "float32", part=0)
                            T.simd.vsts(sq_ub[i * LANES], back)
                    T.copy(sq_ub, sq[sbase : sbase + span])

    return main


def rawsq_r128(cols=128, rows_per_tile=32, cores=CORES):
    """sq = RNE_f32(a*a) over flat bf16 [rows*cols]; one pass (Cast+Square)."""
    span = rows_per_tile * cols

    @T.prim_func
    def main(sh: T.handle, xh: T.handle, rows: T.int64):
        sq = T.match_buffer(sh, (rows * cols,), "float32")
        x = T.match_buffer(xh, (rows * cols,), "bfloat16")
        tiles = rows // rows_per_tile
        with T.Kernel(cores) as core:
            x_ub = T.alloc_shared((span,), "bfloat16")
            sq_ub = T.alloc_shared((span,), "float32")
            T.annotate_buffer_versions({x_ub: 2, sq_ub: 2})
            for step in T.Pipelined(T.ceildiv(tiles, cores), num_stages=2):
                tile = step * cores + core
                if tile < tiles:
                    base = T.cast(tile * span, "int32")
                    T.copy(x[base : base + span], x_ub)
                    with T.SimdVF():
                        m = T.simd.pset(32)
                        for i in range(span // LANES):
                            a = T.simd.vcvt(
                                T.simd.vld(x_ub[i * LANES], dist="UNPK_B16"),
                                "float32", part=0)
                            p = T.simd.vmul(a, a, m)
                            T.simd.vsts(sq_ub[i * LANES], p)
                    T.copy(sq_ub, sq[base : base + span])

    return main


def build_programs():
    yield "qksq_c6144", qksq_c6144()
    yield "rawsq_r128", rawsq_r128()
