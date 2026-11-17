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

"""soup2: second-round soup kills for the wall-x 2B PUSH-FIAFIX stack (SOUP2-ASCENDC).

Site map (MSPROF-SCAN-R5 chosen steady window, per request):
- qksq_c6144 : GDN prefill qk stats pre-pass. The released chain is
               Mul(qk, qk) [bf16, strided rows of x_t [1,T,6144]] followed by
               the ReduceSum's implicit bf16->f32 cast. This kernel writes
               f32(RNE_bf16(a*a)) in one pass -- bit-identical bits into the
               untouched aclnn ReduceSum/Adds/Rsqrt, one launch instead of two
               and one GM read instead of three.
- rawsq_r128 : GDN gdrtailmix raw stats pre-pass. Cast(raw f32) + Square ->
               one pass writing RNE_f32(a*a); aclnn Mean/Adds/Rsqrt read the
               same bits (bitwise).

All kernels are TileLang multi-block SIMD (dconv4/soupform form -- the only
fast-and-stable vector-machine shape on this platform per ascendc-action /
move-r2 hard laws). fp32 vector exp/div route through the tilelang codegen
aliases (safe routines), never raw ccec intrinsics.
"""

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


def quant_u8(span=1024, cores=CORES):
    """out_u8 = uint8(clamp(floor(x + 0.5), 0, 255)); the resize quantize tail.

    Replaces the in-place add_/floor_/clamp_/to(uint8) chain (four aclnn
    launches, four passes) with one pass; clamp-before-floor equals
    floor-before-clamp for integer bounds, and every value entering the final
    conversion is an exact integer in [0, 255] (bitwise on finite inputs).
    """
    @T.prim_func
    def main(oh: T.handle, xh: T.handle, tiles: T.int64):
        out = T.match_buffer(oh, (tiles * span,), "uint8")
        x = T.match_buffer(xh, (tiles * span,), "float32")
        with T.Kernel(cores) as core:
            x_ub = T.alloc_shared((span,), "float32")
            o_ub = T.alloc_shared((span,), "uint8")
            T.annotate_buffer_versions({x_ub: 2, o_ub: 2})
            for step in T.Pipelined(T.ceildiv(tiles, cores), num_stages=2):
                tile = step * cores + core
                if tile < tiles:
                    base = T.cast(tile * span, "int32")
                    T.copy(x[base : base + span], x_ub)
                    with T.SimdVF():
                        m = T.simd.pset(32)
                        for i in range(span // LANES):
                            y = T.simd.vadds(
                                T.simd.vld(x_ub[i * LANES]), 0.5, m)
                            yc = T.simd.vmins(T.simd.vmaxs(y, 0.0, m), 255.0, m)
                            fi = T.simd.vcvt(yc, "int32", round="ROUND_F")
                            T.simd.vsts(
                                o_ub[i * LANES],
                                T.simd.vcvt(fi, "uint8", round="ROUND_Z"),
                                dist="PK4_B32")
                    T.copy(o_ub, out[base : base + span])

    return main


def build_programs():
    yield "qksq_c6144", qksq_c6144()
    yield "rawsq_r128", rawsq_r128()
    yield "quant_u8", quant_u8()
