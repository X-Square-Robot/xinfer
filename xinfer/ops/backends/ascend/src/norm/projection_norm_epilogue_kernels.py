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

"""Elementwise epilogues for Wall-X 2B projections and normalization.

Residual-norm preparation writes a BF16 sum and its FP32 square; apply
multiplies by the reciprocal RMS and zero-centered weight. Z/B/A splitting
scales each row and packs its two outputs, and the SwiGLU epilogue multiplies
the activated gate by the up projection. Vision QKV/RoPE writes directly
in contiguous [3, rows, heads, dim] order."""

import tilelang.ascend.language as T

CORES = 64
LANES = 64


def normprep_c2048(cols=2048, rows_per_tile=4, cores=CORES):
    """(s, sq) = (rne(x+res), f32(s)^2) over bf16 inputs, one pass."""
    span = rows_per_tile * cols

    @T.prim_func
    def main(sh: T.handle, sqh: T.handle, xh: T.handle, rh: T.handle, rows: T.int64):
        s = T.match_buffer(sh, (rows * cols,), "bfloat16")
        sq = T.match_buffer(sqh, (rows * cols,), "float32")
        x = T.match_buffer(xh, (rows * cols,), "bfloat16")
        r = T.match_buffer(rh, (rows * cols,), "bfloat16")
        tiles = rows // rows_per_tile
        with T.Kernel(cores) as core:
            x_ub = T.alloc_shared((span,), "bfloat16")
            r_ub = T.alloc_shared((span,), "bfloat16")
            s_ub = T.alloc_shared((span,), "bfloat16")
            sq_ub = T.alloc_shared((span,), "float32")
            T.annotate_buffer_versions({x_ub: 2, r_ub: 2, s_ub: 2, sq_ub: 2})
            for step in T.Pipelined(T.ceildiv(tiles, cores), num_stages=2):
                tile = step * cores + core
                if tile < tiles:
                    base = T.cast(tile * span, "int32")
                    T.copy(x[base : base + span], x_ub)
                    T.copy(r[base : base + span], r_ub)
                    with T.SimdVF():
                        m = T.simd.pset(32)
                        for i in range(span // LANES):
                            a = T.simd.vcvt(
                                T.simd.vld(x_ub[i * LANES], dist="UNPK_B16"),
                                "float32", part=0)
                            b = T.simd.vcvt(
                                T.simd.vld(r_ub[i * LANES], dist="UNPK_B16"),
                                "float32", part=0)
                            T.simd.vsts(
                                s_ub[i * LANES],
                                T.simd.vcvt(T.simd.vadd(a, b, m),
                                           "bfloat16", round="ROUND_R"),
                                dist="PK_B32")
                            sf = T.simd.vcvt(
                                T.simd.vld(s_ub[i * LANES], dist="UNPK_B16"),
                                "float32", part=0)
                            T.simd.vsts(sq_ub[i * LANES],
                                        T.simd.vmul(sf, sf, m))
                    T.copy(s_ub, s[base : base + span])
                    T.copy(sq_ub, sq[base : base + span])

    return main


def apply_rstd_c2048(cols=2048, rows_per_tile=4, cores=CORES):
    """out = rne((f32(x) * rstd[row]) * (1 + w)); elementwise only."""
    span = rows_per_tile * cols

    @T.prim_func
    def main(oh: T.handle, xh: T.handle, rh: T.handle, wh: T.handle, rows: T.int64):
        out = T.match_buffer(oh, (rows * cols,), "bfloat16")
        x = T.match_buffer(xh, (rows * cols,), "bfloat16")
        rstd = T.match_buffer(rh, (rows,), "float32")
        w = T.match_buffer(wh, (cols,), "float32")
        tiles = rows // rows_per_tile
        with T.Kernel(cores) as core:
            x_ub = T.alloc_shared((span,), "bfloat16")
            r_ub = T.alloc_shared((rows_per_tile,), "float32")
            w_ub = T.alloc_shared((cols,), "float32")
            out_ub = T.alloc_shared((span,), "bfloat16")
            T.annotate_buffer_versions({x_ub: 2, r_ub: 2, w_ub: 2, out_ub: 2})
            T.copy(w, w_ub)
            for step in T.Pipelined(T.ceildiv(tiles, cores), num_stages=2):
                tile = step * cores + core
                if tile < tiles:
                    base = T.cast(tile * span, "int32")
                    rbase = T.cast(tile * rows_per_tile, "int32")
                    T.copy(x[base : base + span], x_ub)
                    T.copy(rstd[rbase : rbase + rows_per_tile], r_ub)
                    with T.SimdVF():
                        m = T.simd.pset(32)
                        one = T.simd.vdup(1.0, "float32", m)
                        for ri in T.serial(rows_per_tile):
                            for i in range(cols // LANES):
                                a = T.simd.vcvt(
                                    T.simd.vld(x_ub[ri * cols + i * LANES],
                                               dist="UNPK_B16"),
                                    "float32", part=0)
                                wv = T.simd.vld(w_ub[i * LANES])
                                t = T.simd.vmuls(a, r_ub[ri], m)
                                u = T.simd.vadd(one, wv, m)
                                T.simd.vsts(
                                    out_ub[ri * cols + i * LANES],
                                    T.simd.vcvt(T.simd.vmul(t, u, m),
                                                "bfloat16", round="ROUND_R"),
                                    dist="PK_B32")
                    T.copy(out_ub, out[base : base + span])

    return main


def zbasplit(conv=6144, tail=2080, rows_per_tile=1, cores=CORES):
    """o1 = rne(acc[:, :conv] * keep), o2 = rne(acc[:, conv:] * keep), one pass.

    conv = 6144 (= 96 x 64) streams in full vectors; tail = 2080 pads its UB
    tile to 2112 (= 33 x 64) so the last vector over-reads padded lanes only,
    and the global T.copy stores exactly 2080 elements per row.
    """
    width = conv + tail
    span = rows_per_tile * width
    tail_pad = (tail + LANES - 1) // LANES * LANES

    @T.prim_func
    def main(o1h: T.handle, o2h: T.handle, ah: T.handle, kh: T.handle, rows: T.int64):
        o1 = T.match_buffer(o1h, (rows * conv,), "bfloat16")
        o2 = T.match_buffer(o2h, (rows * tail,), "bfloat16")
        acc = T.match_buffer(ah, (rows * width,), "float32")
        keep = T.match_buffer(kh, (rows,), "float32")
        tiles = rows // rows_per_tile
        with T.Kernel(cores) as core:
            acc_ub = T.alloc_shared((span,), "float32")
            keep_ub = T.alloc_shared((rows_per_tile,), "float32")
            o1_ub = T.alloc_shared((rows_per_tile * conv,), "bfloat16")
            o2_ub = T.alloc_shared((rows_per_tile * tail_pad,), "bfloat16")
            T.annotate_buffer_versions({acc_ub: 2, keep_ub: 2, o1_ub: 2, o2_ub: 2})
            for step in T.Pipelined(T.ceildiv(tiles, cores), num_stages=2):
                tile = step * cores + core
                if tile < tiles:
                    base = T.cast(tile * span, "int32")
                    rbase = T.cast(tile * rows_per_tile, "int32")
                    T.copy(acc[base : base + span], acc_ub)
                    T.copy(keep[rbase : rbase + rows_per_tile], keep_ub)
                    with T.SimdVF():
                        m = T.simd.pset(32)
                        for ri in T.serial(rows_per_tile):
                            for i in range(conv // LANES):
                                v = T.simd.vmuls(
                                    T.simd.vld(acc_ub[ri * width + i * LANES]),
                                    keep_ub[ri], m)
                                T.simd.vsts(
                                    o1_ub[ri * conv + i * LANES],
                                    T.simd.vcvt(v, "bfloat16", round="ROUND_R"),
                                    dist="PK_B32")
                            for i in range(tail_pad // LANES):
                                v = T.simd.vmuls(
                                    T.simd.vld(acc_ub[ri * width + conv + i * LANES]),
                                    keep_ub[ri], m)
                                T.simd.vsts(
                                    o2_ub[ri * tail_pad + i * LANES],
                                    T.simd.vcvt(v, "bfloat16", round="ROUND_R"),
                                    dist="PK_B32")
                    o1base = T.cast(tile * rows_per_tile * conv, "int32")
                    o2base = T.cast(tile * rows_per_tile * tail, "int32")
                    T.copy(o1_ub, o1[o1base : o1base + rows_per_tile * conv])
                    T.copy(o2_ub[0 : rows_per_tile * tail],
                           o2[o2base : o2base + rows_per_tile * tail])

    return main


def mulcast(half=6144, rows_per_tile=1, cores=CORES):
    """out = rne(g * up) with g contiguous fp32 and up the high half of acc."""
    width = 2 * half

    @T.prim_func
    def main(oh: T.handle, gh: T.handle, ah: T.handle, rows: T.int64):
        out = T.match_buffer(oh, (rows * half,), "bfloat16")
        g = T.match_buffer(gh, (rows * half,), "float32")
        acc = T.match_buffer(ah, (rows * width,), "float32")
        tiles = rows // rows_per_tile
        with T.Kernel(cores) as core:
            g_ub = T.alloc_shared((rows_per_tile * half,), "float32")
            a_ub = T.alloc_shared((rows_per_tile * half,), "float32")
            out_ub = T.alloc_shared((rows_per_tile * half,), "bfloat16")
            T.annotate_buffer_versions({g_ub: 2, a_ub: 2, out_ub: 2})
            for step in T.Pipelined(T.ceildiv(tiles, cores), num_stages=2):
                tile = step * cores + core
                if tile < tiles:
                    gbase = T.cast(tile * rows_per_tile * half, "int32")
                    abase = T.cast(tile * rows_per_tile * width + rows_per_tile * half, "int32")
                    obase = T.cast(tile * rows_per_tile * half, "int32")
                    T.copy(g[gbase : gbase + rows_per_tile * half], g_ub)
                    T.copy(acc[abase : abase + rows_per_tile * half], a_ub)
                    with T.SimdVF():
                        m = T.simd.pset(32)
                        for i in range(rows_per_tile * half // LANES):
                            v = T.simd.vmul(T.simd.vld(g_ub[i * LANES]),
                                            T.simd.vld(a_ub[i * LANES]), m)
                            T.simd.vsts(out_ub[i * LANES],
                                        T.simd.vcvt(v, "bfloat16", round="ROUND_R"),
                                        dist="PK_B32")
                    T.copy(out_ub, out[obase : obase + rows_per_tile * half])

    return main


def vision_qkv_rope_packed(heads=16, dim=64, half=32, rows_per_tile=2, cores=CORES):
    """Apply vision QKV bias and RoPE, storing contiguous [3,rows,heads,dim] output."""
    n = 3 * heads * dim
    sec = heads * dim
    span = rows_per_tile * n
    cspan = rows_per_tile * 2 * half
    GUARD = 32

    @T.prim_func
    def main(oh: T.handle, ah: T.handle, bh: T.handle, ch: T.handle, sh: T.handle,
             rows: T.int64):
        out = T.match_buffer(oh, (rows * n,), "bfloat16")
        acc = T.match_buffer(ah, (rows * n,), "float32")
        bias = T.match_buffer(bh, (n,), "bfloat16")
        cos = T.match_buffer(ch, (rows * 2 * half,), "float32")
        sin = T.match_buffer(sh, (rows * 2 * half,), "float32")
        tiles = rows // rows_per_tile
        with T.Kernel(cores) as core:
            acc_ub = T.alloc_shared((span + 4 * GUARD,), "float32")
            out_ub = T.alloc_shared((span,), "bfloat16")
            bias_ub = T.alloc_shared((n + 4 * GUARD,), "bfloat16")
            cos_ub = T.alloc_shared((cspan,), "float32")
            sin_ub = T.alloc_shared((cspan,), "float32")
            T.annotate_buffer_versions({acc_ub: 2, out_ub: 2, cos_ub: 2, sin_ub: 2})
            T.copy(bias, bias_ub[GUARD : GUARD + n])
            for step in T.Pipelined(T.ceildiv(tiles, cores), num_stages=2):
                tile = step * cores + core
                if tile < tiles:
                    base = T.cast(tile * span, "int32")
                    cbase = T.cast(tile * cspan, "int32")
                    row0 = tile * rows_per_tile
                    T.copy(acc[base : base + span], acc_ub[GUARD : GUARD + span])
                    T.copy(cos[cbase : cbase + cspan], cos_ub)
                    T.copy(sin[cbase : cbase + cspan], sin_ub)
                    with T.SimdVF():
                        m = T.simd.pset(32)
                        v = T.simd.vci(0, "float32")
                        lo = T.simd.vcmps(v, 32.0, m, op="lt")
                        hi = T.simd.vcmps(v, 32.0, m, op="ge")
                        for r in T.serial(rows_per_tile):
                            row_off = r * n
                            croff = r * 2 * half
                            c2 = T.simd.vld(cos_ub[croff])
                            s2 = T.simd.vld(sin_ub[croff])
                            for s in T.serial(2):
                                for h in T.serial(heads):
                                    off = row_off + s * sec + h * dim
                                    boff = GUARD + s * sec + h * dim
                                    goff = GUARD + off
                                    u = T.simd.vadd(
                                        T.simd.vld(acc_ub[goff]),
                                        T.simd.vcvt(T.simd.vld(bias_ub[boff], dist="UNPK_B16"),
                                                    "float32", part=0),
                                        m,
                                    )
                                    w = T.simd.vadd(
                                        T.simd.vld(acc_ub[goff - half]),
                                        T.simd.vcvt(T.simd.vld(bias_ub[boff - half], dist="UNPK_B16"),
                                                    "float32", part=0),
                                        m,
                                    )
                                    z = T.simd.vadd(
                                        T.simd.vld(acc_ub[goff + half]),
                                        T.simd.vcvt(T.simd.vld(bias_ub[boff + half], dist="UNPK_B16"),
                                                    "float32", part=0),
                                        m,
                                    )
                                    t1 = T.simd.vmul(u, c2, m)
                                    res = T.simd.vsub(
                                        T.simd.vadd(t1, T.simd.vmul(w, s2, hi), m),
                                        T.simd.vmul(z, s2, lo),
                                        m,
                                    )
                                    T.simd.vsts(
                                        out_ub[off],
                                        T.simd.vcvt(res, "bfloat16", round="ROUND_R"),
                                        dist="PK_B32",
                                    )
                            for h in T.serial(heads):
                                off = row_off + 2 * sec + h * dim
                                boff = GUARD + 2 * sec + h * dim
                                goff = GUARD + off
                                value = T.simd.vadd(
                                    T.simd.vld(acc_ub[goff]),
                                    T.simd.vcvt(T.simd.vld(bias_ub[boff], dist="UNPK_B16"),
                                                "float32", part=0),
                                    m,
                                )
                                T.simd.vsts(
                                    out_ub[off],
                                    T.simd.vcvt(value, "bfloat16", round="ROUND_R"),
                                    dist="PK_B32",
                                )
                    # packed store: one contiguous sec-element chunk per
                    # (row, section), destination [section, row, head, dim]
                    for r in T.serial(rows_per_tile):
                        for s in T.serial(3):
                            src0 = r * n + s * sec
                            dst0 = T.cast(s * rows * sec + (row0 + r) * sec, "int32")
                            T.copy(out_ub[src0 : src0 + sec], out[dst0 : dst0 + sec])

    return main


def build_programs():
    yield "normprep_c2048", normprep_c2048()
    yield "apply_rstd_c2048", apply_rstd_c2048()
    yield "zbasplit_c8224", zbasplit()
    yield "mulcast_c6144", mulcast()
    yield "vision_qkv_rope_packed", vision_qkv_rope_packed()
