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

"""BF16 projection epilogues for SwiGLU, row-scaled QKV/ZBA splitting, vision
QKV/RoPE, bias plus residual, and GELU. Inputs have already been rounded to
BF16 by the preceding matmul. Each kernel widens its inputs to FP32 for
the epilogue and rounds the result to BF16 on store."""

import tilelang.ascend.language as T

CORES = 64
LANES = 64  # FP32 lanes per CCE vector register


def glu_split_bf16(half=6144, rows_per_tile=1, cores=CORES):
    """out[m,half] = rne(silu(f32(g_bf16)) * f32(u_bf16)) over the BF16 store."""
    span = rows_per_tile * 2 * half
    ospan = rows_per_tile * half

    @T.prim_func
    def main(oh: T.handle, ah: T.handle, rows: T.int64):
        out = T.match_buffer(oh, (rows * half,), "bfloat16")
        acc = T.match_buffer(ah, (rows * 2 * half,), "bfloat16")
        tiles = rows // rows_per_tile
        with T.Kernel(cores) as core:
            acc_ub = T.alloc_shared((span,), "bfloat16")
            out_ub = T.alloc_shared((ospan,), "bfloat16")
            T.annotate_buffer_versions({acc_ub: 3, out_ub: 3})
            for step in T.Pipelined(T.ceildiv(tiles, cores), num_stages=2):
                tile = step * cores + core
                if tile < tiles:
                    base = T.cast(tile * span, "int32")
                    obase = T.cast(tile * ospan, "int32")
                    T.copy(acc[base : base + span], acc_ub)
                    with T.SimdVF():
                        m = T.simd.pset(32)
                        for r in T.serial(rows_per_tile):
                            for c in range(half // LANES):
                                g = T.simd.vcvt(
                                    T.simd.vld(acc_ub[r * 2 * half + c * LANES],
                                               dist="UNPK_B16"),
                                    "float32", part=0)
                                u = T.simd.vcvt(
                                    T.simd.vld(acc_ub[r * 2 * half + half + c * LANES],
                                               dist="UNPK_B16"),
                                    "float32", part=0)
                                one = T.simd.vdup(1.0, "float32", m)
                                sig = T.simd.vdiv(
                                    one, T.simd.vadd(one, T.simd.vexp(T.simd.vneg(g, m), m), m), m)
                                value = T.simd.vmul(T.simd.vmul(g, sig, m), u, m)
                                T.simd.vsts(
                                    out_ub[r * half + c * LANES],
                                    T.simd.vcvt(value, "bfloat16", round="ROUND_R"),
                                    dist="PK_B32")
                    T.copy(out_ub, out[obase : obase + ospan])

    return main


def zbasplit_bf16(conv=6144, tail=2080, rows_per_tile=1, cores=CORES):
    """o1 = rne(f32(acc16[:, :conv]) * keep), o2 = rne(f32(acc16[:, conv:]) * keep)."""
    width = conv + tail
    span = rows_per_tile * width
    tail_pad = (tail + LANES - 1) // LANES * LANES

    @T.prim_func
    def main(o1h: T.handle, o2h: T.handle, ah: T.handle, kh: T.handle, rows: T.int64):
        o1 = T.match_buffer(o1h, (rows * conv,), "bfloat16")
        o2 = T.match_buffer(o2h, (rows * tail,), "bfloat16")
        acc = T.match_buffer(ah, (rows * width,), "bfloat16")
        keep = T.match_buffer(kh, (rows,), "float32")
        tiles = rows // rows_per_tile
        with T.Kernel(cores) as core:
            acc_ub = T.alloc_shared((span,), "bfloat16")
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
                                    T.simd.vcvt(
                                        T.simd.vld(acc_ub[ri * width + i * LANES],
                                                   dist="UNPK_B16"),
                                        "float32", part=0),
                                    keep_ub[ri], m)
                                T.simd.vsts(
                                    o1_ub[ri * conv + i * LANES],
                                    T.simd.vcvt(v, "bfloat16", round="ROUND_R"),
                                    dist="PK_B32")
                            for i in range(tail_pad // LANES):
                                v = T.simd.vmuls(
                                    T.simd.vcvt(
                                        T.simd.vld(acc_ub[ri * width + conv + i * LANES],
                                                   dist="UNPK_B16"),
                                        "float32", part=0),
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


def vision_qkv_rope_bf16_packed(heads=16, dim=64, half=32, rows_per_tile=2, cores=CORES):
    """Apply Q/K rotate-half RoPE after BF16 projection and pack Q/K/V as [3,rows,heads,dim]."""
    n = 3 * heads * dim
    sec = heads * dim
    span = rows_per_tile * n
    cspan = rows_per_tile * 2 * half
    GUARD = 32

    @T.prim_func
    def main(oh: T.handle, ah: T.handle, bh: T.handle, ch: T.handle, sh: T.handle,
             rows: T.int64):
        out = T.match_buffer(oh, (rows * n,), "bfloat16")
        acc = T.match_buffer(ah, (rows * n,), "bfloat16")
        bias = T.match_buffer(bh, (n,), "bfloat16")
        cos = T.match_buffer(ch, (rows * 2 * half,), "float32")
        sin = T.match_buffer(sh, (rows * 2 * half,), "float32")
        tiles = rows // rows_per_tile
        with T.Kernel(cores) as core:
            acc_ub = T.alloc_shared((span + 4 * GUARD,), "bfloat16")
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
                                        T.simd.vcvt(
                                            T.simd.vld(acc_ub[goff], dist="UNPK_B16"),
                                            "float32", part=0),
                                        T.simd.vcvt(
                                            T.simd.vld(bias_ub[boff], dist="UNPK_B16"),
                                            "float32", part=0),
                                        m,
                                    )
                                    w = T.simd.vadd(
                                        T.simd.vcvt(
                                            T.simd.vld(acc_ub[goff - half], dist="UNPK_B16"),
                                            "float32", part=0),
                                        T.simd.vcvt(
                                            T.simd.vld(bias_ub[boff - half], dist="UNPK_B16"),
                                            "float32", part=0),
                                        m,
                                    )
                                    z = T.simd.vadd(
                                        T.simd.vcvt(
                                            T.simd.vld(acc_ub[goff + half], dist="UNPK_B16"),
                                            "float32", part=0),
                                        T.simd.vcvt(
                                            T.simd.vld(bias_ub[boff + half], dist="UNPK_B16"),
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
                                    T.simd.vcvt(
                                        T.simd.vld(acc_ub[goff], dist="UNPK_B16"),
                                        "float32", part=0),
                                    T.simd.vcvt(
                                        T.simd.vld(bias_ub[boff], dist="UNPK_B16"),
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


def finish_br_bf16(cols=1024, rows_per_tile=4, cores=CORES):
    """Widen the BF16 operands, add bias before residual and round the sum once to BF16."""
    span = rows_per_tile * cols

    @T.prim_func
    def main(oh: T.handle, ah: T.handle, bh: T.handle, rh: T.handle, rows: T.int64):
        out = T.match_buffer(oh, (rows * cols,), "bfloat16")
        acc = T.match_buffer(ah, (rows * cols,), "bfloat16")
        bias = T.match_buffer(bh, (cols,), "bfloat16")
        res = T.match_buffer(rh, (rows * cols,), "bfloat16")
        tiles = rows // rows_per_tile
        with T.Kernel(cores) as core:
            acc_ub = T.alloc_shared((span,), "bfloat16")
            out_ub = T.alloc_shared((span,), "bfloat16")
            res_ub = T.alloc_shared((span,), "bfloat16")
            bias_ub = T.alloc_shared((cols,), "bfloat16")
            T.annotate_buffer_versions({acc_ub: 3, out_ub: 3, res_ub: 3})
            T.copy(bias, bias_ub)
            for step in T.Pipelined(T.ceildiv(tiles, cores), num_stages=3):
                tile = step * cores + core
                if tile < tiles:
                    base = T.cast(tile * span, "int32")
                    T.copy(acc[base : base + span], acc_ub)
                    T.copy(res[base : base + span], res_ub)
                    with T.SimdVF():
                        m = T.simd.pset(32)
                        for r in T.serial(rows_per_tile):
                            row_off = r * cols
                            for c in range(cols // LANES):
                                value = T.simd.vcvt(
                                    T.simd.vld(acc_ub[row_off + c * LANES], dist="UNPK_B16"),
                                    "float32", part=0)
                                value = T.simd.vadd(
                                    value,
                                    T.simd.vcvt(
                                        T.simd.vld(bias_ub[c * LANES], dist="UNPK_B16"),
                                        "float32", part=0),
                                    m,
                                )
                                value = T.simd.vadd(
                                    value,
                                    T.simd.vcvt(
                                        T.simd.vld(res_ub[row_off + c * LANES], dist="UNPK_B16"),
                                        "float32", part=0),
                                    m,
                                )
                                T.simd.vsts(
                                    out_ub[row_off + c * LANES],
                                    T.simd.vcvt(value, "bfloat16", round="ROUND_R"),
                                    dist="PK_B32",
                                )
                    T.copy(out_ub, out[base : base + span])

    return main


def gelu_bf16(cols=4096, rows_per_tile=2, cores=CORES):
    """out = rne(gelu_tanh(f32(acc16) + f32(bias))); div form, no clamp, no FMA."""
    K = 0.7978845608028654
    KA = 0.044715 * 0.7978845608028654
    span = rows_per_tile * cols

    @T.prim_func
    def main(oh: T.handle, ah: T.handle, bh: T.handle, rows: T.int64):
        out = T.match_buffer(oh, (rows * cols,), "bfloat16")
        acc = T.match_buffer(ah, (rows * cols,), "bfloat16")
        bias = T.match_buffer(bh, (cols,), "bfloat16")
        tiles = rows // rows_per_tile
        with T.Kernel(cores) as core:
            acc_ub = T.alloc_shared((span,), "bfloat16")
            out_ub = T.alloc_shared((span,), "bfloat16")
            bias_ub = T.alloc_shared((cols,), "bfloat16")
            T.annotate_buffer_versions({acc_ub: 3, out_ub: 3})
            T.copy(bias, bias_ub)
            for step in T.Pipelined(T.ceildiv(tiles, cores), num_stages=3):
                tile = step * cores + core
                if tile < tiles:
                    base = T.cast(tile * span, "int32")
                    T.copy(acc[base : base + span], acc_ub)
                    with T.SimdVF():
                        m = T.simd.pset(32)
                        ck = T.simd.vdup(K, "float32", m)
                        cka = T.simd.vdup(KA, "float32", m)
                        cn2 = T.simd.vdup(-2.0, "float32", m)
                        c1 = T.simd.vdup(1.0, "float32", m)
                        for r in T.serial(rows_per_tile):
                            row_off = r * cols
                            for c in range(cols // LANES):
                                v = T.simd.vadd(
                                    T.simd.vcvt(
                                        T.simd.vld(acc_ub[row_off + c * LANES],
                                                   dist="UNPK_B16"),
                                        "float32", part=0),
                                    T.simd.vcvt(
                                        T.simd.vld(bias_ub[c * LANES], dist="UNPK_B16"),
                                        "float32", part=0),
                                    m)
                                cube = T.simd.vmul(T.simd.vmul(v, v, m), v, m)
                                z = T.simd.vadd(T.simd.vmul(v, ck, m),
                                                T.simd.vmul(cube, cka, m), m)
                                e = T.simd.vexp(T.simd.vmul(z, cn2, m), m,
                                                precision="ftz_false")
                                d = T.simd.vadd(c1, e, m)
                                value = T.simd.vdiv(v, d, m)
                                T.simd.vsts(out_ub[row_off + c * LANES],
                                            T.simd.vcvt(value, "bfloat16", round="ROUND_R"),
                                            dist="PK_B32")
                    T.copy(out_ub, out[base : base + span])

    return main


def build_programs():
    yield "glu_split_bf16_c6144", glu_split_bf16()
    yield "glu_split_bf16_rows2", glu_split_bf16(rows_per_tile=2)
    yield "zbasplit_bf16_c8224", zbasplit_bf16()
    yield "zbasplit_bf16_rows2", zbasplit_bf16(rows_per_tile=2)
    yield "vision_qkv_rope_bf16_packed", vision_qkv_rope_bf16_packed()
    yield "finish_br_bf16_c1024", finish_br_bf16()
    yield "gelu_bf16_c4096", gelu_bf16()
    yield "gelu_bf16_rows4", gelu_bf16(rows_per_tile=4)
