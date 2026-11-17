"""soupkill-r1: single-pass BF16 finishes over the FP32 ACLNN accumulator.

Each kernel replaces a multi-kernel torch tail with one pass (UB-FINISH r5
recipe: T.copy DMA staging + CCE SIMD vector math + fixed 64-core grid; the
vector unit stays off the scalar SimtVF path).

* ``finish_br_c{cols}``: ``out = rne(acc + widen(bias) + widen(res))``. The
  two fp32 adds happen in the same order and rounding as ``_finish``'s torch
  chain (bias first, then residual), with one final RNE.
* ``vision_qkv_rope``: ``out = rne(rope_half(acc + widen(bias)))`` for the 2B
  ViT QKV layout ``[rows, 3, 16, 64]`` with ``rotary_dim = head_dim = 64``.
  Per head ``[a*c - b*s | b*c + a*s]`` uses separate vmul/vsub/vadd (no FMA
  contraction, matching the torch chain's per-op fp32 rounding); the V
  section is copied through with the same single RNE.
* ``glu_split_c{half}``: ``out = rne(silu(acc[:, :half]) * acc[:, half:])``.
"""

import tilelang.ascend.language as T

CORES = 64
LANES = 64  # FP32 lanes per CCE vector register


def finish_bias_res(cols, rows_per_tile, has_bias=True, has_res=True, cores=CORES):
    """Default rows_per_tile keeps the staged UB tile inside the 248 KiB budget."""
    """out[m,cols] = rne(acc + bias + res) with fp32 adds in that order."""
    span = rows_per_tile * cols

    @T.prim_func
    def main(oh: T.handle, ah: T.handle, bh: T.handle, rh: T.handle, rows: T.int64):
        out = T.match_buffer(oh, (rows * cols,), "bfloat16")
        acc = T.match_buffer(ah, (rows * cols,), "float32")
        bias = T.match_buffer(bh, (cols,), "bfloat16")
        res = T.match_buffer(rh, (rows * cols,), "bfloat16")
        tiles = rows // rows_per_tile
        with T.Kernel(cores) as core:
            acc_ub = T.alloc_shared((span,), "float32")
            out_ub = T.alloc_shared((span,), "bfloat16")
            res_ub = T.alloc_shared((span,), "bfloat16")
            bias_ub = T.alloc_shared((cols,), "bfloat16")
            T.annotate_buffer_versions({acc_ub: 3, out_ub: 3, res_ub: 3})
            if has_bias:
                T.copy(bias, bias_ub)
            for step in T.Pipelined(T.ceildiv(tiles, cores), num_stages=3):
                tile = step * cores + core
                if tile < tiles:
                    base = T.cast(tile * span, "int32")
                    T.copy(acc[base : base + span], acc_ub)
                    if has_res:
                        T.copy(res[base : base + span], res_ub)
                    with T.SimdVF():
                        m = T.simd.pset(32)
                        for r in T.serial(rows_per_tile):
                            row_off = r * cols
                            for c in range(cols // LANES):
                                value = T.simd.vld(acc_ub[row_off + c * LANES])
                                if has_bias:
                                    value = T.simd.vadd(
                                        value,
                                        T.simd.vcvt(
                                            T.simd.vld(bias_ub[c * LANES], dist="UNPK_B16"),
                                            "float32",
                                            part=0,
                                        ),
                                        m,
                                    )
                                if has_res:
                                    value = T.simd.vadd(
                                        value,
                                        T.simd.vcvt(
                                            T.simd.vld(res_ub[row_off + c * LANES], dist="UNPK_B16"),
                                            "float32",
                                            part=0,
                                        ),
                                        m,
                                    )
                                T.simd.vsts(
                                    out_ub[row_off + c * LANES],
                                    T.simd.vcvt(value, "bfloat16", round="ROUND_R"),
                                    dist="PK_B32",
                                )
                    T.copy(out_ub, out[base : base + span])

    return main


def vision_qkv_rope(heads=16, dim=64, half=32, rows_per_tile=2, cores=CORES):
    """out = rne(rope_half(acc + bias)) for [rows, 3*heads*dim], rotary = dim."""
    n = 3 * heads * dim
    sec = heads * dim
    span = rows_per_tile * n
    cos_span = rows_per_tile * half

    @T.prim_func
    def main(
        oh: T.handle,
        ah: T.handle,
        bh: T.handle,
        ch: T.handle,
        sh: T.handle,
        rows: T.int64,
    ):
        out = T.match_buffer(oh, (rows * n,), "bfloat16")
        acc = T.match_buffer(ah, (rows * n,), "float32")
        bias = T.match_buffer(bh, (n,), "bfloat16")
        cos = T.match_buffer(ch, (rows * half,), "float32")
        sin = T.match_buffer(sh, (rows * half,), "float32")
        tiles = rows // rows_per_tile
        with T.Kernel(cores) as core:
            acc_ub = T.alloc_shared((span,), "float32")
            out_ub = T.alloc_shared((span,), "bfloat16")
            bias_ub = T.alloc_shared((n,), "bfloat16")
            cos_ub = T.alloc_shared((cos_span,), "float32")
            sin_ub = T.alloc_shared((cos_span,), "float32")
            T.annotate_buffer_versions({acc_ub: 2, out_ub: 2, cos_ub: 2, sin_ub: 2})
            T.copy(bias, bias_ub)
            for step in T.Pipelined(T.ceildiv(tiles, cores), num_stages=2):
                tile = step * cores + core
                if tile < tiles:
                    base = T.cast(tile * span, "int32")
                    cbase = T.cast(tile * cos_span, "int32")
                    T.copy(acc[base : base + span], acc_ub)
                    T.copy(cos[cbase : cbase + cos_span], cos_ub)
                    T.copy(sin[cbase : cbase + cos_span], sin_ub)
                    with T.SimdVF():
                        m = T.simd.pset(32)
                        for r in T.serial(rows_per_tile):
                            row_off = r * n
                            croff = r * half
                            for s in T.serial(2):
                                for h in T.serial(heads):
                                    off = row_off + s * sec + h * dim
                                    a = T.simd.vadd(
                                        T.simd.vld(acc_ub[off]),
                                        T.simd.vcvt(
                                            T.simd.vld(bias_ub[s * sec + h * dim], dist="UNPK_B16"),
                                            "float32",
                                            part=0,
                                        ),
                                        m,
                                    )
                                    b = T.simd.vadd(
                                        T.simd.vld(acc_ub[off + half]),
                                        T.simd.vcvt(
                                            T.simd.vld(bias_ub[s * sec + h * dim + half], dist="UNPK_B16"),
                                            "float32",
                                            part=0,
                                        ),
                                        m,
                                    )
                                    cv = T.simd.vld(cos_ub[croff])
                                    sv = T.simd.vld(sin_ub[croff])
                                    o0 = T.simd.vsub(T.simd.vmul(a, cv, m), T.simd.vmul(b, sv, m), m)
                                    o1 = T.simd.vadd(T.simd.vmul(b, cv, m), T.simd.vmul(a, sv, m), m)
                                    T.simd.vsts(
                                        out_ub[off],
                                        T.simd.vcvt(o0, "bfloat16", round="ROUND_R"),
                                        mask=m,
                                        dist="PK_B32",
                                    )
                                    T.simd.vsts(
                                        out_ub[off + half],
                                        T.simd.vcvt(o1, "bfloat16", round="ROUND_R"),
                                        mask=m,
                                        dist="PK_B32",
                                    )
                            for h in T.serial(heads):
                                off = row_off + 2 * sec + h * dim
                                value = T.simd.vadd(
                                    T.simd.vld(acc_ub[off]),
                                    T.simd.vcvt(
                                        T.simd.vld(bias_ub[2 * sec + h * dim], dist="UNPK_B16"),
                                        "float32",
                                        part=0,
                                    ),
                                    m,
                                )
                                T.simd.vsts(
                                    out_ub[off],
                                    T.simd.vcvt(value, "bfloat16", round="ROUND_R"),
                                    dist="PK_B32",
                                )
                    T.copy(out_ub, out[base : base + span])

    return main


def glu_split(half, rows_per_tile=1, cores=CORES):
    """out[m,half] = rne(silu(acc[m,:half]) * acc[m,half:])."""
    span = rows_per_tile * 2 * half
    ospan = rows_per_tile * half

    @T.prim_func
    def main(oh: T.handle, ah: T.handle, rows: T.int64):
        out = T.match_buffer(oh, (rows * half,), "bfloat16")
        acc = T.match_buffer(ah, (rows * 2 * half,), "float32")
        tiles = rows // rows_per_tile
        with T.Kernel(cores) as core:
            acc_ub = T.alloc_shared((span,), "float32")
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
                                g = T.simd.vld(acc_ub[r * 2 * half + c * LANES])
                                u = T.simd.vld(acc_ub[r * 2 * half + half + c * LANES])
                                one = T.simd.vdup(1.0, "float32", m)
                                sig = T.simd.vdiv(
                                    one, T.simd.vadd(one, T.simd.vexp(T.simd.vneg(g, m), m), m), m
                                )
                                value = T.simd.vmul(T.simd.vmul(g, sig, m), u, m)
                                T.simd.vsts(
                                    out_ub[r * half + c * LANES],
                                    T.simd.vcvt(value, "bfloat16", round="ROUND_R"),
                                    dist="PK_B32",
                                )
                    T.copy(out_ub, out[obase : obase + ospan])

    return main


def build_programs():
    yield "finish_br_c1024", finish_bias_res(1024, 4)
    yield "vision_qkv_rope", vision_qkv_rope()
    yield "glu_split_c6144", glu_split(6144, 1)
