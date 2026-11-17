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
- fc1_gelu_c4096: vision fc1 finish tail. bias-add (bitwise, same fp32 vadd)
               + tanh-form gelu + single bf16 RNE, replacing aclnnAdd +
               aclnnGeluV2 + aclnnCast. NOT bitwise vs aclnnGeluV2 (vendor
               transcendental); adjudicated by the action gate, env-gated.

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


def fc1_gelu_c4096(cols=4096, cores=CORES):
    """out = rne_bf16(0.5*t*(1+tanh(c*(t+0.044715*t^3)))), t = acc + bias.

    tanh is composed as (e^{2u}-1)/(e^{2u}+1) with u clamped to +-10
    (tanh(10) rounds to 1.0f; clamping keeps vexp finite for any t).
    bias rides a hoisted UB row. Not bitwise vs aclnnGeluV2 (gate-adjudicated).
    """
    span = cols

    @T.prim_func
    def main(oh: T.handle, ah: T.handle, bh: T.handle, rows: T.int64):
        out = T.match_buffer(oh, (rows * cols,), "bfloat16")
        acc = T.match_buffer(ah, (rows * cols,), "float32")
        bias = T.match_buffer(bh, (cols,), "float32")
        with T.Kernel(cores) as core:
            acc_ub = T.alloc_shared((span,), "float32")
            bias_ub = T.alloc_shared((span,), "float32")
            out_ub = T.alloc_shared((span,), "bfloat16")
            T.annotate_buffer_versions({acc_ub: 2, out_ub: 2, bias_ub: 2})
            T.copy(bias, bias_ub)
            for step in T.Pipelined(T.ceildiv(rows, cores), num_stages=2):
                row = step * cores + core
                if row < rows:
                    base = T.cast(row * cols, "int32")
                    T.copy(acc[base : base + span], acc_ub)
                    with T.SimdVF():
                        m = T.simd.pset(32)
                        c0 = 0.7978845608028654
                        c1 = 0.044715
                        one = T.simd.vdup(1.0, "float32", m)
                        for i in range(span // LANES):
                            t = T.simd.vadd(
                                T.simd.vld(acc_ub[i * LANES]),
                                T.simd.vld(bias_ub[i * LANES]), m)
                            k = T.simd.vmuls(t, c1, m)
                            k = T.simd.vmul(k, t, m)
                            k = T.simd.vmul(k, t, m)
                            v = T.simd.vadd(t, k, m)
                            u = T.simd.vmuls(v, c0, m)
                            u = T.simd.vmins(T.simd.vmaxs(u, -10.0, m), 10.0, m)
                            e2u = T.simd.vexp(T.simd.vmuls(u, 2.0, m), m,
                                              precision="vexp_1ulp_ftz_false")
                            num = T.simd.vsub(e2u, one, m)
                            den = T.simd.vadd(e2u, one, m)
                            th = T.simd.vdiv(num, den, m, precision="exact")
                            half = T.simd.vadd(one, th, m)
                            g = T.simd.vmul(T.simd.vmuls(t, 0.5, m), half, m)
                            T.simd.vsts(
                                out_ub[i * LANES],
                                T.simd.vcvt(g, "bfloat16", round="ROUND_R"),
                                dist="PK_B32")
                    T.copy(out_ub, out[base : base + span])

    return main




def probe_vdiv(cores=CORES):
    cols = 4096

    @T.prim_func
    def main(oh: T.handle, ah: T.handle, bh: T.handle, rows: T.int64):
        out = T.match_buffer(oh, (rows * cols,), "float32")
        a = T.match_buffer(ah, (rows * cols,), "float32")
        b = T.match_buffer(bh, (rows * cols,), "float32")
        with T.Kernel(cores) as core:
            a_ub = T.alloc_shared((cols,), "float32")
            b_ub = T.alloc_shared((cols,), "float32")
            o_ub = T.alloc_shared((cols,), "float32")
            T.annotate_buffer_versions({a_ub: 2, b_ub: 2, o_ub: 2})
            for step in T.Pipelined(T.ceildiv(rows, cores), num_stages=2):
                row = step * cores + core
                if row < rows:
                    base = T.cast(row * cols, "int32")
                    T.copy(a[base : base + cols], a_ub)
                    T.copy(b[base : base + cols], b_ub)
                    with T.SimdVF():
                        m = T.simd.pset(32)
                        for i in range(cols // LANES):
                            va = T.simd.vld(a_ub[i * LANES])
                            vb = T.simd.vld(b_ub[i * LANES])
                            T.simd.vsts(o_ub[i * LANES], T.simd.vdiv(vb, va, m))
                    T.copy(o_ub, out[base : base + cols])

    return main




def _gelu_micro(sfu, cores=CORES):
    """sfu: 0 = alu only, 1 = +vexp, 2 = +vexp+vdiv-exact, 3 = +vexp+newton-recip."""
    cols = 4096

    @T.prim_func
    def main(oh: T.handle, ah: T.handle, bh: T.handle, rows: T.int64):
        out = T.match_buffer(oh, (rows * cols,), "bfloat16")
        acc = T.match_buffer(ah, (rows * cols,), "float32")
        bias = T.match_buffer(bh, (cols,), "float32")
        with T.Kernel(cores) as core:
            acc_ub = T.alloc_shared((cols,), "float32")
            bias_ub = T.alloc_shared((cols,), "float32")
            out_ub = T.alloc_shared((cols,), "bfloat16")
            T.annotate_buffer_versions({acc_ub: 2, out_ub: 2, bias_ub: 2})
            T.copy(bias, bias_ub)
            for step in T.Pipelined(T.ceildiv(rows, cores), num_stages=2):
                row = step * cores + core
                if row < rows:
                    base = T.cast(row * cols, "int32")
                    T.copy(acc[base : base + cols], acc_ub)
                    with T.SimdVF():
                        m = T.simd.pset(32)
                        c0 = 0.7978845608028654
                        c1 = 0.044715
                        one = T.simd.vdup(1.0, "float32", m)
                        for i in range(cols // LANES):
                            t = T.simd.vadd(T.simd.vld(acc_ub[i * LANES]),
                                            T.simd.vld(bias_ub[i * LANES]), m)
                            k = T.simd.vmuls(t, c1, m)
                            k = T.simd.vmul(k, t, m)
                            k = T.simd.vmul(k, t, m)
                            v = T.simd.vadd(t, k, m)
                            u = T.simd.vmuls(v, c0, m)
                            g = t
                            if sfu >= 1:
                                w = T.simd.vabs(T.simd.vmuls(u, 2.0, m), m)
                                q = T.simd.vexp(T.simd.vmuls(w, -1.0, m), m,
                                                precision="vexp_1ulp_ftz_false")
                                d = T.simd.vadd(q, one, m)
                                if sfu == 2:
                                    th = T.simd.vdiv(one, d, m, precision="exact")
                                    g = T.simd.vmul(t, th, m)
                                elif sfu == 3:
                                    r = T.simd.vsub(T.simd.vmuls(d, 1.5, m), T.simd.vmuls(d, 0.25, m), m)
                                    r = T.simd.vmul(r, r, m)
                                    r = T.simd.vmul(T.simd.vsub(T.simd.vmuls(one, 2.0, m), T.simd.vmul(d, r, m), m), r, m)
                                    r = T.simd.vmul(T.simd.vsub(T.simd.vmuls(one, 2.0, m), T.simd.vmul(d, r, m), m), r, m)
                                    r = T.simd.vmul(T.simd.vsub(T.simd.vmuls(one, 2.0, m), T.simd.vmul(d, r, m), m), r, m)
                                    r = T.simd.vmul(T.simd.vsub(T.simd.vmuls(one, 2.0, m), T.simd.vmul(d, r, m), m), r, m)
                                    g = T.simd.vmul(t, r, m)
                                else:
                                    g = T.simd.vmul(t, q, m)
                            T.simd.vsts(out_ub[i * LANES],
                                        T.simd.vcvt(g, "bfloat16", round="ROUND_R"),
                                        dist="PK_B32")
                    T.copy(out_ub, out[base : base + cols])

    return main


def gelu_micro0():
    return _gelu_micro(0)


def gelu_micro1():
    return _gelu_micro(1)


def gelu_micro2():
    return _gelu_micro(2)


def gelu_micro3():
    return _gelu_micro(3)




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
    yield "fc1_gelu_c4096", fc1_gelu_c4096()
    yield "probe_vdiv", probe_vdiv()
    yield "gelu_micro0", gelu_micro0()
    yield "gelu_micro1", gelu_micro1()
    yield "gelu_micro2", gelu_micro2()
    yield "gelu_micro3", gelu_micro3()
    yield "quant_u8", quant_u8()
