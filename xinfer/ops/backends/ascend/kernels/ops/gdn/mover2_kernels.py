"""MOVE-R2 kernels (final): elementwise pack/tail fuses, dconv4-class TileLang.

Engine history (documented negatives):
- TileLang SimtVF-scalar (v1): correct, 461/254us (strided global scalar loads).
- TileLang fragment+reducer (v2): correct, 230/107us (codegen overhead).
- AscendC ccec (v3): in-kernel fp32 Exp/Div kills the vector pipe; the
  dav-3510 multi-block vector-op hazard (ascendc-action r1: blocks 1+ produce
  garbage on Cast/Mul/Add; single-block only) rules out per-token AIV blocks.
Final architecture: the REDUCTIONS (qk square sums / raw mean+rsqrt) and the
transcendentals (silu, the beta/g gates) stay on aclnn (bitwise-identical
inputs); these kernels fuse the pure elementwise remainders. Only mul/cast
on aclnn-identical inputs => bitwise-expected vs the released chains.

gdrapply_bf16: q/k TND packs + v copy from the SiLU'd token-major x_t and
  the aclnn rstd [T, 32] fp32:
    q[t,h,d]  = RNE_bf16(f32(x[t, h*128+d])      * rstd[t, h])
    k[t,h,d]  = RNE_bf16(f32(x[t, 2048+h*128+d]) * rstd[t, 16+h])
    v[t,h,d]  = x[t, 4096+h*128+d]
  replaces Slice + the n Mul_Cast + 2x mul-by-one packs + AsStrided(v).

gdrtailmix_bf16: the raw-side elementwise tail from the aclnn rstd [T, 16]
  fp32, gate fp32 (the silu chain output) and w bf16:
    nbf = RNE_bf16(f32(raw) * rstd); wn = RNE_bf16(f32(w) * f32(nbf));
    out = RNE_bf16(f32(wn) * gate)
  replaces raw-cast + Mul + Cast + w-Mul + gate-Mul + Cast (6 aclnn ops).

Idioms copied verbatim from soup-form (apply_rstd_c2048 vmuls/vcvt/vsts)
and conv-gemm (dconv4 row-wise guarded tile copies).
"""

import tilelang.ascend.language as T

CORES = 64
LANES = 64
ROWS = 8           # tokens per tile (T in {416..464} all divisible by 8)
COLS = 1024        # column block = 8 heads x 128


def gdrapply_bf16(cols=6144, rows_per_tile=ROWS, cores=CORES):
    NCB = cols // COLS  # 6 col blocks

    @T.prim_func
    def main(
        xh: T.handle,
        rh: T.handle,
        qh: T.handle,
        kh: T.handle,
        vh: T.handle,
        tokens: T.int64,
    ):
        x = T.match_buffer(xh, (tokens * 6144,), "bfloat16")
        rstd = T.match_buffer(rh, (tokens * 32,), "float32")
        q = T.match_buffer(qh, (tokens * 2048,), "bfloat16")
        k = T.match_buffer(kh, (tokens * 2048,), "bfloat16")
        v = T.match_buffer(vh, (tokens * 2048,), "bfloat16")
        row_tiles = T.ceildiv(tokens, rows_per_tile)
        tiles = row_tiles * NCB
        in_span = rows_per_tile * COLS
        with T.Kernel(cores) as core:
            in_ub = T.alloc_shared((in_span,), "bfloat16")
            r_ub = T.alloc_shared((rows_per_tile * 32,), "float32")
            out_ub = T.alloc_shared((in_span,), "bfloat16")
            T.annotate_buffer_versions({in_ub: 2, r_ub: 2, out_ub: 2})
            for step in T.Pipelined(T.ceildiv(tiles, cores), num_stages=2):
                tile = step * cores + core
                if tile < tiles:
                    rt = tile // NCB
                    cb = tile % NCB
                    base = rt * rows_per_tile
                    for r in T.serial(rows_per_tile):
                        if base + r < tokens:
                            src = T.cast((base + r) * 6144 + cb * COLS, "int32")
                            T.copy(x[src : src + COLS],
                                   in_ub[r * COLS : (r + 1) * COLS])
                            rsrc = T.cast((base + r) * 32, "int32")
                            T.copy(rstd[rsrc : rsrc + 32],
                                   r_ub[r * 32 : (r + 1) * 32])
                    if cb < 4:
                        with T.SimdVF():
                            m = T.simd.pset(32)
                            for r in T.serial(rows_per_tile):
                                if base + r < tokens:
                                    for h in T.serial(8):
                                        for c in range(2):
                                            off = r * COLS + h * 128 + c * LANES
                                            xv = T.simd.vcvt(
                                                T.simd.vld(in_ub[off],
                                                           dist="UNPK_B16"),
                                                "float32", part=0)
                                            sv = T.simd.vmuls(
                                                xv,
                                                r_ub[r * 32 + cb * 8 + h], m)
                                            T.simd.vsts(
                                                out_ub[off],
                                                T.simd.vcvt(
                                                    sv, "bfloat16",
                                                    round="ROUND_R"),
                                                dist="PK_B32")
                    for r in T.serial(rows_per_tile):
                        if base + r < tokens:
                            dst = T.cast((base + r) * 2048 + (cb % 2) * COLS,
                                         "int32")
                            seg = out_ub[r * COLS : (r + 1) * COLS]
                            if cb < 4:
                                if cb < 2:
                                    T.copy(seg, q[dst : dst + COLS])
                                else:
                                    T.copy(seg, k[dst : dst + COLS])
                            else:
                                T.copy(in_ub[r * COLS : (r + 1) * COLS],
                                       v[dst : dst + COLS])

    return main


def gdrtailmix_bf16(width=2048, rows_per_tile=ROWS, cores=CORES):
    NCB = width // COLS  # 2 col blocks

    @T.prim_func
    def main(
        outh: T.handle,
        rawh: T.handle,
        rh: T.handle,
        gh: T.handle,
        wh: T.handle,
        tokens: T.int64,
    ):
        out = T.match_buffer(outh, (tokens * 2048,), "bfloat16")
        raw = T.match_buffer(rawh, (tokens * 2048,), "bfloat16")
        rstd = T.match_buffer(rh, (tokens * 16,), "float32")
        gate = T.match_buffer(gh, (tokens * 2048,), "float32")
        w = T.match_buffer(wh, (128,), "bfloat16")
        row_tiles = T.ceildiv(tokens, rows_per_tile)
        tiles = row_tiles * NCB
        in_span = rows_per_tile * COLS
        with T.Kernel(cores) as core:
            raw_ub = T.alloc_shared((in_span,), "bfloat16")
            gate_ub = T.alloc_shared((in_span,), "float32")
            nbf_ub = T.alloc_shared((in_span,), "bfloat16")
            wn_ub = T.alloc_shared((in_span,), "bfloat16")
            out_ub = T.alloc_shared((in_span,), "bfloat16")
            w_ub = T.alloc_shared((128,), "bfloat16")
            r_ub = T.alloc_shared((rows_per_tile * 16,), "float32")
            T.annotate_buffer_versions(
                {raw_ub: 2, gate_ub: 2, nbf_ub: 2, wn_ub: 2, out_ub: 2,
                 w_ub: 2, r_ub: 2})
            T.copy(w[:128], w_ub)
            for step in T.Pipelined(T.ceildiv(tiles, cores), num_stages=2):
                tile = step * cores + core
                if tile < tiles:
                    rt = tile // NCB
                    cb = tile % NCB
                    base = rt * rows_per_tile
                    for r in T.serial(rows_per_tile):
                        if base + r < tokens:
                            src = T.cast((base + r) * 2048 + cb * COLS, "int32")
                            T.copy(raw[src : src + COLS],
                                   raw_ub[r * COLS : (r + 1) * COLS])
                            T.copy(gate[src : src + COLS],
                                   gate_ub[r * COLS : (r + 1) * COLS])
                            rsrc = T.cast((base + r) * 16, "int32")
                            T.copy(rstd[rsrc : rsrc + 16],
                                   r_ub[r * 16 : (r + 1) * 16])
                    with T.SimdVF():
                        m = T.simd.pset(32)
                        # pass 1: nbf = RNE(f32(raw) * rstd[row, head])
                        for r in T.serial(rows_per_tile):
                            if base + r < tokens:
                                for h in T.serial(8):
                                    for c in range(2):
                                        off = r * COLS + h * 128 + c * LANES
                                        rv = T.simd.vcvt(
                                            T.simd.vld(raw_ub[off],
                                                       dist="UNPK_B16"),
                                            "float32", part=0)
                                        sv = T.simd.vmuls(
                                            rv, r_ub[r * 16 + cb * 8 + h], m)
                                        T.simd.vsts(
                                            nbf_ub[off],
                                            T.simd.vcvt(sv, "bfloat16",
                                                        round="ROUND_R"),
                                            dist="PK_B32")
                        # pass 2: wn = RNE(f32(w) * f32(nbf))
                        for i in range(in_span // LANES):
                            wv = T.simd.vcvt(
                                T.simd.vld(w_ub[(i * LANES) % 128],
                                           dist="UNPK_B16"),
                                "float32", part=0)
                            nv = T.simd.vcvt(
                                T.simd.vld(nbf_ub[i * LANES], dist="UNPK_B16"),
                                "float32", part=0)
                            T.simd.vsts(
                                wn_ub[i * LANES],
                                T.simd.vcvt(T.simd.vmul(wv, nv, m),
                                            "bfloat16", round="ROUND_R"),
                                dist="PK_B32")
                        # pass 3: out = RNE(f32(wn) * gate)
                        for i in range(in_span // LANES):
                            wnv = T.simd.vcvt(
                                T.simd.vld(wn_ub[i * LANES], dist="UNPK_B16"),
                                "float32", part=0)
                            gv = T.simd.vld(gate_ub[i * LANES])
                            T.simd.vsts(
                                out_ub[i * LANES],
                                T.simd.vcvt(T.simd.vmul(wnv, gv, m),
                                            "bfloat16", round="ROUND_R"),
                                dist="PK_B32")
                    for r in T.serial(rows_per_tile):
                        if base + r < tokens:
                            dst = T.cast((base + r) * 2048 + cb * COLS, "int32")
                            T.copy(out_ub[r * COLS : (r + 1) * COLS],
                                   out[dst : dst + COLS])

    return main


def build_programs():
    yield "gdrapply_bf16", gdrapply_bf16()
    yield "gdrtailmix_bf16", gdrtailmix_bf16()
