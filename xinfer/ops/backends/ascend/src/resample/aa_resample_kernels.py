"""aa_resample: separable antialiased-bicubic resize kernels for the 2B device
image chain (AA-RESAMPLE).

Semantics (extracted and validated against the fp64 torch oracle on
2026-11-16, probes aa_weight_fit / aa_precision1: the separable formula equals
the torch antialiased-bicubic oracle to 3e-13 in fp64, and its fp32 emulation
is ~200x closer to the oracle than the aclnnUpsampleBicubic2dAA vendor):

  r = hin/hout; src_j = r*(j+0.5)-0.5; fs = 1/r if r>1 else 1
  w_j(k) = fs * cubic_A=-0.5(fs*(k - src_j))  on |fs*(k-src_j)| < 2
  out-of-range taps contribute zero (zero-fill border, NOT replicate), and the
  1D weights are normalized per output index by the sum of in-range weights.

The kernels are the two separable passes of one resize step
(f32 [3,Hin,Win] -> u8 [3,hout,wout]; the host mount performs the released
chain's own exact u8->f32 cast), the chain_view/_resize step semantics
`F.interpolate(bicubic, antialias=True, align_corners=False)` +
`add_(0.5).floor_().clamp_(0,255).to(uint8)` fused:

  aa_v_*: vertical pass, f32 in -> fp32 mid [3,hout,Win]
          (one block per output row; per-tap runtime-row window loads into a
           WINP=pad64(WIN)-wide win_ub with clamped row index and zero
           weights on out-of-range taps; the WINP bound fixes the
           WIN%64!=0 truncation that left the 224-wide up legs with 32
           unwritten lanes per row (iso4/iso5 evidence: dev=69 = prefill -7
           kept, written rows 0); the [WIN, WINP) pad lanes of win_ub stay
           uninitialized -- their garbage only ever reaches the acc pad
           lanes through lane-independent vmuls/vadd and only acc[0:WIN] is
           stored, so the output is exact; a zero-init pass for those lanes
           was tried and DROPPED: the extra SimdVF region makes the
           auto-scheduler emit a pre-SimdVF V_MTE2 notify plus doubled MTE2
           waits whose FIFO flag matching deadlocks the vector core
           (real9 hang, aivec timeout; the minimal real6 flag form does
           not). real6-proven fence discipline otherwise: per-tap
           V_MTE2/MTE2_V pairs order win_ub reuse, the MTE3_V pair drains
           the previous channel's GM store before the vector unit rewrites
           acc_ub, and VST_VLD mem_bars order the vector store->load
           through acc_ub. The interrupted real7 session had also started a
           batched-taps redesign -- dropped: python-list tap weights cannot
           be indexed with a traced loop Var, and the 25x1280 fp32 window
           would need ~125KB UB)
  aa_h_*: horizontal pass + the strict quantize law, fp32 mid -> u8 out
          (one block per output row; per-lane vgather2 windows along the mid
           row with host-precomputed per-tap index tables; register-file
           accumulation inside ONE SimdVF region per channel (adanorm-r2
           alloc_local idiom) so no cross-block fence is needed; WOUTP
           pad64(WOUT) with zero-weight index-0 gathers covers the tail;
           explicit MTE2_V fence after the mid row load and V_MTE3 before the
           store, plus the MTE3_V pair guarding out_ub reuse across channels;
           then y=acc+0.5 -> trunc int32 (== floor for y>=0; negatives
           collapse through the int clamp) -> clamp [0,255] -> u8 ROUND_Z ->
           PK4_B32 store)

All shapes are closure constants (fully specialized kernels); the weight and
index tables are runtime fp32/uint32 GM buffers built by the host mount
(glue/aa_resample.py) from the validated formula.
"""

import tilelang.ascend.language as T

LANES = 64


def _v_kernel(HIN, WIN, HOUT, TAPS):
    """in f32 [3,HIN,WIN] -> mid f32 [3,HOUT,WIN]; block = one output row.

    WINP = pad64(WIN): both win_ub and acc_ub are WINP lanes wide so every
    WIN element is covered by a full 64-lane group; only acc_ub[0:WIN] is
    stored. The [WIN, WINP) win_ub pad lanes are never written (no zero-init
    pass: it deadlocks the auto-scheduler's flag stream, see module doc) --
    their garbage reaches only the acc pad lanes through lane-independent
    vector ops and is never copied out.
    """

    WINP = (WIN + LANES - 1) // LANES * LANES
    NG = WINP // LANES

    @T.prim_func
    def main(ih: T.handle, mh: T.handle, wyh: T.handle, byh: T.handle):
        inbuf = T.match_buffer(ih, (3 * HIN * WIN,), "float32")
        mid = T.match_buffer(mh, (3 * HOUT * WIN,), "float32")
        wy = T.match_buffer(wyh, (HOUT * TAPS,), "float32")
        by = T.match_buffer(byh, (HOUT,), "int32")
        with T.Kernel(HOUT) as bj:
            win_ub = T.alloc_shared((WINP,), "float32")
            acc_ub = T.alloc_shared((WINP,), "float32")
            wy_ub = T.alloc_shared((TAPS,), "float32")
            T.ascend_set_flag("V_MTE2", 0)
            T.ascend_set_flag("MTE3_V", 0)
            T.copy(wy[bj * TAPS : bj * TAPS + TAPS], wy_ub)
            T.ascend_set_flag("MTE2_S", 0)
            base = by[bj]
            for c in range(3):
                T.ascend_wait_flag("MTE3_V", 0)
                with T.SimdVF():
                    m = T.simd.pset(32)
                    for xg in range(NG):
                        T.simd.vsts(acc_ub[xg * LANES],
                                    T.simd.vdup(0.0, "float32"))
                for t in range(TAPS):
                    T.ascend_wait_flag("V_MTE2", 0)
                    row = T.min(T.max(base + t, 0), HIN - 1)
                    T.copy(inbuf[c * HIN * WIN + row * WIN :
                                  c * HIN * WIN + row * WIN + WIN],
                           win_ub[0:WIN])
                    T.ascend_set_flag("MTE2_V", 0)
                    if c == 0 and t == 0:
                        T.ascend_wait_flag("MTE2_S", 0)
                    w = wy_ub[t]
                    T.ascend_wait_flag("MTE2_V", 0)
                    with T.SimdVF():
                        m = T.simd.pset(32)
                        T.simd.mem_bar("VST_VLD")
                        for xg in range(NG):
                            T.simd.vsts(
                                acc_ub[xg * LANES],
                                T.simd.vadd(
                                    T.simd.vmuls(
                                        T.simd.vld(win_ub[xg * LANES]),
                                        w, m),
                                    T.simd.vld(acc_ub[xg * LANES]), m))
                    T.ascend_set_flag("V_MTE2", 0)
                T.ascend_set_flag("V_MTE3", 0)
                T.ascend_wait_flag("V_MTE3", 0)
                T.copy(acc_ub[0:WIN],
                       mid[(c * HOUT + bj) * WIN :
                           (c * HOUT + bj) * WIN + WIN])
                T.ascend_set_flag("MTE3_V", 0)
            T.ascend_wait_flag("V_MTE2", 0)
            T.ascend_wait_flag("MTE3_V", 0)

    return main


def _h_kernel(WIN, HOUT, WOUT, TAPS):
    """mid f32 [3,HOUT,WIN] -> out u8 [3,HOUT,WOUT]; block = one output row."""
    WOUTP = (WOUT + LANES - 1) // LANES * LANES
    NG = WOUTP // LANES

    @T.prim_func
    def main(oh: T.handle, mh: T.handle, wxh: T.handle, ixh: T.handle):
        out = T.match_buffer(oh, (3 * HOUT * WOUT,), "uint8")
        mid = T.match_buffer(mh, (3 * HOUT * WIN,), "float32")
        wx = T.match_buffer(wxh, (TAPS * WOUTP,), "float32")
        ix = T.match_buffer(ixh, (TAPS * WOUTP,), "uint32")
        with T.Kernel(HOUT) as bj:
            mid_ub = T.alloc_shared((WIN,), "float32")
            wx_ub = T.alloc_shared((TAPS * WOUTP,), "float32")
            ix_ub = T.alloc_shared((TAPS * WOUTP,), "uint32")
            out_ub = T.alloc_shared((WOUTP,), "uint8")
            T.copy(wx, wx_ub)
            T.copy(ix, ix_ub)
            T.ascend_set_flag("MTE2_V", 1)
            T.ascend_wait_flag("MTE2_V", 1)
            T.ascend_set_flag("MTE3_V", 9)
            for c in range(3):
                T.copy(mid[(c * HOUT + bj) * WIN : (c * HOUT + bj) * WIN + WIN],
                       mid_ub)
                T.ascend_set_flag("MTE2_V", 2)
                T.ascend_wait_flag("MTE2_V", 2)
                T.ascend_wait_flag("MTE3_V", 9)
                with T.SimdVF():
                    m = T.simd.pset(32)
                    acc = T.simd.alloc_local(NG, "float32")
                    for wg in range(NG):
                        acc[wg] = T.simd.vdup(0.0, "float32")
                    for t in range(TAPS):
                        for wg in range(NG):
                            iv = T.simd.vld(ix_ub[t * WOUTP + wg * LANES])
                            g = T.simd.vgather2(mid_ub[0], iv)
                            wv = T.simd.vld(wx_ub[t * WOUTP + wg * LANES])
                            acc[wg] = T.simd.vadd(T.simd.vmul(wv, g), acc[wg])
                    for wg in range(NG):
                        # strict chain law: floor(y) with y=acc+0.5 for y>=0;
                        # ROUND_Z (truncate) == floor there, and negatives
                        # collapse through the int clamp to 0, so the result
                        # equals add_(0.5).floor_().clamp_(0,255).to(u8) for
                        # every acc.
                        y = T.simd.vadds(acc[wg], 0.5)
                        yi = T.simd.vcvt(y, "int32", round="ROUND_Z")
                        yi = T.simd.vmin(
                            T.simd.vmax(yi, T.simd.vdup(0, "int32")),
                            T.simd.vdup(255, "int32"))
                        T.simd.vsts(out_ub[wg * LANES],
                                    T.simd.vcvt(yi, "uint8", round="ROUND_Z"),
                                    dist="PK4_B32")
                T.ascend_set_flag("V_MTE3", 3)
                T.ascend_wait_flag("V_MTE3", 3)
                T.copy(out_ub[0:WOUT],
                       out[(c * HOUT + bj) * WOUT :
                           (c * HOUT + bj) * WOUT + WOUT])
                T.ascend_set_flag("MTE3_V", 9)

    return main


# deployment resize steps: (tag, HIN, WIN, HOUT, WOUT, TAPS)
STEPS = [
    ("720_1280_128_224", 720, 1280, 128, 224, 25),   # heavy AA down (front 720p)
    ("480_640_160_224", 480, 640, 160, 224, 13),     # medium AA down (desktop)
    ("128_224_224_352", 128, 224, 224, 352, 5),      # AA up leg (min_pixels)
    ("160_224_224_320", 160, 224, 224, 320, 5),      # AA up leg
]


def build_programs():
    for tag, hin, win, hout, wout, taps in STEPS:
        yield f"aa_v_{tag}", _v_kernel(hin, win, hout, taps)
        yield f"aa_h_{tag}", _h_kernel(win, hout, wout, taps)


COMPILE_OPTIONS = ()
