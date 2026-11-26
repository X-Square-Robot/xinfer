// Copyright 2026 X-Square Robot. SPDX-License-Identifier: Apache-2.0
#include "__clang_rpu_builtin_vars.h"
#if STREAMED_K <= 0 || STREAMED_N <= 0 || STREAMED_TM <= 0 || STREAMED_TN <= 0 || \
    STREAMED_K % 16 || STREAMED_N % 16 || STREAMED_TM % 16 || STREAMED_TN % 16
#error "Language GEMM dimensions and tiles must be positive multiples of 16"
#endif
#if STREAMED_TK != STREAMED_K
#error "Language GEMM requires full-K FP32 accumulation"
#endif
#if STREAMED_PIPELINE != 1 && STREAMED_PIPELINE != 2
#error "Language GEMM supports pipeline 1 or 2"
#endif
#if STREAMED_TM > 384 || STREAMED_TN > STREAMED_N || \
    ((STREAMED_TM / 16) * (STREAMED_K / 16) + (STREAMED_TN / 16) * (STREAMED_K / 16) + \
     (STREAMED_TM / 16) * (STREAMED_TN / 16)) > 400
#error "Language GEMM tile exceeds the workload or the 400-vector VLM budget"
#endif
// M=64..384 in physical 64-row prefix buckets, FP16 input/output,
// full-K FP32 accumulation (no FP16 partial sums).
// W: [N_padded/16][K/16][8][16 N][16 K]; A/C row-major in per-core SPM.
// ABI: A byte address [0:1], W DDR/256 [2:3], C byte address [4:5], M [6].
// Broadcast the complete grid to all eight cores. O/down return TP partials.
// Keep a complete N tile of weights while iterating over M tiles.
template <short AM>
__attribute__((always_inline)) inline void streamed_full_k(
    half *av, DDR_large wv, half *rv, unsigned short rows,
    __local__ f16v16 *a, __local__ f16v16 *b, __local__ f16v16 *r) {
  const short AK = STREAMED_K / 16;
  const short AN = STREAMED_TN / 16;
  const short m_blocks = (rows + 15) / 16;
  const short n_base = (short)blockIdx.x * AN;
  const short n_count = STREAMED_N / 16 - n_base < AN
                            ? STREAMED_N / 16 - n_base : AN;
  short lp[3];

  DDR_large bg = wv + (int)n_base * AK * 16 + (int)coreId_u * 2;
  LoopStepInfo bd[2] = {{&lp[1], AK}, {&lp[0], 1}};
  FragVData bv(b, bd);
  LoopStepInfo bs[2] = {
      {&lp[1], (int)AK * (int)4096, rpu::__AGran::a_g1B},
      {&lp[0], (int)4096, rpu::__AGran::a_g1B}};
  FragSData bw(bg, bs);
#pragma rpu_hwloop_ul
  for (lp[1] = 0; lp[1] < n_count; ++lp[1]) {
#pragma rpu_hwloop_ul
    for (lp[0] = 0; lp[0] < AK; ++lp[0])
      rpu::vldExecute_async(bv, bw, (short)(threadId * 32));
  }

#pragma rpu_hwloop_ul
  for (short m0 = 0; m0 < m_blocks; m0 += AM) {
    short m_count = m_blocks - m0 < AM ? m_blocks - m0 : AM;
    short live_m = (rows + 15 - threadId) / 16 - m0;
    short valid_m = live_m < AM ? live_m : AM;
    half *ag = av + (int)m0 * 16 * STREAMED_K;
    LoopStepInfo ad[2] = {{&lp[1], AK}, {&lp[0], 1}};
    FragVData va(a, ad);
    LoopStepInfo as[2] = {{&lp[1], ((int)16 * STREAMED_K)}, {&lp[0], (int)16}};
    FragSData sa(ag + (int)threadId * STREAMED_K, as);
#pragma rpu_hwloop_ul
    for (lp[1] = 0; lp[1] < valid_m; ++lp[1]) {
#pragma rpu_hwloop_ul
      for (lp[0] = 0; lp[0] < AK; ++lp[0])
        rpu::vldExecute_async(va, sa, (short)0);
    }
    rpu::fenceVld(0);
#if STREAMED_PIPELINE != 1
    // A staging can overlap the previous tile's C writeback. Wait before
    // VMAT reuses r[], and drain the final store after the outer loop.
    rpu::fenceVst(0);
#endif

    LoopStepInfo cs[3] = {{&lp[2], AN}, {&lp[1], 1}, {&lp[0], 0}};
    FragVData vc(r, cs);
    LoopStepInfo ac[3] = {{&lp[2], AK}, {&lp[1], 0}, {&lp[0], 1}};
    FragVData aa(a, ac);
    LoopStepInfo bc[3] = {{&lp[2], 0}, {&lp[1], AK}, {&lp[0], 1}};
    FragVData bb(b, bc);
    LoopOutInfo oi[1] = {{&lp[0]}};
    // Clear at the first K tile of each independent output tile.
    LoopClrInfo ci[1] = {{&lp[0]}};
    LoopOutLPAttr oa(oi);
    LoopClrLPAttr ca(ci);
#pragma rpu_hwloop_ul
    for (lp[2] = 0; lp[2] < m_count; ++lp[2]) {
#pragma rpu_hwloop_ul
      for (lp[1] = 0; lp[1] < n_count; ++lp[1]) {
#pragma rpu_hwloop_ul
        for (lp[0] = 0; lp[0] < AK; ++lp[0])
          rpu::vmatExecute_async(vc, aa, bb, &oa, &ca);
      }
    }
    rpu::fenceVmat(0);

    half *cg = rv + (int)m0 * 16 * STREAMED_N + (int)n_base * 16;
    LoopStepInfo rs[2] = {{&lp[1], AN}, {&lp[0], 1}};
    FragVData vr(r, rs);
    LoopStepInfo ds[2] = {{&lp[1], ((int)16 * STREAMED_N)}, {&lp[0], (int)16}};
    FragSData sc(cg + (int)threadId * STREAMED_N, ds);
#pragma rpu_hwloop_ul
    for (lp[1] = 0; lp[1] < valid_m; ++lp[1]) {
#pragma rpu_hwloop_ul
      for (lp[0] = 0; lp[0] < n_count; ++lp[0])
        rpu::vstExecute_async(vr, sc, (short)0);
    }
#if STREAMED_PIPELINE == 1
    rpu::fenceVst(0);
#endif
  }
#if STREAMED_PIPELINE != 1
  rpu::fenceVst(0);
#endif
}

__rprog__ void STREAMED_ENTRY(half *av, DDR_large wv, half *rv, unsigned short rows) {
  const short AK = STREAMED_K / 16, AN = STREAMED_TN / 16, AM = STREAMED_TM / 16;
  __local__ f16v16 a[AM * AK], b[AN * AK], r[AM * AN];
  streamed_full_k<AM>(av, wv, rv, rows, a, b, r);
}
