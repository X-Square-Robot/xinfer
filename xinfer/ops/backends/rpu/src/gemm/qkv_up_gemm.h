// Copyright 2026 X-Square Robot. SPDX-License-Identifier: Apache-2.0
#include "__clang_rpu_builtin_vars.h"

// Full-K Vision GEMM. Fixed N/K; M is any integer in [1,868]. FP16
// operands/output, FP32 VMAT accumulation over the whole K in one pass.
// The QKV/up entries stream K instead (qkv_up_gemm_stream.h) and use this
// path only above their streamed row limit.
// ABI: A LocalSPM bytes [0:1], W DDR/256 [2:3], C LocalSPM bytes [4:5],
// bias LocalSPM bytes [6:7], M [8]. QKV appends K [10:11], V [12:13];
// slot 9 is compiler alignment padding. Bias is a compact FP16 vector [N].
// W layout: [N/16][K/16][8 cores][16 output columns][16 K values].
// Each core computes a complete independent local GEMM. grid={ceil(N/TN),1,1}.
// A/C are compact M-row matrices; QKV has separate compact Q/K/V [M,128].
// Per-thread loop bounds mask arbitrary tail rows. Bias initializes the
// accumulator; no extra output pass.
// O/down return partials: supply nonzero bias only on core 0 before all-reduce.
// Every weight is read once and reused for all M rows owned by the warp.
// Optional fused epilogue for the full-K and K128-pipeline paths:
// QKV_UP_TILE_EPILOGUE(r, scratch, n_base, m_tile, m_count, rows) runs on
// the FP16 output tiles r[m * TN/16 + n] of M16 tiles [m_tile, m_tile +
// m_count) before they are stored; `scratch` is the A buffer, free there.
#ifndef QKV_UP_TILE_EPILOGUE
#define QKV_UP_TILE_EPILOGUE(r, scratch, n_base, m_tile, m_count, rows)
#endif
// A fused entry may append kernel arguments (after K/V) that its epilogue
// reads: QKV_UP_EXTRA_PARAMS (", type name...") and QKV_UP_EXTRA_ARGS.
#ifndef QKV_UP_EXTRA_PARAMS
#define QKV_UP_EXTRA_PARAMS
#define QKV_UP_EXTRA_ARGS
#endif
#if QKV_UP_SPLIT_QKV
#define QKV_UP_OUTPUT_PARAMS , half *kv, half *vv QKV_UP_EXTRA_PARAMS
#define QKV_UP_OUTPUT_ARGS , kv, vv QKV_UP_EXTRA_ARGS
// A block's N range may cross a 128-column projection boundary. Split its
// writeback into at most two segments; each value is stored exactly once.
template <short AM>
__attribute__((always_inline)) inline void qkv_up_store_segment(
    __local__ f16v16 *r, half *out, short *m0, short rows, short columns) {
  short lp[2];
  LoopStepInfo rs[2] = {{&lp[1], QKV_UP_TN / 16}, {&lp[0], 1}};
  FragVData vr(r, rs);
  LoopStepInfo rs_tiled[3] = {{m0, 0}, {&lp[1], QKV_UP_TN / 16}, {&lp[0], 1}};
  FragVData vr_tiled(r, rs_tiled);
  // Hardware outer-loop address stepping avoids rebuilding two 32-bit output
  // addresses for every M tile. The step is one complete AM-row-block tile.
  LoopStepInfo ds[3] = {{m0, (int)(AM * 16 * 128)},
                        {&lp[1], (int)(16 * 128)}, {&lp[0], (int)16}};
  LoopStepInfo ds0[2] = {{&lp[1], (int)(16 * 128)}, {&lp[0], (int)16}};
  FragSData sc(out, ds);
  FragSData sc0(out, ds0);
#pragma rpu_hwloop_ul
  for (lp[1] = 0; lp[1] < rows; ++lp[1]) {
#pragma rpu_hwloop_ul
    for (lp[0] = 0; lp[0] < columns; ++lp[0])
      if (AM)
        rpu::vstExecute_async(vr_tiled, sc, (short)(threadId * 128));
      else
        rpu::vstExecute_async(vr, sc0, (short)(threadId * 128));
  }
}
template <short AM>
__attribute__((always_inline)) inline void qkv_up_store_qkv(
    __local__ f16v16 *r, half *qv, half *kv, half *vv,
    short *m0, short valid_m, short n_base, short n_count) {
  half *first = n_base < 8 ? qv : n_base < 16 ? kv : vv;
  const short local_n = n_base & 7;
  const short first_count = n_count < 8 - local_n ? n_count : 8 - local_n;
  qkv_up_store_segment<AM>(r, first + (unsigned int)local_n * 16, m0,
                           valid_m, first_count);
  if (first_count < n_count) {
    half *second = n_base < 8 ? kv : vv;
    qkv_up_store_segment<AM>(r + first_count, second, m0,
                             valid_m, n_count - first_count);
  }
}
#else
#define QKV_UP_OUTPUT_PARAMS QKV_UP_EXTRA_PARAMS
#define QKV_UP_OUTPUT_ARGS QKV_UP_EXTRA_ARGS
#endif
#ifdef QKV_UP_PIPELINE_ROWS
// Rows <= QKV_UP_PIPELINE_ROWS: two banks overlap the next A/W K chunk with
// the current VMAT. Each chunk uses FP32 VMAT accumulation, with FP16
// partial sums held between chunks.
__attribute__((always_inline)) inline void qkv_up_load_k(
    __local__ f16v16 *a, __local__ f16v16 *b, half *av, DDR_large bg,
    short k0, short valid_m, short n_count) {
  const short CK = QKV_UP_TK / 16, AK = QKV_UP_K / 16;
  short lp[2];
  LoopStepInfo ad[2] = {{&lp[1], CK}, {&lp[0], 1}};
  FragVData va(a, ad);
  LoopStepInfo as[2] = {{&lp[1], (int)(16 * QKV_UP_K)}, {&lp[0], (int)16}};
  FragSData sa(av + (int)k0 * 16, as);
#pragma rpu_hwloop_ul
  for (lp[1] = 0; lp[1] < valid_m; ++lp[1]) {
#pragma rpu_hwloop_ul
    for (lp[0] = 0; lp[0] < CK; ++lp[0])
      rpu::vldExecute_async(va, sa, (short)(threadId * QKV_UP_K));
  }
  LoopStepInfo bd[2] = {{&lp[1], CK}, {&lp[0], 1}};
  FragVData vb(b, bd);
  LoopStepInfo bs[2] = {
      {&lp[1], (int)AK * (int)4096, rpu::__AGran::a_g1B},
      {&lp[0], (int)4096, rpu::__AGran::a_g1B}};
  FragSData sb(bg + (int)k0 * 16, bs);
#pragma rpu_hwloop_ul
  for (lp[1] = 0; lp[1] < n_count; ++lp[1]) {
#pragma rpu_hwloop_ul
    for (lp[0] = 0; lp[0] < CK; ++lp[0])
      rpu::vldExecute_async(vb, sb, (short)(threadId * 32));
  }
}
__attribute__((always_inline)) inline void qkv_up_k_pipeline(
    half *av, DDR_large wv, half *rv, half *bias, unsigned short rows,
    __local__ f16v16 *a, __local__ f16v16 *b, __local__ f16v16 *r QKV_UP_OUTPUT_PARAMS) {
  const short AK = QKV_UP_K / 16, CK = QKV_UP_TK / 16;
  const short AN = QKV_UP_TN / 16, AM = QKV_UP_PIPELINE_ROWS / 16;
  const short m_blocks = (rows + 15) / 16;
  const short valid_m = (rows + 15 - threadId) / 16;
  const short n_base = (short)blockIdx.x * AN;
  const short n_count = QKV_UP_N / 16 - n_base < AN
                            ? QKV_UP_N / 16 - n_base : AN;
  short lp[3];
  // Seed the existing partial-sum tile directly from SPM. This replaces the
  // zeroing VALU pass and shares the first A/W load's fence.
  LoopStepInfo init_dst[2] = {{&lp[1], AN}, {&lp[0], 1}};
  FragVData init_r(r, init_dst);
  LoopStepInfo init_src[2] = {{&lp[1], (int)0}, {&lp[0], (int)16}};
  FragSData init_bias(bias + (int)n_base * 16, init_src);
#pragma rpu_hwloop_ul
  for (lp[1] = 0; lp[1] < m_blocks; ++lp[1]) {
#pragma rpu_hwloop_ul
    for (lp[0] = 0; lp[0] < n_count; ++lp[0])
      rpu::vldExecute_async(init_r, init_bias, (short)0);
  }
  DDR_large bg = wv + (int)n_base * AK * 16 + (int)coreId_u * 2;
  qkv_up_load_k(a, b, av, bg, 0, valid_m, n_count);
  rpu::fenceVld(0);
#pragma rpu_hwloop_ul
  for (short k0 = 0; k0 < AK; k0 += CK) {
    short bank = (k0 / CK) & 1;
    if (k0 + CK < AK)
      qkv_up_load_k(a + (1 - bank) * AM * CK, b + (1 - bank) * AN * CK,
                    av, bg, k0 + CK, valid_m, n_count);
    LoopStepInfo cs[3] = {{&lp[2], AN}, {&lp[1], 1}, {&lp[0], 0}};
    FragVData vc(r, cs);
    LoopStepInfo ac[3] = {{&lp[2], CK}, {&lp[1], 0}, {&lp[0], 1}};
    FragVData aa(a + bank * AM * CK, ac);
    LoopStepInfo bc[3] = {{&lp[2], 0}, {&lp[1], CK}, {&lp[0], 1}};
    FragVData bb(b + bank * AN * CK, bc);
    LoopStepInfo init_steps[2] = {{&lp[2], AN}, {&lp[1], 1}};
    FragVData init(r, init_steps);
    LoopOutInfo oi[1] = {{&lp[0]}};
    LoopOutLPAttr oa(oi);
#pragma rpu_hwloop_ul
    for (lp[2] = 0; lp[2] < m_blocks; ++lp[2]) {
#pragma rpu_hwloop_ul
      for (lp[1] = 0; lp[1] < n_count; ++lp[1]) {
        rpu::vcfg_set_vmat(rpu::vcfg_vmat_target::acc_init_fp16, init);
#pragma rpu_hwloop_ul
        for (lp[0] = 0; lp[0] < CK; ++lp[0])
          rpu::vmatExecute_async(vc, aa, bb, &oa);
      }
    }
    rpu::fenceVmat(0);
    rpu::fenceVld(0);
  }
  QKV_UP_TILE_EPILOGUE(r, a, n_base, 0, m_blocks, rows);
#if QKV_UP_SPLIT_QKV
  qkv_up_store_qkv<0>(r, rv, kv, vv, nullptr, valid_m, n_base, n_count);
#else
  LoopStepInfo rs[2] = {{&lp[1], AN}, {&lp[0], 1}};
  FragVData vr(r, rs);
  LoopStepInfo ds[2] = {{&lp[1], (int)(16 * QKV_UP_N)}, {&lp[0], (int)16}};
  FragSData sc(rv + (int)n_base * 16, ds);
#pragma rpu_hwloop_ul
  for (lp[1] = 0; lp[1] < valid_m; ++lp[1]) {
#pragma rpu_hwloop_ul
    for (lp[0] = 0; lp[0] < n_count; ++lp[0])
      rpu::vstExecute_async(vr, sc, (short)(threadId * QKV_UP_N));
  }
#endif
  rpu::fenceVst(0);
}
#endif

// Keep a complete N tile of weights while iterating over M tiles.
template <short AM>
__attribute__((always_inline)) inline void qkv_up_full_k(
    half *av, DDR_large wv, half *rv, half *bias, unsigned short rows,
    __local__ f16v16 *a, __local__ f16v16 *b, __local__ f16v16 *r,
    __local__ f16v16 *biasv QKV_UP_OUTPUT_PARAMS) {
  const short AK = QKV_UP_K / 16;
  const short AN = QKV_UP_TN / 16;
  const short m_blocks = (rows + 15) / 16;
  const short n_base = (short)blockIdx.x * AN;
  const short n_count = QKV_UP_N / 16 - n_base < AN
                            ? QKV_UP_N / 16 - n_base : AN;
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
  // One bias tile stays resident alongside W across every M tile.
  LoopStepInfo bias_dst[1] = {{&lp[0], 1}};
  FragVData vbias(biasv, bias_dst);
  LoopStepInfo bias_src[1] = {{&lp[0], (int)16}};
  FragSData sbias(bias + (int)n_base * 16, bias_src);
#pragma rpu_hwloop_ul
  for (lp[0] = 0; lp[0] < n_count; ++lp[0])
    rpu::vldExecute_async(vbias, sbias, (short)0);

#pragma rpu_hwloop_ul
  for (short m0 = 0; m0 < m_blocks; m0 += AM) {
    short m_count = m_blocks - m0 < AM ? m_blocks - m0 : AM;
    // Since m0 < ceil(rows/16), live_m is nonnegative even for an inactive
    // last-row lane. Keep every lane participating in W loads and VMAT: an
    // early return for threadId >= rows would lose required weight columns.
    short live_m = (rows + 15 - threadId) / 16 - m0;
    short valid_m = live_m < AM ? live_m : AM;
    half *ag = av + (int)m0 * 16 * QKV_UP_K;
    LoopStepInfo ad[2] = {{&lp[1], AK}, {&lp[0], 1}};
    FragVData va(a, ad);
    LoopStepInfo as[2] = {{&lp[1], (int)(16 * QKV_UP_K)}, {&lp[0], (int)16}};
    FragSData sa(ag, as);
#pragma rpu_hwloop_ul
    for (lp[1] = 0; lp[1] < valid_m; ++lp[1]) {
#pragma rpu_hwloop_ul
      for (lp[0] = 0; lp[0] < AK; ++lp[0])
        rpu::vldExecute_async(va, sa, (short)(threadId * QKV_UP_K));
    }
    rpu::fenceVld(0);
#if QKV_UP_PIPELINE != 1
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
    LoopOutLPAttr oa(oi);
    LoopStepInfo bias_steps[1] = {{&lp[1], 1}};
    FragVData init(biasv, bias_steps);
#pragma rpu_hwloop_ul
    for (lp[2] = 0; lp[2] < m_count; ++lp[2]) {
#pragma rpu_hwloop_ul
      for (lp[1] = 0; lp[1] < n_count; ++lp[1]) {
        rpu::vcfg_set_vmat(rpu::vcfg_vmat_target::acc_init_fp16, init);
#pragma rpu_hwloop_ul
        for (lp[0] = 0; lp[0] < AK; ++lp[0])
          rpu::vmatExecute_async(vc, aa, bb, &oa, rpu::__ClrMode::clroff);
      }
    }
    rpu::fenceVmat(0);
    QKV_UP_TILE_EPILOGUE(r, a, n_base, m0, m_count, rows);

#if QKV_UP_SPLIT_QKV
    qkv_up_store_qkv<AM>(r, rv, kv, vv, &m0, valid_m, n_base, n_count);
#else
    half *cg = rv + (int)m0 * 16 * QKV_UP_N + (int)n_base * 16;
    LoopStepInfo rs[2] = {{&lp[1], AN}, {&lp[0], 1}};
    FragVData vr(r, rs);
    LoopStepInfo ds[2] = {{&lp[1], (int)(16 * QKV_UP_N)}, {&lp[0], (int)16}};
    FragSData sc(cg, ds);
#pragma rpu_hwloop_ul
    for (lp[1] = 0; lp[1] < valid_m; ++lp[1]) {
#pragma rpu_hwloop_ul
      for (lp[0] = 0; lp[0] < n_count; ++lp[0])
        rpu::vstExecute_async(vr, sc, (short)(threadId * QKV_UP_N));
    }
#endif
#if QKV_UP_PIPELINE == 1
    rpu::fenceVst(0);
#endif
  }
#if QKV_UP_PIPELINE != 1
  rpu::fenceVst(0);
#endif
}

#ifndef QKV_UP_STREAM_ROWS
__rprog__ void QKV_UP_ENTRY(half *av, DDR_large wv, half *rv, half *bias,
                            unsigned short rows QKV_UP_OUTPUT_PARAMS) {
  const short AK = QKV_UP_K / 16, AN = QKV_UP_TN / 16;
  const short AM = QKV_UP_TM / 16;
  __local__ f16v16 a[AM * AK], b[AN * AK], r[AM * AN], biasv[AN];
  qkv_up_full_k<AM>(av, wv, rv, bias, rows, a, b, r, biasv QKV_UP_OUTPUT_ARGS);
}
#endif
