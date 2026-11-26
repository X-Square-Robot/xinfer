// Copyright 2026 X-Square Robot. SPDX-License-Identifier: Apache-2.0
// K-streamed Vision GEMM (QKV/up): C = A @ W.T + bias, W read from DDR once
// and overlapped with VMAT. M is any integer in [1,868].
// ABI: A LocalSPM bytes [0:1], W DDR/256 [2:3], C [4:5], bias [6:7], M [8];
// QKV appends K [10:11], V [12:13]. W: [N/16][K/16][8][16 N][16 K].
//
// Rows <= QKV_UP_STREAM_ROWS: a block owns QKV_UP_STREAM_TN columns (default
// QKV_UP_TN; blocks beyond N exit) for every row
// and keeps their FP32 accumulators in VLM. W advances in K chunks of
// QKV_UP_TK, double buffered; each chunk runs P phases over contiguous M16
// row ranges: QKV_UP_PHASES, or two up to QKV_UP_SMALL_ROWS (if defined)
// where A is small and fewer fences keep DDR busier. Phase p issues its share of the next W chunk
// (DDR), then the next phase's A rows (SPM), then its VMAT stream, then
// fences. Resident warps run in lockstep, so every phase carries the same
// DDR share. The final chunk writes each phase's rows back during the next.
// The bias seeds each accumulator once and every K16 step accumulates in
// the same order as the full-K kernel, so results match it bit for bit.
// Larger M uses the full-K kernel (QKV_UP_TN columns per block). With
// QKV_UP_PIPELINE_ROWS defined, rows up to it keep the K128 pipeline with
// FP16 partials between chunks (qkv_up_gemm.h, QKV_UP_TN columns) instead.
// The grid is N / QKV_UP_TN.
#define QKV_UP_STREAM_ROWS 400
#ifndef QKV_UP_STREAM_TN
#define QKV_UP_STREAM_TN QKV_UP_TN
#endif
// Optional fused epilogue (fusion/): QKV_UP_STREAM_EPILOGUE(result,
// scratch, n_base, m0, m1, rows) runs on the rounded FP16 rows [m0, m1)
// (low slot of each accumulator pair; the high slot is free scratch) before
// they are stored. QKV_UP_EPILOGUE_TILE_VLM reserves extra VLM per M16 tile of
// one phase at `scratch`. QKV_UP_ENTRY_PROLOGUE runs once at entry.
#ifndef QKV_UP_STREAM_EPILOGUE
#define QKV_UP_STREAM_EPILOGUE(result, scratch, n_base, m0, m1, rows)
#endif
#ifndef QKV_UP_EPILOGUE_TILE_VLM
#define QKV_UP_EPILOGUE_TILE_VLM 0
#endif
#ifndef QKV_UP_ENTRY_PROLOGUE
#define QKV_UP_ENTRY_PROLOGUE
#endif
#include "qkv_up_gemm.h"

constexpr short VS_AK = QKV_UP_K / 16, VS_CK = QKV_UP_TK / 16;
constexpr short VS_AN = QKV_UP_STREAM_TN / 16;
// Geometry of the P-phase path for up to MaxRows rows.
template <short P, short MaxRows> struct VsShape {
  static constexpr short Phases = P;
  static constexpr short MT = (MaxRows + 15) / 16;
  static constexpr short PM = (MT + P - 1) / P;  // M16 tiles per phase
  static constexpr short BPP = VS_AN * VS_CK / P;  // W vectors per phase
  // FP32 accumulator pairs, W and A double buffers, bias, two zeros.
  static constexpr short VLM = 2 * MT * VS_AN + 2 * VS_AN * VS_CK +
                               2 * PM * VS_CK + VS_AN + 2 +
                               PM * QKV_UP_EPILOGUE_TILE_VLM;
  // Keep at least two phases in this load schedule.
  static_assert(P >= 2 && VS_AN * VS_CK % P == 0, "chunk/phase split");
  static_assert(BPP <= VS_CK ? VS_CK % BPP == 0 : BPP % VS_CK == 0,
                "a phase loads whole tiles or an exact fraction of one");
};
using VsLarge = VsShape<QKV_UP_PHASES, QKV_UP_STREAM_ROWS>;
constexpr short vs_max(short x, short y) { return x > y ? x : y; }
#ifdef QKV_UP_SMALL_ROWS
using VsSmall = VsShape<2, QKV_UP_SMALL_ROWS>;
constexpr short VS_SMALL_VLM = VsSmall::VLM;
#else
constexpr short VS_SMALL_VLM = 0;
#endif
#ifdef QKV_UP_PIPELINE_ROWS
// K128 pipeline: output partials, A and W double buffers.
constexpr short VS_PIPE_AM = QKV_UP_PIPELINE_ROWS / 16;
constexpr short VS_PIPE_VLM = VS_PIPE_AM * (QKV_UP_TN / 16) +
                              2 * VS_PIPE_AM * VS_CK + 2 * (QKV_UP_TN / 16) * VS_CK;
#else
constexpr short VS_PIPE_VLM = 0;
#endif
// Full-K fallback (QKV_UP_TN columns): output, bias, A, W.
constexpr short VS_FULL_AM = QKV_UP_TM / 16, VS_FULL_AN = QKV_UP_TN / 16;
constexpr short VS_FULL_VLM = VS_FULL_AM * VS_FULL_AN + VS_FULL_AN +
                              VS_FULL_AM * VS_AK + VS_FULL_AN * VS_AK;
// One VLM arena shared by all paths.
constexpr short VS_ARENA = vs_max(vs_max(VS_SMALL_VLM, VS_PIPE_VLM),
                                  vs_max(VsLarge::VLM, VS_FULL_VLM));
static_assert(QKV_UP_N % QKV_UP_STREAM_TN == 0 && VS_AK % VS_CK == 0, "whole tiles");
static_assert(VS_ARENA <= 400, "VLM budget");

__attribute__((always_inline)) inline void vs_repeat_b32() {
  constexpr short format =
      rpu::acc_format::acc_fp32 | rpu::dst_format::dst_B32 |
      rpu::rs0_format::rs0_fp16 | rpu::rs1_format::rs1_fp16;
  __builtin_rhvm_vmat_set_tsize(VS_CK, 1, format);
  __builtin_rhvm_vmat_set_tstep0(0, 1, 1, format);
  __builtin_rhvm_vmat_set_tstep1(0, 0, 0, format);
}

__attribute__((always_inline)) inline void vs_repeat_fp16() {
  rpu::vmatSetTsize(VS_CK, 1);
  rpu::vmatSetTstep0(0, 1, 1);
  rpu::vmatSetTstep1(0, 0, 0);
}

// A rows of M16 tiles [m0, m1) for K chunk kc. rows_t is this thread's
// number of valid M16 tiles, so tail rows are never read.
__attribute__((always_inline)) inline void vs_load_a(
    half *av, __local__ f16v16 *buffer, short m0, short m1, short rows_t,
    short kc, short *lp) {
  const short end = m1 < rows_t ? m1 : rows_t;
  const short count = end - m0;
  LoopStepInfo ad[2] = {{&lp[1], VS_CK}, {&lp[0], 1}};
  FragVData va(buffer, ad);
  LoopStepInfo as[2] = {{&lp[1], (int)16 * (int)QKV_UP_K}, {&lp[0], (int)16}};
  FragSData sa(av + (int)kc * VS_CK * 16 +
                   ((int)m0 * 16 + (int)threadId) * QKV_UP_K, as);
#pragma rpu_hwloop_ul
  for (lp[1] = 0; lp[1] < count; ++lp[1]) {
#pragma rpu_hwloop_ul
    for (lp[0] = 0; lp[0] < VS_CK; ++lp[0])
      rpu::vldExecute_async(va, sa, (short)0);
  }
}

// Load W vectors [v, v + Count) of chunk kc, where v = tile * CK + k16.
// Keep Count a compile-time constant.
template <short Count>
__attribute__((always_inline)) inline void vs_load_w_run(
    DDR_large bg, __local__ f16v16 *dst, short tile, short k16, short kc,
    short *lp) {
  LoopStepInfo bd[1] = {{&lp[0], 1}};
  FragVData vb(dst, bd);
  LoopStepInfo bs[1] = {{&lp[0], (int)4096, rpu::__AGran::a_g1B}};
  FragSData sb(bg + ((int)tile * VS_AK + (int)kc * VS_CK + k16) * 16, bs);
#pragma rpu_hwloop_ul
  for (lp[0] = 0; lp[0] < Count; ++lp[0])
    rpu::vldExecute_async(vb, sb, (short)(threadId * 32));
}

// Phase p's share of chunk kc into the chunk buffer.
template <class S>
__attribute__((always_inline)) inline void vs_load_w(
    DDR_large bg, __local__ f16v16 *chunk, short p, short kc, short *lp) {
  if constexpr (S::BPP <= VS_CK) {
    const short v = p * S::BPP;
    vs_load_w_run<S::BPP>(bg, chunk + v, v / VS_CK, v % VS_CK, kc, lp);
  } else {
#pragma rpu_hwloop_ul
    for (lp[4] = 0; lp[4] < S::BPP / VS_CK; ++lp[4]) {
      const short tile = p * (S::BPP / VS_CK) + lp[4];
      vs_load_w_run<VS_CK>(bg, chunk + tile * VS_CK, tile, 0, kc, lp);
    }
  }
}

// Non-final chunks write FP32 partials; the final chunk rounds once into
// the pair's low FP16 slot. Configure the Repeat shape once per phase.
template <bool Final>
__attribute__((always_inline)) inline void vs_vmat_phase(
    __local__ i32v16 *acc, __local__ f16v16 *buffer, __local__ f16v16 *b,
    short m0, short count, short *lp) {
  auto *result = reinterpret_cast<__local__ f16v16 *>(acc);
  LoopStepInfo seed_steps[2] = {{&lp[2], 2 * VS_AN}, {&lp[1], 2}};
  FragVData seed_lo((__local__ i16v16 *)acc + 2 * m0 * VS_AN, seed_steps);
  FragVData seed_hi((__local__ i16v16 *)acc + 2 * m0 * VS_AN + 1, seed_steps);
  LoopOutInfo oi[1] = {{(short)0}};
  LoopOutLPAttr oa(oi);
  if constexpr (Final)
    vs_repeat_fp16();
  else
    vs_repeat_b32();
#pragma rpu_hwloop_ul
  for (lp[2] = 0; lp[2] < count; ++lp[2]) {
#pragma rpu_hwloop_ul
    for (lp[1] = 0; lp[1] < VS_AN; ++lp[1]) {
      const short idx = (m0 + lp[2]) * VS_AN + lp[1];
      rpu::vcfg_set_vmat(rpu::vcfg_vmat_target::acc_init_fp32_l, seed_lo);
      rpu::vcfg_set_vmat(rpu::vcfg_vmat_target::acc_init_fp32_h, seed_hi);
      if constexpr (Final)
        rpu::vmatRepeat_async(result + 2 * idx, buffer + lp[2] * VS_CK,
                              b + lp[1] * VS_CK, &oa);
      else
        rpu::vmatRepeat_async(acc + idx, buffer + lp[2] * VS_CK,
                              b + lp[1] * VS_CK, &oa);
    }
  }
}

// Seed every FP32 accumulator with the bias: acc = bias + sum((-0) * (+0)).
// Adding -0 is exact for every value, including -0, so the partial equals
// the FP16 bias widened to FP32, as acc_init_fp16 in the full-K kernel.
__attribute__((always_inline)) inline void vs_seed_bias(
    __local__ i32v16 *acc, __local__ f16v16 *biasv, __local__ f16v16 *zeros,
    short mt, short *lp) {
  LoopStepInfo bias_steps[1] = {{&lp[1], 1}};
  FragVData seed_bias(biasv, bias_steps);
  LoopOutInfo oi[1] = {{(short)0}};
  LoopOutLPAttr oa(oi);
  constexpr short format =
      rpu::acc_format::acc_fp32 | rpu::dst_format::dst_B32 |
      rpu::rs0_format::rs0_fp16 | rpu::rs1_format::rs1_fp16;
  __builtin_rhvm_vmat_set_tsize(1, 1, format);
  __builtin_rhvm_vmat_set_tstep0(0, 0, 0, format);
  __builtin_rhvm_vmat_set_tstep1(0, 0, 0, format);
#pragma rpu_hwloop_ul
  for (lp[2] = 0; lp[2] < mt; ++lp[2]) {
#pragma rpu_hwloop_ul
    for (lp[1] = 0; lp[1] < VS_AN; ++lp[1]) {
      rpu::vcfg_set_vmat(rpu::vcfg_vmat_target::acc_init_fp16, seed_bias);
      rpu::vmatRepeat_async(acc + lp[2] * VS_AN + lp[1], zeros, zeros + 1, &oa);
    }
  }
  rpu::fenceVmat(0);
}

// FP16 results (low slot of each accumulator pair) of M16 tiles [m0, m1)
// and `columns` N16 tiles from local tile `first` into a compact
// [M, stride] output. rows_t masks this thread's tail rows.
__attribute__((always_inline)) inline void vs_store(
    __local__ f16v16 *result, half *out, short first, short columns,
    short m0, short m1, short rows_t, int stride) {
  const short end = m1 < rows_t ? m1 : rows_t;
  const short count = end - m0;
  short row, column;
  LoopStepInfo rs[2] = {{&row, 2 * VS_AN}, {&column, 2}};
  FragVData vr(result + 2 * (m0 * VS_AN + first), rs);
  LoopStepInfo ds[2] = {{&row, (int)16 * stride}, {&column, (int)16}};
  FragSData output(out + ((int)m0 * 16 + (int)threadId) * stride, ds);
#pragma rpu_hwloop_ul
  for (row = 0; row < count; ++row) {
#pragma rpu_hwloop_ul
    for (column = 0; column < columns; ++column)
      rpu::vstExecute_async(vr, output, (short)0);
  }
}

// Rows [m0, m1) of this block's N range.
__attribute__((always_inline)) inline void vs_store_rows(
    __local__ i32v16 *acc, half *rv QKV_UP_OUTPUT_PARAMS, short n_base,
    short m0, short m1, short rows_t, __local__ f16v16 *scratch,
    unsigned short rows) {
  auto *result = reinterpret_cast<__local__ f16v16 *>(acc);
  QKV_UP_STREAM_EPILOGUE(result, scratch, n_base, m0, m1, rows);
#if QKV_UP_SPLIT_QKV
  // A block crosses at most one 128-column projection boundary: write two
  // segments; each value is stored exactly once.
  half *first = n_base < 8 ? rv : n_base < 16 ? kv : vv;
  const short local_n = n_base & 7;
  const short first_count = VS_AN < 8 - local_n ? VS_AN : 8 - local_n;
  vs_store(result, first + (int)local_n * 16, 0, first_count, m0, m1, rows_t, 128);
  if (first_count < VS_AN)
    vs_store(result, n_base < 8 ? kv : vv, first_count, VS_AN - first_count,
             m0, m1, rows_t, 128);
#else
  vs_store(result, rv + (int)n_base * 16, 0, VS_AN, m0, m1, rows_t, QKV_UP_N);
#endif
}

// One K chunk. The fences at the end of phase p also retire the VMAT that
// last read the A buffer phase p+1 overwrites.
template <class S, bool Final>
__attribute__((always_inline)) inline void vs_chunk(
    half *av, DDR_large bg, __local__ i32v16 *acc, __local__ f16v16 *a,
    __local__ f16v16 *b, short c, short mt, short rows_t, short *lp,
    half *rv QKV_UP_OUTPUT_PARAMS, short n_base, __local__ f16v16 *scratch,
    unsigned short rows) {
  auto *bcur = b + (c & 1) * VS_AN * VS_CK;
  auto *bnext = b + ((c + 1) & 1) * VS_AN * VS_CK;
  constexpr short P = S::Phases;
#pragma rpu_hwloop_ul
  for (lp[3] = 0; lp[3] < P; ++lp[3]) {
    const short phase = c * P + lp[3];
    const short m0 = lp[3] * mt / P, m1 = (lp[3] + 1) * mt / P;
    const bool wrap = lp[3] == P - 1;
    if constexpr (!Final) vs_load_w<S>(bg, bnext, lp[3], c + 1, lp);
    if (!(Final && wrap)) {
      const short n0 = wrap ? 0 : m1;
      const short n1 = wrap ? mt / P : (lp[3] + 2) * mt / P;
      if (n1 > n0)
        vs_load_a(av, a + ((phase + 1) & 1) * S::PM * VS_CK, n0, n1, rows_t,
                  wrap ? c + 1 : c, lp);
    }
    if (m1 > m0)
      vs_vmat_phase<Final>(acc, a + (phase & 1) * S::PM * VS_CK, bcur, m0,
                           m1 - m0, lp);
    // The previous phase's rows were rounded before the last fence.
    if (Final && lp[3] > 0) {
      const short s0 = (lp[3] - 1) * mt / P;
      if (m0 > s0)
        vs_store_rows(acc, rv QKV_UP_OUTPUT_ARGS, n_base, s0, m0, rows_t, scratch, rows);
    }
    rpu::fenceVld(0);
    rpu::fenceVmat(0);
  }
}

template <class S>
__attribute__((always_inline)) inline void qkv_up_k_stream(
    half *av, DDR_large wv, half *rv, half *bias, unsigned short rows,
    __local__ f16v16 *arena QKV_UP_OUTPUT_PARAMS) {
  constexpr short chunks = VS_AK / VS_CK, P = S::Phases;
  const short n_base = (short)blockIdx.x * VS_AN;
  const short mt = (rows + 15) / 16;
  const short rows_t = (rows + 15 - threadId) / 16;
  DDR_large bg = wv + (int)n_base * VS_AK * 16 + (int)coreId_u * 2;
  auto *acc = reinterpret_cast<__local__ i32v16 *>(arena);
  auto *b = arena + 2 * S::MT * VS_AN;
  auto *a = b + 2 * VS_AN * VS_CK;
  auto *biasv = a + 2 * S::PM * VS_CK;
  auto *zeros = biasv + VS_AN;
  auto *scratch = zeros + 2;
  short lp[5];

  // Prologue: all of W chunk 0, the bias tile and A for chunk 0 phase 0.
  short p;
#pragma rpu_hwloop_ul
  for (p = 0; p < P; ++p) vs_load_w<S>(bg, b, p, 0, lp);
  LoopStepInfo bias_dst[1] = {{&lp[0], 1}};
  FragVData vbias(biasv, bias_dst);
  LoopStepInfo bias_src[1] = {{&lp[0], (int)16}};
  FragSData sbias(bias + (int)n_base * 16, bias_src);
#pragma rpu_hwloop_ul
  for (lp[0] = 0; lp[0] < VS_AN; ++lp[0])
    rpu::vldExecute_async(vbias, sbias, (short)0);
  if (mt / P > 0) vs_load_a(av, a, 0, mt / P, rows_t, 0, lp);
  // VSFU bypass copies the scalar bits: -0 for the A side, +0 for W.
  rpu::vsfuExecute(zeros, (half)-0.0f, (half)0.0f, rpu::__ShuffleMode::Bypass);
  rpu::vsfuExecute(zeros + 1, (half)0.0f, (half)0.0f, rpu::__ShuffleMode::Bypass);
  rpu::fenceVld(0);
  vs_seed_bias(acc, biasv, zeros, mt, lp);

  short c;
#pragma rpu_hwloop_ul
  for (c = 0; c < chunks - 1; ++c)
    vs_chunk<S, false>(av, bg, acc, a, b, c, mt, rows_t, lp, rv QKV_UP_OUTPUT_ARGS, n_base,
                       scratch, rows);
  vs_chunk<S, true>(av, bg, acc, a, b, chunks - 1, mt, rows_t, lp, rv QKV_UP_OUTPUT_ARGS, n_base,
                    scratch, rows);
  vs_store_rows(acc, rv QKV_UP_OUTPUT_ARGS, n_base, (P - 1) * mt / P, mt, rows_t, scratch, rows);
  rpu::fenceVst(0);
}

__rprog__ void QKV_UP_ENTRY(half *av, DDR_large wv, half *rv, half *bias,
                            unsigned short rows QKV_UP_OUTPUT_PARAMS) {
  // Use one explicit scratch allocation shared by every path.
  // Each path ignores blocks beyond N, including for a larger supplied grid.
  __local__ f16v16 arena[VS_ARENA];
  QKV_UP_ENTRY_PROLOGUE
#if defined(QKV_UP_PIPELINE_ROWS)
  if (rows <= QKV_UP_PIPELINE_ROWS) {
    // Small VMAT base offsets first: partials, A, then W.
    auto *r = arena, *a = r + VS_PIPE_AM * (QKV_UP_TN / 16);
    auto *b = a + 2 * VS_PIPE_AM * VS_CK;
    qkv_up_k_pipeline(av, wv, rv, bias, rows, a, b, r QKV_UP_OUTPUT_ARGS);
    return;
  }
#elif defined(QKV_UP_SMALL_ROWS)
  if (rows <= QKV_UP_SMALL_ROWS) {
    if ((short)blockIdx.x * VS_AN >= QKV_UP_N / 16) return;
    qkv_up_k_stream<VsSmall>(av, wv, rv, bias, rows, arena QKV_UP_OUTPUT_ARGS);
    return;
  }
#endif
  if (rows <= QKV_UP_STREAM_ROWS) {
    if ((short)blockIdx.x * VS_AN >= QKV_UP_N / 16) return;
    qkv_up_k_stream<VsLarge>(av, wv, rv, bias, rows, arena QKV_UP_OUTPUT_ARGS);
  } else {
    if ((short)blockIdx.x * VS_FULL_AN >= QKV_UP_N / 16) return;
    // Small VMAT base offsets first: output, bias, A, then W.
    auto *r = arena, *biasv = r + VS_FULL_AM * VS_FULL_AN;
    auto *a = biasv + VS_FULL_AN, *b = a + VS_FULL_AM * VS_AK;
    qkv_up_full_k<VS_FULL_AM>(av, wv, rv, bias, rows, a, b, r, biasv
                              QKV_UP_OUTPUT_ARGS);
  }
}
