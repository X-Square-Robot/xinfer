// Copyright 2026 X-Square Robot. SPDX-License-Identifier: Apache-2.0
#include "__clang_rpu_builtin_vars.h"

#if (STREAMED_N != 1024 && STREAMED_N != 1056) || STREAMED_K != 2048 || \
    STREAMED_TN != 48 || STREAMED_TM != 384 || STREAMED_TK != 128 || \
    STREAMED_PIPELINE != 6
#error "Streamed QKV requires N1024/1056 K2048 M384 N48 K128 pipeline 6"
#endif

#if STREAMED_QKV_ZBA
#define STREAMED_QKV_OUTPUT_PARAMETERS \
  half *qv, half *kv, half *vv, half *zv, half *bv, half *av_out
#define STREAMED_QKV_OUTPUT_ARGUMENTS qv, kv, vv, zv, bv, av_out
#else
#define STREAMED_QKV_OUTPUT_PARAMETERS \
  half *qv, half *qgatev, half *kv, half *vv
#define STREAMED_QKV_OUTPUT_ARGUMENTS qv, qgatev, kv, vv
#endif

// FP16 A/W and output with FP32 accumulation. One K128 chunk occupies 192 A
// vectors plus 24 B vectors. The 72 FP32 accumulators occupy 144 FP16-vector
// slots, for 360/400 total VLM slots. Each accumulator owns the adjacent
// half-vector pair [2*idx, 2*idx+1]. On the final K chunk VMAT writes the FP16
// result only to the low half of that same pair; the store pass gathers those
// low halves with a stride of two. This avoids a dense FP16 result overlay on
// still-live FP32 accumulator pairs.

__attribute__((always_inline)) inline void streamed_qkv_load_k(
    __local__ f16v16 *a, __local__ f16v16 *b, half *av, DDR_large bg,
    short k0, short valid_m, short n_count) {
  const short CK = STREAMED_TK / 16, AK = STREAMED_K / 16;
  short lp[2];
  LoopStepInfo ad[2] = {{&lp[1], CK}, {&lp[0], 1}};
  FragVData va(a, ad);
  LoopStepInfo as[2] = {
      {&lp[1], ((int)16 * STREAMED_K)}, {&lp[0], (int)16}};
  FragSData sa(av + (int)k0 * 16 + (int)threadId * STREAMED_K, as);
#pragma rpu_hwloop_ul
  for (lp[1] = 0; lp[1] < valid_m; ++lp[1]) {
#pragma rpu_hwloop_ul
    for (lp[0] = 0; lp[0] < CK; ++lp[0])
      rpu::vldExecute_async(va, sa, (short)0);
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

template <typename OutputPtr>
__attribute__((always_inline)) inline void streamed_qkv_repeat4_tile(
    __local__ f16v16 *a_row, __local__ f16v16 *b_tile,
    __local__ i32v16 *acc_tile, __local__ i16v16 *acc_seed,
    OutputPtr second_output) {
  // Configure Repeat4 for each half of K128.
  // The first command retains its FP32 result inside VMAT; the second writes
  // either the next FP32 partial or the final FP16 output selected by the
  // pointer type of second_output.
  short kl0, kl1;
  LoopStepInfo fixed0[1] = {{&kl0, 0}}, reduce0[1] = {{&kl0, 1}};
  LoopStepInfo fixed1[1] = {{&kl1, 0}}, reduce1[1] = {{&kl1, 1}};
  FragVData first_half(acc_tile, fixed0);
  FragVData second_half(second_output, fixed1);
  FragVData aa0(a_row, reduce0), bb0(b_tile, reduce0);
  FragVData aa1(a_row + 4, reduce1), bb1(b_tile + 4, reduce1);
  LoopOutInfo output_info[1] = {{&kl1}};
  LoopOutLPAttr output(output_info);

  // All output tiles in one K128 chunk seed from disjoint VLM accumulator
  // pairs produced by the preceding, already-fenced chunk. VMAT snapshots the
  // two acc-init addresses with each queued command, so changing them for the
  // next output tile does not require draining the current tile. The caller
  // fences once after the complete chunk before any partial is consumed.
  rpu::vcfg_set_vmat(rpu::vcfg_vmat_target::acc_init_fp32_l, acc_seed);
  rpu::vcfg_set_vmat(rpu::vcfg_vmat_target::acc_init_fp32_h, acc_seed + 1);
#pragma rpu_hwloop_ul
  for (kl0 = 0; kl0 < 4; ++kl0)
    rpu::vmatExecute_async(first_half, aa0, bb0, rpu::__OutMode::outoff);
#pragma rpu_hwloop_ul
  for (kl1 = 0; kl1 < 4; ++kl1)
    rpu::vmatExecute_async(second_half, aa1, bb1, &output);
}

__attribute__((always_inline)) inline void streamed_qkv_store_output_vector(
    __local__ f16v16 *result_half, short result_vector, half *dst,
    short dst_vector, short dst_columns, short valid_m) {
  constexpr short AN = STREAMED_TN / 16;
  short row;
  LoopStepInfo result_steps[1] = {{&row, 2 * AN}};
  FragVData result(result_half + 2 * result_vector, result_steps);
  LoopStepInfo dst_steps[1] = {{&row, (int)16 * dst_columns}};
  FragSData output(dst + (int)threadId * dst_columns +
                         (int)dst_vector * 16,
                   dst_steps);
#pragma rpu_hwloop_ul
  for (row = 0; row < valid_m; ++row)
    rpu::vstExecute_async(result, output, (short)0);
}

__attribute__((always_inline)) inline void streamed_qkv_store_planar(
    __local__ f16v16 *result_half, short result_vector, short global_vector,
    short valid_m, STREAMED_QKV_OUTPUT_PARAMETERS) {
#if STREAMED_QKV_ZBA
  if (global_vector < 16)
    streamed_qkv_store_output_vector(result_half, result_vector, qv,
                                 global_vector, 256, valid_m);
  else if (global_vector < 32)
    streamed_qkv_store_output_vector(result_half, result_vector, kv,
                                 global_vector - 16, 256, valid_m);
  else if (global_vector < 48)
    streamed_qkv_store_output_vector(result_half, result_vector, vv,
                                 global_vector - 32, 256, valid_m);
  else if (global_vector < 64)
    streamed_qkv_store_output_vector(result_half, result_vector, zv,
                                 global_vector - 48, 256, valid_m);
  else if (global_vector == 64)
    streamed_qkv_store_output_vector(result_half, result_vector, bv,
                                 0, 16, valid_m);
  else
    streamed_qkv_store_output_vector(result_half, result_vector, av_out,
                                 0, 16, valid_m);
#else
  if (global_vector < 16)
    streamed_qkv_store_output_vector(result_half, result_vector, qv,
                                 global_vector, 256, valid_m);
  else if (global_vector < 32)
    streamed_qkv_store_output_vector(result_half, result_vector, qgatev,
                                 global_vector - 16, 256, valid_m);
  else if (global_vector < 48)
    streamed_qkv_store_output_vector(result_half, result_vector, kv,
                                 global_vector - 32, 256, valid_m);
  else
    streamed_qkv_store_output_vector(result_half, result_vector, vv,
                                 global_vector - 48, 256, valid_m);
#endif
}

template <bool full_m384>
__attribute__((always_inline)) inline void streamed_qkv_k_pipeline(
    half *av, DDR_large wv, STREAMED_QKV_OUTPUT_PARAMETERS,
    unsigned short rows, __local__ f16v16 *a, __local__ f16v16 *b,
    __local__ i32v16 *acc) {
  const short AK = STREAMED_K / 16, CK = STREAMED_TK / 16;
  const short AN = STREAMED_TN / 16, AM = STREAMED_TM / 16;
  // The host ABI accepts complete 64-row physical buckets.
  // Every lane owns rows / 16 tiles; keep the VMAT loop bound warp-uniform.
  const short live_m = rows / 16;
  const short valid_m = full_m384 ? AM : (live_m < AM ? live_m : AM);
  const short n_base = (short)blockIdx.x * AN;
  const short remaining_n = STREAMED_N / 16 - n_base;
  const short n_count = remaining_n < AN ? remaining_n : AN;
  short lp[3];

  auto *acc_half = reinterpret_cast<__local__ i16v16 *>(acc);
  auto *result_half = reinterpret_cast<__local__ f16v16 *>(acc);
  LoopStepInfo zero_step[1] = {{&lp[0], 1}};
  FragVData zero_acc(acc_half, zero_step);
#pragma rpu_hwloop_ul
  for (lp[0] = 0; lp[0] < 2 * valid_m * AN; ++lp[0])
    rpu::valuExecute_async(zero_acc, (short)0, (short)0,
                           rpu::__ValuOpc::add);
  rpu::fenceValu(0);

  DDR_large bg = wv + (int)n_base * AK * 16 + (int)coreId_u * 2;
  // Keep the final K128 chunk outside the repeated partial loop.
  // Use the FP32 VMAT mode for both commands in each partial chunk.
#pragma rpu_hwloop_ul
  for (short k0 = 0; k0 < AK - CK; k0 += CK) {
    streamed_qkv_load_k(a, b, av, bg, k0, valid_m, n_count);
    rpu::fenceVld(0);
    if (n_count == AN) {
#pragma rpu_hwloop_ul
      for (short mi = 0; mi < valid_m; ++mi) {
        const short row_idx = mi * AN;
        streamed_qkv_repeat4_tile(a + mi * CK, b, acc + row_idx,
                              acc_half + 2 * row_idx, acc + row_idx);
        streamed_qkv_repeat4_tile(a + mi * CK, b + CK, acc + row_idx + 1,
                              acc_half + 2 * (row_idx + 1),
                              acc + row_idx + 1);
        streamed_qkv_repeat4_tile(a + mi * CK, b + 2 * CK, acc + row_idx + 2,
                              acc_half + 2 * (row_idx + 2),
                              acc + row_idx + 2);
      }
    } else {
#pragma rpu_hwloop_ul
      for (short mi = 0; mi < valid_m; ++mi) {
        const short row_idx = mi * AN;
        streamed_qkv_repeat4_tile(a + mi * CK, b, acc + row_idx,
                              acc_half + 2 * row_idx, acc + row_idx);
#if STREAMED_N == 1040
        streamed_qkv_repeat4_tile(a + mi * CK, b + CK, acc + row_idx + 1,
                              acc_half + 2 * (row_idx + 1),
                              acc + row_idx + 1);
#endif
      }
    }
    rpu::fenceVmat(0);
    rpu::fenceVld(0);
  }

  const short final_k0 = AK - CK;
  streamed_qkv_load_k(a, b, av, bg, final_k0, valid_m, n_count);
  rpu::fenceVld(0);
  if (n_count == AN) {
#pragma rpu_hwloop_ul
    for (short mi = 0; mi < valid_m; ++mi) {
      const short row_idx = mi * AN;
      streamed_qkv_repeat4_tile(a + mi * CK, b, acc + row_idx,
                            acc_half + 2 * row_idx,
                            result_half + 2 * row_idx);
      streamed_qkv_repeat4_tile(a + mi * CK, b + CK, acc + row_idx + 1,
                            acc_half + 2 * (row_idx + 1),
                            result_half + 2 * (row_idx + 1));
      streamed_qkv_repeat4_tile(a + mi * CK, b + 2 * CK, acc + row_idx + 2,
                            acc_half + 2 * (row_idx + 2),
                            result_half + 2 * (row_idx + 2));
    }
  } else {
#pragma rpu_hwloop_ul
    for (short mi = 0; mi < valid_m; ++mi) {
      const short row_idx = mi * AN;
      streamed_qkv_repeat4_tile(a + mi * CK, b, acc + row_idx,
                            acc_half + 2 * row_idx,
                            result_half + 2 * row_idx);
#if STREAMED_N == 1040
      streamed_qkv_repeat4_tile(a + mi * CK, b + CK, acc + row_idx + 1,
                            acc_half + 2 * (row_idx + 1),
                            result_half + 2 * (row_idx + 1));
#endif
    }
  }
  rpu::fenceVmat(0);
  rpu::fenceVld(0);

  // Each final FP16 vector is the low half of its FP32 pair. Route each
  // N16 vector directly into its framework-owned contiguous output plane.
  streamed_qkv_store_planar(result_half, 0, n_base, valid_m,
                        STREAMED_QKV_OUTPUT_ARGUMENTS);
  if (n_count > 1)
    streamed_qkv_store_planar(result_half, 1, n_base + 1, valid_m,
                          STREAMED_QKV_OUTPUT_ARGUMENTS);
  if (n_count > 2)
    streamed_qkv_store_planar(result_half, 2, n_base + 2, valid_m,
                          STREAMED_QKV_OUTPUT_ARGUMENTS);
  rpu::fenceVst(0);
}

__rprog__ void STREAMED_ENTRY(half *av, DDR_large wv,
                           STREAMED_QKV_OUTPUT_PARAMETERS,
                           unsigned short rows) {
  const short AM = STREAMED_TM / 16, AN = STREAMED_TN / 16;
  const short CK = STREAMED_TK / 16;
  __local__ f16v16 a[AM * CK], b[AN * CK];
  __local__ i32v16 acc[AM * AN];
  if (rows == STREAMED_TM)
    streamed_qkv_k_pipeline<true>(av, wv, STREAMED_QKV_OUTPUT_ARGUMENTS,
                              rows, a, b, acc);
  else
    streamed_qkv_k_pipeline<false>(av, wv, STREAMED_QKV_OUTPUT_ARGUMENTS,
                               rows, a, b, acc);
}

#undef STREAMED_QKV_OUTPUT_PARAMETERS
#undef STREAMED_QKV_OUTPUT_ARGUMENTS
