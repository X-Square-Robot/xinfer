// Copyright 2026 X-Square Robot. SPDX-License-Identifier: Apache-2.0
#include "__clang_rpu_builtin_vars.h"

#ifndef IO_TN
#define IO_TN 32
#endif
#if IO_N % IO_TN || IO_TN % 16 || IO_K % 16
#error "Action I/O GEMM requires complete vector tiles"
#endif

// Fixed M32, single-core FP16/ACC32 projection.  A/C/bias are compact
// LocalSPM tensors and W is the num_cores=1 tp_col_swizzle_mc_weight layout:
// [N/16][K/16][16 N][16 K].
// Pointer ABI: A[0:1], W/256[2:3], C[4:5], bias[6:7].
__rprog__ void IO_ENTRY(
    half *av, DDR_large wv, half *cv, half *bias) {
  const short vlen = 16;
  const short AM = 2;
  const short AN = IO_TN / 16;
  const short AK = IO_K / 16;
  const short n0 = (short)blockIdx.x * AN;

  // Load A, W and bias together and accumulate in FP32.
  __local__ f16v16 a[AM * AK];
  __local__ f16v16 b[AN * AK];
  __local__ f16v16 r[AM * AN];
  __local__ f16v16 biasv[AN];
  short loop[3];

  LoopStepInfo a_dst_steps[2] = {
      {&loop[1], AK}, {&loop[0], (short)1}};
  FragVData a_dst(a, a_dst_steps);
  LoopStepInfo a_src_steps[2] = {
      {&loop[1], (int)(vlen * IO_K)},
      {&loop[0], (int)vlen}};
  FragSData a_src(av, a_src_steps);
#pragma rpu_hwloop_ul
  for (loop[1] = 0; loop[1] < AM; ++loop[1]) {
#pragma rpu_hwloop_ul
    for (loop[0] = 0; loop[0] < AK; ++loop[0])
      rpu::vldExecute_async(
          a_dst, a_src, (short)(threadId * IO_K));
  }

  // With one transformed-weight core, each 16x16 block occupies 512 bytes.
  // DDR_large arithmetic is in 256-byte units.
  DDR_large weight = wv + (int)n0 * AK * 2;
  LoopStepInfo b_dst_steps[2] = {
      {&loop[1], AK}, {&loop[0], (short)1}};
  FragVData b_dst(b, b_dst_steps);
  LoopStepInfo b_src_steps[2] = {
      {&loop[1], (int)AK * 512, rpu::__AGran::a_g1B},
      {&loop[0], (int)512, rpu::__AGran::a_g1B}};
  FragSData b_src(weight, b_src_steps);
#pragma rpu_hwloop_ul
  for (loop[1] = 0; loop[1] < AN; ++loop[1]) {
#pragma rpu_hwloop_ul
    for (loop[0] = 0; loop[0] < AK; ++loop[0])
      rpu::vldExecute_async(
          b_dst, b_src, (short)(threadId * vlen * sizeof(half)));
  }

  LoopStepInfo bias_dst_steps[1] = {{&loop[0], (short)1}};
  FragVData bias_dst(biasv, bias_dst_steps);
  LoopStepInfo bias_src_steps[1] = {{&loop[0], (int)vlen}};
  FragSData bias_src(bias + (int)n0 * vlen, bias_src_steps);
#pragma rpu_hwloop_ul
  for (loop[0] = 0; loop[0] < AN; ++loop[0])
    rpu::vldExecute_async(bias_dst, bias_src, (short)0);
  rpu::fenceVld((short)0);

  LoopStepInfo c_steps[3] = {
      {&loop[2], AN}, {&loop[1], (short)1}, {&loop[0], (short)0}};
  FragVData c_dst(r, c_steps);
  LoopStepInfo ac_steps[3] = {
      {&loop[2], AK}, {&loop[1], (short)0}, {&loop[0], (short)1}};
  FragVData ac(a, ac_steps);
  LoopStepInfo bc_steps[3] = {
      {&loop[2], (short)0}, {&loop[1], AK}, {&loop[0], (short)1}};
  FragVData bc(b, bc_steps);
  LoopStepInfo init_steps[1] = {{&loop[1], (short)1}};
  FragVData init(biasv, init_steps);
  LoopOutInfo out_info[1] = {{&loop[0]}};
  LoopOutLPAttr out_attr(out_info);
#pragma rpu_hwloop_ul
  for (loop[2] = 0; loop[2] < AM; ++loop[2]) {
#pragma rpu_hwloop_ul
    for (loop[1] = 0; loop[1] < AN; ++loop[1]) {
      rpu::vcfg_set_vmat(rpu::vcfg_vmat_target::acc_init_fp16, init);
#pragma rpu_hwloop_ul
      for (loop[0] = 0; loop[0] < AK; ++loop[0])
        rpu::vmatExecute_async(
            c_dst, ac, bc, &out_attr, rpu::__ClrMode::clroff);
    }
  }
  rpu::fenceVmat((short)0);

#ifdef IO_OUTPUT_SCALE
  // The Action input path exposes the projection output as FP16 before its
  // residual-stream scale. r[] is already the rounded VMAT result, so this
  // fused epilogue preserves that boundary and removes only the extra launch.
  rpu::valuSetTsize(AM * AN, 1);
  // Keep the scalar factor fixed across all 16-lane output chunks.
  rpu::valuSetTstep0(1, 1, 0);
  rpu::valuSetTstep1(0, 0, 0);
  rpu::valuRepeat(r, r, (half)IO_OUTPUT_SCALE, rpu::__ValuOpc::mul);
#endif

  LoopStepInfo r_src_steps[2] = {
      {&loop[1], AN}, {&loop[0], (short)1}};
  FragVData r_src(r, r_src_steps);
  LoopStepInfo r_dst_steps[2] = {
      {&loop[1], (int)(vlen * IO_N)},
      {&loop[0], (int)vlen}};
  FragSData r_dst(cv + (int)n0 * vlen, r_dst_steps);
#pragma rpu_hwloop_ul
  for (loop[1] = 0; loop[1] < AM; ++loop[1]) {
#pragma rpu_hwloop_ul
    for (loop[0] = 0; loop[0] < AN; ++loop[0])
      rpu::vstExecute_async(
          r_src, r_dst, (short)(threadId * IO_N));
  }
  rpu::fenceVst((short)0);
}
