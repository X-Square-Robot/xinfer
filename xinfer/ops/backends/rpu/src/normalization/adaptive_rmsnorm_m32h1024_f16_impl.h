// Copyright 2026 X-Square Robot. SPDX-License-Identifier: Apache-2.0
#pragma once
#include "__clang_rpu_builtin_vars.h"
using namespace rpu;

// Action adaptive normalization, per core:
//   y = rmsnorm(x, modulation[0:1024]) + modulation[1024:2048]
// for the exact local FP16 shape [32,1024]. VMAT accumulates the square sum
// in FP32. Reciprocal square root and the RMSNorm / gamma multiplications
// use hardware opcodes (__ValuOpc::sqrt_reciprocal + valuRepeat), so the
// multiply-by-rsqrt, multiply-by-weight, and shift add remain distinct
// FP16 VLM boundaries, matching the unfused operator chain.
__rprog__ void XINFER_ACTION_NORM_KERNEL(
    half *dst, half *src, half *modulation,
    unsigned short mean_scale_low, unsigned short mean_scale_high,
    half epsilon) {
  constexpr short chunks = 64;
  // Each block owns 16 complete rows; two blocks cover 32 rows. Keep the
  // bounds guard warp-uniform so every lane participates in the VMAT
  // square-sum operations below.
  if ((unsigned short)blockIdx.x >= 2) return;
  unsigned short row = (unsigned short)blockIdx.x * 16 + threadId;

  __local__ f16v16 x[chunks], gamma[chunks], shift[chunks];
  __local__ f16v16 gram[1], norm[1], scratch[1];

  vcfg_set_valu(vcfg_valu_target::imm0, (half)1.0f);
  vcfg_set_valu(vcfg_valu_target::imm1, (half)1.0f);
  vcfg_set_valu(vcfg_valu_target::imm2, (half)1.0f);
  vcfg_set_vmat(vcfg_vmat_target::acc_init_fp32_l, (short)0);
  vcfg_set_vmat(vcfg_vmat_target::acc_init_fp32_h, (short)0);
  vcfg_set_vmat(vcfg_vmat_target::scale_allch_l, (short)mean_scale_low);
  vcfg_set_vmat(vcfg_vmat_target::scale_allch_h, (short)mean_scale_high);

  vldSetTsize32B(chunks, 1);
  vldSetTstep32B_0(1, 16);
  vldSetTstep32B_1(0, 0);
  vldRepeat_async(x, src, (int)row * 1024);
  vldRepeat_async(gamma, modulation, (int)0);
  vldRepeat_async(shift, modulation + 1024, (int)0);
  fenceVld(0);

  vmatSetTsize(chunks, 1);
  vmatSetTstep0(0, 1, 1);
  vmatSetTstep1(0, 0, 0);
  LoopOutInfo outlp[1] = {{(short)0}};
  LoopClrInfo clrlp[1] = {{(short)0}};
  LoopOutLPAttr out(outlp);
  LoopClrLPAttr clr(clrlp);
  vmatRepeat(gram, x, x, &out, &clr);

  half mean_square;
  vcfg_set_valu(vcfg_valu_target::imm2, gram, (short)threadId);
  valuExecute(scratch, (half)1.0f, (half)0.0f, __ValuOpc::add);
  vcfg_set_valu(vcfg_valu_target::imm2, (half)1.0f);
  vsfuScalar(mean_square, scratch, (short)0, __ShuffleMode::Psel);
  const unsigned short bits = reinterpret_cast<unsigned short&>(mean_square);
  const unsigned short any_small = sumint(bits);
  const unsigned short any_large = sumint((unsigned short)(0xffffu - bits));

  valuSetTsize(chunks, 1);
  valuSetTstep0(1, 1, 0);
  valuSetTstep1(0, 0, 0);
  half adjusted_epsilon = epsilon;
  if (any_small < 0x0400u || any_large <= 0x83ffu) {
    half scale = (half)1.0f;
    if (bits < 0x0400u) {
      scale = (half)128.0f;
      adjusted_epsilon = epsilon * (half)16384.0f;
    } else if (bits >= 0x7c00u) {
      scale = (half)0.00390625f;
      adjusted_epsilon = (half)0.0f;
    }
    valuRepeat(x, x, scale, __ValuOpc::mul);
    vmatRepeat(gram, x, x, &out, &clr);
    vcfg_set_valu(vcfg_valu_target::imm2, gram, (short)threadId);
    valuExecute(scratch, (half)1.0f, (half)0.0f, __ValuOpc::add);
    vcfg_set_valu(vcfg_valu_target::imm2, (half)1.0f);
    vsfuScalar(mean_square, scratch, (short)0, __ShuffleMode::Psel);
  }

  valuExecute(norm, mean_square, adjusted_epsilon, __ValuOpc::add);
  valuExecute(norm, norm, (half)0.0f, __ValuOpc::sqrt_reciprocal);
  valuRepeat(x, x, norm, __ValuOpc::mul);
  valuSetTstep0(1, 1, 1);
  valuRepeat(x, x, gamma, __ValuOpc::mul);
  valuRepeat(x, x, shift, __ValuOpc::add);

  vstSetTsize32B(chunks, 1);
  vstSetTstep32B_0(1, 16);
  vstSetTstep32B_1(0, 0);
  vstRepeat(x, dst, (int)row * 1024);
}
