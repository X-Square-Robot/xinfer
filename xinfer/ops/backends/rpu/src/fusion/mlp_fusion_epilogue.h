// Copyright 2026 X-Square Robot. SPDX-License-Identifier: Apache-2.0
#pragma once
// VLM epilogues shared by the Vision fused GEMM entries. Each
// repeats the exact FP16 VALU sequence of the standalone kernel it replaces
// (gelu_f16 tanh fast path, rope_f16 vision path) on the rounded FP16 GEMM
// output, so a fused result is bit-identical to GEMM -> SPM -> epilogue.
#include "__clang_rpu_builtin_vars.h"

// gelu_f16's VALU scale registers: unit input scales for add (imm0, imm1);
// imm2 is the VALU output scale, left at 1 after every epilogue.
__attribute__((always_inline)) inline void vf_gelu_config() {
  rpu::vcfg_set_valu(rpu::vcfg_valu_target::imm0, (half)1.0f);
  rpu::vcfg_set_valu(rpu::vcfg_valu_target::imm1, (half)1.0f);
  rpu::vcfg_set_valu(rpu::vcfg_valu_target::imm2, (half)1.0f);
}

// x[i * XS] = gelu_tanh(x[i * XS]) for i < n, scratch t[i * TS]:
// x * sigmoid((1 + 0.044715 x^2) * 2 sqrt(2/pi) * x), gelu_f16's Repeat order.
template <short XS, short TS>
__attribute__((always_inline)) inline void vf_gelu(
    __local__ f16v16 *x, __local__ f16v16 *t, short n) {
  if (n <= 0) return;
  rpu::valuSetTsize(n, 1);
  rpu::valuSetTstep1(0, 0, 0);
  rpu::valuSetTstep0(TS, XS, XS);
  rpu::vcfg_set_valu(rpu::vcfg_valu_target::imm2, (half)0.044715f);
  rpu::valuRepeat(t, x, x, rpu::__ValuOpc::mul);
  rpu::vcfg_set_valu(rpu::vcfg_valu_target::imm2, (half)1.5957691216f);
  rpu::valuSetTstep0(TS, TS, 0);
  rpu::valuRepeat(t, t, (half)1.0f, rpu::__ValuOpc::add);
  rpu::vcfg_set_valu(rpu::vcfg_valu_target::imm2, (half)1.0f);
  rpu::valuSetTstep0(TS, TS, XS);
  rpu::valuRepeat(t, t, x, rpu::__ValuOpc::mul);
  rpu::valuSetTstep0(TS, TS, 0);
  rpu::valuRepeat(t, t, (half)0.0f, rpu::__ValuOpc::sigmoid);
  rpu::valuSetTstep0(XS, XS, TS);
  rpu::valuRepeat(x, x, t, rpu::__ValuOpc::mul);
}

// Position-selected [table_rows, 32] FP16 cos/sin rows of M16 tiles
// [m_first, m_first + count): c[2m + j] = cos[token][16j..16j+15]. Tail
// threads re-read the last valid row instead of reading past the table.
__attribute__((always_inline)) inline void vf_rope_tables(
    half *cos_table, half *sin_table, __local__ f16v16 *c,
    __local__ f16v16 *s, short m_first, short count, unsigned short rows) {
#pragma rpu_hwloop_ul
  for (short m = 0; m < count; ++m) {
    unsigned short token = (unsigned short)((m_first + m) * 16) + threadId;
    unsigned short row = token < rows ? token : rows - 1;
    int offset = (int)row * 32;
    rpu::vldExecute_async(c + 2 * m, cos_table, offset);
    rpu::vldExecute_async(c + 2 * m + 1, cos_table + 16, offset);
    rpu::vldExecute_async(s + 2 * m, sin_table, offset);
    rpu::vldExecute_async(s + 2 * m + 1, sin_table + 16, offset);
  }
  rpu::fenceVld(0);
}

// Rotate-half RoPE of `count` M16 tiles of one 64-column head in place:
// x[m][n] at lo[m * LM + n * LN] (n = 0..3, column 16n..16n+15), scratch
// hi[] with the same geometry, cos/sin c[2m + j], s[2m + j]. rope_f16 forms
// ac = x_j c_j, bs = x_{j+2} s_j, as = x_j s_j, bc = x_{j+2} c_j, then
// ac - bs and bc + as, each rounded to FP16; this applies the same products
// in the same order and moves ac - bs back with an exact multiply by one.
template <short LN, short LM>
__attribute__((always_inline)) inline void vf_rope(
    __local__ f16v16 *lo, __local__ f16v16 *hi, __local__ f16v16 *c,
    __local__ f16v16 *s, short count) {
  if (count <= 0) return;
  rpu::valuSetTsize(2, count);
  rpu::valuSetTstep0(LN, LN, 1);
  rpu::valuSetTstep1(LM, LM, 2);
  rpu::valuRepeat(hi, lo, c, rpu::__ValuOpc::mul);                    // ac
  rpu::valuRepeat(hi + 2 * LN, lo + 2 * LN, s, rpu::__ValuOpc::mul);  // bs
  rpu::valuSetTstep0(LN, LN, LN);
  rpu::valuSetTstep1(LM, LM, LM);
  rpu::valuRepeat(hi, hi, hi + 2 * LN, rpu::__ValuOpc::sub);          // ac - bs
  rpu::valuSetTstep0(LN, LN, 1);
  rpu::valuSetTstep1(LM, LM, 2);
  rpu::valuRepeat(hi + 2 * LN, lo + 2 * LN, c, rpu::__ValuOpc::mul);  // bc
  rpu::valuRepeat(lo + 2 * LN, lo, s, rpu::__ValuOpc::mul);           // as
  rpu::valuSetTstep0(LN, LN, LN);
  rpu::valuSetTstep1(LM, LM, LM);
  rpu::valuRepeat(lo + 2 * LN, hi + 2 * LN, lo + 2 * LN,
                  rpu::__ValuOpc::add);                                // bc + as
  rpu::valuSetTstep0(LN, LN, 0);
  rpu::valuSetTstep1(LM, LM, 0);
  rpu::valuRepeat(lo, hi, (half)1.0f, rpu::__ValuOpc::mul);
}

// x[i] = gelu_erf(x[i]) for i < n with scratch t[0..3]: gelu_f16's
// approximate=0 per-vector sequence (A&S 26.2.17 normal CDF, evaluated as
// max(x,0) - |x| Phi(-|x|) on the input clamped to [-10,10]), imm2 = 1.
__attribute__((always_inline)) inline void vf_gelu_erf(
    __local__ f16v16 *x, __local__ f16v16 *t, short n) {
  auto *a = t, *s = t + 1, *p = t + 2, *q = t + 3;
  rpu::vcfg_set_valu(rpu::vcfg_valu_target::imm2, (half)1.0f);
#pragma rpu_hwloop_ul
  for (short i = 0; i < n; ++i) {
    rpu::valuExecute(a, x + i, (half)-10.0f, rpu::__ValuOpc::max);
    rpu::valuExecute(a, a, (half)10.0f, rpu::__ValuOpc::min);
    rpu::valuExecute(s, (half)0.0f, a, rpu::__ValuOpc::sub);
    rpu::valuExecute(a, a, s, rpu::__ValuOpc::max);
    rpu::valuExecute(s, a, (half)0.2316419f, rpu::__ValuOpc::mul);
    rpu::valuExecute(s, s, (half)1.0f, rpu::__ValuOpc::add);
    rpu::valuExecute(s, s, (half)0.0f, rpu::__ValuOpc::reciprocal);
    rpu::valuExecute(p, s, (half)1.330274429f, rpu::__ValuOpc::mul);
    rpu::valuExecute(p, p, (half)-1.821255978f, rpu::__ValuOpc::add);
    rpu::valuExecute(p, p, s, rpu::__ValuOpc::mul);
    rpu::valuExecute(p, p, (half)1.781477937f, rpu::__ValuOpc::add);
    rpu::valuExecute(p, p, s, rpu::__ValuOpc::mul);
    rpu::valuExecute(p, p, (half)-0.356563782f, rpu::__ValuOpc::add);
    rpu::valuExecute(p, p, s, rpu::__ValuOpc::mul);
    rpu::valuExecute(p, p, (half)0.319381530f, rpu::__ValuOpc::add);
    rpu::valuExecute(p, p, s, rpu::__ValuOpc::mul);
    rpu::valuExecute(q, a, a, rpu::__ValuOpc::mul);
    rpu::valuExecute(q, q, (half)-0.5f, rpu::__ValuOpc::mul);
    rpu::valuExecute(q, q, (half)0.0f, rpu::__ValuOpc::exp);
    rpu::valuExecute(p, p, q, rpu::__ValuOpc::mul);
    rpu::valuExecute(p, p, (half)0.3989422804f, rpu::__ValuOpc::mul);
    rpu::valuExecute(p, p, a, rpu::__ValuOpc::mul);
    rpu::valuExecute(x + i, x + i, (half)0.0f, rpu::__ValuOpc::max);
    rpu::valuExecute(x + i, x + i, p, rpu::__ValuOpc::sub);
  }
}
