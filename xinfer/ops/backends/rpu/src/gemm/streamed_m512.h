// Copyright 2026 X-Square Robot. SPDX-License-Identifier: Apache-2.0
#include "__clang_rpu_builtin_vars.h"

// Physical prefix buckets above 384 rows (448, 512) for the column-split
// language projections. The M384 kernels keep the whole FP32 M x N tile in
// VLM; at M512 that accumulator no longer fits beside A and B. This variant
// keeps the same single DDR weight stream (each weight element is read once)
// and narrows a block to N64: the FP32 M512 x N64 tile uses 256 of the 400 VLM
// slots, B 32 and A 64. A is streamed from SPM once per N64 block.
//
// Entry wrappers define STREAMED_ENTRY, STREAMED_N, STREAMED_K and exactly one output
// routing: STREAMED_OUTPUT_GATEUP, STREAMED_OUTPUT_QKV or STREAMED_OUTPUT_QKV_ZBA.
#define STREAMED_TN 64
#define STREAMED_TM 512
#define STREAMED_TK 64
#define STREAMED_PIPELINE 4

#if STREAMED_K != 2048 || (STREAMED_N != 1536 && STREAMED_N != 1024 && STREAMED_N != 1056)
#error "M512 streamed projection requires K2048 and N1536/1024/1056"
#endif
#if defined(STREAMED_OUTPUT_GATEUP) + defined(STREAMED_OUTPUT_QKV) + \
    defined(STREAMED_OUTPUT_QKV_ZBA) != 1
#error "M512 streamed projection requires exactly one output routing"
#endif

constexpr short STREAMED_AN = STREAMED_TN / 16;
constexpr short STREAMED_AM = STREAMED_TM / 16;
constexpr short STREAMED_CK = STREAMED_TK / 16;
constexpr short STREAMED_AK = STREAMED_K / 16;
constexpr short STREAMED_PHASES = STREAMED_AN;
constexpr short STREAMED_PM = (STREAMED_AM + STREAMED_PHASES - 1) / STREAMED_PHASES;
// N16 tiles of the last block; 2 for QKV-ZBA (its padded B and A), else 0.
constexpr short STREAMED_TAIL = (STREAMED_N / 16) % STREAMED_AN;
static_assert(STREAMED_AN == 4 && (STREAMED_TAIL == 0 || STREAMED_TAIL == 2),
              "stores are unrolled for N64 blocks and a two-tile tail");
// One reviewed arena keeps every VMAT/VLD base below the 8-bit constant base.
constexpr short STREAMED_B_VECTORS = 2 * STREAMED_AN * STREAMED_CK;
constexpr short STREAMED_A_VECTORS = 2 * STREAMED_PM * STREAMED_CK;
constexpr short STREAMED_ACC_SLOTS = 2 * STREAMED_AM * STREAMED_AN;
static_assert(STREAMED_B_VECTORS + STREAMED_A_VECTORS < 256,
              "B and A bases must stay below the 8-bit VMAT constant base");
static_assert(STREAMED_B_VECTORS + STREAMED_A_VECTORS + STREAMED_ACC_SLOTS <= 400,
              "M512 projection exceeds the 400-vector VLM");

#if defined(STREAMED_OUTPUT_GATEUP)
#define STREAMED_OUTPUT_PARAMETERS half *gatev, half *upv
#define STREAMED_OUTPUT_ARGUMENTS gatev, upv
#elif defined(STREAMED_OUTPUT_QKV)
#define STREAMED_OUTPUT_PARAMETERS half *qv, half *qgatev, half *kv, half *vv
#define STREAMED_OUTPUT_ARGUMENTS qv, qgatev, kv, vv
#else
#define STREAMED_OUTPUT_PARAMETERS \
  half *qv, half *kv, half *vv, half *zv, half *bv, half *av_out
#define STREAMED_OUTPUT_ARGUMENTS qv, kv, vv, zv, bv, av_out
#endif

__attribute__((always_inline)) inline void lang_m512_repeat4_i32() {
  constexpr short format =
      rpu::acc_format::acc_fp32 | rpu::dst_format::dst_B32 |
      rpu::rs0_format::rs0_fp16 | rpu::rs1_format::rs1_fp16;
  __builtin_rhvm_vmat_set_tsize(4, 1, format);
  __builtin_rhvm_vmat_set_tstep0(0, 1, 1, format);
  __builtin_rhvm_vmat_set_tstep1(0, 0, 0, format);
}

__attribute__((always_inline)) inline void lang_m512_repeat4_fp16() {
  rpu::vmatSetTsize(4, 1);
  rpu::vmatSetTstep0(0, 1, 1);
  rpu::vmatSetTstep1(0, 0, 0);
}

// Stage A rows [16*m0, 16*(m0+count)) of K chunk k0 (K16 units) from SPM.
__attribute__((always_inline)) inline void lang_m512_load_a(
    half *av, __local__ f16v16 *buffer, short m0, short count, short k0,
    short *lp) {
  LoopStepInfo ad[2] = {{&lp[1], STREAMED_CK}, {&lp[0], 1}};
  FragVData va_load(buffer, ad);
  LoopStepInfo as[2] = {
      {&lp[1], (int)16 * (int)STREAMED_K}, {&lp[0], (int)16}};
  FragSData sa(av + (int)k0 * 16 + ((int)m0 * 16 + (int)threadId) * STREAMED_K,
               as);
#pragma rpu_hwloop_ul
  for (lp[1] = 0; lp[1] < count; ++lp[1]) {
#pragma rpu_hwloop_ul
    for (lp[0] = 0; lp[0] < STREAMED_CK; ++lp[0])
      rpu::vldExecute_async(va_load, sa, (short)0);
  }
}

// Stage N16 weight tile n of chunk k0 from DDR.
// Keep the K16 trip count a compile-time constant and the tile index explicit.
__attribute__((always_inline)) inline void lang_m512_load_b_tile(
    DDR_large bg, __local__ f16v16 *dst, short k0, short n, short *lp) {
  LoopStepInfo bd[1] = {{&lp[0], 1}};
  FragVData vb_load(dst, bd);
  LoopStepInfo bs[1] = {{&lp[0], (int)4096, rpu::__AGran::a_g1B}};
  FragSData sb(bg + (int)k0 * 16 + (int)n * STREAMED_AK * 16, bs);
#pragma rpu_hwloop_ul
  for (lp[0] = 0; lp[0] < STREAMED_CK; ++lp[0])
    rpu::vldExecute_async(vb_load, sb, (short)(threadId * 32));
}

// Load the NT tiles of chunk 0 (prologue only).
template <short NT>
__attribute__((always_inline)) inline void lang_m512_load_b_all(
    DDR_large bg, __local__ f16v16 *buffer, short *lp) {
  LoopStepInfo bd[2] = {{&lp[1], STREAMED_CK}, {&lp[0], 1}};
  FragVData vb_load(buffer, bd);
  LoopStepInfo bs[2] = {
      {&lp[1], (int)STREAMED_AK * (int)4096, rpu::__AGran::a_g1B},
      {&lp[0], (int)4096, rpu::__AGran::a_g1B}};
  FragSData sb(bg, bs);
#pragma rpu_hwloop_ul
  for (lp[1] = 0; lp[1] < NT; ++lp[1]) {
#pragma rpu_hwloop_ul
    for (lp[0] = 0; lp[0] < STREAMED_CK; ++lp[0])
      rpu::vldExecute_async(vb_load, sb, (short)(threadId * 32));
  }
}

// One A phase against the block's NT columns of the current B chunk.
// Non-final chunks write FP32 partials; the final chunk rounds once into each
// accumulator pair's low FP16 slot. The Repeat shape is configured once.
template <bool Final, short NT>
__attribute__((always_inline)) inline void lang_m512_vmat_phase(
    __local__ i32v16 *acc, __local__ f16v16 *buffer, __local__ f16v16 *b,
    short m0, short count, short *lp) {
  auto *result = reinterpret_cast<__local__ f16v16 *>(acc);
  LoopStepInfo seed_steps[2] = {{&lp[2], 2 * STREAMED_AN}, {&lp[1], 2}};
  FragVData seed_lo((__local__ i16v16 *)acc + 2 * m0 * STREAMED_AN, seed_steps);
  FragVData seed_hi((__local__ i16v16 *)acc + 2 * m0 * STREAMED_AN + 1,
                    seed_steps);
  LoopOutInfo oi[1] = {{(short)0}};
  LoopOutLPAttr oa(oi);
  if constexpr (Final)
    lang_m512_repeat4_fp16();
  else
    lang_m512_repeat4_i32();
#pragma rpu_hwloop_ul
  for (lp[2] = 0; lp[2] < count; ++lp[2]) {
#pragma rpu_hwloop_ul
    for (lp[1] = 0; lp[1] < NT; ++lp[1]) {
      const short idx = (m0 + lp[2]) * STREAMED_AN + lp[1];
      rpu::vcfg_set_vmat(rpu::vcfg_vmat_target::acc_init_fp32_l, seed_lo);
      rpu::vcfg_set_vmat(rpu::vcfg_vmat_target::acc_init_fp32_h, seed_hi);
      if constexpr (Final)
        rpu::vmatRepeat_async(result + 2 * idx, buffer + lp[2] * STREAMED_CK,
                              b + lp[1] * STREAMED_CK, &oa);
      else
        rpu::vmatRepeat_async(acc + idx, buffer + lp[2] * STREAMED_CK,
                              b + lp[1] * STREAMED_CK, &oa);
    }
  }
}

// Chunk c: phase p issues VMAT on A phase p (buffer p&1) against B(c), then
// B(c+1) tile p (when the block owns it), then A phase p+1 (the next chunk's
// phase 0 after the last phase) into the other buffer. The fences at the end
// of phase p also retire the VMAT that last read the buffer being refilled.
template <bool Final, short NT>
__attribute__((always_inline)) inline void lang_m512_chunk(
    half *av, DDR_large bg, __local__ i32v16 *acc, __local__ f16v16 *a,
    __local__ f16v16 *b, short c, short mt, short *lp) {
  constexpr short P = STREAMED_PHASES, PM = STREAMED_PM;
  auto *bcur = b + (c & 1) * STREAMED_AN * STREAMED_CK;
  auto *bnext = b + ((c + 1) & 1) * STREAMED_AN * STREAMED_CK;
  const short k0 = c * STREAMED_CK, knext = k0 + STREAMED_CK;
#pragma rpu_hwloop_ul
  for (lp[3] = 0; lp[3] < P; ++lp[3]) {
    const short m0 = lp[3] * mt / P, m1 = (lp[3] + 1) * mt / P;
    if (m1 > m0)
      lang_m512_vmat_phase<Final, NT>(acc, a + (lp[3] & 1) * PM * STREAMED_CK,
                                      bcur, m0, m1 - m0, lp);
    const bool wrap = lp[3] == P - 1;
    const short n0 = wrap ? 0 : m1;
    const short n1 = wrap ? mt / P : (lp[3] + 2) * mt / P;
    if constexpr (!Final) {
      if (lp[3] < NT)
        lang_m512_load_b_tile(bg, bnext + lp[3] * STREAMED_CK, knext, lp[3], lp);
    }
    if (!(Final && wrap) && n1 > n0)
      lang_m512_load_a(av, a + ((lp[3] + 1) & 1) * PM * STREAMED_CK, n0, n1 - n0,
                       wrap ? knext : k0, lp);
    rpu::fenceVld(0);
    rpu::fenceVmat(0);
  }
}

__attribute__((always_inline)) inline void lang_m512_store_vector(
    __local__ f16v16 *result, short vector, half *dst, short dst_vector,
    short dst_columns, short mt) {
  short row;
  LoopStepInfo rs[1] = {{&row, 2 * STREAMED_AN}};
  FragVData vr(result + 2 * vector, rs);
  LoopStepInfo ds[1] = {{&row, (int)16 * dst_columns}};
  FragSData output(dst + (int)threadId * dst_columns + (int)dst_vector * 16,
                   ds);
#pragma rpu_hwloop_ul
  for (row = 0; row < mt; ++row)
    rpu::vstExecute_async(vr, output, (short)0);
}

// Route block vector j (global N16 vector g) into its framework-owned plane.
__attribute__((always_inline)) inline void lang_m512_store(
    __local__ f16v16 *result, short j, short g, short mt,
    STREAMED_OUTPUT_PARAMETERS) {
#if defined(STREAMED_OUTPUT_GATEUP)
  if (g < 48)
    lang_m512_store_vector(result, j, gatev, g, 768, mt);
  else
    lang_m512_store_vector(result, j, upv, g - 48, 768, mt);
#elif defined(STREAMED_OUTPUT_QKV)
  if (g < 16)
    lang_m512_store_vector(result, j, qv, g, 256, mt);
  else if (g < 32)
    lang_m512_store_vector(result, j, qgatev, g - 16, 256, mt);
  else if (g < 48)
    lang_m512_store_vector(result, j, kv, g - 32, 256, mt);
  else
    lang_m512_store_vector(result, j, vv, g - 48, 256, mt);
#else
  if (g < 16)
    lang_m512_store_vector(result, j, qv, g, 256, mt);
  else if (g < 32)
    lang_m512_store_vector(result, j, kv, g - 16, 256, mt);
  else if (g < 48)
    lang_m512_store_vector(result, j, vv, g - 32, 256, mt);
  else if (g < 64)
    lang_m512_store_vector(result, j, zv, g - 48, 256, mt);
  else if (g == 64)
    lang_m512_store_vector(result, j, bv, 0, 16, mt);
  else
    lang_m512_store_vector(result, j, av_out, 0, 16, mt);
#endif
}

template <short NT>
__attribute__((always_inline)) inline void lang_m512_block(
    half *av, DDR_large bg, __local__ f16v16 *arena, short n_base,
    unsigned short rows, STREAMED_OUTPUT_PARAMETERS) {
  constexpr short chunks = STREAMED_AK / STREAMED_CK;
  auto *b = arena;
  auto *a = arena + STREAMED_B_VECTORS;
  auto *acc = reinterpret_cast<__local__ i32v16 *>(arena + STREAMED_B_VECTORS +
                                                   STREAMED_A_VECTORS);
  auto *result = reinterpret_cast<__local__ f16v16 *>(acc);
  // The ABI admits whole 64-row buckets: every thread owns the same M16
  // tiles, and loop bounds stay warp-level.
  const short mt = (short)(rows / 16);
  const short first = mt / STREAMED_PHASES;
  short lp[5];

  LoopStepInfo zero_steps[1] = {{&lp[0], 1}};
  // Clear the FP32 accumulator bits through matching integer VALU operands.
  FragVData zero(reinterpret_cast<__local__ i16v16 *>(acc), zero_steps);
#pragma rpu_hwloop_ul
  for (lp[0] = 0; lp[0] < 2 * mt * STREAMED_AN; ++lp[0])
    rpu::valuExecute_async(zero, (short)0, (short)0, rpu::__ValuOpc::add);
  lang_m512_load_b_all<NT>(bg, b, lp);
  lang_m512_load_a(av, a, 0, first, 0, lp);
  rpu::fenceValu(0);
  rpu::fenceVld(0);
  short c;
#pragma rpu_hwloop_ul
  for (c = 0; c < chunks - 1; ++c)
    lang_m512_chunk<false, NT>(av, bg, acc, a, b, c, mt, lp);
  lang_m512_chunk<true, NT>(av, bg, acc, a, b, chunks - 1, mt, lp);

  // Unrolled like the M384 planar stores: routing branches stay outside any
  // hardware loop.
  lang_m512_store(result, 0, n_base, mt, STREAMED_OUTPUT_ARGUMENTS);
  lang_m512_store(result, 1, n_base + 1, mt, STREAMED_OUTPUT_ARGUMENTS);
  if constexpr (NT > 2) {
    lang_m512_store(result, 2, n_base + 2, mt, STREAMED_OUTPUT_ARGUMENTS);
    lang_m512_store(result, 3, n_base + 3, mt, STREAMED_OUTPUT_ARGUMENTS);
  }
  rpu::fenceVst(0);
}

// Grid: ceil(N/64) blocks per core; each weight element is read from DDR once.
__rprog__ void STREAMED_ENTRY(half *av, DDR_large wv, STREAMED_OUTPUT_PARAMETERS,
                           unsigned short rows) {
  __local__ f16v16 arena[STREAMED_B_VECTORS + STREAMED_A_VECTORS + STREAMED_ACC_SLOTS];
  const short n_base = (short)blockIdx.x * STREAMED_AN;
  // A stale larger grid must not touch memory beyond N.
  if (n_base >= STREAMED_N / 16) return;
  DDR_large bg = wv + (int)n_base * STREAMED_AK * 16 + (int)coreId_u * 2;
  if constexpr (STREAMED_TAIL != 0) {
    if (n_base + STREAMED_AN > STREAMED_N / 16) {
      lang_m512_block<STREAMED_TAIL>(av, bg, arena, n_base, rows,
                                 STREAMED_OUTPUT_ARGUMENTS);
      return;
    }
  }
  lang_m512_block<STREAMED_AN>(av, bg, arena, n_base, rows, STREAMED_OUTPUT_ARGUMENTS);
}

#undef STREAMED_OUTPUT_PARAMETERS
#undef STREAMED_OUTPUT_ARGUMENTS
