// Copyright 2026 X-Square Robot. SPDX-License-Identifier: Apache-2.0
#include "__clang_rpu_builtin_vars.h"

// Column-split language projections for physical prefix buckets of 448 or 512 rows.
// Each block owns N64 columns and streams K128 chunks of A and B.
// The phase schedule bounds scratch use while retaining the FP32 accumulator.
// Build one tiling per entry; the host chooses the entry from the row count.
// Keep accumulator initialization inside the row/tile loops.
//
// Entry wrappers define STREAMED_ENTRY, STREAMED_N, STREAMED_K and one output
// routing: STREAMED_OUTPUT_GATEUP, STREAMED_OUTPUT_QKV or STREAMED_OUTPUT_QKV_ZBA.
// STREAMED_TN/STREAMED_TM describe the launch: the host grid is ceil(N/STREAMED_TN)
// blocks and every block owns STREAMED_TN/TN16 N16 tiles.
#ifndef STREAMED_TN
#define STREAMED_TN 64
#endif
#ifndef STREAMED_TM
#define STREAMED_TM 448
#endif
#ifndef STREAMED_TK
#define STREAMED_TK 128
#endif
#ifndef STREAMED_PIPELINE
#define STREAMED_PIPELINE 4
#endif
// Initialize the FP32 accumulator within the row/tile loop that owns its loop variables.

#if STREAMED_K != 2048 || (STREAMED_N != 1536 && STREAMED_N != 1024 && STREAMED_N != 1056)
#error "M512 streamed projection requires K2048 and N1536/1024/1056"
#endif
#if defined(STREAMED_OUTPUT_GATEUP) + defined(STREAMED_OUTPUT_QKV) + \
    defined(STREAMED_OUTPUT_QKV_ZBA) != 1
#error "M512 streamed projection requires exactly one output routing"
#endif
#if STREAMED_TM != 448 && STREAMED_TM != 512
#error "the streamed projection serves the 448 and 512 row buckets"
#endif

constexpr short STREAMED_CK = STREAMED_TK / 16;
constexpr short STREAMED_AK = STREAMED_K / 16;
// The phase count is a property of the row bucket, not a free knob: it is the
// largest split whose arena fits the 400-vector VLM with a K128 staging arena.
constexpr short STREAMED_PHASES = STREAMED_TM == 448 ? 4 : 8;

// One reviewed arena keeps every VMAT/VLD base below the 8-bit constant base:
// B, then A, then the accumulators.
template <short TN, short TM, short PH>
struct Tiling {
  static constexpr short AN = TN / 16;
  static constexpr short AM = TM / 16;
  static constexpr short PHASES = PH;
  static constexpr short PM = (AM + PHASES - 1) / PHASES;
  // N16 tiles of the last block (zero when N divides evenly).
  static constexpr short TAIL = (STREAMED_N / 16) % AN;
  static constexpr short BLOCKS = (STREAMED_N / 16 + AN - 1) / AN;
  static constexpr short B_VECTORS = 2 * AN * STREAMED_CK;
  static constexpr short A_VECTORS = 2 * PM * STREAMED_CK;
  static constexpr short ACC_SLOTS = 2 * AM * AN;
  static constexpr short ARENA = B_VECTORS + A_VECTORS + ACC_SLOTS;
  static_assert(AN == 4, "stores are unrolled for N64 blocks");
  static_assert(TAIL == 0 || TAIL == 2,
                "N1536/1024 have no tail, N1056 leaves two N16 tiles");
  static_assert(B_VECTORS + A_VECTORS < 256,
                "B and A bases must stay below the 8-bit VMAT constant base");
  static_assert(ARENA <= 400, "projection tile exceeds the 400-vector VLM");
};
using Tile = Tiling<STREAMED_TN, STREAMED_TM, STREAMED_PHASES>;

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

__attribute__((always_inline)) inline void lang_m512_repeat_i32() {
  constexpr short format =
      rpu::acc_format::acc_fp32 | rpu::dst_format::dst_B32 |
      rpu::rs0_format::rs0_fp16 | rpu::rs1_format::rs1_fp16;
  __builtin_rhvm_vmat_set_tsize(STREAMED_CK, 1, format);
  __builtin_rhvm_vmat_set_tstep0(0, 1, 1, format);
  __builtin_rhvm_vmat_set_tstep1(0, 0, 0, format);
}

__attribute__((always_inline)) inline void lang_m512_repeat_fp16() {
  rpu::vmatSetTsize(STREAMED_CK, 1);
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
template <class T, bool Final, short NT>
__attribute__((always_inline)) inline void lang_m512_vmat_phase(
    __local__ i32v16 *acc, __local__ f16v16 *buffer, __local__ f16v16 *b,
    short m0, short count, short *lp) {
  auto *result = reinterpret_cast<__local__ f16v16 *>(acc);
  LoopOutInfo oi[1] = {{(short)0}};
  LoopOutLPAttr oa(oi);
  if constexpr (Final)
    lang_m512_repeat_fp16();
  else
    lang_m512_repeat_i32();
  LoopStepInfo seed_steps[2] = {{&lp[2], 2 * T::AN}, {&lp[1], 2}};
  FragVData seed_lo((__local__ i16v16 *)acc + 2 * m0 * T::AN, seed_steps);
  FragVData seed_hi((__local__ i16v16 *)acc + 2 * m0 * T::AN + 1, seed_steps);
#pragma rpu_hwloop_ul
  for (lp[2] = 0; lp[2] < count; ++lp[2]) {
#pragma rpu_hwloop_ul
    for (lp[1] = 0; lp[1] < NT; ++lp[1]) {
      const short idx = (m0 + lp[2]) * T::AN + lp[1];
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
template <class T, bool Final, short NT>
__attribute__((always_inline)) inline void lang_m512_chunk(
    half *av, DDR_large bg, __local__ i32v16 *acc, __local__ f16v16 *a,
    __local__ f16v16 *b, short c, short mt, short *lp) {
  constexpr short P = T::PHASES, PM = T::PM;
  auto *bcur = b + (c & 1) * T::AN * STREAMED_CK;
  auto *bnext = b + ((c + 1) & 1) * T::AN * STREAMED_CK;
  const short k0 = c * STREAMED_CK, knext = k0 + STREAMED_CK;
  // Even phase counts restart the A buffers at phase 0 every chunk; an odd
  // count needs the parity offset.
  const short parity = (P & 1) ? (short)(c & 1) : (short)0;
#pragma rpu_hwloop_ul
  for (lp[3] = 0; lp[3] < P; ++lp[3]) {
    const short m0 = lp[3] * mt / P, m1 = (lp[3] + 1) * mt / P;
    if (m1 > m0)
      lang_m512_vmat_phase<T, Final, NT>(
          acc, a + ((lp[3] + parity) & 1) * PM * STREAMED_CK, bcur, m0, m1 - m0,
          lp);
    const bool wrap = lp[3] == P - 1;
    const short n0 = wrap ? 0 : m1;
    const short n1 = wrap ? mt / P : (lp[3] + 2) * mt / P;
    if constexpr (!Final) {
      if (lp[3] < NT)
        lang_m512_load_b_tile(bg, bnext + lp[3] * STREAMED_CK, knext, lp[3], lp);
    }
    if (!(Final && wrap) && n1 > n0)
      lang_m512_load_a(av, a + ((lp[3] + 1 + parity) & 1) * PM * STREAMED_CK, n0,
                       n1 - n0, wrap ? knext : k0, lp);
    rpu::fenceVld(0);
    rpu::fenceVmat(0);
  }
}

template <class T>
__attribute__((always_inline)) inline void lang_m512_store_vector(
    __local__ f16v16 *result, short vector, half *dst, short dst_vector,
    short dst_columns, short mt) {
  short row;
  LoopStepInfo rs[1] = {{&row, 2 * T::AN}};
  FragVData vr(result + 2 * vector, rs);
  LoopStepInfo ds[1] = {{&row, (int)16 * dst_columns}};
  FragSData output(dst + (int)threadId * dst_columns + (int)dst_vector * 16,
                   ds);
#pragma rpu_hwloop_ul
  for (row = 0; row < mt; ++row)
    rpu::vstExecute_async(vr, output, (short)0);
}

// Route block vector j (global N16 vector g) into its framework-owned plane.
template <class T>
__attribute__((always_inline)) inline void lang_m512_store(
    __local__ f16v16 *result, short j, short g, short mt,
    STREAMED_OUTPUT_PARAMETERS) {
#if defined(STREAMED_OUTPUT_GATEUP)
  if (g < 48)
    lang_m512_store_vector<T>(result, j, gatev, g, 768, mt);
  else
    lang_m512_store_vector<T>(result, j, upv, g - 48, 768, mt);
#elif defined(STREAMED_OUTPUT_QKV)
  if (g < 16)
    lang_m512_store_vector<T>(result, j, qv, g, 256, mt);
  else if (g < 32)
    lang_m512_store_vector<T>(result, j, qgatev, g - 16, 256, mt);
  else if (g < 48)
    lang_m512_store_vector<T>(result, j, kv, g - 32, 256, mt);
  else
    lang_m512_store_vector<T>(result, j, vv, g - 48, 256, mt);
#else
  if (g < 16)
    lang_m512_store_vector<T>(result, j, qv, g, 256, mt);
  else if (g < 32)
    lang_m512_store_vector<T>(result, j, kv, g - 16, 256, mt);
  else if (g < 48)
    lang_m512_store_vector<T>(result, j, vv, g - 32, 256, mt);
  else if (g < 64)
    lang_m512_store_vector<T>(result, j, zv, g - 48, 256, mt);
  else if (g == 64)
    lang_m512_store_vector<T>(result, j, bv, 0, 16, mt);
  else
    lang_m512_store_vector<T>(result, j, av_out, 0, 16, mt);
#endif
}

template <class T, short NT>
__attribute__((always_inline)) inline void lang_m512_block(
    half *av, DDR_large bg, __local__ f16v16 *arena, short n_base,
    unsigned short rows, STREAMED_OUTPUT_PARAMETERS) {
  constexpr short chunks = STREAMED_AK / STREAMED_CK;
  auto *b = arena;
  auto *a = arena + T::B_VECTORS;
  auto *acc = reinterpret_cast<__local__ i32v16 *>(arena + T::B_VECTORS +
                                                   T::A_VECTORS);
  auto *result = reinterpret_cast<__local__ f16v16 *>(acc);
  // The ABI admits whole 64-row buckets: every thread owns the same M16
  // tiles, and loop bounds stay warp-level.
  const short mt = (short)(rows / 16);
  const short first = mt / T::PHASES;
  short lp[5];

  LoopStepInfo zero_steps[1] = {{&lp[0], 1}};
  FragVData zero(result, zero_steps);
#pragma rpu_hwloop_ul
  for (lp[0] = 0; lp[0] < 2 * mt * T::AN; ++lp[0])
    rpu::valuExecute_async(zero, (half)0.0f, (half)0.0f, rpu::__ValuOpc::add);
  lang_m512_load_b_all<NT>(bg, b, lp);
  lang_m512_load_a(av, a, 0, first, 0, lp);
  rpu::fenceValu(0);
  rpu::fenceVld(0);
  short c;
#pragma rpu_hwloop_ul
  for (c = 0; c < chunks - 1; ++c)
    lang_m512_chunk<T, false, NT>(av, bg, acc, a, b, c, mt, lp);
  lang_m512_chunk<T, true, NT>(av, bg, acc, a, b, chunks - 1, mt, lp);

  // Unrolled like the M384 planar stores: routing branches stay outside any
  // hardware loop. Every store is guarded by NT: the N1056 tail block owns two
  // N16 vectors and an unguarded store would write a zeroed, never-computed
  // tile past N.
  lang_m512_store<T>(result, 0, n_base, mt, STREAMED_OUTPUT_ARGUMENTS);
  if constexpr (NT > 1)
    lang_m512_store<T>(result, 1, n_base + 1, mt, STREAMED_OUTPUT_ARGUMENTS);
  if constexpr (NT > 2) {
    lang_m512_store<T>(result, 2, n_base + 2, mt, STREAMED_OUTPUT_ARGUMENTS);
    lang_m512_store<T>(result, 3, n_base + 3, mt, STREAMED_OUTPUT_ARGUMENTS);
  }
  rpu::fenceVst(0);
}

// Grid: ceil(N/64) blocks per core, each weight element read from DDR once. A
// stale larger grid must not touch memory beyond N.
__rprog__ void STREAMED_ENTRY(half *av, DDR_large wv, STREAMED_OUTPUT_PARAMETERS,
                           unsigned short rows) {
  __local__ f16v16 arena[Tile::ARENA];
  const short n_base = (short)blockIdx.x * Tile::AN;
  if (n_base >= STREAMED_N / 16) return;
  DDR_large bg = wv + (int)n_base * STREAMED_AK * 16 + (int)coreId_u * 2;
  if constexpr (Tile::TAIL != 0) {
    if (n_base + Tile::AN > STREAMED_N / 16) {
      lang_m512_block<Tile, Tile::TAIL>(
          av, bg, arena, n_base, rows, STREAMED_OUTPUT_ARGUMENTS);
      return;
    }
  }
  lang_m512_block<Tile, Tile::AN>(
      av, bg, arena, n_base, rows, STREAMED_OUTPUT_ARGUMENTS);
}

#undef STREAMED_OUTPUT_PARAMETERS
#undef STREAMED_OUTPUT_ARGUMENTS
