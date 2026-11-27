// Copyright 2026 X-Square Robot. SPDX-License-Identifier: Apache-2.0
#include "__clang_rpu_builtin_vars.h"

#if STREAMED_N != 1536 || STREAMED_K != 2048 || STREAMED_TN != 96 || STREAMED_TM != 384 || \
    STREAMED_TK != 64 || STREAMED_PIPELINE != 6
#error "Streamed gate/up requires N1536 K2048 M384 N96 K64 pipeline 6"
#endif

// A block owns N96 (six N16 tiles) over the whole M prefix, so each core
// streams A from SPM 16 times instead of 32 (N48). The FP32 M x N96 tile stays
// in VLM; B moves in K64 chunks, double-buffered. Each chunk runs one phase
// per N16 tile: phase p covers M16 tiles [p*MT/6, (p+1)*MT/6) of A (at most
// four for M384) and prefetches B tile p of the next chunk. Resident warps
// run in lockstep, so every phase issues the same DDR share; bunching B into
// fewer phases leaves DDR idle in the others.
constexpr short STREAMED_PHASES = STREAMED_TN / 16;
constexpr short STREAMED_PM = (STREAMED_TM / 16 + STREAMED_PHASES - 1) / STREAMED_PHASES;

__attribute__((always_inline)) inline void streamed_set_repeat4_i32() {
  constexpr short format =
      rpu::acc_format::acc_fp32 | rpu::dst_format::dst_B32 |
      rpu::rs0_format::rs0_fp16 | rpu::rs1_format::rs1_fp16;
  __builtin_rhvm_vmat_set_tsize(4, 1, format);
  __builtin_rhvm_vmat_set_tstep0(0, 1, 1, format);
  __builtin_rhvm_vmat_set_tstep1(0, 0, 0, format);
}

__attribute__((always_inline)) inline void streamed_set_repeat4_fp16() {
  rpu::vmatSetTsize(4, 1);
  rpu::vmatSetTstep0(0, 1, 1);
  rpu::vmatSetTstep1(0, 0, 0);
}

// Stage A rows [16*m0, 16*(m0+count)) of K chunk k0 (K16 units) from SPM.
__attribute__((always_inline)) inline void streamed_gateup_load_a(
    half *av, __local__ f16v16 *buffer, short m0, short count, short k0,
    short *lp) {
  constexpr short CK = STREAMED_TK / 16;
  LoopStepInfo ad[2] = {{&lp[1], CK}, {&lp[0], 1}};
  FragVData va_load(buffer, ad);
  LoopStepInfo as[2] = {
      {&lp[1], (int)16 * (int)STREAMED_K}, {&lp[0], (int)16}};
  FragSData sa(av + (int)k0 * 16 + ((int)m0 * 16 + (int)threadId) * STREAMED_K,
               as);
#pragma rpu_hwloop_ul
  for (lp[1] = 0; lp[1] < count; ++lp[1]) {
#pragma rpu_hwloop_ul
    for (lp[0] = 0; lp[0] < CK; ++lp[0])
      rpu::vldExecute_async(va_load, sa, (short)0);
  }
}

// Stage N16 weight tile n (K16 blocks [k0, k0+KB)) from DDR into dst.
// Keep the tile count constant and the tile index explicit.
template <short KB = STREAMED_TK / 16>
__attribute__((always_inline)) inline void streamed_gateup_load_b_tile(
    DDR_large bg, __local__ f16v16 *dst, short k0, short n, short *lp) {
  constexpr short AK = STREAMED_K / 16;
  LoopStepInfo bd[1] = {{&lp[0], 1}};
  FragVData vb_load(dst, bd);
  LoopStepInfo bs[1] = {{&lp[0], (int)4096, rpu::__AGran::a_g1B}};
  FragSData sb(bg + (int)k0 * 16 + (int)n * AK * 16, bs);
#pragma rpu_hwloop_ul
  for (lp[0] = 0; lp[0] < KB; ++lp[0])
    rpu::vldExecute_async(vb_load, sb, (short)(threadId * 32));
}

__attribute__((always_inline)) inline void streamed_gateup_load_b(
    DDR_large bg, __local__ f16v16 *buffer, short k0, short n, short *lp) {
  streamed_gateup_load_b_tile(bg, buffer + n * (STREAMED_TK / 16), k0, n, lp);
}

// Load all N16 tiles of chunk k0 (prologue only).
__attribute__((always_inline)) inline void streamed_gateup_load_b_all(
    DDR_large bg, __local__ f16v16 *buffer, short k0, short *lp) {
  constexpr short AK = STREAMED_K / 16, CK = STREAMED_TK / 16, AN = STREAMED_TN / 16;
  LoopStepInfo bd[2] = {{&lp[1], CK}, {&lp[0], 1}};
  FragVData vb_load(buffer, bd);
  LoopStepInfo bs[2] = {
      {&lp[1], (int)AK * (int)4096, rpu::__AGran::a_g1B},
      {&lp[0], (int)4096, rpu::__AGran::a_g1B}};
  FragSData sb(bg + (int)k0 * 16, bs);
#pragma rpu_hwloop_ul
  for (lp[1] = 0; lp[1] < AN; ++lp[1]) {
#pragma rpu_hwloop_ul
    for (lp[0] = 0; lp[0] < CK; ++lp[0])
      rpu::vldExecute_async(vb_load, sb, (short)(threadId * 32));
  }
}

// One A phase against all N96 columns of the current B chunk. Non-final
// chunks write FP32 partials; the final chunk rounds once into each
// accumulator pair's low FP16 slot. Every command has the same Repeat shape,
// so one configuration serves the whole phase.
template <bool Final>
__attribute__((always_inline)) inline void streamed_gateup_vmat_phase(
    __local__ i32v16 *acc, __local__ f16v16 *buffer, __local__ f16v16 *b,
    short m0, short count, short *lp) {
  constexpr short CK = STREAMED_TK / 16, AN = STREAMED_TN / 16;
  auto *result = reinterpret_cast<__local__ f16v16 *>(acc);
  LoopStepInfo seed_steps[2] = {{&lp[2], 2 * AN}, {&lp[1], 2}};
  FragVData seed_lo((__local__ i16v16 *)acc + 2 * m0 * AN, seed_steps);
  FragVData seed_hi((__local__ i16v16 *)acc + 2 * m0 * AN + 1, seed_steps);
  LoopOutInfo oi[1] = {{(short)0}};
  LoopOutLPAttr oa(oi);
  if constexpr (Final)
    streamed_set_repeat4_fp16();
  else
    streamed_set_repeat4_i32();
#pragma rpu_hwloop_ul
  for (lp[2] = 0; lp[2] < count; ++lp[2]) {
#pragma rpu_hwloop_ul
    for (lp[1] = 0; lp[1] < AN; ++lp[1]) {
      const short idx = (m0 + lp[2]) * AN + lp[1];
      rpu::vcfg_set_vmat(rpu::vcfg_vmat_target::acc_init_fp32_l, seed_lo);
      rpu::vcfg_set_vmat(rpu::vcfg_vmat_target::acc_init_fp32_h, seed_hi);
      if constexpr (Final)
        rpu::vmatRepeat_async(result + 2 * idx, buffer + lp[2] * CK,
                              b + lp[1] * CK, &oa);
      else
        rpu::vmatRepeat_async(acc + idx, buffer + lp[2] * CK,
                              b + lp[1] * CK, &oa);
    }
  }
}

// Chunk c: phase p issues VMAT on A phase p (buffer p&1) against B(c), then
// B(c+1) tile p, then A phase p+1 (the next chunk's phase 0 after p=5) into
// the other buffer. The whole VMAT stream goes first so the shared VMAT unit
// never waits behind load issue; the DDR request precedes the SPM loads in the
// load queue. The fences at the end of phase p also retire the VMAT that last
// read the buffer phase p+1 overwrites.
template <bool Final>
__attribute__((always_inline)) inline void streamed_gateup_chunk(
    half *av, DDR_large bg, __local__ i32v16 *acc, __local__ f16v16 *a,
    __local__ f16v16 *b, short c, short mt, short *lp) {
  constexpr short CK = STREAMED_TK / 16, AN = STREAMED_TN / 16, PM = STREAMED_PM;
  constexpr short P = STREAMED_PHASES;
  auto *bcur = b + (c & 1) * AN * CK;
  auto *bnext = b + ((c + 1) & 1) * AN * CK;
  const short k0 = c * CK, knext = k0 + CK;
#pragma rpu_hwloop_ul
  for (lp[3] = 0; lp[3] < P; ++lp[3]) {
    const short m0 = lp[3] * mt / P, m1 = (lp[3] + 1) * mt / P;
    if (m1 > m0)
      streamed_gateup_vmat_phase<Final>(acc, a + (lp[3] & 1) * PM * CK, bcur,
                                    m0, m1 - m0, lp);
    const bool wrap = lp[3] == P - 1;
    const short n0 = wrap ? 0 : m1;
    const short n1 = wrap ? mt / P : (lp[3] + 2) * mt / P;
    if constexpr (!Final) streamed_gateup_load_b(bg, bnext, knext, lp[3], lp);
    if (!(Final && wrap) && n1 > n0)
      streamed_gateup_load_a(av, a + ((lp[3] + 1) & 1) * PM * CK, n0, n1 - n0,
                         wrap ? knext : k0, lp);
    rpu::fenceVld(0);
    rpu::fenceVmat(0);
  }
}

// Rows <= 64 (at most four M16 tiles) are pure weight streaming: A and VMAT
// are small. Run the block as two N48 passes of K128 chunks so every fence
// covers three whole K128 tile runs (32 KiB contiguous across the eight cores);
// the six-phase K64 schedule cannot fill DDR at this size. This is the
// previous N48 schedule and fits the N96 arrays (B 24 of 48, A 32 of 32).
template <bool Final>
__attribute__((always_inline)) inline void streamed_gateup_small_chunk(
    half *av, DDR_large bg, __local__ i32v16 *acc, __local__ f16v16 *a,
    __local__ f16v16 *b, short k0, short n_off, short mt, short *lp) {
  constexpr short SK = 8, SN = 3, AN = STREAMED_TN / 16;
  auto *result = reinterpret_cast<__local__ f16v16 *>(acc);
#pragma rpu_hwloop_ul
  for (lp[4] = 0; lp[4] < SN; ++lp[4])
    streamed_gateup_load_b_tile<SK>(bg, b + lp[4] * SK, k0, n_off + lp[4], lp);
  LoopStepInfo ad[2] = {{&lp[1], SK}, {&lp[0], 1}};
  FragVData va_load(a, ad);
  LoopStepInfo as[2] = {
      {&lp[1], (int)16 * (int)STREAMED_K}, {&lp[0], (int)16}};
  FragSData sa(av + (int)k0 * 16 + (int)threadId * STREAMED_K, as);
#pragma rpu_hwloop_ul
  for (lp[1] = 0; lp[1] < mt; ++lp[1]) {
#pragma rpu_hwloop_ul
    for (lp[0] = 0; lp[0] < SK; ++lp[0])
      rpu::vldExecute_async(va_load, sa, (short)0);
  }
  rpu::fenceVld(0);
  LoopStepInfo seed_steps[2] = {{&lp[2], 2 * AN}, {&lp[1], 2}};
  FragVData seed_lo((__local__ i16v16 *)acc + 2 * n_off, seed_steps);
  FragVData seed_hi((__local__ i16v16 *)acc + 2 * n_off + 1, seed_steps);
  LoopOutInfo oi[1] = {{(short)0}};
  LoopOutLPAttr oa(oi);
  // Only the final chunk switches formats inside a pair; elsewhere one
  // Repeat setup serves every command.
  if constexpr (!Final) streamed_set_repeat4_i32();
#pragma rpu_hwloop_ul
  for (lp[2] = 0; lp[2] < mt; ++lp[2]) {
#pragma rpu_hwloop_ul
    for (lp[1] = 0; lp[1] < SN; ++lp[1]) {
      const short idx = lp[2] * AN + n_off + lp[1];
      rpu::vcfg_set_vmat(rpu::vcfg_vmat_target::acc_init_fp32_l, seed_lo);
      rpu::vcfg_set_vmat(rpu::vcfg_vmat_target::acc_init_fp32_h, seed_hi);
      if constexpr (Final) streamed_set_repeat4_i32();
      rpu::vmatRepeat_async(acc + idx, a + lp[2] * SK, b + lp[1] * SK,
                            rpu::__OutMode::outoff);
      if constexpr (Final) {
        streamed_set_repeat4_fp16();
        rpu::vmatRepeat_async(result + 2 * idx, a + lp[2] * SK + 4,
                              b + lp[1] * SK + 4, &oa);
      } else {
        rpu::vmatRepeat_async(acc + idx, a + lp[2] * SK + 4,
                              b + lp[1] * SK + 4, &oa);
      }
    }
  }
  rpu::fenceVmat(0);
}

__attribute__((always_inline)) inline void streamed_gateup_small_m(
    half *av, DDR_large bg, __local__ i32v16 *acc, __local__ f16v16 *a,
    __local__ f16v16 *b, short mt, short *lp) {
  constexpr short AK = STREAMED_K / 16, SK = 8;
  short pass, k0;
#pragma rpu_hwloop_ul
  for (pass = 0; pass < 2; ++pass) {
    const short n_off = pass * 3;
#pragma rpu_hwloop_ul
    for (k0 = 0; k0 < AK - SK; k0 += SK)
      streamed_gateup_small_chunk<false>(av, bg, acc, a, b, k0, n_off, mt, lp);
    streamed_gateup_small_chunk<true>(av, bg, acc, a, b, AK - SK, n_off, mt, lp);
  }
}

__attribute__((always_inline)) inline void streamed_gateup_store_plane(
    __local__ f16v16 *result, half *dst, short plane_n_base,
    short valid_m, short n_count) {
  constexpr short AN = STREAMED_TN / 16;
  short row, column;
  LoopStepInfo rs[2] = {{&row, 2 * AN}, {&column, 2}};
  FragVData vr(result, rs);
  LoopStepInfo ds[2] = {
      {&row, (int)(16 * 768)}, {&column, (int)16}};
  FragSData output(dst + (int)threadId * 768 +
                         (int)plane_n_base * 16,
                   ds);
#pragma rpu_hwloop_ul
  for (row = 0; row < valid_m; ++row) {
#pragma rpu_hwloop_ul
    for (column = 0; column < n_count; ++column)
      rpu::vstExecute_async(vr, output, (short)0);
  }
}

// Grid: N1536/N96 = 16 blocks per core, two waves of eight resident warps.
// Each weight element is read from DDR once.
__rprog__ void STREAMED_ENTRY(half *av, DDR_large wv, half *gatev, half *upv,
                           unsigned short rows) {
  constexpr short AK = STREAMED_K / 16, CK = STREAMED_TK / 16;
  constexpr short AN = STREAMED_TN / 16, AM = STREAMED_TM / 16, PM = STREAMED_PM;
  constexpr short chunks = AK / CK;
  // The ABI admits rows 64..384 in multiples of 64: every thread owns the
  // same complete M16 tiles, and loop bounds stay warp-level.
  const short mt = rows < STREAMED_TM ? (short)(rows / 16) : AM;
  const short first = mt / STREAMED_PHASES;
  const short n_base = (short)blockIdx.x * AN;
  // A stale grid larger than N/TN must not touch memory beyond N1536.
  if (n_base >= STREAMED_N / 16) return;
  DDR_large bg = wv + (int)n_base * AK * 16 + (int)coreId_u * 2;

  __local__ f16v16 b[2 * AN * CK];
  __local__ f16v16 a[2 * PM * CK];
  __local__ i32v16 acc[AM * AN];
  auto *result = reinterpret_cast<__local__ f16v16 *>(acc);
  short lp[5];

  LoopStepInfo zero_steps[1] = {{&lp[0], 1}};
  // Clear the FP32 accumulator bits through matching integer VALU operands.
  FragVData zero(reinterpret_cast<__local__ i16v16 *>(acc), zero_steps);
#pragma rpu_hwloop_ul
  for (lp[0] = 0; lp[0] < 2 * mt * AN; ++lp[0])
    rpu::valuExecute_async(zero, (short)0, (short)0,
                           rpu::__ValuOpc::add);
  if (mt <= 4) {
    rpu::fenceValu(0);
    streamed_gateup_small_m(av, bg, acc, a, b, mt, lp);
  } else {
    streamed_gateup_load_b_all(bg, b, 0, lp);
    if (first > 0) streamed_gateup_load_a(av, a, 0, first, 0, lp);
    rpu::fenceValu(0);
    rpu::fenceVld(0);
    short c;
#pragma rpu_hwloop_ul
    for (c = 0; c < chunks - 1; ++c)
      streamed_gateup_chunk<false>(av, bg, acc, a, b, c, mt, lp);
    streamed_gateup_chunk<true>(av, bg, acc, a, b, chunks - 1, mt, lp);
  }

  constexpr short plane_vectors = 768 / 16;
  if (n_base < plane_vectors)
    streamed_gateup_store_plane(result, gatev, n_base, mt, AN);
  else
    streamed_gateup_store_plane(result, upv, n_base - plane_vectors, mt, AN);
  rpu::fenceVst(0);
}
