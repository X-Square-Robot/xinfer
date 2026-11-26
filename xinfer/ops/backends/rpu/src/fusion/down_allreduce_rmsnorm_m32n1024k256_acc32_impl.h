// Copyright 2026 X-Square Robot. SPDX-License-Identifier: Apache-2.0
#pragma once
#if XINFER_ACTION_DOWN_WINDOW
#define XINFER_ACTION_DOWN_FORMALS , unsigned short local_counter0, unsigned short local_counter1
#define XINFER_ACTION_DOWN_ARGS , local_counter0, local_counter1
#define XINFER_ACTION_DOWN_LOCAL0 ((short)local_counter0)
#define XINFER_ACTION_DOWN_LOCAL1 ((short)local_counter1)
#else
#define XINFER_ACTION_DOWN_FORMALS
#define XINFER_ACTION_DOWN_ARGS
#define XINFER_ACTION_DOWN_LOCAL0 0
#define XINFER_ACTION_DOWN_LOCAL1 1
#endif
// Action fused TP8 row-parallel output stage:
//   P_c    = A_c x W_c^T                 local partial [32,1024], K=256
//   S      = sum_c P_c                    all_reduce_residual FP16 tree
//   X      = S * gate + residual          residual_gate_m32h1024_f16
//   Y      = rmsnorm(X, gamma) + shift    adaptive_rmsnorm_m32h1024_f16
// X and Y end replicated in every core's LocalSPM. Replaces
// gemm_ddr_row + all_reduce_residual + residual_gate + the next modulation
// DMA + adaptive_rmsnorm, and is bit-identical to that chain: the GEMM keeps
// the production VMAT order, each core sums its owned rows in the
// all_reduce_residual cyclic tree order, and the gate/norm code is unchanged.
// MLP use: A=silu out, residual=residual2, X->residual1, Y->input_norm
// (Y may alias residual). Attention o use: A=attention out,
// residual=residual1, X->residual2, Y->residual1.
//
// Identity-next mode (identity_next=1) also folds the following identity
// layer's pre-MLP chain, bit-identical to residual_gate + adaptive_rmsnorm:
//   Z      = Y * gate' + X               gate' = next_row[2048:3072]
//   W      = rmsnorm(Z, gamma'') + shift''   from next2_row
// Z goes to norm_out (may alias residual) and W to w_out; X stays in stream as
// replicated scratch. Use: X->scratch, Z->residual2, W->MLP input.
//
// Communication only READS peer SPM; every store targets this core's own
// SPM through a local pointer. Phases (block 0 leads every rendezvous; S1,
// S2 and S3 are eight-core, the joins core-local):
//   P1 GEMM (block b owns columns [128b,+128)); block 7 also stages P4's
//      modulation vectors from DDR                              -> S1
//   P2 core c owns rows [4c,4c+4): blocks 1..7 bulk-copy one peer's partial
//      rows [4c,4c+4) into local SPM (-> join); every block then sums its
//      pieces in the all_reduce_residual order, gates and stores X rows
//                                                               -> S2
//   P3 blocks 0..6 pull the other cores' X rows; blocks 0,1 also preload
//      the staged modulation vectors                            -> join
//   P4 blocks 0,1 normalize rows [16b,+16)                      -> S3
// The final barrier keeps the partial and X rows live until all peers finish
// reading. Each launch starts with zeroed counters on all eight cores.

#include "__clang_rpu_builtin_vars.h"
using namespace rpu;

// P2 transport: 1 = bulk-copy peer partial blocks into local SPM, then reduce
// locally; 0 = reduce straight from eight small remote reads per thread.
#ifndef FUSION_P2_BULK
#define FUSION_P2_BULK 1
#endif

// Pointer into the per-thread VLM pool (address space only, no storage).
typedef __attribute__((address_space(1))) f16v16 vlm_f16v16;

static __attribute__((always_inline)) __SRAM half *peer_address(
    __SRAM half *base, unsigned short peer, unsigned short stride_hi) {
  int bank_bytes = (int)(unsigned short)(peer * stride_hi) << 16;
  return reinterpret_cast<__SRAM half *>(reinterpret_cast<int>(base) + bank_bytes);
}

// Split-phase rendezvous. Every block adds one core arrival; the leader
// (block 0) waits for this local phase's eight arrivals, joins global phase
// global_phase unless it is 0 (a core-local barrier), and bumps the core
// release counter that all blocks wait on.
static __attribute__((always_inline)) void rendezvous(
    short block, unsigned short phase, unsigned short global_phase,
    unsigned short global_counter XINFER_ACTION_DOWN_FORMALS) {
  watom(XINFER_ACTION_DOWN_LOCAL0, 1, field_mode::core, modify_mode::add);
  if (block == 0) {
    watoms(XINFER_ACTION_DOWN_LOCAL0, 0, (short)(8 * phase - 1), field_mode::core, cmp_mode::gt,
           modify_mode::add);
    if (global_phase) {
      watom((short)global_counter, 1, field_mode::global, modify_mode::add);
      watoms((short)global_counter, 0, (short)(8 * global_phase - 1),
             field_mode::global, cmp_mode::gt, modify_mode::add);
    }
    watom(XINFER_ACTION_DOWN_LOCAL1, 1, field_mode::core, modify_mode::add);
  }
  watoms(XINFER_ACTION_DOWN_LOCAL1, 0, (short)(phase - 1), field_mode::core, cmp_mode::gt,
         modify_mode::add);
  fenceAtom((short)0);
}

// adaptive_rmsnorm_m32h1024_f16 for one row per thread, in place on x. The
// caller loads x, gamma and shift and sets the VALU/VMAT immediates. The
// small/large rescale decision is warp-wide, so warp b must own rows
// [16b,16b+16) exactly like the production kernel.
static __attribute__((always_inline)) void adaptive_rmsnorm_row(
    vlm_f16v16 *x, vlm_f16v16 *gamma, vlm_f16v16 *shift, vlm_f16v16 *gram,
    vlm_f16v16 *norm, vlm_f16v16 *scratch, half epsilon) {
  constexpr short chunks = 64;
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
}

__rprog__ void XINFER_ACTION_DOWN_KERNEL(
    half *av, DDR_large wv, half *partial, __SRAM half *partial_global,
    half *residual, DDR_large gate_row, DDR_large next_row,
    half *stream, __SRAM half *stream_global, half *norm_out,
    unsigned short core_stride_hi, unsigned short mean_scale_low,
    unsigned short mean_scale_high, half epsilon,
    unsigned short global_counter, unsigned short identity_next,
    DDR_large next2_row, half *w_out
#if XINFER_ACTION_DOWN_RSQRT_LUT
    , half *rsqrt_lut
#endif
    XINFER_ACTION_DOWN_FORMALS
) {
  if (global_counter < 3 || global_counter > 50) return;
#if XINFER_ACTION_DOWN_WINDOW
  if (local_counter0 < 2 || local_counter0 > 32 || (local_counter0 & 1) ||
      local_counter1 != local_counter0 + 1 ||
      global_counter != 34 + (local_counter0 - 2) / 2) return;
#endif
  const short vlen = 16;
  const short a_ld = 256;
  const short r_ld = 1024;
  const short M = 2;   // 2 x 16 rows
  const short N = 8;   // 8 x 16 columns per grid block
  const short K = 16;  // 16 x 16 local reduction blocks
  const short block = (short)blockIdx.x;
  const unsigned short owner = coreId_u;

  // One VLM pool, carved per phase: P1 176 (+16 staging), P2 20, P3 16 +
  // preloads, P4 195 (identity 387) vectors.
  __local__ f16v16 pool[388];
  // Residual-gate vectors of this block's P2 columns, loaded from DDR while
  // streaming the GEMM weights.
  __local__ f16v16 gate_v[2];
  short loop[3];
  const int p2_vector = (int)owner * 16 + (int)block * 2;
  const int stage_offset = (int)((4 * owner + 4) & 31) * 1024;
  const int p2_column = (p2_vector & 3) * 256 + (int)threadId * 16;

  // ---------------- P1: production row GEMM into local partial ----------
  {
    vlm_f16v16 *a = pool;           // M*K = 32
    vlm_f16v16 *b = pool + M * K;   // N*K = 128
    vlm_f16v16 *r = b + N * K;      // M*N = 16
    LoopStepInfo a_dst_steps[2] = {
        {&loop[1], (short)K}, {&loop[0], (short)1}};
    FragVData a_dst(a, a_dst_steps);
    LoopStepInfo a_src_steps[2] = {
        {&loop[1], (int)(vlen * a_ld)}, {&loop[0], (int)vlen}};
    FragSData a_src(av, a_src_steps);
#pragma rpu_hwloop_ul
    for (loop[1] = 0; loop[1] < M; ++loop[1]) {
#pragma rpu_hwloop_ul
      for (loop[0] = 0; loop[0] < K; ++loop[0])
        vldExecute_async(a_dst, a_src, (short)(threadId * a_ld));
    }
    fenceVld((short)0);

    const short core = (short)owner;
    const int local_nblock = (int)block * N;
    const int weight_base_large = local_nblock * (int)K * 16 + (int)core * 2;
    DDR_large b_base = wv + weight_base_large;
    LoopStepInfo b_dst_steps[2] = {
        {&loop[1], (short)K}, {&loop[0], (short)1}};
    FragVData b_dst(b, b_dst_steps);
    LoopStepInfo b_src_steps[2] = {
        {&loop[1], (int)K * 8 * 256 * (int)sizeof(half), __AGran::a_g1B},
        {&loop[0], (int)(8 * 256 * sizeof(half)), __AGran::a_g1B}};
    FragSData b_src(b_base, b_src_steps);
#pragma rpu_hwloop_ul
    for (loop[1] = 0; loop[1] < N; ++loop[1]) {
#pragma rpu_hwloop_ul
      for (loop[0] = 0; loop[0] < K; ++loop[0])
        vldExecute_async(b_dst, b_src, (short)(threadId * vlen * sizeof(half)));
    }
    // Gate = modulation[2048:3072]; the row's gate starts 4 KiB (16 units) in.
    vldExecute_async(gate_v + 0, gate_row + 16, p2_column * 2);
    vldExecute_async(gate_v + 1, gate_row + 16, p2_column * 2 + 512);
    // Block 7 stages what P4 needs (4 vectors per thread of each 1024-value
    // row) into memory no other core touches: gamma|shift into norm_out rows
    // 4c+4, 4c+5 (never read as residual here); in identity-next mode gamma,
    // shift, gate', gamma'' into w_out rows 0..3. The P4 warps preload it
    // into VLM during P3; shift'' follows in P3 (w_out rows are overwritten
    // by P4).
    vlm_f16v16 *stage = pool + 176;
    if (block == 7) {
#pragma rpu_hwloop_ul
      for (short j = 0; j < 4; ++j) {
        vldExecute_async(stage + j, next_row, ((int)threadId * 4 + (int)j) * 32);
        vldExecute_async(stage + 4 + j, next_row + 8, ((int)threadId * 4 + (int)j) * 32);
        if (identity_next) {
          vldExecute_async(stage + 8 + j, next_row + 16, ((int)threadId * 4 + (int)j) * 32);
          vldExecute_async(stage + 12 + j, next2_row, ((int)threadId * 4 + (int)j) * 32);
        }
      }
    }
    fenceVld((short)0);
    if (block == 7) {
      vstSetTsize32B(4, 1);
      vstSetTstep32B_0(1, 16);
      vstSetTstep32B_1(0, 0);
      if (identity_next) {
        vstRepeat_async(stage, w_out, (int)threadId * 64);
        vstRepeat_async(stage + 4, w_out, 1024 + (int)threadId * 64);
        vstRepeat_async(stage + 8, w_out, 2048 + (int)threadId * 64);
        vstRepeat_async(stage + 12, w_out, 3072 + (int)threadId * 64);
      } else {
        vstRepeat_async(stage, norm_out, stage_offset + (int)threadId * 64);
        vstRepeat_async(stage + 4, norm_out, stage_offset + 1024 + (int)threadId * 64);
      }
      fenceVst((short)0);
    }

    LoopStepInfo c_steps[3] = {
        {&loop[2], (short)N}, {&loop[1], (short)1},
        {&loop[0], (short)0}};
    FragVData c_dst(r, c_steps);
    LoopStepInfo ac_steps[3] = {
        {&loop[2], (short)K}, {&loop[1], (short)0},
        {&loop[0], (short)1}};
    FragVData ac(a, ac_steps);
    LoopStepInfo bc_steps[3] = {
        {&loop[2], (short)0}, {&loop[1], (short)K},
        {&loop[0], (short)1}};
    FragVData bc(b, bc_steps);
    LoopOutInfo out_info[1] = {{&loop[0]}};
    // Reset for each N output tile when its K reduction begins.
    LoopClrInfo clear_info[1] = {{&loop[0]}};
    LoopOutLPAttr out_attr(out_info);
    LoopClrLPAttr clear_attr(clear_info);
#pragma rpu_hwloop_ul
    for (loop[2] = 0; loop[2] < M; ++loop[2]) {
#pragma rpu_hwloop_ul
      for (loop[1] = 0; loop[1] < N; ++loop[1]) {
#pragma rpu_hwloop_ul
        for (loop[0] = 0; loop[0] < K; ++loop[0])
          vmatExecute_async(c_dst, ac, bc, &out_attr, &clear_attr);
      }
    }
    fenceVmat((short)0);

    half *block_r = partial + (int)block * 128;
    LoopStepInfo r_src_steps[2] = {
        {&loop[1], (short)N}, {&loop[0], (short)1}};
    FragVData r_src(r, r_src_steps);
    LoopStepInfo r_dst_steps[2] = {
        {&loop[1], (int)(vlen * r_ld)}, {&loop[0], (int)vlen}};
    FragSData r_dst(block_r, r_dst_steps);
#pragma rpu_hwloop_ul
    for (loop[1] = 0; loop[1] < M; ++loop[1]) {
#pragma rpu_hwloop_ul
      for (loop[0] = 0; loop[0] < N; ++loop[0])
        vstExecute_async(r_src, r_dst, (short)(threadId * r_ld));
    }
    fenceVst((short)0);
  }
  rendezvous(block, 1, 1, global_counter XINFER_ACTION_DOWN_ARGS);

  // ---------------- P2: owned rows, FP16 tree sum, residual gate ---------
  // Core c owns the 16 contiguous 256-value vectors [16c,16c+16) = rows
  // [4c,4c+4); block b sums vectors 16c+2b and 16c+2b+1.
  // Bulk transport: block b>0 first copies peer p=(c+b)%8's partial rows
  // [4c,4c+4) into this core's stream rows [4p,4p+4), which receive p's X
  // rows only in P3; after a core-local barrier the sum reads local SPM.
  constexpr short bulk = FUSION_P2_BULK ? 1 : 0;
  if (bulk) {
    if (block > 0) {
      vlm_f16v16 *buf = pool;  // 16 vectors = 256 contiguous values/thread
      const unsigned short peer = (owner + (unsigned short)block) & 7;
      vldSetTsize32B(16, 1);
      vldSetTstep32B_0(1, 16);
      vldSetTstep32B_1(0, 0);
      vldRepeat_async(buf, peer_address(partial_global, peer, core_stride_hi),
                      (int)owner * 4096 + (int)threadId * 256);
      fenceVld((short)0);
      vstSetTsize32B(16, 1);
      vstSetTstep32B_0(1, 16);
      vstSetTstep32B_1(0, 0);
      vstRepeat(buf, stream, (int)peer * 4096 + (int)threadId * 256);
      fenceVst((short)0);
    }
    rendezvous(block, 2, 0, global_counter XINFER_ACTION_DOWN_ARGS);
  }
  {
    vlm_f16v16 *x0 = pool, *x1 = pool + 2, *x2 = pool + 4, *x3 = pool + 6;
    vlm_f16v16 *x4 = pool + 8, *x5 = pool + 10, *x6 = pool + 12, *x7 = pool + 14;
    vlm_f16v16 *rr = pool + 16;
    const int offset = p2_vector * 256 + (int)threadId * 16;

    vcfg_set_valu(vcfg_valu_target::imm2, (half)1.0f);
    vldSetTsize32B(2, 1);
    vldSetTstep32B_0(1, 256);
    vldSetTstep32B_1(0, 0);
    if (bulk) {
      // Peer p's copy of this piece sits at stream row block 4p.
      const int piece = offset - (int)owner * 4096;
      vldRepeat_async(x0, partial, offset);
      vldRepeat_async(x1, stream, (int)((owner + 1) & 7) * 4096 + piece);
      vldRepeat_async(x2, stream, (int)((owner + 2) & 7) * 4096 + piece);
      vldRepeat_async(x3, stream, (int)((owner + 3) & 7) * 4096 + piece);
      vldRepeat_async(x4, stream, (int)((owner + 4) & 7) * 4096 + piece);
      vldRepeat_async(x5, stream, (int)((owner + 5) & 7) * 4096 + piece);
      vldRepeat_async(x6, stream, (int)((owner + 6) & 7) * 4096 + piece);
      vldRepeat_async(x7, stream, (int)((owner + 7) & 7) * 4096 + piece);
    } else {
      vldRepeat_async(x0, peer_address(partial_global, owner, core_stride_hi), offset);
      vldRepeat_async(x1, peer_address(partial_global, (owner + 1) & 7, core_stride_hi), offset);
      vldRepeat_async(x2, peer_address(partial_global, (owner + 2) & 7, core_stride_hi), offset);
      vldRepeat_async(x3, peer_address(partial_global, (owner + 3) & 7, core_stride_hi), offset);
      vldRepeat_async(x4, peer_address(partial_global, (owner + 4) & 7, core_stride_hi), offset);
      vldRepeat_async(x5, peer_address(partial_global, (owner + 5) & 7, core_stride_hi), offset);
      vldRepeat_async(x6, peer_address(partial_global, (owner + 6) & 7, core_stride_hi), offset);
      vldRepeat_async(x7, peer_address(partial_global, (owner + 7) & 7, core_stride_hi), offset);
    }
    vldRepeat_async(rr, residual, offset);
    fenceVld((short)0);

    valuSetTsize(2, 1);
    valuSetTstep0(1, 1, 1);
    valuSetTstep1(0, 0, 0);
    valuRepeat_async(x0, x0, x1, __ValuOpc::add);
    valuRepeat_async(x2, x2, x3, __ValuOpc::add);
    valuRepeat_async(x4, x4, x5, __ValuOpc::add);
    valuRepeat_async(x6, x6, x7, __ValuOpc::add);
    fenceValu((short)0);
    valuRepeat_async(x0, x0, x2, __ValuOpc::add);
    valuRepeat_async(x4, x4, x6, __ValuOpc::add);
    fenceValu((short)0);
    valuRepeat(x0, x0, x4, __ValuOpc::add);
    // residual_gate: branch = branch * gate; out = branch + residual.
    valuRepeat(x0, x0, gate_v, __ValuOpc::mul);

    valuRepeat(x0, x0, rr, __ValuOpc::add);

    vstSetTsize32B(2, 1);
    vstSetTstep32B_0(1, 256);
    vstSetTstep32B_1(0, 0);
    vstRepeat(x0, stream, offset);
    fenceVst((short)0);
  }
  rendezvous(block, 2 + bulk, 2, global_counter XINFER_ACTION_DOWN_ARGS);

  // ---------------- P3: pull the other seven cores' X rows ---------------
  // Each thread copies 256 contiguous values (16 vectors). Blocks 0,1 also
  // preload P4's staged modulation vectors with the same VLD configuration.
  vldSetTsize32B(16, 1);
  vldSetTstep32B_0(1, 16);
  vldSetTstep32B_1(0, 0);
  if (block < 7) {
    vlm_f16v16 *buf = pool;  // 16 vectors = one peer's four rows
    const unsigned short peer = (owner + 1 + (unsigned short)block) & 7;
    const int offset = (int)peer * 16 * 256 + (int)threadId * 256;
    vldRepeat_async(buf, peer_address(stream_global, peer, core_stride_hi), offset);
    if (block < 2) {
      // P4 layout: gamma 128, shift 192 in both modes; identity additionally
      // keeps gate' 256 and gamma'' 320 (x, y and scratch stay free).
#pragma rpu_hwloop_ul
      for (short q = 0; q < 4; ++q) {
        if (identity_next) {
          vldRepeat_async(pool + 128 + 16 * q, w_out, 256 * (int)q);
          vldRepeat_async(pool + 192 + 16 * q, w_out, 1024 + 256 * (int)q);
          vldRepeat_async(pool + 256 + 16 * q, w_out, 2048 + 256 * (int)q);
          vldRepeat_async(pool + 320 + 16 * q, w_out, 3072 + 256 * (int)q);
        } else {
          // Same P4 layout in normal/identity modes, so the norm body
          // has one call site in a fixed two-iteration scalar loop.
          vldRepeat_async(pool + 128 + 16 * q, norm_out, stage_offset + 256 * (int)q);
          vldRepeat_async(pool + 192 + 16 * q, norm_out, stage_offset + 1024 + 256 * (int)q);
        }
      }
    }
    fenceVld((short)0);
    vstSetTsize32B(16, 1);
    vstSetTstep32B_0(1, 16);
    vstSetTstep32B_1(0, 0);
    vstRepeat(buf, stream, offset);
    fenceVst((short)0);
  } else if (identity_next) {
    // Block 7 (no peer to pull) stages shift'' into partial[0:1024], free
    // since S2 (every peer has finished reading it).
    vlm_f16v16 *stage = pool;
#pragma rpu_hwloop_ul
    for (short j = 0; j < 4; ++j)
      vldExecute_async(stage + j, next2_row + 8, ((int)threadId * 4 + (int)j) * 32);
    fenceVld((short)0);
    vstSetTsize32B(4, 1);
    vstSetTstep32B_0(1, 16);
    vstSetTstep32B_1(0, 0);
    vstRepeat(stage, partial, (int)threadId * 64);
    fenceVst((short)0);
  }
  // Local join: P4 needs every pulled row and the staged gamma/shift.
  watom(XINFER_ACTION_DOWN_LOCAL0, 1, field_mode::core, modify_mode::add);
  if (block == 0) {
    watoms(XINFER_ACTION_DOWN_LOCAL0, 0, (short)(23 + 8 * bulk), field_mode::core, cmp_mode::gt, modify_mode::add);
    watom(XINFER_ACTION_DOWN_LOCAL1, 1, field_mode::core, modify_mode::add);
  }
  watoms(XINFER_ACTION_DOWN_LOCAL1, 0, (short)(2 + bulk), field_mode::core, cmp_mode::gt, modify_mode::add);
  fenceAtom((short)0);

  // ---------------- P4: adaptive RMSNorm, row per thread -----------------
  if (block < 2) {
    constexpr short chunks = 64;
    const int row_offset = ((int)block * 16 + (int)threadId) * 1024;
    vlm_f16v16 *x = pool, *y = pool + 64;
    vlm_f16v16 *gamma1 = pool + 128, *shift1 = pool + 192;
    vlm_f16v16 *gate1 = pool + 256, *gamma2 = pool + 320;
    vlm_f16v16 *gram = pool + 384, *norm = pool + 385, *scratch = pool + 386;

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
    vstSetTsize32B(chunks, 1);
    vstSetTstep32B_0(1, 16);
    vstSetTstep32B_1(0, 0);
    if (identity_next) vldRepeat_async(x, stream, row_offset);
    vldRepeat_async(y, stream, row_offset);
    fenceVld(0);
    // All lanes use the same call site for norm materialization.
    // The uniform identity_next flag selects the second pass.
    for (short pass = 0; pass < 2; ++pass) {
      if (pass == 0 || identity_next) {
        vlm_f16v16 *gamma = pass == 0 ? gamma1 : gamma2;
        adaptive_rmsnorm_row(y, gamma, shift1, gram, norm, scratch, epsilon);
        if (pass == 0 && identity_next) {
          vldSetTsize32B(chunks, 1);
          vldRepeat_async(shift1, partial, (int)0);
          valuSetTsize(chunks, 1);
          valuSetTstep0(1, 1, 1);
          valuSetTstep1(0, 0, 0);
          valuRepeat(y, y, gate1, __ValuOpc::mul);
          valuRepeat(gamma1, y, x, __ValuOpc::add);
          valuRepeat(y, y, x, __ValuOpc::add);
          vstRepeat_async(gamma1, norm_out, row_offset);
          fenceVld(0);
        } else {
          vstRepeat(y, pass == 0 ? norm_out : w_out, row_offset);
          fenceVst((short)0);
        }
      }
    }
  }

  // S3: every block arrives once more and never waits on a core counter
  // again; the leader then joins the last global phase and waits for all
  // eight cores. No core leaves (and later overwrites X) while a peer may
  // still read its X rows, and once any core leaves, nothing still depends on
  // launch-initialized core counters.
  watom(XINFER_ACTION_DOWN_LOCAL0, 1, field_mode::core, modify_mode::add);
  if (block == 0) {
    watoms(XINFER_ACTION_DOWN_LOCAL0, 0, (short)(31 + 8 * bulk), field_mode::core, cmp_mode::gt, modify_mode::add);
    watom((short)global_counter, 1, field_mode::global, modify_mode::add);
    watoms((short)global_counter, 0, (short)23, field_mode::global,
           cmp_mode::gt, modify_mode::add);
    fenceAtom((short)0);
  }
}
