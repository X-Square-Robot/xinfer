// Copyright 2026 X-Square Robot. SPDX-License-Identifier: Apache-2.0
#pragma once
#if XINFER_VISION_REDUCE_WINDOW
#define XINFER_VISION_REDUCE_FORMALS , unsigned short local_counter0, unsigned short local_counter1
#define XINFER_VISION_REDUCE_ARGS , local_counter0, local_counter1
#define XINFER_VISION_REDUCE_LOCAL0 ((short)local_counter0)
#define XINFER_VISION_REDUCE_LOCAL1 ((short)local_counter1)
#else
#define XINFER_VISION_REDUCE_FORMALS
#define XINFER_VISION_REDUCE_ARGS
#define XINFER_VISION_REDUCE_LOCAL0 0
#define XINFER_VISION_REDUCE_LOCAL1 1
#endif
#include "__clang_rpu_builtin_vars.h"
using namespace rpu;

// all_reduce_residual + layernorm_f16 for the TP8 Vision
// residual stream ([rows, 1024] FP16, rows 1..868), sequence parallel:
//   X = sum_c P_c + R          (each core keeps only its own X shard)
//   N = LayerNorm(X; gamma, beta, eps), replicated on all eight cores.
// The standalone chain reduces, all-gathers X, and then every core
// normalizes all rows. Here each core normalizes only the rows of its own
// shard and the all-gather moves N instead of X: the same link traffic, an
// eighth of the LayerNorm work and one launch less. R is read only at this
// core's shard (as all_reduce_residual does), so the next reduction can use
// the shard-valid X as its residual.
//
// Bit-identical to the chain. P1 is all_reduce_residual's reduce-scatter
// unchanged (same owner shards and FP16 tree; X shard c stored locally).
// LayerNorm works on the same global 16-row tiles as layernorm_f16 (rows
// 16k..16k+15, tail rows re-reading the last row) and runs its per-tile
// H <= 2048 code unchanged, including the warp-wide variance-overflow choice.
// Each core normalizes every tile that overlaps its all-reduce shard (the two
// neighbours both compute a shared boundary tile, bit-identically), pulling
// the tile's rows outside the shard from cores owner - 1 and owner + 1. Every
// core then holds N for exactly its own equal shard, and P2 is
// all_reduce_residual's balanced all-gather (owning whole tiles instead gave
// core 0 a third more vectors and slowed every gather step to its pace).
//   entry (global 1)  every producer has finished P
//   P1                reduce-scatter steps, X shard stored locally
//   mid (global 2)    every X shard is final
//   LN                block b normalizes overlapping tile kfirst + b; the
//                     first/last tile's block first pulls its outside rows
//   ln (global 3)     every N tile is final on its owner
//   P2                all-gather of N tiles, one source core per step
//   exit (global 4)   no peer still reads this core's X or N
// Preserve the declared barrier schedule and admitted row envelope.
// Step barriers pace peer reads; the final two gathers share a step.
// Keep peer and local loads in complete, explicitly separated phases.
// Communication only reads peer SPM; every store is local. N may alias P (no
// peer reads P after mid) or R (R is read only in P1); X must not alias P,
// R or N. ABI: X global [0:1], P global [2:3], R local [4:5], rows [6], core
// stride/64KiB [7], launched blocks/core [8] (must be 8), global sync
// counter [9] (3..50), N global [10:11], X local [12:13], P local [14:15],
// N local [16:17], gamma [18:19], beta [20:21], 1/1024 FP32 bits low [22]
// and high [23], epsilon [24]. grid.x = 8, cores 0..7, all blocks resident;
// core counters 0, 1 and the selected global counter must be zero.

// Pointer into the per-thread VLM pool (address space only, no storage).
typedef __attribute__((address_space(1))) f16v16 vlm_f16v16;

static __attribute__((always_inline)) __SRAM half *peer_address(
    __SRAM half *base, unsigned short peer, unsigned short stride_hi) {
  int bank_bytes = (int)(unsigned short)(peer * stride_hi) << 16;
  return reinterpret_cast<__SRAM half *>(reinterpret_cast<int>(base) + bank_bytes);
}

// all_reduce_residual's barrier: core counter 0 counts arrivals, core
// counter 1 releases; GPHASE > 0 makes the leader join the global counter.
template <short PHASE, short GPHASE, bool RELEASE = true>
__attribute__((always_inline)) inline void barrier(unsigned short block,
                                                   unsigned short counter XINFER_VISION_REDUCE_FORMALS) {
  watom(XINFER_VISION_REDUCE_LOCAL0, 1, field_mode::core, modify_mode::add);
  if (block == 0) {
    watoms(XINFER_VISION_REDUCE_LOCAL0, 0, (short)(8 * PHASE - 1), field_mode::core, cmp_mode::gt,
           modify_mode::add);
    if (GPHASE > 0) {
      watom((short)counter, 1, field_mode::global, modify_mode::add);
      watoms((short)counter, 0, (short)(8 * GPHASE - 1), field_mode::global,
             cmp_mode::gt, modify_mode::add);
    }
    if (RELEASE) watom(XINFER_VISION_REDUCE_LOCAL1, 1, field_mode::core, modify_mode::add);
  }
  if (RELEASE)
    watoms(XINFER_VISION_REDUCE_LOCAL1, 0, (short)(PHASE - 1), field_mode::core, cmp_mode::gt,
           modify_mode::add);
  fenceAtom((short)0);
}

__rprog__ void XINFER_VISION_REDUCE_KERNEL(
    __SRAM half *x_global, __SRAM half *partial, half *residual,
    unsigned short rows, unsigned short core_stride_hi,
    unsigned short blocks_per_core, unsigned short global_sync_counter,
    __SRAM half *norm_global, half *x_local, half *partial_local,
    half *norm_local, half *gamma_src, half *beta_src,
    unsigned short mean_scale_low, unsigned short mean_scale_high,
    half epsilon XINFER_VISION_REDUCE_FORMALS) {
  if (rows < 128 || rows > 868 || (rows & 3) != 0 || blocks_per_core != 8 ||
      global_sync_counter < 3 || global_sync_counter > 50) return;
#if XINFER_VISION_REDUCE_WINDOW
  if (local_counter0 < 2 || local_counter0 > 32 || (local_counter0 & 1) ||
      local_counter1 != local_counter0 + 1 ||
      global_sync_counter != 34 + (local_counter0 - 2) / 2) return;
#endif
  const unsigned short total_vectors = rows * 4;
  const unsigned short base_vectors = total_vectors / 8;
  const unsigned short extra = total_vectors & 7;
  const unsigned short owner = coreId_u;
  const unsigned short block = (unsigned short)blockIdx.x;
  const unsigned short G = global_sync_counter;
  auto shard_start = [&](unsigned short p) -> unsigned short {
    return p * base_vectors + (p < extra ? p : extra);
  };
  auto shard_end = [&](unsigned short p) -> unsigned short {
    return shard_start(p) + base_vectors + (p < extra);
  };
  // This block's slice of shard p (all_reduce_residual's split).
  auto slice_count = [&](unsigned short p) -> short {
    const unsigned short n = base_vectors + (p < extra);
    return (short)(n / 8 + (block < (n & 7)));
  };
  auto slice_offset = [&](unsigned short p) -> int {
    const unsigned short n = base_vectors + (p < extra);
    const unsigned short v = shard_start(p) + block * (n / 8) +
                             (block < (n & 7) ? block : (n & 7));
    return (int)v * 256 + (int)threadId * 16;
  };
  auto x = [&](unsigned short i) {
    return peer_address(partial, (owner + i) & 7, core_stride_hi);
  };

  // One VLM arena: P1 uses four 64-vector operands (slices <= 55 vectors at
  // 868 rows), all VLD Repeat destinations wholly below vector 256; the
  // residual is read into B after the tree. LayerNorm uses X, gamma and beta
  // (64 vectors each) plus four row vectors; P2 two 64-vector buffers.
  // Reuse the explicit arena across phases.
  __local__ f16v16 arena[256];
  auto *A = arena, *B = arena + 64, *C = arena + 128, *D = arena + 192;
  const short count = slice_count(owner);  // >= 8 in the admitted envelope
  const int offset = slice_offset(owner);

  barrier<1, 1>(block, G XINFER_VISION_REDUCE_ARGS);

  // ---------------- P1: all_reduce_residual's reduce-scatter -------------
  vcfg_set_valu(vcfg_valu_target::imm2, (half)1.0f);
  vldSetTsize32B(count, 1);
  vldSetTstep32B_0(1, 256);
  vldSetTstep32B_1(0, 0);
  valuSetTsize(count, 1);
  valuSetTstep0(1, 1, 1);
  valuSetTstep1(0, 0, 0);
  vldRepeat_async(A, partial_local, offset);
  vldRepeat_async(B, x(1), offset);
  fenceVld((short)0);
  barrier<2, 0>(block, G XINFER_VISION_REDUCE_ARGS);
  vldRepeat_async(C, x(2), offset);
  valuRepeat(A, A, B, __ValuOpc::add);              // t01
  fenceVld((short)0);
  barrier<3, 0>(block, G XINFER_VISION_REDUCE_ARGS);
  vldRepeat_async(B, x(3), offset);
  fenceVld((short)0);
  barrier<4, 0>(block, G XINFER_VISION_REDUCE_ARGS);
  vldRepeat_async(D, x(4), offset);
  valuRepeat(C, C, B, __ValuOpc::add);              // t23
  valuRepeat(A, A, C, __ValuOpc::add);              // t0123
  fenceVld((short)0);
  barrier<5, 0>(block, G XINFER_VISION_REDUCE_ARGS);
  vldRepeat_async(B, x(5), offset);
  fenceVld((short)0);
  barrier<6, 0>(block, G XINFER_VISION_REDUCE_ARGS);
  vldRepeat_async(C, x(6), offset);
  valuRepeat(D, D, B, __ValuOpc::add);              // t45
  fenceVld((short)0);
  barrier<7, 0>(block, G XINFER_VISION_REDUCE_ARGS);
  vldRepeat_async(B, x(7), offset);
  fenceVld((short)0);
  valuRepeat(C, C, B, __ValuOpc::add);              // t67
  valuRepeat(D, D, C, __ValuOpc::add);              // t4567
  valuRepeat(A, A, D, __ValuOpc::add);
  vldRepeat_async(B, residual, offset);
  fenceVld((short)0);
  valuRepeat(A, A, B, __ValuOpc::add);
  vstSetTsize32B(count, 1);
  vstSetTstep32B_0(1, 256);
  vstSetTstep32B_1(0, 0);
  vstRepeat(A, x_local, offset);
  fenceVst((short)0);

  barrier<8, 2>(block, G XINFER_VISION_REDUCE_ARGS);

  // Tiles overlapping this shard: k in [kfirst, klast).
  const unsigned short my_start = shard_start(owner), my_end = shard_end(owner);
  const unsigned short kfirst = my_start / 64, klast = (my_end + 63) / 64;
  const unsigned short need_start = kfirst * 64;
  const unsigned short need_end =
      klast * 64 < total_vectors ? klast * 64 : total_vectors;

  // ---------------- LayerNorm: layernorm_f16's per-tile code ---------------
  if (kfirst + block < klast) {
    // Rows of the first/last tile outside this shard: in the envelope each
    // neighbouring shard has >= 64 vectors, so they lie entirely in core
    // owner - 1 (leading) or owner + 1 (trailing). Each copy leaves its
    // branch in the local VLD space the other path is in.
    vldSetTstep32B_0(1, 256);
    vldSetTstep32B_1(0, 0);
    vstSetTstep32B_0(1, 256);
    vstSetTstep32B_1(0, 0);
    if (block == 0 && need_start < my_start) {
      const int at = (int)need_start * 256 + (int)threadId * 16;
      vldSetTsize32B((short)(my_start - need_start), 1);
      vldRepeat_async(arena, peer_address(x_global, owner - 1, core_stride_hi), at);
      fenceVld((short)0);
      vstSetTsize32B((short)(my_start - need_start), 1);
      vstRepeat(arena, x_local, at);
      fenceVst((short)0);
      vldSetTsize32B(1, 1);
      vldRepeat(arena, x_local, (int)threadId * 16);
    }
    if (kfirst + block + 1 == klast && need_end > my_end) {
      const int at = (int)my_end * 256 + (int)threadId * 16;
      vldSetTsize32B((short)(need_end - my_end), 1);
      vldRepeat_async(arena, peer_address(x_global, owner + 1, core_stride_hi), at);
      fenceVld((short)0);
      vstSetTsize32B((short)(need_end - my_end), 1);
      vstRepeat(arena, x_local, at);
      fenceVst((short)0);
      vldSetTsize32B(1, 1);
      vldRepeat(arena, x_local, (int)threadId * 16);
    }
    const unsigned short tid = threadId;
    const unsigned short chunks = 64;
    auto *xv = arena, *gamma = arena + 64, *beta = arena + 128;
    auto *ones = arena + 192, *mean_hi = arena + 193, *mean_lo = arena + 194;
    auto *inv_std = arena + 195;
    vldSetTsize32B(chunks, 1);
    vldSetTstep32B_0(1, 16);
    vldSetTstep32B_1(0, 0);
    vldRepeat(gamma, gamma_src, (int)0);
    vldRepeat(beta, beta_src, (int)0);
    valuExecute(ones, (half)1.0f, (half)0.0f, __ValuOpc::add);
    vcfg_set_vmat(vcfg_vmat_target::scale_allch_l, (short)mean_scale_low);
    vcfg_set_vmat(vcfg_vmat_target::scale_allch_h, (short)mean_scale_high);
    vmatSetTsize(chunks, 1);
    vmatSetTstep1(0, 0, 0);
    valuSetTsize(chunks, 1);
    valuSetTstep1(0, 0, 0);
    vstSetTsize32B(chunks, 1);
    vstSetTstep32B_0(1, 16);
    vstSetTstep32B_1(0, 0);
    unsigned int row = (unsigned int)(kfirst + block) * 16 + tid;
    unsigned int load_row = row < rows ? row : (unsigned int)rows - 1;
    int row_offset = (int)(load_row * 1024u);
    vldRepeat(xv, x_local, row_offset);
    vcfg_set_valu(vcfg_valu_target::imm2, (half)1.0f);
    vmatSetTstep0(0, 1, 0);
    LoopOutInfo outlp[1] = {{(short)0}};
    LoopClrInfo clrlp[1] = {{(short)0}};
    LoopOutLPAttr out(outlp);
    LoopClrLPAttr clr(clrlp);
    vmatRepeat(mean_hi, xv, ones, &out, &clr);
    valuSetTstep0(1, 1, 0);
    valuRepeat(xv, xv, mean_hi, __ValuOpc::sub);
    vmatRepeat(mean_lo, xv, ones, &out, &clr);
    valuRepeat(xv, xv, mean_lo, __ValuOpc::sub);
    vmatSetTstep0(0, 1, 1);
    vmatRepeat(inv_std, xv, xv, &out, &clr);
    half variance;
    vcfg_set_valu(vcfg_valu_target::imm2, inv_std, (short)tid);
    valuExecute(mean_hi, (half)1.0f, (half)0.0f, __ValuOpc::add);
    vcfg_set_valu(vcfg_valu_target::imm2, (half)1.0f);
    vsfuScalar(variance, mean_hi, (short)0, __ShuffleMode::Psel);
    unsigned short variance_bits = reinterpret_cast<unsigned short&>(variance);
    unsigned short min_complement = sumint((unsigned short)(0xffffu - variance_bits));
    if (min_complement < 0x8400u) {
      vcfg_set_vmat(vcfg_vmat_target::scale_allch_h,
                    (short)(mean_scale_high - 0x0800u));
      vmatRepeat(mean_lo, xv, xv, &out, &clr);
      vcfg_set_vmat(vcfg_vmat_target::scale_allch_h, (short)mean_scale_high);
      vcfg_set_valu(vcfg_valu_target::imm2, mean_lo, (short)tid);
      valuExecute(mean_hi, (half)1.0f, (half)0.0f, __ValuOpc::add);
      vcfg_set_valu(vcfg_valu_target::imm2, (half)1.0f);
      half scaled_variance;
      vsfuScalar(scaled_variance, mean_hi, (short)0, __ShuffleMode::Psel);
      half chosen_variance = variance_bits >= 0x7c00u ? scaled_variance : variance;
      half epsilon_scale = variance_bits >= 0x7c00u ? (half)0.0000152587890625f : (half)1.0f;
      half inverse_scale = variance_bits >= 0x7c00u ? (half)0.00390625f : (half)1.0f;
      valuExecute(mean_hi, epsilon, epsilon_scale, __ValuOpc::mul);
      valuExecute(inv_std, chosen_variance, mean_hi, __ValuOpc::add);
      valuExecute(inv_std, inv_std, (half)0.0f, __ValuOpc::sqrt_reciprocal);
      valuExecute(inv_std, inv_std, inverse_scale, __ValuOpc::mul);
    } else {
      valuExecute(inv_std, inv_std, epsilon, __ValuOpc::add);
      valuExecute(inv_std, inv_std, (half)0.0f, __ValuOpc::sqrt_reciprocal);
    }
    vcfg_set_valu(vcfg_valu_target::imm2, inv_std, (short)tid);
    valuSetTstep0(1, 1, 1);
    valuRepeat(xv, xv, gamma, __ValuOpc::mul);
    vcfg_set_valu(vcfg_valu_target::imm2, (half)1.0f);
    valuRepeat(xv, xv, beta, __ValuOpc::add);
    if (row < rows) vstRepeat(xv, norm_local, row_offset);
  }

  barrier<9, 3>(block, G XINFER_VISION_REDUCE_ARGS);

  // ---------------- P2: all_reduce_residual's all-gather, on N -------------
  vldSetTstep32B_0(1, 256);
  vldSetTstep32B_1(0, 0);
  vstSetTstep32B_0(1, 256);
  vstSetTstep32B_1(0, 0);
  auto gather = [&](unsigned short j, vlm_f16v16 *buf) {
    const unsigned short p = (owner + j) & 7;
    const short m = slice_count(p);  // >= 8 vectors in the envelope
    const int off = slice_offset(p);
    vldSetTsize32B(m, 1);
    vldRepeat_async(buf, peer_address(norm_global, p, core_stride_hi), off);
    fenceVst((short)0);
    fenceVld((short)0);
    vstSetTsize32B(m, 1);
    vstRepeat_async(buf, norm_local, off);
  };
  gather(1, A); barrier<10, 0>(block, G XINFER_VISION_REDUCE_ARGS);
  gather(2, B); barrier<11, 0>(block, G XINFER_VISION_REDUCE_ARGS);
  gather(3, A); barrier<12, 0>(block, G XINFER_VISION_REDUCE_ARGS);
  gather(4, B); barrier<13, 0>(block, G XINFER_VISION_REDUCE_ARGS);
  gather(5, A); barrier<14, 0>(block, G XINFER_VISION_REDUCE_ARGS);
  gather(6, B);
  gather(7, A);
  fenceVst((short)0);

  barrier<15, 4, false>(block, G XINFER_VISION_REDUCE_ARGS);
}
