// Copyright 2026 X-Square Robot. SPDX-License-Identifier: Apache-2.0
#pragma once
#if XINFER_REDUCE_WINDOW
#define XINFER_REDUCE_FORMALS , unsigned short local_counter0, unsigned short local_counter1
#define XINFER_REDUCE_ARGS , local_counter0, local_counter1
#define XINFER_REDUCE_LOCAL0 ((short)local_counter0)
#define XINFER_REDUCE_LOCAL1 ((short)local_counter1)
#else
#define XINFER_REDUCE_FORMALS
#define XINFER_REDUCE_ARGS
#define XINFER_REDUCE_LOCAL0 0
#define XINFER_REDUCE_LOCAL1 1
#endif
#include "__clang_rpu_builtin_vars.h"
using namespace rpu;

// Pointer into the per-thread VLM pool (address space only, no storage).
typedef __attribute__((address_space(1))) f16v16 vlm_f16v16;

static __attribute__((always_inline)) __SRAM half *peer_address(
    __SRAM half *base, unsigned short peer, unsigned short stride_hi) {
  // Compute the peer offset from the declared stride.
  int bank_bytes = (int)(unsigned short)(peer * stride_hi) << 16;
  return reinterpret_cast<__SRAM half *>(reinterpret_cast<int>(base) + bank_bytes);
}

// Barrier PHASE of the eight resident blocks. Core counter 0 counts arrivals
// (8 per barrier) and core counter 1 releases (1 per barrier). If GPHASE > 0
// the leader (block 0) also joins global barrier GPHASE on the occurrence
// counter (8 core arrivals per barrier). Without RELEASE the other blocks
// leave at once; the core still completes only when its leader leaves.
// Barrier thresholds are compile-time constants.
template <short PHASE, short GPHASE, bool RELEASE = true>
__attribute__((always_inline)) inline void barrier(unsigned short block,
                                                   unsigned short counter XINFER_REDUCE_FORMALS) {
  watom(XINFER_REDUCE_LOCAL0, 1, field_mode::core, modify_mode::add);
  if (block == 0) {
    watoms(XINFER_REDUCE_LOCAL0, 0, (short)(8 * PHASE - 1), field_mode::core, cmp_mode::gt,
           modify_mode::add);
    if (GPHASE > 0) {
      watom((short)counter, 1, field_mode::global, modify_mode::add);
      watoms((short)counter, 0, (short)(8 * GPHASE - 1), field_mode::global,
             cmp_mode::gt, modify_mode::add);
    }
    if (RELEASE) watom(XINFER_REDUCE_LOCAL1, 1, field_mode::core, modify_mode::add);
  }
  if (RELEASE)
    watoms(XINFER_REDUCE_LOCAL1, 0, (short)(PHASE - 1), field_mode::core, cmp_mode::gt,
           modify_mode::add);
  fenceAtom((short)0);
}

// FP16 TP8 sum of row-major [rows, cols], optionally plus one residual.
// H1024 rows=1..868; H2048 rows=1..512. The result is replicated on all eight
// cores. Owner c reduces the contiguous 256-element vectors of shard c (the
// first total_vectors%8 owners receive one extra vector) in the balanced FP16
// tree over the inputs of cores c, c+1, ..., c+7, then adds the residual.
// Inputs and residual are never written; the output must be disjoint.
//
// Communication reads peer SPM and stores locally.
// Transfers move in permutation steps: at step j every block of core c
// reads core c+j, and a core-local barrier keeps the blocks on one peer.
// Block b handles 1/8 of every shard, keeping partial sums in VLM.
//   entry (global)   every producer has finished its input
//   P1 reduce-scatter x0 local, x1..x7 from cores c+1..c+7 (steps 2..7),
//                    tree, residual, local store of shard c
//   mid (global)     every shard is final
//   P2 all-gather    step j copies shard c+j from that core (steps 9..14)
//   exit (global)    no peer can still read this core's output
// One global counter advances 0->8->16->24.
//
// ABI slots: output global core-0 [0:1], input global core-0 [2:3], residual
// LOCAL [4:5], rows [6], core stride/64KiB [7], reserved [8], cols [9],
// with_residual [10], launched blocks/core [11] (must be 8), global sync
// counter [12], output LOCAL [14:15], input LOCAL [16:17].
// Paired address arguments use even slots. grid.x=8, grid.y=z=1, cores=0..7; all
// eight blocks must be resident. Core counters 0, 1 and the selected global
// counter must be zero in every launch packet.
__rprog__ void XINFER_REDUCE_KERNEL(__SRAM half *output,
                                   __SRAM half *input,
                                   half *residual,
                                   unsigned short rows,
                                   unsigned short core_stride_hi,
                                   unsigned short reserved,
                                   unsigned short cols,
                                   unsigned short with_residual,
                                   unsigned short blocks_per_core,
                                   unsigned short global_sync_counter,
                                   half *output_local,
                                   half *input_local XINFER_REDUCE_FORMALS) {
  if (rows == 0 || rows > 868 || (cols != 1024 && cols != 2048) ||
      (cols == 2048 && rows > 512) || with_residual > 1 ||
      blocks_per_core != 8 || global_sync_counter < 3 ||
      global_sync_counter > 50) return;
  (void)reserved;
#if XINFER_REDUCE_WINDOW
  if (local_counter0 < 2 || local_counter0 > 30 || (local_counter0 & 1) ||
      local_counter1 != local_counter0 + 1) return;
#endif
  const unsigned short total_vectors = rows * (cols / 256);
  const unsigned short base_vectors = total_vectors / 8;
  const unsigned short extra = total_vectors & 7;
  const unsigned short owner = coreId_u;
  const unsigned short block = (unsigned short)blockIdx.x;
  const unsigned short G = global_sync_counter;

  // This block's slice of shard p: count vectors at a per-thread offset.
  auto slice_count = [&](unsigned short p) -> short {
    const unsigned short n = base_vectors + (p < extra);
    return (short)(n / 8 + (block < (n & 7)));
  };
  auto slice_offset = [&](unsigned short p) -> int {
    const unsigned short n = base_vectors + (p < extra);
    const unsigned short s = p * base_vectors + (p < extra ? p : extra);
    const unsigned short v =
        s + block * (n / 8) + (block < (n & 7) ? block : (n & 7));
    return (int)v * 256 + (int)threadId * 16;
  };
  auto x = [&](unsigned short i) {
    return peer_address(input, (owner + i) & 7, core_stride_hi);
  };

  // A running sum, B/C/D tree operands: 64 vectors each cover the largest
  // slice (H2048 M512: 512 vectors per shard; H1024 M868: 55). Every operand
  // is a VLD Repeat destination and must lie wholly below vector 256, so the
  // four live operands fill exactly [0,256) of one reviewed arena and the
  // residual is read into B once B is free, after the tree. The FP16 sum
  // order (tree, then + residual) is unchanged.
  __local__ f16v16 arena[256];
  auto *A = arena, *B = arena + 64, *C = arena + 128, *D = arena + 192;
  const short count = slice_count(owner);
  const int offset = slice_offset(owner);
  const bool live = count > 0;

  barrier<1, 1>(block, G XINFER_REDUCE_ARGS);

  // ---------------- P1: reduce-scatter, FP16 tree ------------------------
  // Each step issues one peer read and combines operands that already
  // arrived; the step barrier holds every block on that peer.
  vcfg_set_valu(vcfg_valu_target::imm2, (half)1.0f);
  vldSetTsize32B(count, 1);
  vldSetTstep32B_0(1, 256);
  vldSetTstep32B_1(0, 0);
  valuSetTsize(count, 1);
  valuSetTstep0(1, 1, 1);
  valuSetTstep1(0, 0, 0);
  if (live) {
    vldRepeat_async(A, input_local, offset);
    vldRepeat_async(B, x(1), offset);
    fenceVld((short)0);
  }
  barrier<2, 0>(block, G XINFER_REDUCE_ARGS);
  if (live) {
    vldRepeat_async(C, x(2), offset);
    valuRepeat(A, A, B, __ValuOpc::add);            // t01
    fenceVld((short)0);
  }
  barrier<3, 0>(block, G XINFER_REDUCE_ARGS);
  if (live) {
    vldRepeat_async(B, x(3), offset);
    fenceVld((short)0);
  }
  barrier<4, 0>(block, G XINFER_REDUCE_ARGS);
  if (live) {
    vldRepeat_async(D, x(4), offset);
    valuRepeat(C, C, B, __ValuOpc::add);            // t23
    valuRepeat(A, A, C, __ValuOpc::add);            // t0123
    fenceVld((short)0);
  }
  barrier<5, 0>(block, G XINFER_REDUCE_ARGS);
  if (live) {
    vldRepeat_async(B, x(5), offset);
    fenceVld((short)0);
  }
  barrier<6, 0>(block, G XINFER_REDUCE_ARGS);
  if (live) {
    vldRepeat_async(C, x(6), offset);
    valuRepeat(D, D, B, __ValuOpc::add);            // t45
    fenceVld((short)0);
  }
  barrier<7, 0>(block, G XINFER_REDUCE_ARGS);
  if (live) {
    vldRepeat_async(B, x(7), offset);
    fenceVld((short)0);
    valuRepeat(C, C, B, __ValuOpc::add);            // t67
    valuRepeat(D, D, C, __ValuOpc::add);            // t4567
    valuRepeat(A, A, D, __ValuOpc::add);
    if (with_residual) {
      vldRepeat_async(B, residual, offset);
      fenceVld((short)0);
      valuRepeat(A, A, B, __ValuOpc::add);
    }
    vstSetTsize32B(count, 1);
    vstSetTstep32B_0(1, 256);
    vstSetTstep32B_1(0, 0);
    vstRepeat(A, output_local, offset);
    fenceVst((short)0);
  }

  barrier<8, 2>(block, G XINFER_REDUCE_ARGS);

  // ---------------- P2: all-gather, one source core per step -------------
  // Step j reads shard c+j into one buffer while the previous step's local
  // store drains the other.
  vldSetTstep32B_0(1, 256);
  vldSetTstep32B_1(0, 0);
  vstSetTstep32B_0(1, 256);
  vstSetTstep32B_1(0, 0);
  auto gather = [&](unsigned short j, vlm_f16v16 *buf) {
    const unsigned short p = (owner + j) & 7;
    const short n = slice_count(p);
    const int off = slice_offset(p);
    if (n > 0) {
      vldSetTsize32B(n, 1);
      vldRepeat_async(buf, peer_address(output, p, core_stride_hi), off);
    }
    fenceVst((short)0);
    fenceVld((short)0);
    if (n > 0) {
      vstSetTsize32B(n, 1);
      vstRepeat_async(buf, output_local, off);
    }
  };
  gather(1, A); barrier<9, 0>(block, G XINFER_REDUCE_ARGS);
  gather(2, B); barrier<10, 0>(block, G XINFER_REDUCE_ARGS);
  gather(3, A); barrier<11, 0>(block, G XINFER_REDUCE_ARGS);
  gather(4, B); barrier<12, 0>(block, G XINFER_REDUCE_ARGS);
  gather(5, A); barrier<13, 0>(block, G XINFER_REDUCE_ARGS);
  gather(6, B); barrier<14, 0>(block, G XINFER_REDUCE_ARGS);
  gather(7, A);
  fenceVst((short)0);

#if XINFER_REDUCE_WINDOW
  barrier<15, 3, true>(block, G XINFER_REDUCE_ARGS);
#else
  barrier<15, 3, false>(block, G XINFER_REDUCE_ARGS);
#endif
}
