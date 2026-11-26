// Copyright 2026 X-Square Robot. SPDX-License-Identifier: Apache-2.0
#pragma once
#include "__clang_rpu_builtin_vars.h"
using namespace rpu;

// Action residual chain for local [16,1024] FP16 tensors:
//   branch = round_fp16(branch * gate[1,1024])
//   dst    = round_fp16(branch + residual)
// dst may alias branch. One warp owns 16 rows and keeps both FP16 operation
// boundaries in VLM before the final store; multiplications and additions
// use hardware opcodes directly.
__rprog__ void XINFER_RESIDUAL_GATE_KERNEL(
    half *dst, half *branch, half *residual, half *gate) {
  constexpr short chunks = 64;
  unsigned short row = (unsigned short)blockIdx.x * 16 + threadId;
  if (row >= 16) return;
  __local__ f16v16 x[chunks], r[chunks], g[chunks];

  vldSetTsize32B(chunks, 1);
  vldSetTstep32B_0(1, 16);
  vldSetTstep32B_1(0, 0);
  vldRepeat_async(x, branch, (int)row * 1024);
  vldRepeat_async(r, residual, (int)row * 1024);
  vldRepeat_async(g, gate, (int)0);
  fenceVld(0);

  valuSetTsize(chunks, 1);
  valuSetTstep0(1, 1, 1);
  valuSetTstep1(0, 0, 0);
  valuRepeat(x, x, g, __ValuOpc::mul);
  valuRepeat(x, x, r, __ValuOpc::add);

  vstSetTsize32B(chunks, 1);
  vstSetTstep32B_0(1, 16);
  vstSetTstep32B_1(0, 0);
  vstRepeat(x, dst, (int)row * 1024);
}
