// Copyright 2026 X-Square Robot
// SPDX-License-Identifier: Apache-2.0
#pragma once

// Host launch plans for the fused kernels in fusion/. SPM arguments are
// per-core LocalSPM byte offsets, identical on all eight cores (broadcast).
// The public GELU arithmetic sequence requires numerical qualification with
// the selected device assets; fusion alone does not establish bitwise parity.

#include "common/rpu/attention_layout.hpp"
#include "common/rpu/kernel_abi.hpp"
#include "common/rpu/mlp_projection.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace xinfer::rpu::abi {

// qkv_bias_n384k1024 followed by in-place rope_f16 on Q and K (2 heads, D64, rotary
// 64, position_offset 0). cosine/sine are the position-selected FP16
// [table_rows, 32] tables; rows <= table_rows.
struct QkvRopeBiasN384K1024Args {
  std::uint32_t q, src, bias, rows;
  std::uint64_t weight_ddr_bytes;
  std::uint32_t k, v, cosine, sine, table_rows;
};
inline LaunchPlan qkv_rope_bias_n384k1024(const QkvRopeBiasN384K1024Args& a) {
  using namespace detail;
  auto p = qkv_up_gemm({Entry::QkvBiasN384K1024, a.q, a.src, a.bias, a.rows,
                        a.weight_ddr_bytes, a.k, a.v});
  require(a.rows <= a.table_rows && a.table_rows <= 65535,
          "Vision RoPE table is shorter than the projected rows");
  const std::uint64_t table_bytes = 64ULL * a.table_rows;
  const std::uint64_t part_bytes = 256ULL * a.rows;
  span(a.cosine, table_bytes); span(a.sine, table_bytes);
  for (const auto out : {a.q, a.k, a.v}) {
    disjoint(out, part_bytes, a.cosine, table_bytes);
    disjoint(out, part_bytes, a.sine, table_bytes);
  }
  p.entry = Entry::QkvRopeBiasN384K1024Fusion;
  u32(p, 14, a.cosine); u32(p, 16, a.sine); p.register_count = 18;
  p.grid[0] = 6;  // one N64 block per local Q/K/V head
  return p;
}

// up_n512k1024 followed by in-place gelu_f16 with approximate = 1 (tanh).
inline LaunchPlan up_gelu_n512k1024(std::uint32_t dst, std::uint32_t src,
                                 std::uint32_t bias, std::uint32_t rows,
                                 std::uint64_t weight_ddr_bytes) {
  auto p = qkv_up_gemm({Entry::UpN512K1024, dst, src, bias, rows, weight_ddr_bytes});
  p.entry = Entry::UpGeluN512K1024Fusion;
  return p;
}

// 3 x (kcache_insert_f16 + vcache_insert_f16) + flash_attention_varlen_f16
// with K/V read in place from the compact [S,2,64] Q/K/V projection outputs.
// Images are packed in order: image i starts at row sum(lengths[0..i)).
struct FlashAttentionSpmD384Args {
  std::uint32_t dst, query, key, value;
  std::vector<unsigned> lengths;
};
inline LaunchPlan flash_attention_spm_d384(const FlashAttentionSpmD384Args& a) {
  using namespace detail;
  const auto rows = attention::packed_rows(a.lengths);
  const auto bytes = 256ULL * rows;
  span(a.dst, bytes); span(a.query, bytes); span(a.key, bytes); span(a.value, bytes);
  for (const auto in : {a.query, a.key, a.value}) disjoint(a.dst, bytes, in, bytes);
  LaunchPlan p{Entry::FlashAttentionSpmD384Fusion};
  u32(p, 0, a.dst); u32(p, 2, a.query); u32(p, 4, a.key); u32(p, 6, a.value);
  for (unsigned i = 0; i < a.lengths.size(); ++i)
    p.regs[8 + i] = static_cast<std::uint16_t>(a.lengths[i]);
  // Eight resident blocks split the (image, head, Q16 group) list evenly.
  p.register_count = 11; p.cores = 8; p.grid = {8, 1, 1};
  return p;
}

// all_reduce_residual + layernorm_f16 on the [rows, 1024] residual stream
// (rows % 4 == 0 in [128, 868]; use the unfused pair otherwise), sequence
// parallel: x receives this core's shard of X = sum P + R (the
// residual is read only at that shard), norm receives LayerNorm(X)
// replicated. norm may equal partial or residual; x must be disjoint from
// all three. sync_epoch selects global counter 50 - sync_epoch, as for
// all_reduce_residual. Broadcast grid 8 to cores 0..7.
struct AllReduceResidualLayerNormArgs {
  std::uint32_t x, partial, residual, norm, gamma, beta, rows;
  float epsilon = 1.0e-6F;
  unsigned sync_epoch = 0;
};
inline LaunchPlan all_reduce_residual_layernorm(const AllReduceResidualLayerNormArgs& a) {
  using namespace detail;
  require(a.rows >= 128 && a.rows <= 868 && a.rows % 4 == 0,
          "Vision all-reduce LayerNorm needs rows % 4 == 0 in [128,868]");
  require(a.sync_epoch <= 47, "all-reduce sync epoch exceeds the graph envelope");
  const std::uint64_t bytes = 2048ULL * a.rows, affine = 2048;
  span(a.x, bytes); span(a.partial, bytes); span(a.residual, bytes); span(a.norm, bytes);
  span(a.gamma, affine); span(a.beta, affine);
  disjoint(a.x, bytes, a.partial, bytes); disjoint(a.x, bytes, a.residual, bytes);
  disjoint(a.x, bytes, a.norm, bytes); disjoint(a.partial, bytes, a.residual, bytes);
  if (a.norm != a.partial && a.norm != a.residual) {
    disjoint(a.norm, bytes, a.partial, bytes); disjoint(a.norm, bytes, a.residual, bytes);
  }
  for (const auto affine_at : {a.gamma, a.beta}) {
    disjoint(affine_at, affine, a.x, bytes); disjoint(affine_at, affine, a.norm, bytes);
  }
  const auto eps = fp32_to_fp16(a.epsilon);
  require(positive_finite_fp16(eps), "LayerNorm epsilon must be positive finite FP16");
  LaunchPlan p{Entry::AllReduceResidualLayerNormFusion};
  u32(p, 0, global_spm_base + a.x); u32(p, 2, global_spm_base + a.partial);
  u32(p, 4, a.residual); p.regs[6] = static_cast<std::uint16_t>(a.rows);
  p.regs[7] = core_stride_bytes >> 16; p.regs[8] = 8;
  p.regs[9] = static_cast<std::uint16_t>(50 - a.sync_epoch);
  u32(p, 10, global_spm_base + a.norm); u32(p, 12, a.x); u32(p, 14, a.partial);
  u32(p, 16, a.norm); u32(p, 18, a.gamma); u32(p, 20, a.beta);
  const float scale = 1.0F / 1024;
  std::uint32_t scale_bits;
  std::memcpy(&scale_bits, &scale, sizeof(scale_bits));
  u32(p, 22, scale_bits); p.regs[24] = eps;
  p.register_count = 25; p.cores = 8; p.grid = {8, 1, 1};
  return p;
}

// fc1_n512k4096 followed by in-place gelu_f16 with approximate = 0 (erf).
inline LaunchPlan fc1_gelu_n512k4096(std::uint32_t dst, std::uint32_t src,
                                  std::uint32_t bias, std::uint32_t rows,
                                  std::uint64_t weight_ddr_bytes) {
  auto p = mlp_projection({Entry::Fc1N512K4096, dst, src, bias, rows, weight_ddr_bytes});
  p.entry = Entry::Fc1GeluN512K4096Fusion;
  return p;
}


}  // namespace xinfer::rpu::abi
