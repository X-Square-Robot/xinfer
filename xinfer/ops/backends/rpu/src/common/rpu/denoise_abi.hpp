// Copyright 2026 X-Square Robot
// SPDX-License-Identifier: Apache-2.0
#pragma once

// Host launch plans for the Action fused kernels in
// src/fusion/; the catalog declares each public entry signature.
// Register layouts follow each kernel's declared launch ABI. SPM arguments are per-core
// LocalSPM byte offsets that must be identical on all eight cores (broadcast
// launch); DDR arguments are 256-byte aligned device addresses. apply() also
// clears the atomic counters before each launch so graphs can be replayed.

#include "common/rpu/fp16.hpp"
#include "common/rpu/kernel_abi.hpp"

#include <array>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

// DownAllReduceNormArgs::rtc_rows: per-row clean/step modulation in the M16
// fused down program (training-time RTC).
#define XINFER_RPU_OPS_HAS_ACTION_FUSION_RTC 1

namespace xinfer::rpu::denoise {

struct Plan {
  std::string name;
  std::array<std::uint16_t, 64> regs{};
  unsigned register_count = 0;
  std::array<std::uint16_t, 3> grid{8, 1, 1};
  unsigned cores = 8;
  std::vector<unsigned> zero_counters;  // atomic counter ids
};

namespace detail {
using xinfer::rpu::abi::detail::require;
inline void u32(Plan& p, unsigned reg, std::uint32_t v) {
  p.regs[reg] = static_cast<std::uint16_t>(v);
  p.regs[reg + 1] = static_cast<std::uint16_t>(v >> 16);
}
inline std::uint32_t ddr_units(std::uint64_t address, std::uint64_t bytes) {
  xinfer::rpu::abi::detail::ddr_span(address, bytes);
  return static_cast<std::uint32_t>(address >> 8);
}
inline void spm(std::uint32_t offset, std::uint64_t bytes) {
  xinfer::rpu::abi::detail::span(offset, bytes);
}
inline void disjoint(std::uint32_t a, std::uint64_t an, std::uint32_t b, std::uint64_t bn) {
  xinfer::rpu::abi::detail::disjoint(a, an, b, bn);
}
inline std::uint32_t fp32_bits(float v) {
  std::uint32_t bits;
  std::memcpy(&bits, &v, sizeof(bits));
  return bits;
}
inline std::uint32_t global_spm(std::uint32_t local) {
  return xinfer::rpu::abi::global_spm_base + local;
}
}  // namespace detail

inline constexpr std::uint64_t kTensor = 2ULL * 32 * 1024;   // [32,1024] FP16
inline constexpr std::uint64_t kLocal = 2ULL * 32 * 256;     // [32,256] FP16
inline constexpr std::uint64_t kWeight = 2ULL * 1024 * 2048; // 4 MiB TP8 weight

// gate/up column GEMMs + silu_mul: out = silu(A Wg^T) * (A Wu^T).
struct GateUpSiluArgs {
  std::uint32_t a, out;
  std::uint64_t gate_weight, up_weight;  // production transformed DDR weights
  std::uint32_t rows = 32;
};
inline Plan gate_up_silu(const GateUpSiluArgs& x) {
  using namespace detail;
  xinfer::rpu::abi::detail::compact_row_bucket_rows(x.rows);
  const std::uint64_t kTensor = 2ULL * x.rows * 1024;
  const std::uint64_t kLocal = 2ULL * x.rows * 256;
  spm(x.a, kTensor); spm(x.out, kLocal); disjoint(x.a, kTensor, x.out, kLocal);
  Plan p;
  p.name = x.rows == 16 ? "gate_up_silu_m16n256k1024_acc32"
                        : "gate_up_silu_m32n256k1024_acc32";
  u32(p, 0, x.a);
  u32(p, 2, ddr_units(x.gate_weight, kWeight));
  u32(p, 4, ddr_units(x.up_weight, kWeight));
  u32(p, 6, x.out);
  p.register_count = 8;
  return p;
}

// gate_up_silu uses hardware sigmoid + valuRepeat multiplies on both m16 and
// m32. The plain kernel is the only implementation; no RNE/LUT variant.

// q/k/v/gate column GEMMs + q/k RMSNorm and M-RoPE.
struct QkvRopeArgs {
  std::uint32_t a, planes, qk_weight;  // planes: [4][32,256]; qk_weight: [2][256]
  std::uint64_t weights;               // q,k,v,g transformed weights back to back
  std::uint64_t cosine, sine;          // DDR FP16 [table_rows,32]
  std::uint32_t table_rows, position_offset = 0;
  float epsilon = 1.0e-6F;
  std::uint32_t rows = 32;
};
inline Plan qkv_rope(const QkvRopeArgs& x) {
  using namespace detail;
  xinfer::rpu::abi::detail::compact_row_bucket_rows(x.rows);
  const std::uint64_t kTensor = 2ULL * x.rows * 1024;
  const std::uint64_t kLocal = 2ULL * x.rows * 256;
  require(x.table_rows >= x.rows && x.position_offset <= x.table_rows - x.rows &&
              x.position_offset <= 65535,
          "qkv_rope M-RoPE table does not cover action rows");
  require(x.epsilon == 1.0e-6F, "Action QK RMSNorm requires epsilon=1e-6");
  spm(x.a, kTensor); spm(x.planes, 4 * kLocal); spm(x.qk_weight, 2 * 256 * 2);
  disjoint(x.a, kTensor, x.planes, 4 * kLocal);
  disjoint(x.planes, 4 * kLocal, x.qk_weight, 2 * 256 * 2);
  const std::uint64_t table_bytes = 2ULL * x.table_rows * 32;
  Plan p;
  p.name = x.rows == 16 ? "qknorm_qkv_rope_m16k1024_acc32"
                        : "qknorm_qkv_rope_m32k1024_acc32";
  u32(p, 0, x.a);
  u32(p, 2, ddr_units(x.weights, 4 * kWeight));
  u32(p, 4, x.planes);
  u32(p, 6, x.qk_weight);
  u32(p, 8, ddr_units(x.cosine, table_bytes));
  u32(p, 10, ddr_units(x.sine, table_bytes));
  p.regs[12] = static_cast<std::uint16_t>(x.position_offset);
  u32(p, 13, fp32_bits(1.0F / 256.0F));
  p.regs[15] = xinfer::rpu::fp32_to_fp16(x.epsilon);
  p.register_count = 16;
  p.zero_counters = {0, 1};
  return p;
}

// Row GEMM + TP8 all-reduce + residual gate + next adaptive RMSNorm. With
// next2_row set, also the following identity layer's residual gate (Z, into
// norm_out) and adaptive RMSNorm (W, into w_out); stream is then scratch.
// RTC (M16 only): rows [0, rtc_rows) use the clean (tau=1) modulation. The
// modulation rows then address the clean rows of the [.., 2, 3072] pair
// layout, whose step-condition rows follow at +kRtcNoisyOffset bytes.
inline constexpr std::uint32_t kRtcNoisyOffset = 2U * 3 * 1024;
inline constexpr std::uint32_t kRtcMaxRows = 8;
struct DownAllReduceNormArgs {
  std::uint32_t a;          // [32,256] GEMM input
  std::uint32_t partial;    // [32,1024] scratch
  std::uint32_t residual;   // [32,1024]; may alias norm_out
  std::uint32_t stream;     // [32,1024] X = sum*gate + residual
  std::uint32_t norm_out;   // [32,1024] Y; must NOT alias stream
  std::uint64_t weight;     // production row-transformed DDR weight
  std::uint64_t gate_row;   // DDR modulation row [3,1024]; gate = [2048:3072]
  std::uint64_t next_row;   // DDR modulation row; gamma [0:1024], shift [1024:2048]
  float epsilon;            // adaptive RMSNorm epsilon
  unsigned global_counter;  // 3..50, distinct across occurrences in flight
  std::uint64_t next2_row = 0;  // identity-next: DDR modulation row after next_row
  std::uint32_t w_out = 0;      // identity-next: [32,1024] W
  std::uint32_t rows = 32;
  float normalization_scale = 1.0F;  // S/4 for the M16 residual storage scale
  std::uint32_t rtc_rows = 0;        // RTC clean prefix rows, 0..8 (M16 only)
};
namespace detail {
inline Plan down_plan(const DownAllReduceNormArgs& x) {
  xinfer::rpu::abi::detail::compact_row_bucket_rows(x.rows);
  xinfer::rpu::abi::detail::row_bucket_normalization_scale(x.rows, x.normalization_scale);
  const std::uint64_t kTensor = 2ULL * x.rows * 1024;
  const std::uint64_t kLocal = 2ULL * x.rows * 256;
  require(x.global_counter >= 3 && x.global_counter <= 50,
          "fused all-reduce global counter must be in 3..50");
  const auto eps_bits = xinfer::rpu::fp32_to_fp16(x.epsilon);
  require(xinfer::rpu::abi::detail::positive_finite_fp16(eps_bits),
          "adaptive RMSNorm requires a positive finite FP16 epsilon");
  spm(x.a, kLocal); spm(x.partial, kTensor); spm(x.residual, kTensor);
  spm(x.stream, kTensor); spm(x.norm_out, kTensor);
  disjoint(x.a, kLocal, x.partial, kTensor);
  disjoint(x.partial, kTensor, x.residual, kTensor);
  disjoint(x.partial, kTensor, x.stream, kTensor);
  disjoint(x.partial, kTensor, x.norm_out, kTensor);
  disjoint(x.stream, kTensor, x.residual, kTensor);
  disjoint(x.stream, kTensor, x.norm_out, kTensor);  // peers read stream rows
  disjoint(x.a, kLocal, x.stream, kTensor);
  require(x.norm_out == x.residual || (x.norm_out + kTensor <= x.residual ||
                                       x.residual + kTensor <= x.norm_out),
          "norm_out must equal or be disjoint from residual");
  require(x.rtc_rows <= kRtcMaxRows && (x.rtc_rows == 0 || x.rows == 16),
          "fused RTC split requires M16 and at most 8 clean rows");
  // With RTC every modulation row is a clean/step pair.
  const std::uint64_t row_bytes = 2ULL * 3 * 1024 * (x.rtc_rows ? 2 : 1);
  Plan p;
  p.name = x.rows == 16 ? "down_allreduce_rmsnorm_m16n1024k256_acc32"
                        : "down_allreduce_rmsnorm_m32n1024k256_acc32";
  u32(p, 0, x.a);
  u32(p, 2, ddr_units(x.weight, kWeight));
  u32(p, 4, x.partial);
  u32(p, 6, global_spm(x.partial));
  u32(p, 8, x.residual);
  u32(p, 10, ddr_units(x.gate_row, row_bytes));
  u32(p, 12, ddr_units(x.next_row, row_bytes));
  u32(p, 14, x.stream);
  u32(p, 16, global_spm(x.stream));
  u32(p, 18, x.norm_out);
  p.regs[20] = static_cast<std::uint16_t>(xinfer::rpu::abi::core_stride_bytes >> 16);
  u32(p, 21, fp32_bits(x.normalization_scale * x.normalization_scale / 1024.0F));
  p.regs[23] = eps_bits;
  p.regs[24] = static_cast<std::uint16_t>(x.global_counter);
  p.register_count = 25;
  if (x.next2_row != 0) {
    spm(x.w_out, kTensor);
    for (auto other : {x.a, x.partial, x.residual, x.stream, x.norm_out}) {
      const std::uint64_t bytes = other == x.a ? kLocal : kTensor;
      disjoint(x.w_out, kTensor, other, bytes);
    }
    p.regs[25] = 1;
    u32(p, 26, ddr_units(x.next2_row, row_bytes));
    u32(p, 28, x.w_out);
    p.register_count = 30;
  }
  if (x.rows == 16) {
    p.regs[30] = xinfer::rpu::fp32_to_fp16(x.normalization_scale);
    p.register_count = 31;
  }
  if (x.rtc_rows != 0) {
    p.regs[31] = static_cast<std::uint16_t>(x.rtc_rows);
    p.regs[32] = static_cast<std::uint16_t>(kRtcNoisyOffset);
    p.register_count = 33;
  }
  p.zero_counters = {0, 1, x.global_counter};
  return p;
}
}  // namespace detail
inline Plan down_allreduce_rmsnorm(const DownAllReduceNormArgs& x) {
  return detail::down_plan(x);
}

// HW ABI-padded alias: the kernel body is the hardware sqrt_reciprocal path.
// The trailing LUT slot is reserved as ABI padding (always 0); both M16 and
// M32 route to their respective _hw_ kernel names.
inline Plan down_allreduce_rmsnorm_hw(const DownAllReduceNormArgs& x) {
  using namespace detail;
  auto p = detail::down_plan(x);
  p.name = x.rows == 16 ? "down_allreduce_rmsnorm_hw_m16n1024k256_acc32"
                        : "down_allreduce_rmsnorm_hw_m32n1024k256_acc32";
  // The M16 RTC scalar at slot 32 leaves slot 33 as pointer-alignment padding;
  // M32's epsilon at slot 8 is followed by padding slot 9. Append 0 as ABI
  // padding for the (unused) rsqrt_lut pointer.
  const unsigned slot = x.rows == 16 ? 34 : 30;
  u32(p, slot, 0u);
  p.register_count = slot + 2;
  return p;
}

template <class Kernel>
std::uint32_t apply(Kernel& kernel, const Plan& plan) {
  auto rc = kernel.reset_regs();
  if (rc) return rc;
  for (unsigned i = 0; i < plan.register_count; ++i) {
    rc = kernel.set_regs(i, plan.regs[i]);
    if (rc) return rc;
  }
  for (unsigned id : plan.zero_counters) {
    rc = kernel.set_regs(128 + 2 * id, std::uint32_t{0});
    if (rc) return rc;
  }
  return 0;
}

}  // namespace xinfer::rpu::denoise
