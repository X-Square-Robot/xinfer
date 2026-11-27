// Copyright 2026 X-Square Robot
// SPDX-License-Identifier: Apache-2.0
#pragma once

#define XINFER_RPU_OPS_HAS_ACTION_ROWS 1
#define XINFER_RPU_OPS_HAS_ACTION_NORMALIZATION_SCALE 1

#include "common/rpu/fp16.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>

// Host launch contract for the source kernels. No allocation, DMA or enqueue
// occurs here: the inference framework owns buffers, queues and Graph replay.
namespace xinfer::rpu::abi {

inline constexpr unsigned version = 11;
inline constexpr std::uint64_t spm_bytes = 8U * 1024U * 1024U;
inline constexpr std::uint64_t working_bytes = 7U * 1024U * 1024U;
inline constexpr std::uint32_t global_spm_base = 0x40000000U;
inline constexpr std::uint32_t core_stride_bytes = 0x01000000U;

enum class Entry { LayerNorm, Rope, Gelu, AllReduceResidual,
                   QkvBiasN384K1024, ON1024K128, UpN512K1024, DownN1024K512,
                   QkvZba, Qkv, ON2048K256, GateUp, DownN2048K768,
                   PatchEmbed, Fc1N512K4096, DownN2048K512,
                   KcacheInsert, VcacheInsert, FlashAttentionVarlen,
                   AddVpb, Fill, AllGather, CopyPhysicalPrefixF16,
                   GatherEmbeddingF16,
                   KcacheInsertD256, VcacheInsertD256, CausalSdpaD256,
                   RMSNormF16, CausalDepthwiseConvPrefill,
                   CausalDepthwiseConvStep, SiluMulFp16,
                   BmmFp16Spm, BmmFp16SpmColCol128, BmmFp16SpmRowRow64,
                   BmmFp16SpmRowCol64K64, BmmFp16SpmRowCol64K128,
                   UnitTrilInvFp16, CumsumFp16, TpQkvCompactF16,
                   QkvExpandKvF16, CausalSdpaPrefixSpmD256,
                   VcacheInsertStagedD256, GemmDdrColM32N256K1024Acc32,
                   GemmDdrRowM32N1024K256Acc32,
                   GemmDdrBiasM32N1024K64Acc32,
                   GemmDdrBiasM32N64K1024Acc32,
                   GemmDdrBiasScaleM32N1024K64Acc32,
                   AdaptiveRmsnormM32H1024F16,
                   ResidualGateM32H1024F16,
                   SigmoidMulM32D256F16,
                   EulerUpdateM32D64F16,
                   QkRmsnormMropeM32D256F16,
                   SdpaMaskedM32D256F16, RMSNormGdnF16, SliceLastF16, TransposeNbcC128F16, TransposeNbcC1F16,
                   BinaryAddF16, BinarySubF16,
                   BinaryMulF16, BinaryMaxF16, ScalarMulF16,
                   UnaryF16, BcastRowsF16, BcastTilesF16,
                   L2normF16,
                   DownAllReduceRmsnormFusion, GateUpSiluFusion,
                   QkvRopeFusionM1024, ConvSiluPrefillFusion,
                   GdrPrefillFusion, Reserved71,
                   Reserved72,
                   GemmDdrColM16N256K1024Acc32,
                   GemmDdrRowM16N1024K256Acc32,
                   GemmDdrBiasM16N1024K64Acc32,
                   GemmDdrBiasM16N64K1024Acc32,
                   GemmDdrBiasScaleM16N1024K64Acc32,
                   AdaptiveRmsnormM16H1024F16,
                   ResidualGateM16H1024F16,
                   SigmoidMulM16D256F16,
                   EulerUpdateM16D64F16,
                   QkRmsnormMropeM16D256F16,
                   SdpaMaskedM16D256F16,
                   DownAllReduceRmsnormFusionM16,
                   GateUpSiluFusionM16,
                   QkvRopeFusionM16, QkvRopeBiasN384K1024Fusion,
                   UpGeluN512K1024Fusion, FlashAttentionSpmD384Fusion,
                   AllReduceResidualLayerNormFusion, Fc1GeluN512K4096Fusion,
                   GatherEmbeddingTp8F16, QkvZbaM512, QkvQgateM512,
                   GateUpM512,
                   AllReduceResidualWindow, Reserved97,
                   QkvZbaM448K128, QkvQgateM448K128, GateUpM448K128, QkvZbaM512K128, QkvQgateM512K128, GateUpM512K128, RmsNormCoeffH2048F16,
                   DownAllReduceRmsnormHwM16,
                   DownAllReduceRmsnormHwWindowM16,
                   AdaptiveRmsnormHwM16H1024F16,
                   ResidualGateHwM16H1024F16,
                   AdaptiveRmsnormHwM32H1024F16,
                   ResidualGateHwM32H1024F16,
                   DownAllReduceRmsnormHwM32,
                   DownAllReduceRmsnormHwWindowM32 };
inline constexpr std::array<const char*, 113> names = {
    "layernorm_f16", "rope_f16", "gelu_f16", "all_reduce_residual",
    "qkv_bias_n384k1024", "o_n1024k128", "up_n512k1024", "down_n1024k512",
    "qkv_zba_n1056k2048", "qkv_qgate_n1024k2048", "o_n2048k256", "gateup_n1536k2048", "down_n2048k768",
    "patch_embed_n128k1536", "fc1_n512k4096", "down_n2048k512",
    "kcache_insert_f16", "vcache_insert_f16", "flash_attention_varlen_f16",
    "add_vpb_f16", "fill_f16", "all_gather_f16",
    "copy_physical_prefix_f16", "gather_embedding_f16",
    "kcache_insert_d256_f16", "vcache_insert_d256_f16",
    "causal_sdpa_d256_f16", "rmsnorm_f16",
    "causal_depthwise_conv1d_prefill_f16",
    "causal_depthwise_conv1d_step_f16", "silu_mul_fp16",
    "bmm_fp16_spm", "bmm_fp16_spm_128x128x64_colcol",
    "bmm_fp16_spm_64x64x128_rowrow", "bmm_fp16_spm_64x128x64_rowcol",
    "bmm_fp16_spm_64x128x128_rowcol", "unit_tril_inv_fp16",
    "cumsum_fp16", "tp_qkv_compact_f16", "qkv_expand_kv_f16",
    "causal_sdpa_prefix_spm_d256_f16", "vcache_insert_staged_d256_f16",
    "gemm_ddr_col_m32n256k1024_acc32",
    "gemm_ddr_row_m32n1024k256_acc32",
    "gemm_ddr_bias_m32n1024k64_acc32",
    "gemm_ddr_bias_m32n64k1024_acc32",
    "gemm_ddr_bias_scale_m32n1024k64_acc32",
    "adaptive_rmsnorm_m32h1024_f16",
    "residual_gate_m32h1024_f16",
    "sigmoid_mul_m32d256_f16",
    "euler_update_m32d64_f16",
    "qk_rmsnorm_mrope_m32d256_f16",
    "sdpa_masked_m32d256_f16", "rmsnorm_gdn_f16",
    "slice_last_f16", "transpose_nbc_c128_f16",
    "transpose_nbc_c1_f16", "binary_add_f16",
    "binary_sub_f16", "binary_mul_f16",
    "binary_max_f16", "scalar_mul_f16",
    "unary_f16", "bcast_rows_f16", "bcast_tiles_f16",
    "l2norm_f16",
    "down_allreduce_rmsnorm_m32n1024k256_acc32",
    "gate_up_silu_m32n256k1024_acc32",
    "qknorm_qkv_rope_m32k1024_acc32",
    "conv_silu_prefill_f16", "gdr_prefill_f16",
    nullptr, nullptr,
    "gemm_ddr_col_m16n256k1024_acc32",
    "gemm_ddr_row_m16n1024k256_acc32",
    "gemm_ddr_bias_m16n1024k64_acc32",
    "gemm_ddr_bias_m16n64k1024_acc32",
    "gemm_ddr_bias_scale_m16n1024k64_acc32",
    "adaptive_rmsnorm_m16h1024_f16",
    "residual_gate_m16h1024_f16",
    "sigmoid_mul_m16d256_f16",
    "euler_update_m16d64_f16",
    "qk_rmsnorm_mrope_m16d256_f16",
    "sdpa_masked_m16d256_f16",
    "down_allreduce_rmsnorm_m16n1024k256_acc32",
    "gate_up_silu_m16n256k1024_acc32",
    "qknorm_qkv_rope_m16k1024_acc32",
    "qkv_rope_bias_n384k1024", "up_gelu_n512k1024", "flash_attention_spm_d384",
    "all_reduce_residual_layernorm", "fc1_gelu_n512k4096",
    "gather_embedding_tp8_f16", "qkv_zba_m512", "qkv_qgate_m512",
    "gateup_m512",
    "all_reduce_residual_window", nullptr,
    "qkv_zba_m448_k128", "qkv_qgate_m448_k128", "gateup_m448_k128", "qkv_zba_m512_k128", "qkv_qgate_m512_k128", "gateup_m512_k128", "rmsnorm_coeff_h2048_f16",
    "down_allreduce_rmsnorm_hw_m16n1024k256_acc32",
    "down_allreduce_rmsnorm_hw_window_m16n1024k256_acc32",
    "adaptive_rmsnorm_hw_m16h1024_f16",
    "residual_gate_hw_m16h1024_f16",
    "adaptive_rmsnorm_hw_m32h1024_f16",
    "residual_gate_hw_m32h1024_f16",
    "down_allreduce_rmsnorm_hw_m32n1024k256_acc32",
    "down_allreduce_rmsnorm_hw_window_m32n1024k256_acc32"};

inline const char* name(Entry entry) {
  const auto i = static_cast<std::size_t>(entry);
  if (i >= names.size() || !names[i]) throw std::invalid_argument("unknown xinfer::rpu entry");
  return names[i];
}

struct LaunchPlan {
  Entry entry;
  std::array<std::uint16_t, 64> regs{};
  unsigned register_count = 0;
  std::array<std::uint16_t, 3> grid{1, 1, 1};
  unsigned cores = 1;
};

namespace detail {
inline void require(bool ok, const char* message) {
  if (!ok) throw std::invalid_argument(message);
}
inline void u32(LaunchPlan& p, unsigned reg, std::uint32_t value) {
  p.regs[reg] = static_cast<std::uint16_t>(value);
  p.regs[reg + 1] = static_cast<std::uint16_t>(value >> 16);
}
inline void span(std::uint32_t offset, std::uint64_t bytes) {
  require(offset % 32 == 0 && bytes > 0 && offset < spm_bytes &&
          bytes <= spm_bytes - offset,
          "xinfer::rpu requires aligned local SPM byte offsets within one core");
}
inline bool overlap(std::uint32_t a, std::uint64_t an,
                    std::uint32_t b, std::uint64_t bn) {
  return a < std::uint64_t(b) + bn && b < std::uint64_t(a) + an;
}
inline void disjoint(std::uint32_t a, std::uint64_t an,
                     std::uint32_t b, std::uint64_t bn) {
  require(!overlap(a, an, b, bn), "xinfer::rpu output overlaps a live input");
}
inline void elementwise_alias(std::uint32_t dst, std::uint32_t src,
                             std::uint64_t bytes) {
  if (dst != src) disjoint(dst, bytes, src, bytes);
}
inline void compact_row_bucket_rows(std::uint32_t rows) {
  require(rows == 16 || rows == 32, "Action supports rows 16 or 32");
}
inline bool positive_finite_fp16(std::uint16_t bits) {
  // Validate the actual scalar sent to the device. A finite FP32 value can
  // round to FP16 infinity; integer bits also retain this guard under fast-math.
  return bits > 0 && bits < 0x7c00U;
}

inline void row_bucket_normalization_scale(std::uint32_t rows, float scale) {
  require(scale == 1.0F || (rows == 16 && scale == 64.0F),
          "Action normalization scale requires 1, or 64 for M16");
}
inline void shape16(std::uint32_t rows, std::uint32_t cols) {
  require(rows > 0 && rows <= 65535 && cols > 0 && cols <= 65520 &&
          cols % 16 == 0, "invalid xinfer::rpu FP16 shape");
}
inline void ddr_span(std::uint64_t address, std::uint64_t bytes) {
  constexpr std::uint64_t limit = 1ULL << 40;
  require(address != 0 && address % 256 == 0 && address < limit &&
          bytes > 0 && bytes <= limit - address,
          "xinfer::rpu requires a nonzero 256-byte-aligned DDR range below 1 TiB");
}
inline bool ddr_overlap(std::uint64_t a, std::uint64_t an,
                        std::uint64_t b, std::uint64_t bn) {
  return a < b + bn && b < a + an;
}
}  // namespace detail

// Private batch compilation only. A completed initializer must zero 0..49
// before a window; body launches must not write atomic registers.
// Each occurrence owns two local counters and one distinct global counter.
inline LaunchPlan collective_window(LaunchPlan p, unsigned occurrence) {
  detail::require(occurrence < 15, "collective window exceeds 15 occurrences");
  if (p.entry == Entry::DownAllReduceRmsnormHwM16) {
    // Hardware-arithmetic twin of the LUT window ABI: the hw kernel keeps
    // the LUT variant's formal list (slot 34 = unused lut pointer), so the
    // calibrated 24/33/36 rewrite applies verbatim.
    detail::require(p.register_count == 36, "invalid M16 DownNorm hw window prefix");
    detail::require(p.cores == 8 && p.grid == std::array<std::uint16_t, 3>{8, 1, 1},
                    "M16 collective window requires eight blocks on eight cores");
    p.entry = Entry::DownAllReduceRmsnormHwWindowM16;
    p.regs[24] = static_cast<std::uint16_t>(34 + occurrence);
    p.regs[33] = static_cast<std::uint16_t>(2 + 2 * occurrence);
    p.regs[36] = static_cast<std::uint16_t>(3 + 2 * occurrence);
    p.register_count = 37;
    return p;
  }
  // M32 HW has no calibrated window ABI yet; callers only exercise it through
  // the plain (non-window) family entry path.
  detail::require(p.entry == Entry::AllReduceResidual,
                  "entry has no calibrated AR or M16 DownNorm private window ABI");
  detail::require(p.register_count == 18, "invalid all-reduce window prefix");
  detail::require(p.cores == 8 && p.grid == std::array<std::uint16_t, 3>{8, 1, 1},
                  "collective window requires eight resident blocks on eight cores");
  p.entry = Entry::AllReduceResidualWindow;
  // Occurrences 0..14 use globals 34..48 with local pairs 2..29;
  // all identifiers are included in the initialization range 0..50.
  p.regs[12] = static_cast<std::uint16_t>(34 + occurrence);
  // The two added u16 arguments occupy positions 13 and 18; the first uses
  // alignment padding before the 32-bit pointer arguments.
  p.regs[13] = static_cast<std::uint16_t>(2 + 2 * occurrence);
  p.regs[18] = static_cast<std::uint16_t>(3 + 2 * occurrence);
  p.register_count = 19;
  return p;
}

// Convert a framework core-0 absolute SPM address to the local byte offset
// consumed by half* kernels. Addresses in other cores are intentionally rejected.
inline std::uint32_t local_spm_offset(std::uint32_t address) {
  if (address >= global_spm_base && address < global_spm_base + spm_bytes)
    return address - global_spm_base;
  detail::require(address < spm_bytes, "expected local or core-0 SPM address");
  return address;
}

struct CumsumArgs {
  std::uint32_t dst, src, rows;
};

// One launch per core-local contiguous [rows,64] tensor. The profile
// supplies rows = 2 * (outer_chunk / 64), hence 2..16 even rows.
inline LaunchPlan cumsum_fp16(const CumsumArgs& a) {
  using namespace detail;
  require(a.rows >= 1 && a.rows <= 16, "cumsum rows outside [1,16]");
  const std::uint64_t bytes = 2ULL * a.rows * 64;
  span(a.src, bytes); span(a.dst, bytes);
  elementwise_alias(a.dst, a.src, bytes);
  LaunchPlan p{Entry::CumsumFp16};
  u32(p, 0, a.src); u32(p, 2, a.dst); p.regs[4] = a.rows;
  p.register_count = 5; p.grid[0] = (a.rows + 15) / 16;
  p.cores = 8;
  return p;
}

enum class BmmOrientation : unsigned { RowByCol = 0, RowByRow = 1, ColByCol = 3 };
struct BmmArgs {
  std::uint32_t dst, a, b, M, N, K, batch;
  BmmOrientation orientation;
};
namespace detail {
// Shared geometry, memory and register contract. The generic device entry has
// a smaller batch limit than the fixed-shape entries.
inline LaunchPlan bmm_fp16_spm_plan(const BmmArgs& x, Entry entry) {
  require(x.batch >= 2 && x.batch <= 16 && x.batch % 2 == 0,
          "BMM batch must be even in [2,16]");
  const bool row_row = x.M == 64 && x.N == 64 && x.K == 128 &&
                       x.orientation == BmmOrientation::RowByRow;
  const bool row_col = x.M == 64 && x.N == 128 &&
                       (x.K == 64 || (x.K == 128 && x.batch == 2)) &&
                       x.orientation == BmmOrientation::RowByCol;
  const bool col_col = x.M == 128 && x.N == 128 && x.K == 64 &&
                       x.batch == 2 && x.orientation == BmmOrientation::ColByCol;
  require(row_row || row_col || col_col, "unsupported GDN BMM geometry/orientation");
  const std::uint64_t ab = 2ULL * x.batch * x.M * x.K;
  const std::uint64_t bb = 2ULL * x.batch * x.K * x.N;
  const std::uint64_t cb = 2ULL * x.batch * x.M * x.N;
  require(ab + bb + cb <= working_bytes, "BMM exceeds working budget");
  span(x.a, ab); span(x.b, bb); span(x.dst, cb);
  disjoint(x.dst, cb, x.a, ab); disjoint(x.dst, cb, x.b, bb);
  LaunchPlan p{entry};
  u32(p, 0, x.a); u32(p, 2, x.b); u32(p, 4, x.dst);
  p.regs[6] = x.M; p.regs[7] = x.N; p.regs[8] = x.K;
  p.regs[9] = x.batch; p.regs[10] = static_cast<unsigned>(x.orientation);
  p.register_count = 11; p.grid[2] = x.batch; p.cores = 8;
  return p;
}
}  // namespace detail

// Exact FP16 GDN chunk GEMMs with contiguous per-batch storage. Orientation
// selects A[M,K] @ B[K,N], A[M,K] @ B[N,K]^T, or A[K,M]^T @ B[K,N].
// The generic device entry returns without writing for batch > 12, so reject
// those requests before the caller can enqueue a silently ineffective launch.
inline LaunchPlan bmm_fp16_spm(const BmmArgs& x) {
  detail::require(x.batch >= 2 && x.batch <= 12 && x.batch % 2 == 0,
                  "generic BMM batch must be even in [2,12]; use an explicit "
                  "supported fixed-shape entry for 14 or 16");
  return detail::bmm_fp16_spm_plan(x, Entry::BmmFp16Spm);
}

// Specialized fixed shape for the column-by-column [128,128,64] product.
// The argument registers are identical to the generic BMM kernel.
inline LaunchPlan bmm_fp16_spm_colcol128(const BmmArgs& x) {
  detail::require(x.M == 128 && x.N == 128 && x.K == 64 && x.batch == 2 &&
                  x.orientation == BmmOrientation::ColByCol,
                  "specialized BMM requires [128,128,64], batch 2, mode 3");
  return detail::bmm_fp16_spm_plan(x, Entry::BmmFp16SpmColCol128);
}

// Fixed-shape BMM entry, preserving the generic register ABI.
inline LaunchPlan bmm_fp16_spm_rowrow64(const BmmArgs& x) {
  detail::require(x.M == 64 && x.N == 64 && x.K == 128 && x.orientation == BmmOrientation::RowByRow, "specialized BMM geometry/orientation mismatch");
  return detail::bmm_fp16_spm_plan(x, Entry::BmmFp16SpmRowRow64);
}

// Fixed-shape BMM entry, preserving the generic register ABI.
inline LaunchPlan bmm_fp16_spm_rowcol64k64(const BmmArgs& x) {
  detail::require(x.M == 64 && x.N == 128 && x.K == 64 && x.orientation == BmmOrientation::RowByCol, "specialized BMM geometry/orientation mismatch");
  return detail::bmm_fp16_spm_plan(x, Entry::BmmFp16SpmRowCol64K64);
}

// Fixed-shape BMM entry, preserving the generic register ABI.
inline LaunchPlan bmm_fp16_spm_rowcol64k128(const BmmArgs& x) {
  detail::require(x.M == 64 && x.N == 128 && x.K == 128 && x.batch == 2 && x.orientation == BmmOrientation::RowByCol, "specialized BMM geometry/orientation mismatch");
  return detail::bmm_fp16_spm_plan(x, Entry::BmmFp16SpmRowCol64K128);
}

struct UnitTrilInvArgs {
  std::uint32_t dst, src, batch;
};
// Exact GDN chunk: M is contiguous strict-lower FP16 [batch,64,64].
// The result (I-M)^-1 may replace M in place, or use disjoint output.
inline LaunchPlan unit_tril_inv_fp16(const UnitTrilInvArgs& a) {
  using namespace detail;
  require(a.batch >= 2 && a.batch <= 16 && a.batch % 2 == 0,
          "unit lower inverse batch must be even in [2,16]");
  const std::uint64_t bytes = 2ULL * a.batch * 64 * 64;
  span(a.src, bytes); span(a.dst, bytes);
  elementwise_alias(a.dst, a.src, bytes);
  LaunchPlan p{Entry::UnitTrilInvFp16};
  u32(p, 0, a.src); u32(p, 2, a.dst); p.regs[4] = a.batch;
  p.register_count = 5; p.grid[0] = a.batch; p.cores = 8;
  return p;
}

struct LayerNormArgs {
  std::uint32_t dst, src, weight, bias;
  std::uint32_t rows, cols;
  bool with_bias = true;
  float epsilon = 1.0e-5F;
};

inline LaunchPlan layernorm(const LayerNormArgs& a) {
  using namespace detail;
  shape16(a.rows, a.cols);
  const std::uint64_t bytes = 2ULL * a.rows * a.cols;
  const std::uint64_t affine_bytes = 2ULL * a.cols;
  require(2 * bytes + (a.with_bias ? 2 : 1) * affine_bytes <= working_bytes,
          "LayerNorm exceeds the 7 MiB working budget");
  span(a.dst, bytes); span(a.src, bytes); span(a.weight, affine_bytes);
  disjoint(a.dst, bytes, a.src, bytes);
  disjoint(a.dst, bytes, a.weight, affine_bytes);
  if (a.with_bias) {
    span(a.bias, affine_bytes); disjoint(a.dst, bytes, a.bias, affine_bytes);
  }
  const auto eps = fp32_to_fp16(a.epsilon);
  require(positive_finite_fp16(eps),
          "LayerNorm epsilon must be positive finite FP16");
  LaunchPlan p{Entry::LayerNorm};
  u32(p, 0, a.dst); u32(p, 2, a.src); u32(p, 4, a.weight);
  u32(p, 6, a.with_bias ? a.bias : 0);
  p.regs[8] = a.rows; p.regs[9] = a.cols;
  p.regs[10] = a.cols; p.regs[11] = a.with_bias ? a.cols : 0;
  const float scale = 1.0F / a.cols;
  std::uint32_t scale_bits;
  std::memcpy(&scale_bits, &scale, sizeof(scale));
  u32(p, 12, scale_bits); p.regs[14] = eps;
  p.register_count = 15;
  const unsigned tile = a.cols <= 2048 && a.rows > 128 ? 32 : 16;
  p.grid[0] = (a.rows + tile - 1) / tile;
  return p;
}

struct RMSNormF16Args {
  std::uint32_t dst, src, weight;
  std::uint32_t rows, cols;
  float epsilon = 1.0e-6F;
};

// Contiguous FP16 tensors in each core's local SPM. Broadcast this plan to
// all eight cores; the framework owns storage, DMA, and Graph replay.
inline LaunchPlan rmsnorm_f16(const RMSNormF16Args& a) {
  using namespace detail;
  const bool supported =
      (a.cols == 2048 && a.rows >= 64 && a.rows <= 512) ||
      (a.cols == 256 && a.rows >= 64 && a.rows <= 512) ||
      (a.cols == 128 && a.rows >= 128 && a.rows <= 1024) ||
      ((a.rows == 1 || a.rows == 2) &&
       (a.cols == 128 || a.cols == 256 || a.cols == 2048));
  require(supported, "unsupported RMSNorm FP16 shape");
  require(a.epsilon == 1.0e-6F, "RMSNorm FP16 requires epsilon=1e-6");
  const std::uint64_t bytes = 2ULL * a.rows * a.cols;
  const std::uint64_t weight_bytes = 2ULL * a.cols;
  require((a.dst == a.src ? bytes : 2 * bytes) + weight_bytes <= working_bytes,
          "RMSNorm FP16 exceeds the 7 MiB working budget");
  span(a.dst, bytes); span(a.src, bytes); span(a.weight, weight_bytes);
  elementwise_alias(a.dst, a.src, bytes);
  disjoint(a.dst, bytes, a.weight, weight_bytes);
  disjoint(a.src, bytes, a.weight, weight_bytes);
  LaunchPlan p{Entry::RMSNormF16};
  u32(p, 0, a.dst); u32(p, 2, a.src); u32(p, 4, a.weight);
  p.regs[6] = static_cast<std::uint16_t>(a.rows);
  p.regs[7] = static_cast<std::uint16_t>(a.cols);
  const float scale = 1.0F / a.cols;
  std::uint32_t scale_bits;
  std::memcpy(&scale_bits, &scale, sizeof(scale));
  u32(p, 8, scale_bits);
  p.regs[10] = fp32_to_fp16(a.epsilon);
  p.register_count = 11;
  p.grid[0] = static_cast<std::uint16_t>((a.rows + 15) / 16);
  p.cores = 8;
  return p;
}

// Coefficient-first two-native-MUL Text RMS, separately named so the generic
// staged RMSNorm/Action contracts retain their original association.
inline LaunchPlan rmsnorm_coeff_h2048_f16(const RMSNormF16Args& a) {
  detail::require(a.cols == 2048 && a.rows >= 64 && a.rows <= 512 &&
                      a.rows % 64 == 0,
                  "Text coefficient RMSNorm requires C2048/M64..512 bucket");
  auto p = rmsnorm_f16(a);
  p.entry = Entry::RmsNormCoeffH2048F16;
  return p;
}

// Bounded GDN prefill candidate; keep the generic C128 path unchanged.
inline LaunchPlan rmsnorm_gdn_f16(const RMSNormF16Args& a) {
  detail::require(a.cols == 128 && a.rows >= 128 && a.rows <= 1024 &&
                      a.rows % 128 == 0,
                  "GDN RMSNorm requires C128 and bucketed M128..1024");
  auto p = rmsnorm_f16(a);
  p.entry = Entry::RMSNormGdnF16;
  return p;
}

struct SliceLastArgs {
  std::uint32_t dst, src;
  std::uint32_t rows, src_cols, dst_cols, col_offset;
};

// One TP8 broadcast launch replaces eight core-addressed release slices.
// This initial profile covers the two GDN layouts only.
inline LaunchPlan slice_last_f16(const SliceLastArgs& a) {
  using namespace detail;
  require((a.rows >= 64 && a.rows <= 512 && a.rows % 64 == 0 &&
           a.src_cols == 16 && a.dst_cols == 2 && a.col_offset == 0) ||
          (a.rows == 2 && a.src_cols == 64 && a.dst_cols == 1 &&
           a.col_offset == 63),
          "unsupported GDN last-dimension slice profile");
  const std::uint64_t source_bytes = 2ULL * a.rows * a.src_cols;
  const std::uint64_t destination_bytes = 2ULL * a.rows * a.dst_cols;
  span(a.src, source_bytes);
  span(a.dst, destination_bytes);
  disjoint(a.src, source_bytes, a.dst, destination_bytes);
  LaunchPlan p{Entry::SliceLastF16};
  u32(p, 0, a.dst); u32(p, 2, a.src);
  p.regs[4] = static_cast<std::uint16_t>(a.rows);
  p.regs[5] = static_cast<std::uint16_t>(a.src_cols);
  p.regs[6] = static_cast<std::uint16_t>(a.dst_cols);
  p.regs[7] = static_cast<std::uint16_t>(a.col_offset);
  p.register_count = 8;
  p.grid[0] = static_cast<std::uint16_t>((a.rows + 15) / 16);
  p.cores = 8;
  return p;
}

struct TransposeNbcArgs {
  std::uint32_t dst, src;
  std::uint32_t b, n, c;
};

// Exact GDN prefill 3D SPM permutation, broadcast across eight cores.
inline LaunchPlan transpose_nbc_f16(const TransposeNbcArgs& a) {
  using namespace detail;
  require(((a.b == 64 && a.n == 2) || (a.b == 2 && a.n == 64)) &&
              (a.c == 1 || a.c == 128) &&
              (a.c != 1 || a.b == 64),
          "unsupported GDN transpose profile");
  const std::uint64_t bytes = 2ULL * a.b * a.n * a.c;
  span(a.src, bytes); span(a.dst, bytes);
  disjoint(a.src, bytes, a.dst, bytes);
  LaunchPlan p{a.c == 1 ? Entry::TransposeNbcC1F16 :
                        Entry::TransposeNbcC128F16};
  u32(p, 0, a.dst); u32(p, 2, a.src);
  if (a.c == 128) {
    p.regs[4] = static_cast<std::uint16_t>(a.b);
    p.register_count = 5;
  } else {
    p.register_count = 4;
  }
  p.grid[0] = static_cast<std::uint16_t>((a.b * a.n *
      (a.c == 128 ? 8U : 1U) + 15U) / 16U);
  p.cores = 8;
  return p;
}

enum class BinaryOp : unsigned { Add, Sub, Mul, Max };
struct BinaryArgs {
  std::uint32_t dst, lhs, rhs, elements;
  BinaryOp operation;
};

// Exact FP16 local-SPM binary operation, TP8 broadcast. Masked tails support
// short spans and exact/disjoint output aliases, including in-place.
inline LaunchPlan binary_f16(const BinaryArgs& a) {
  using namespace detail;
  require(a.elements > 0 && 2ULL * a.elements <= working_bytes,
          "unsupported language binary element count");
  Entry entry;
  switch (a.operation) {
    case BinaryOp::Add: entry = Entry::BinaryAddF16; break;
    case BinaryOp::Sub: entry = Entry::BinarySubF16; break;
    case BinaryOp::Mul: entry = Entry::BinaryMulF16; break;
    case BinaryOp::Max: entry = Entry::BinaryMaxF16; break;
    default: throw std::invalid_argument("unsupported language binary op");
  }
  const std::uint64_t bytes = 2ULL * a.elements;
  span(a.dst, bytes); span(a.lhs, bytes); span(a.rhs, bytes);
  elementwise_alias(a.dst, a.lhs, bytes);
  elementwise_alias(a.dst, a.rhs, bytes);
  LaunchPlan p{entry};
  u32(p, 0, a.dst); u32(p, 2, a.lhs); u32(p, 4, a.rhs);
  u32(p, 6, a.elements);
  p.register_count = 8;
  p.grid[0] = static_cast<std::uint16_t>((a.elements + 16383) / 16384);
  p.cores = 8;
  return p;
}

struct ScalarMulArgs {
  std::uint32_t dst, src, elements;
  std::uint16_t scalar_bits;
};

// language GDN scalar multiplications use either -1 or FP16(1/sqrt(128)).
// Other constants require an independently validated source profile.
inline LaunchPlan scalar_mul_f16(const ScalarMulArgs& a) {
  using namespace detail;
  require(a.elements > 0 && 2ULL * a.elements <= working_bytes,
          "unsupported language scalar element count");
  require(a.scalar_bits == 0xbc00 || a.scalar_bits == 0x2da8,
          "unsupported language scalar multiplier");
  const std::uint64_t bytes = 2ULL * a.elements;
  span(a.dst, bytes); span(a.src, bytes);
  elementwise_alias(a.dst, a.src, bytes);
  LaunchPlan p{Entry::ScalarMulF16};
  u32(p, 0, a.dst); u32(p, 2, a.src); u32(p, 4, a.elements);
  p.regs[6] = a.scalar_bits;
  p.register_count = 7;
  p.grid[0] = static_cast<std::uint16_t>((a.elements + 4095) / 4096);
  p.cores = 8;
  return p;
}

enum class UnaryOp : std::uint16_t {
  Sigmoid = 0, Exp = 1, Silu = 2, Softplus = 3
};
struct UnaryArgs {
  std::uint32_t dst, src, elements;
  UnaryOp op;
};

// language local-SPM unary work. Each selected launch uses TP8 and the
// same per-core span; in-place output is legal, partial overlap is not.
inline LaunchPlan unary_f16(const UnaryArgs& a) {
  using namespace detail;
  require(a.elements > 0 && 2ULL * a.elements <= working_bytes,
          "unsupported language unary element count");
  require(a.op == UnaryOp::Sigmoid || a.op == UnaryOp::Exp ||
          a.op == UnaryOp::Silu || a.op == UnaryOp::Softplus,
          "unsupported language unary operation");
  const std::uint64_t bytes = 2ULL * a.elements;
  span(a.dst, bytes); span(a.src, bytes);
  elementwise_alias(a.dst, a.src, bytes);
  LaunchPlan p{Entry::UnaryF16};
  u32(p, 0, a.dst); u32(p, 2, a.src); u32(p, 4, a.elements);
  p.regs[6] = static_cast<std::uint16_t>(a.op);
  p.register_count = 7;
  p.grid[0] = static_cast<std::uint16_t>((a.elements + 4095) / 4096);
  p.cores = 8;
  return p;
}

enum class BcastOp : std::uint16_t { Add = 0, Sub = 1, Mul = 2 };
enum class BcastMode : std::uint16_t { Row = 0, Col = 1, GroupCol = 2 };
struct BcastArgs {
  std::uint32_t dst, x, a, rows, cols, group_rows;
  BcastMode mode;
  BcastOp op;
  bool x_first;
};

// language broadcast binary on TP8 local SPM. X[rows,cols] is the full
// operand. A broadcasts along each row (Row, A[rows]), along every row (Col,
// A[cols]) or per group of rows (GroupCol, A[rows/group_rows,cols]). The
// result is A op X, or X op A when x_first is set. Rows of 2 or 16..256
// columns run one row per thread; multiples of 4096 columns run 4096-element
// tiles. Output may alias X exactly and never overlaps A.
inline LaunchPlan bcast_f16(const BcastArgs& a) {
  using namespace detail;
  require(a.op == BcastOp::Add || a.op == BcastOp::Sub ||
          a.op == BcastOp::Mul,
          "unsupported language broadcast operation");
  require(a.mode == BcastMode::Row || a.mode == BcastMode::Col ||
          a.mode == BcastMode::GroupCol,
          "unsupported language broadcast mode");
  require(a.rows > 0 && a.rows <= 65535 && a.cols > 0 && a.cols <= 65535,
          "unsupported language broadcast shape");
  const bool tiles = a.cols % 4096 == 0;
  const bool rows = a.cols == 2 || (a.cols % 16 == 0 && a.cols <= 256);
  require(tiles || rows, "unsupported language broadcast columns");
  require(!tiles || a.mode != BcastMode::GroupCol,
          "language grouped broadcast requires at most 256 columns");
  std::uint64_t a_elements = a.rows;
  if (a.mode == BcastMode::Col) a_elements = a.cols;
  if (a.mode == BcastMode::GroupCol) {
    require(a.group_rows > 0 && a.rows % a.group_rows == 0,
            "language grouped broadcast needs whole row groups");
    a_elements = std::uint64_t(a.rows / a.group_rows) * a.cols;
  }
  const std::uint64_t bytes = 2ULL * a.rows * a.cols;
  require(bytes <= working_bytes,
          "unsupported language broadcast element count");
  span(a.dst, bytes); span(a.x, bytes); span(a.a, 2 * a_elements);
  elementwise_alias(a.dst, a.x, bytes);
  disjoint(a.dst, bytes, a.a, 2 * a_elements);
  const auto control = static_cast<std::uint16_t>(
      static_cast<unsigned>(a.op) | static_cast<unsigned>(a.mode) << 2 |
      (a.x_first ? 1U : 0U) << 4);
  LaunchPlan p{tiles ? Entry::BcastTilesF16 : Entry::BcastRowsF16};
  u32(p, 0, a.dst); u32(p, 2, a.x); u32(p, 4, a.a);
  p.regs[6] = static_cast<std::uint16_t>(a.rows);
  p.regs[7] = static_cast<std::uint16_t>(a.cols);
  if (tiles) {
    const std::uint64_t blocks = std::uint64_t(a.rows) * (a.cols / 4096);
    require(blocks <= 65535, "language broadcast grid overflow");
    p.regs[8] = control;
    p.register_count = 9;
    p.grid[0] = static_cast<std::uint16_t>(blocks);
  } else {
    p.regs[8] = static_cast<std::uint16_t>(
        a.mode == BcastMode::GroupCol ? a.group_rows : 1);
    p.regs[9] = control;
    p.register_count = 10;
    p.grid[0] = static_cast<std::uint16_t>((a.rows + 15) / 16);
  }
  p.cores = 8;
  return p;
}

struct LangL2normArgs {
  std::uint32_t dst, src, rows, cols;
  float epsilon;
};

// GDN q/k L2 normalization over C128 rows on TP8 local SPM:
// Y = X * rsqrt(sum(X^2) + 1e-6). Exact in-place aliasing is legal.
inline LaunchPlan l2norm_f16(const LangL2normArgs& a) {
  using namespace detail;
  require(a.rows > 0 && a.rows <= 65535 && a.cols == 128,
          "language L2Norm requires C128 rows");
  require(a.epsilon == 1.0e-6F, "language L2Norm requires epsilon=1e-6");
  const std::uint64_t bytes = 2ULL * a.rows * a.cols;
  require(bytes <= working_bytes,
          "unsupported language L2Norm element count");
  span(a.dst, bytes); span(a.src, bytes);
  elementwise_alias(a.dst, a.src, bytes);
  LaunchPlan p{Entry::L2normF16};
  u32(p, 0, a.dst); u32(p, 2, a.src);
  p.regs[4] = static_cast<std::uint16_t>(a.rows);
  p.regs[5] = static_cast<std::uint16_t>(a.cols);
  p.regs[6] = fp32_to_fp16(a.epsilon);
  p.register_count = 7;
  p.grid[0] = static_cast<std::uint16_t>((a.rows + 15) / 16);
  p.cores = 8;
  return p;
}

enum class GeluMode : unsigned { Erf = 0, Tanh = 1 };
struct GeluArgs {
  std::uint32_t dst, src, elements;
  GeluMode mode = GeluMode::Tanh;
};

inline LaunchPlan gelu(const GeluArgs& a) {
  using namespace detail;
  require(a.elements > 0 && a.elements % 16 == 0 &&
          4ULL * a.elements + 256 <= working_bytes,
          "invalid GELU element count or working budget");
  require(a.mode == GeluMode::Erf || a.mode == GeluMode::Tanh, "invalid GELU mode");
  const std::uint64_t bytes = 2ULL * a.elements;
  span(a.dst, bytes); span(a.src, bytes); elementwise_alias(a.dst, a.src, bytes);
  LaunchPlan p{Entry::Gelu};
  u32(p, 0, a.dst); u32(p, 2, a.src); u32(p, 4, a.elements);
  p.regs[6] = static_cast<std::uint16_t>(a.mode); p.register_count = 7;
  p.grid[0] = (a.elements + 4095) / 4096;
  return p;
}

// LocalSPM per core. The source evaluates sigmoid(x), rounds to FP16, then
// rounds each multiplication (x * sigmoid(x)) * y to FP16. In-place dst==x or
// dst==y and equal x==y are legal; every other overlap is rejected.
struct SiluMulArgs {
  std::uint32_t dst, x, y, elements;
  unsigned cores = 8;
};

inline LaunchPlan silu_mul(const SiluMulArgs& a) {
  using namespace detail;
  require(a.elements > 0 && a.elements % 256 == 0 &&
          6ULL * a.elements + 256 <= working_bytes,
          "SiLU-mul requires positive 256-aligned elements within SPM budget");
  require(a.cores >= 1 && a.cores <= 8, "SiLU-mul requires 1..8 cores");
  const std::uint64_t bytes = 2ULL * a.elements;
  span(a.dst, bytes); span(a.x, bytes); span(a.y, bytes);
  elementwise_alias(a.dst, a.x, bytes);
  elementwise_alias(a.dst, a.y, bytes);
  elementwise_alias(a.x, a.y, bytes);
  LaunchPlan p{Entry::SiluMulFp16};
  u32(p, 0, a.dst); u32(p, 2, a.x); u32(p, 4, a.y);
  u32(p, 6, a.elements);
  p.register_count = 8;
  p.grid[0] = static_cast<std::uint16_t>((a.elements + 65535) / 65536);
  p.cores = a.cores;
  return p;
}

struct RopeArgs {
  std::uint32_t dst, src, cosine, sine;
  std::uint32_t tokens, heads, head_dim, rotary_dim, table_rows;
  std::uint32_t position_offset = 0;
};

inline LaunchPlan rope(const RopeArgs& a) {
  using namespace detail;
  shape16(a.tokens, a.head_dim);
  require(a.heads > 0 && a.heads <= 65535 && a.rotary_dim > 0 &&
          a.rotary_dim <= a.head_dim && a.rotary_dim % 32 == 0 &&
          a.position_offset <= a.table_rows &&
          a.tokens <= a.table_rows - a.position_offset, "invalid RoPE geometry/table range");
  const std::uint64_t bytes = 2ULL * a.tokens * a.heads * a.head_dim;
  const std::uint64_t table_bytes = std::uint64_t(a.table_rows) * a.rotary_dim;
  require(2 * bytes + 2 * table_bytes + 256 <= working_bytes,
          "RoPE exceeds the 7 MiB working budget");
  span(a.dst, bytes); span(a.src, bytes);
  span(a.cosine, table_bytes); span(a.sine, table_bytes);
  elementwise_alias(a.dst, a.src, bytes);
  disjoint(a.dst, bytes, a.cosine, table_bytes);
  disjoint(a.dst, bytes, a.sine, table_bytes);
  LaunchPlan p{Entry::Rope};
  u32(p, 0, a.dst); u32(p, 2, a.src); u32(p, 4, a.cosine); u32(p, 6, a.sine);
  p.regs[8] = a.tokens; p.regs[9] = a.heads; p.regs[10] = a.head_dim;
  p.regs[11] = a.rotary_dim; u32(p, 12, a.position_offset);
  p.register_count = 14; p.grid[0] = (a.tokens + 15) / 16;
  return p;
}

// TP8 language partial rotary. Each core owns one complete D256 Q or K head;
// the host must refresh the position-selected M-RoPE tables in each core's
// local SPM before Graph replay. The table refresh is part of this boundary.
inline LaunchPlan partial_rope_tp8(const RopeArgs& a) {
  detail::require(a.tokens >= 64 && a.tokens <= 512 && a.tokens % 64 == 0 &&
                  a.heads == 1 && a.head_dim == 256 && a.rotary_dim == 64,
                  "TP8 partial RoPE requires a 64-row D256 physical prefix bucket");
  auto p = rope(a);
  p.cores = 8;
  return p;
}

struct QkvUpShape { unsigned n, k, tile_n; };
inline QkvUpShape qkv_up_shape(Entry entry) {
  switch (entry) {
    case Entry::QkvBiasN384K1024: return {384, 1024, 48};
    case Entry::ON1024K128: return {1024, 128, 128};
    case Entry::UpN512K1024: return {512, 1024, 64};
    case Entry::DownN1024K512: return {1024, 512, 96};
    default: throw std::invalid_argument("expected a Vision GEMM entry");
  }
}
struct QkvUpGemmArgs {
  Entry entry;
  std::uint32_t dst, src, bias, rows;
  std::uint64_t weight_ddr_bytes;
  // QKV requires three independent compact [M,128] outputs: dst is Q.
  std::uint32_t dst_k = 0, dst_v = 0;
};
// A[M,K], C[M,N], bias[N] are compact FP16 LocalSPM at equal core offsets.
// O/down return unreduced partials; the caller supplies zero bias on cores 1..7.
// Broadcast the complete grid to every core. Zero bias still needs a buffer.
inline LaunchPlan qkv_up_gemm(const QkvUpGemmArgs& a) {
  using namespace detail;
  const auto s = qkv_up_shape(a.entry);
  require(a.rows > 0 && a.rows <= 868, "Vision GEMM rows outside [1,868]");
  const bool qkv = a.entry == Entry::QkvBiasN384K1024;
  const auto weight_bytes = 2ULL * 8ULL * s.n * s.k;
  require(a.weight_ddr_bytes % 256 == 0 && a.weight_ddr_bytes < (1ULL << 40) &&
          weight_bytes <= (1ULL << 40) - a.weight_ddr_bytes,
          "Vision GEMM requires a 256-byte aligned DDR range below 1 TiB");
  const auto in_bytes = 2ULL * a.rows * s.k, out_bytes = 2ULL * a.rows * s.n;
  const auto bias_bytes = 2ULL * s.n;
  require(in_bytes + out_bytes + bias_bytes <= working_bytes, "Vision GEMM exceeds SPM working budget");
  span(a.src, in_bytes); span(a.bias, bias_bytes);
  disjoint(a.src, in_bytes, a.bias, bias_bytes);
  if (qkv) {
    const auto part_bytes = out_bytes / 3;
    const std::array<std::uint32_t, 3> outputs{a.dst, a.dst_k, a.dst_v};
    for (unsigned i = 0; i < outputs.size(); ++i) {
      span(outputs[i], part_bytes);
      disjoint(outputs[i], part_bytes, a.src, in_bytes);
      disjoint(outputs[i], part_bytes, a.bias, bias_bytes);
      for (unsigned j = 0; j < i; ++j)
        disjoint(outputs[i], part_bytes, outputs[j], part_bytes);
    }
  } else {
    span(a.dst, out_bytes);
    disjoint(a.dst, out_bytes, a.src, in_bytes);
    disjoint(a.dst, out_bytes, a.bias, bias_bytes);
  }
  LaunchPlan p{a.entry};
  u32(p, 0, a.src); u32(p, 2, a.weight_ddr_bytes / 256);
  const unsigned out_reg = 4;
  u32(p, out_reg, a.dst); u32(p, out_reg + 2, a.bias);
  p.regs[out_reg + 4] = a.rows; p.register_count = out_reg + 5;
  if (qkv) {
    u32(p, out_reg + 6, a.dst_k); u32(p, out_reg + 8, a.dst_v);
    p.register_count = out_reg + 10;
  }
  p.grid[0] = (s.n + s.tile_n - 1) / s.tile_n; p.cores = 8;
  return p;
}

struct StreamedGemmShape {
  std::uint32_t k, columns, valid_columns, tile_n;
  unsigned output_count;
  std::array<std::uint32_t, 6> output_columns;
};
inline StreamedGemmShape streamed_gemm_shape(Entry entry) {
  switch (entry) {
    // QKV-ZBA gives B and A independent padded N16 weight/output blocks.
    case Entry::QkvZba:
      return {2048, 1056, 1028, 48, 6, {256, 256, 256, 256, 16, 16}};
    case Entry::Qkv:
      return {2048, 1024, 1024, 48, 4, {256, 256, 256, 256, 0, 0}};
    // Buckets above 384 rows: N64 blocks keep the FP32 M512 tile in VLM.
    case Entry::QkvZbaM448K128:
    case Entry::QkvZbaM512K128:
    case Entry::QkvZbaM512:
      return {2048, 1056, 1028, 64, 6, {256, 256, 256, 256, 16, 16}};
    case Entry::QkvQgateM448K128:
    case Entry::QkvQgateM512K128:
    case Entry::QkvQgateM512:
      return {2048, 1024, 1024, 64, 4, {256, 256, 256, 256, 0, 0}};
    case Entry::GateUpM448K128:
    case Entry::GateUpM512K128:
    case Entry::GateUpM512:
      return {2048, 1536, 1536, 64, 2, {768, 768, 0, 0, 0, 0}};
    case Entry::ON2048K256:
      return {256, 2048, 2048, 160, 1, {2048, 0, 0, 0, 0, 0}};
    case Entry::GateUp:
      return {2048, 1536, 1536, 96, 2, {768, 768, 0, 0, 0, 0}};
    case Entry::DownN2048K768:
      return {768, 2048, 2048, 96, 1, {2048, 0, 0, 0, 0, 0}};
    case Entry::TpQkvCompactF16:
      return {2048, 640, 640, 32, 1, {640, 0, 0, 0, 0, 0}};
    default: throw std::invalid_argument("expected a language GEMM entry");
  }
}
struct StreamedGemmArgs {
  Entry entry;
  // Planar destinations in semantic order. Unused entries must be zero.
  // QKV: Q,Qgate,K,V. QKV-ZBA: Q,K,V,Z,B,A. GateUp: Gate,Up.
  std::array<std::uint32_t, 6> dst;
  std::uint32_t src, rows;
  std::uint64_t weight_ddr_bytes;
};
// A: [rows,K] in core-local row-major FP16 SPM.
// W: [columns/16][K/16][8 cores][16 columns][16 K], FP16 DDR.
// Each output is an independent contiguous [rows,output_columns[i]] tensor.
// QKV-ZBA's B/A weights and outputs are separately padded to N16; only lanes
// 0..1 are semantic and lanes 2..15 must be zero in each padded weight block.
// Compact TP QKV remains one [Q256,Qgate256,K64,V64] output. Core c owns
// K/V slice c%4 of head c/4; expand with qkv_expand_kv() before consumption.
// O/Down return one unreduced partial output. The caller broadcasts the complete
// grid to all eight cores.
// Buckets above 384 rows select the *_m512 column-split entries; callers keep
// passing the base entry. The *_m512 entries also accept 128..384 rows, where
// they are bit-identical to the base entries.
inline Entry streamed_gemm_entry(Entry entry, std::uint32_t rows) {
  if (rows <= 384) return entry;
  switch (entry) {
    case Entry::QkvZba: return Entry::QkvZbaM512;
    case Entry::Qkv: return Entry::QkvQgateM512;
    case Entry::GateUp: return Entry::GateUpM512;
    default: return entry;
  }
}
inline LaunchPlan streamed_gemm(const StreamedGemmArgs& a) {
  using namespace detail;
  require(a.rows >= 64 && a.rows <= 512 && a.rows % 64 == 0,
          "language GEMM rows must be a physical prefix bucket in [64,512]");
  const Entry entry = streamed_gemm_entry(a.entry, a.rows);
  const bool k128_m448 = entry == Entry::QkvZbaM448K128 ||
      entry == Entry::QkvQgateM448K128 ||
      entry == Entry::GateUpM448K128;
  const bool k128_m512 = entry == Entry::QkvZbaM512K128 ||
      entry == Entry::QkvQgateM512K128 ||
      entry == Entry::GateUpM512K128;
  require(!k128_m448 || a.rows == 448, "K128 M448 entry requires rows448");
  require(!k128_m512 || a.rows == 512, "K128 M512 entry requires rows512");
  const bool m512 = entry == Entry::QkvZbaM512 ||
      entry == Entry::QkvQgateM512 || entry == Entry::GateUpM512;
  require(a.rows <= 384 || m512 || k128_m448 || k128_m512 || entry == Entry::ON2048K256 ||
              entry == Entry::DownN2048K768,
          "language GEMM entry admits physical prefix buckets up to 384");
  require(!m512 || a.rows >= 128,
          "M512 language GEMM entries require at least 128 rows");
  const auto s = streamed_gemm_shape(entry);
  const std::uint64_t ab = 2ULL * a.rows * s.k;
  std::uint64_t cb = 0;
  for (unsigned i = 0; i < s.output_count; ++i)
    cb += 2ULL * a.rows * s.output_columns[i];
  require(ab + cb <= working_bytes, "language GEMM exceeds SPM working budget");
  span(a.src, ab);
  for (unsigned i = 0; i < s.output_count; ++i) {
    const auto bytes = 2ULL * a.rows * s.output_columns[i];
    span(a.dst[i], bytes);
    disjoint(a.dst[i], bytes, a.src, ab);
    for (unsigned j = 0; j < i; ++j)
      disjoint(a.dst[i], bytes, a.dst[j],
               2ULL * a.rows * s.output_columns[j]);
  }
  for (unsigned i = s.output_count; i < a.dst.size(); ++i)
    require(a.dst[i] == 0, "unused language GEMM destinations must be zero");
  const std::uint64_t wb = 2ULL * 8ULL * s.columns * s.k;
  require(a.weight_ddr_bytes % 256 == 0 && a.weight_ddr_bytes < (1ULL << 40) &&
          wb <= (1ULL << 40) - a.weight_ddr_bytes,
          "language GEMM requires a 256-byte-aligned DDR range below 1 TiB");
  LaunchPlan p{entry};
  u32(p, 0, a.src);
  u32(p, 2, static_cast<std::uint32_t>(a.weight_ddr_bytes >> 8));
  const unsigned out_reg = 4;
  for (unsigned i = 0; i < s.output_count; ++i)
    u32(p, out_reg + 2 * i, a.dst[i]);
  const unsigned rows_reg = out_reg + 2 * s.output_count;
  p.regs[rows_reg] = a.rows;
  p.register_count = rows_reg + 1; p.cores = 8;
  p.grid[0] = (s.columns + s.tile_n - 1) / s.tile_n;
  return p;
}

struct QkvExpandKvArgs {
  std::uint32_t dst, src, rows;
};
// Expand eight [rows,640] compact TP outputs into eight [rows,1024] outputs.
// All core-local input/output offsets are identical across cores. Source and
// destination must be disjoint; broadcast the whole grid to all eight cores,
// then wait for every core before consuming the destination. This plan covers
// ordinary parameter registers 0..5 only. After apply() and before adding the
// kernel or building its batch, the expansion host path must set 32-bit register
// 128 to zero. This launch requires an eight-warp queue configuration.
// Validate both initial execution and repeated replay in the consuming runtime.
inline LaunchPlan qkv_expand_kv(const QkvExpandKvArgs& a) {
  using namespace detail;
  require(a.rows >= 64 && a.rows <= 384 && a.rows % 64 == 0,
          "QKV expansion rows must be a physical prefix bucket in [64,384]");
  const std::uint64_t in_bytes = 2ULL * a.rows * 640;
  const std::uint64_t out_bytes = 2ULL * a.rows * 1024;
  require(in_bytes + out_bytes <= working_bytes,
          "QKV expansion exceeds SPM working budget");
  span(a.src, in_bytes); span(a.dst, out_bytes);
  disjoint(a.src, in_bytes, a.dst, out_bytes);
  LaunchPlan p{Entry::QkvExpandKvF16};
  u32(p, 0, global_spm_base + a.dst); u32(p, 2, a.src);
  p.regs[4] = a.rows; p.regs[5] = core_stride_bytes >> 16;
  p.register_count = 6; p.cores = 8;
  p.grid[0] = (a.rows + 15) / 16;
  return p;
}

struct AllReduceArgs {
  std::uint32_t dst, src, residual, rows;
  // Accepted for source compatibility (0 or 1..25) and ignored: the pull
  // schedule always splits each shard over eight resident blocks.
  unsigned vectors_per_block = 0;
  unsigned cols = 1024;
  bool with_residual = true;  // false: pure TP8 sum; residual field is ignored
  unsigned sync_epoch = 0;  // unique occurrence within one resident batch
};
// dst/src/residual are core-local SPM offsets, equal on every core. The
// kernel pulls peer inputs and shards through the global SPM window and
// stores only to local SPM. Broadcast grid 8 to cores 0..7.
inline LaunchPlan all_reduce_residual(const AllReduceArgs& a) {
  using namespace detail;
  require(a.rows >= 1 && ((a.cols == 1024 && a.rows <= 868) || (a.cols == 2048 && a.rows <= 512)),
          "unsupported all-reduce shape");
  require(a.vectors_per_block <= 25, "invalid all-reduce vectors per block");
  require(a.sync_epoch <= 47, "all-reduce sync epoch exceeds the graph envelope");
  const std::uint64_t bytes = 2ULL * a.rows * a.cols;
  span(a.dst, bytes); span(a.src, bytes); disjoint(a.dst, bytes, a.src, bytes);
  if (a.with_residual) {
    span(a.residual, bytes); disjoint(a.dst, bytes, a.residual, bytes);
  }
  LaunchPlan p{Entry::AllReduceResidual};
  u32(p, 0, global_spm_base + a.dst); u32(p, 2, global_spm_base + a.src);
  u32(p, 4, a.with_residual ? a.residual : 0); p.regs[6] = a.rows;
  p.regs[7] = core_stride_bytes >> 16; p.regs[8] = 0;
  p.regs[9] = a.cols; p.regs[10] = a.with_residual;
  p.grid[0] = 8;
  p.regs[11] = p.grid[0];
  // Assign one-shot global counters in descending order. Preserve each
  // occurrence's completion state until its consumers finish.
  p.regs[12] = 50 - a.sync_epoch;
  // The launch ABI aligns the two trailing 32-bit arguments to even slots.
  u32(p, 14, a.dst); u32(p, 16, a.src);
  p.register_count = 18; p.cores = 8;
  return p;
}

inline constexpr unsigned denoise_full_attention_layers = 6;
inline constexpr std::uint64_t denoise_cache_block_bytes = 64U * 1024U;

struct CopyPhysicalPrefixF16Layer {
  std::uint64_t src_k_ddr_bytes;
  std::uint64_t src_v_ddr_bytes;
  std::uint64_t dst_k_ddr_bytes;
  std::uint64_t dst_v_ddr_bytes;
};

struct CopyPhysicalPrefixF16Args {
  std::array<CopyPhysicalPrefixF16Layer,
             denoise_full_attention_layers> layers;
  std::uint32_t prefix_len;
  std::uint32_t bucket_len;
  // Host allocation bound only. The device program receives prefix/bucket.
  std::uint32_t capacity_tokens;
};

// Copy the six full-attention K/V physical prefixes from the base decoder to
// the action decoder. Each tensor has physical FP16 shape
// [1, blocks, 1, 16, 8, 16, 16], hence one 16-token block occupies 64 KiB.
// DDR address arguments use 256-byte units. Device argument order is
// srcK, srcV, dstK, dstV for each layer, then prefix_len and bucket_len.
inline LaunchPlan copy_physical_prefix_f16(
    const CopyPhysicalPrefixF16Args& a) {
  using namespace detail;
  require(a.prefix_len >= 1 && a.prefix_len <= 512,
          "action prefix length must be in [1,512]");
  const std::uint32_t rounded_bucket = (a.prefix_len + 63U) / 64U * 64U;
  require(a.bucket_len == a.prefix_len || a.bucket_len == rounded_bucket,
          "action bucket must be exact or ceil64(prefix)");
  require(a.capacity_tokens > 0 && a.capacity_tokens % 16 == 0 &&
          a.bucket_len <= a.capacity_tokens,
          "action cache capacity must be 16-aligned and cover the bucket");

  const std::uint64_t source_bytes =
      ((std::uint64_t(a.prefix_len) + 15U) / 16U) *
      denoise_cache_block_bytes;
  const std::uint64_t destination_bytes =
      ((std::uint64_t(a.bucket_len) + 15U) / 16U) *
      denoise_cache_block_bytes;
  std::array<std::uint64_t, 2 * denoise_full_attention_layers> sources{};
  std::array<std::uint64_t, 2 * denoise_full_attention_layers> destinations{};
  for (unsigned layer = 0; layer < a.layers.size(); ++layer) {
    const auto& l = a.layers[layer];
    sources[2 * layer] = l.src_k_ddr_bytes;
    sources[2 * layer + 1] = l.src_v_ddr_bytes;
    destinations[2 * layer] = l.dst_k_ddr_bytes;
    destinations[2 * layer + 1] = l.dst_v_ddr_bytes;
  }
  for (const auto address : sources) ddr_span(address, source_bytes);
  for (unsigned i = 0; i < destinations.size(); ++i) {
    ddr_span(destinations[i], destination_bytes);
    for (unsigned j = 0; j < i; ++j)
      require(!ddr_overlap(destinations[i], destination_bytes,
                           destinations[j], destination_bytes),
              "action destination cache ranges must be pairwise disjoint");
    for (const auto source : sources)
      require(!ddr_overlap(destinations[i], destination_bytes,
                           source, source_bytes),
              "action destination cache overlaps a source cache");
  }

  LaunchPlan p{Entry::CopyPhysicalPrefixF16};
  unsigned reg = 0;
  for (const auto& l : a.layers) {
    for (const auto address : {l.src_k_ddr_bytes, l.src_v_ddr_bytes,
                               l.dst_k_ddr_bytes, l.dst_v_ddr_bytes}) {
      u32(p, reg, static_cast<std::uint32_t>(address / 256U));
      reg += 2;
    }
  }
  p.regs[48] = static_cast<std::uint16_t>(a.prefix_len);
  p.regs[49] = static_cast<std::uint16_t>(a.bucket_len);
  p.register_count = 50;
  p.grid = {static_cast<std::uint16_t>((a.bucket_len + 15U) / 16U), 2, 6};
  p.cores = 8;
  return p;
}

inline constexpr std::uint32_t vla_embedding_hidden = 2048;
inline constexpr std::uint32_t vla_embedding_vocab_rows = 256277;
inline constexpr std::uint32_t vla_embedding_max_rows = 512;

struct GatherEmbeddingF16Args {
  std::uint64_t ids_ddr_bytes;
  std::uint64_t vocab_ddr_bytes;
  std::uint32_t dst;
  std::uint32_t vocab_rows;
  std::uint32_t valid_rows;
  std::uint32_t output_rows;
};

// Gather compact int32 ids from DDR into a padded core-0 FP16 SPM tensor.
// The caller validates every id in [0,vocab_rows) before enqueue.  The kernel
// independently turns an invalid id into a zero row, but that is only an OOB
// safety backstop and not a substitute for host validation.
inline LaunchPlan gather_embedding_f16(const GatherEmbeddingF16Args& a) {
  using namespace detail;
  require(a.vocab_rows == vla_embedding_vocab_rows,
          "embedding requires vocab size 256277");
  require(a.valid_rows >= 1 && a.valid_rows <= a.output_rows &&
              a.output_rows <= vla_embedding_max_rows,
          "embedding rows must satisfy 1 <= valid <= output <= 512");
  const std::uint64_t ids_bytes = 4ULL * a.valid_rows;
  const std::uint64_t vocab_bytes =
      2ULL * a.vocab_rows * vla_embedding_hidden;
  const std::uint64_t output_bytes =
      2ULL * a.output_rows * vla_embedding_hidden;
  require(output_bytes <= working_bytes,
          "embedding output exceeds the SPM working budget");
  ddr_span(a.ids_ddr_bytes, ids_bytes);
  ddr_span(a.vocab_ddr_bytes, vocab_bytes);
  require(!ddr_overlap(a.ids_ddr_bytes, ids_bytes,
                       a.vocab_ddr_bytes, vocab_bytes),
          "embedding ids overlap the vocabulary");
  span(a.dst, output_bytes);

  LaunchPlan p{Entry::GatherEmbeddingF16};
  u32(p, 0, static_cast<std::uint32_t>(a.ids_ddr_bytes >> 8));
  u32(p, 2, static_cast<std::uint32_t>(a.vocab_ddr_bytes >> 8));
  u32(p, 4, a.dst);
  u32(p, 6, a.vocab_rows);
  p.regs[8] = static_cast<std::uint16_t>(a.valid_rows);
  p.regs[9] = static_cast<std::uint16_t>(a.output_rows);
  p.register_count = 10;
  p.grid[0] = static_cast<std::uint16_t>((a.output_rows + 31U) / 32U);
  return p;
}

// Eight-core variant: core c gathers rows [c*R, (c+1)*R), R = output_rows/8,
// into its own LocalSPM at dst (compact slice), for one eight-channel
// SPM->DDR scatter. output_rows must be a multiple of 64. Broadcast launch.
inline LaunchPlan gather_embedding_tp8_f16(const GatherEmbeddingF16Args& a) {
  using namespace detail;
  require(a.output_rows % 64 == 0,
          "TP8 embedding requires output rows divisible by 64");
  LaunchPlan p = gather_embedding_f16(a);
  p.entry = Entry::GatherEmbeddingTp8F16;
  p.grid[0] = static_cast<std::uint16_t>(a.output_rows / 64U);
  p.cores = 8;
  return p;
}

enum class IoGemmKind {
  ColumnParallel,
  RowParallel,
  InputProjection,
  InputProjectionScaled,
  OutputProjection,
};

struct AdaptiveRmsnormArgs {
  std::uint32_t dst, src, modulation;
  float epsilon = 1.0e-6F;
  std::uint32_t rows = 32;
  // M16 residual storage may use S=256 instead of the legacy S=4. The
  // normalization factor is S/4; epsilon retains the legacy eps/16 units.
  float normalization_scale = 1.0F;
};

inline LaunchPlan adaptive_rmsnorm(
    const AdaptiveRmsnormArgs& a) {
  using namespace detail;
  compact_row_bucket_rows(a.rows);
  row_bucket_normalization_scale(a.rows, a.normalization_scale);
  const std::uint64_t tensor_bytes = 2ULL * a.rows * 1024;
  constexpr std::uint64_t modulation_bytes = 2ULL * 3 * 1024;
  const auto epsilon_bits = fp32_to_fp16(a.epsilon);
  require(positive_finite_fp16(epsilon_bits),
          "Action adaptive RMSNorm requires positive finite FP16 epsilon");
  span(a.dst, tensor_bytes); span(a.src, tensor_bytes);
  span(a.modulation, modulation_bytes);
  elementwise_alias(a.dst, a.src, tensor_bytes);
  disjoint(a.dst, tensor_bytes, a.modulation, modulation_bytes);
  disjoint(a.src, tensor_bytes, a.modulation, modulation_bytes);
  LaunchPlan p{(a.rows == 16 ? Entry::AdaptiveRmsnormM16H1024F16 : Entry::AdaptiveRmsnormM32H1024F16)};
  u32(p, 0, a.dst); u32(p, 2, a.src); u32(p, 4, a.modulation);
  const float mean_scale = a.normalization_scale * a.normalization_scale / 1024.0F;
  std::uint32_t mean_scale_bits;
  std::memcpy(&mean_scale_bits, &mean_scale, sizeof(mean_scale_bits));
  u32(p, 6, mean_scale_bits);
  p.regs[8] = epsilon_bits;
  p.register_count = 9; p.grid[0] = a.rows / 16; p.cores = 8;
  if (a.rows == 16) {
    p.regs[9] = fp32_to_fp16(a.normalization_scale);
    p.register_count = 10;
  }
  return p;
}

// Plain adaptive RMSNorm entries use hardware sqrt_reciprocal + valuRepeat.
// The _hw_ aliases are kept for compatibility; they select the same kernel
// implementation through the ABI-padded Entry tags.
inline LaunchPlan action_adaptive_rmsnorm_hw(
    const AdaptiveRmsnormArgs& a) {
  auto p = adaptive_rmsnorm(a);
  p.entry = a.rows == 16 ? Entry::AdaptiveRmsnormHwM16H1024F16
                         : Entry::AdaptiveRmsnormHwM32H1024F16;
  return p;
}

struct ResidualGateArgs {
  std::uint32_t dst, branch, residual, gate;
  std::uint32_t rows = 32;
};

inline LaunchPlan residual_gate(const ResidualGateArgs& a) {
  using namespace detail;
  compact_row_bucket_rows(a.rows);
  const std::uint64_t tensor_bytes = 2ULL * a.rows * 1024;
  constexpr std::uint64_t gate_bytes = 2ULL * 1024;
  span(a.dst, tensor_bytes); span(a.branch, tensor_bytes);
  span(a.residual, tensor_bytes); span(a.gate, gate_bytes);
  elementwise_alias(a.dst, a.branch, tensor_bytes);
  disjoint(a.dst, tensor_bytes, a.residual, tensor_bytes);
  disjoint(a.dst, tensor_bytes, a.gate, gate_bytes);
  disjoint(a.branch, tensor_bytes, a.residual, tensor_bytes);
  disjoint(a.branch, tensor_bytes, a.gate, gate_bytes);
  disjoint(a.residual, tensor_bytes, a.gate, gate_bytes);
  LaunchPlan p{(a.rows == 16 ? Entry::ResidualGateM16H1024F16 : Entry::ResidualGateM32H1024F16)};
  u32(p, 0, a.dst); u32(p, 2, a.branch);
  u32(p, 4, a.residual); u32(p, 6, a.gate);
  p.register_count = 8; p.grid[0] = a.rows / 16; p.cores = 8;
  return p;
}

// Plain residual-gate entries use hardware mul+add. The _hw_ aliases share
// the register layout through the ABI-padded Entry tags.
inline LaunchPlan action_residual_gate_hw(const ResidualGateArgs& a) {
  auto p = residual_gate(a);
  p.entry = a.rows == 16 ? Entry::ResidualGateHwM16H1024F16
                         : Entry::ResidualGateHwM32H1024F16;
  return p;
}

struct SigmoidMulArgs {
  std::uint32_t dst, gate, value;
  std::uint32_t rows = 32;
};

inline LaunchPlan sigmoid_mul(const SigmoidMulArgs& a) {
  using namespace detail;
  compact_row_bucket_rows(a.rows);
  const std::uint64_t bytes = 2ULL * a.rows * 256;
  span(a.dst, bytes); span(a.gate, bytes); span(a.value, bytes);
  elementwise_alias(a.dst, a.value, bytes);
  disjoint(a.dst, bytes, a.gate, bytes);
  disjoint(a.gate, bytes, a.value, bytes);
  LaunchPlan p{(a.rows == 16 ? Entry::SigmoidMulM16D256F16 : Entry::SigmoidMulM32D256F16)};
  u32(p, 0, a.dst); u32(p, 2, a.gate); u32(p, 4, a.value);
  p.register_count = 6; p.grid[0] = a.rows / 16; p.cores = 8;
  return p;
}

struct EulerUpdateArgs {
  std::uint32_t action, velocity, keep_mask, padding_velocity;
  float delta_t;
  std::uint32_t rows = 32;
};

inline LaunchPlan euler_update(const EulerUpdateArgs& a) {
  using namespace detail;
  compact_row_bucket_rows(a.rows);
  const std::uint64_t tensor_bytes = 2ULL * a.rows * 64;
  constexpr std::uint64_t mask_bytes = 2ULL * 64;
  span(a.action, tensor_bytes); span(a.velocity, tensor_bytes);
  span(a.keep_mask, mask_bytes); span(a.padding_velocity, tensor_bytes);
  disjoint(a.action, tensor_bytes, a.velocity, tensor_bytes);
  disjoint(a.action, tensor_bytes, a.keep_mask, mask_bytes);
  disjoint(a.action, tensor_bytes, a.padding_velocity, tensor_bytes);
  disjoint(a.velocity, tensor_bytes, a.keep_mask, mask_bytes);
  disjoint(a.velocity, tensor_bytes, a.padding_velocity, tensor_bytes);
  disjoint(a.keep_mask, mask_bytes, a.padding_velocity, tensor_bytes);
  const auto delta_t_bits = fp32_to_fp16(a.delta_t);
  require(positive_finite_fp16(delta_t_bits),
          "Action Euler delta_t must be positive finite FP16");
  LaunchPlan p{(a.rows == 16 ? Entry::EulerUpdateM16D64F16 : Entry::EulerUpdateM32D64F16)};
  u32(p, 0, a.action); u32(p, 2, a.velocity);
  u32(p, 4, a.keep_mask); u32(p, 6, a.padding_velocity);
  p.regs[8] = delta_t_bits;
  p.register_count = 9; p.grid[0] = a.rows / 16; p.cores = 1;
  return p;
}

struct QkRmsnormMropeArgs {
  std::uint32_t dst, src, weight;
  std::uint64_t cosine_ddr_bytes, sine_ddr_bytes;
  std::uint32_t table_rows;
  std::uint32_t position_offset = 0;
  float epsilon = 1.0e-6F;
  std::uint32_t rows = 32;
};

inline LaunchPlan qk_rmsnorm_mrope(
    const QkRmsnormMropeArgs& a) {
  using namespace detail;
  compact_row_bucket_rows(a.rows);
  const std::uint64_t tensor_bytes = 2ULL * a.rows * 256;
  constexpr std::uint64_t weight_bytes = 2ULL * 256;
  require(a.table_rows >= a.rows && a.position_offset <= a.table_rows &&
              a.rows <= a.table_rows - a.position_offset,
          "Action QK M-RoPE table does not cover action rows");
  require(a.position_offset <= 65535,
          "Action QK M-RoPE position exceeds uint16");
  require(a.epsilon == 1.0e-6F,
          "Action QK RMSNorm requires epsilon=1e-6");
  span(a.dst, tensor_bytes); span(a.src, tensor_bytes);
  span(a.weight, weight_bytes);
  elementwise_alias(a.dst, a.src, tensor_bytes);
  disjoint(a.dst, tensor_bytes, a.weight, weight_bytes);
  disjoint(a.src, tensor_bytes, a.weight, weight_bytes);
  const std::uint64_t table_bytes = 2ULL * a.table_rows * 32;
  ddr_span(a.cosine_ddr_bytes, table_bytes);
  ddr_span(a.sine_ddr_bytes, table_bytes);
  require(!ddr_overlap(a.cosine_ddr_bytes, table_bytes,
                       a.sine_ddr_bytes, table_bytes),
          "Action cosine and sine tables overlap");
  LaunchPlan p{(a.rows == 16 ? Entry::QkRmsnormMropeM16D256F16 : Entry::QkRmsnormMropeM32D256F16)};
  u32(p, 0, a.dst); u32(p, 2, a.src); u32(p, 4, a.weight);
  u32(p, 6, static_cast<std::uint32_t>(a.cosine_ddr_bytes >> 8));
  u32(p, 8, static_cast<std::uint32_t>(a.sine_ddr_bytes >> 8));
  p.regs[10] = static_cast<std::uint16_t>(a.position_offset);
  const float mean_scale = 1.0F / 256.0F;
  std::uint32_t mean_scale_bits;
  std::memcpy(&mean_scale_bits, &mean_scale, sizeof(mean_scale_bits));
  u32(p, 11, mean_scale_bits);
  p.regs[13] = fp32_to_fp16(a.epsilon);
  p.register_count = 14; p.grid[0] = a.rows / 16; p.cores = 8;
  return p;
}

struct IoGemmParams {
  IoGemmKind kind;
  std::uint32_t dst;
  std::uint32_t src;
  std::uint64_t weight_ddr_bytes;
  std::uint32_t bias = 0;
  std::uint32_t rows = 32;
};

// Fixed-shape Action FP16 GEMMs. All kernels accumulate with
// VMAT FP32 and round once when storing FP16. ColumnParallel/RowParallel are
// the bias-free TP8 transformer projections. InputProjection/OutputProjection
// are the single-core bias projections at the Action loop boundary. DDR address
// consumes the transformed-weight address in 256-byte units.
inline LaunchPlan io_gemm(const IoGemmParams& a) {
  using namespace detail;
  compact_row_bucket_rows(a.rows);
  const bool valid = a.kind == IoGemmKind::ColumnParallel ||
                     a.kind == IoGemmKind::RowParallel ||
                     a.kind == IoGemmKind::InputProjection ||
                     a.kind == IoGemmKind::InputProjectionScaled ||
                     a.kind == IoGemmKind::OutputProjection;
  require(valid, "unknown Action GEMM kind");
  const bool column = a.kind == IoGemmKind::ColumnParallel;
  const bool row = a.kind == IoGemmKind::RowParallel;
  const bool input = a.kind == IoGemmKind::InputProjection ||
                     a.kind == IoGemmKind::InputProjectionScaled;
  const bool scaled_input =
      a.kind == IoGemmKind::InputProjectionScaled;
  const bool io = input || a.kind == IoGemmKind::OutputProjection;
  const std::uint64_t weight_bytes = io ? 2ULL * 64 * 1024
                                        : 2ULL * 2048 * 1024;
  const std::uint64_t src_bytes = column ? 2ULL * a.rows * 1024
      : row ? 2ULL * a.rows * 256
      : input ? 2ULL * a.rows * 64
      : 2ULL * a.rows * 1024;
  const std::uint64_t dst_bytes = column ? 2ULL * a.rows * 256
      : row ? 2ULL * a.rows * 1024
      : input ? 2ULL * a.rows * 1024
      : 2ULL * a.rows * 64;
  span(a.src, src_bytes);
  span(a.dst, dst_bytes);
  disjoint(a.dst, dst_bytes, a.src, src_bytes);
  ddr_span(a.weight_ddr_bytes, weight_bytes);
  if (io) {
    const std::uint64_t bias_bytes = input ? 2ULL * 1024 : 2ULL * 64;
    span(a.bias, bias_bytes);
    disjoint(a.dst, dst_bytes, a.bias, bias_bytes);
    disjoint(a.src, src_bytes, a.bias, bias_bytes);
  } else {
    require(a.bias == 0, "TP8 Action GEMM does not consume bias");
  }

  const Entry entry = column ? (a.rows == 16 ? Entry::GemmDdrColM16N256K1024Acc32 : Entry::GemmDdrColM32N256K1024Acc32)
      : row ? (a.rows == 16 ? Entry::GemmDdrRowM16N1024K256Acc32 : Entry::GemmDdrRowM32N1024K256Acc32)
      : input ? (scaled_input
          ? (a.rows == 16 ? Entry::GemmDdrBiasScaleM16N1024K64Acc32 : Entry::GemmDdrBiasScaleM32N1024K64Acc32)
          : (a.rows == 16 ? Entry::GemmDdrBiasM16N1024K64Acc32 : Entry::GemmDdrBiasM32N1024K64Acc32))
      : (a.rows == 16 ? Entry::GemmDdrBiasM16N64K1024Acc32 : Entry::GemmDdrBiasM32N64K1024Acc32);
  LaunchPlan p{entry};
  u32(p, 0, a.src);
  u32(p, 2, static_cast<std::uint32_t>(a.weight_ddr_bytes >> 8));
  u32(p, 4, a.dst);
  if (io) u32(p, 6, a.bias);
  p.register_count = io ? 8 : 6;
  p.grid[0] = column ? 2 : row ? 8 : input ? 32 : 2;
  p.cores = io ? 1 : 8;
  return p;
}

// Works with the public Rhino Launch Kernel_t without a header/link dependency.
// Return the first SDK error. Never enqueue a partially configured kernel.
template <class Kernel>
std::uint32_t apply(Kernel& kernel, const LaunchPlan& plan,
                    bool initialize_atomic_registers = true) {
  if (plan.register_count > plan.regs.size())
    throw std::invalid_argument("invalid xinfer::rpu register count");
  auto rc = kernel.reset_regs();
  if (rc) return rc;
  for (unsigned i = 0; i < plan.register_count; ++i) {
    rc = kernel.set_regs(i, plan.regs[i]);
    if (rc) return rc;
  }
  // all_reduce_residual counts block arrivals on core counter 0 and releases
  // blocks on core counter 1; one occurrence-specific global counter carries
  // its entry, mid and exit rendezvous. Occurrences own ids 50..3 in
  // descending order, so the global counter never aliases a core counter and
  // every sync_epoch remains independently launchable. Atomic launch
  // registers are outside the ordinary packed argument range.
  // all_reduce_residual_layernorm uses the same counters; its id is slot 9.
  if (initialize_atomic_registers && (plan.entry == Entry::AllReduceResidual ||
      plan.entry == Entry::AllReduceResidualLayerNormFusion)) {
    rc = kernel.set_regs(128, std::uint32_t{0});
    if (rc) return rc;
    rc = kernel.set_regs(130, std::uint32_t{0});
    if (rc) return rc;
    const unsigned id = plan.regs[plan.entry == Entry::AllReduceResidual ? 12 : 9];
    rc = kernel.set_regs(128 + 2 * id, std::uint32_t{0});
    if (rc) return rc;
  }
  return 0;
}
}  // namespace xinfer::rpu::abi
