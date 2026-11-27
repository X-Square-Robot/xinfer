// Copyright 2026 X-Square Robot
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "common/rpu/causal_attention_layout.hpp"
#include "common/rpu/kernel_abi.hpp"

namespace xinfer::rpu::abi {
inline void causal_cache_range(std::uint64_t address) {
  constexpr std::uint64_t bytes = causal_attention::cache_elements() * 2;
  detail::require(address != 0 && address % 256 == 0 &&
      address < (1ULL << 40) && bytes <= (1ULL << 40) - address,
      "D256 cache DDR range must be 256-byte aligned and below 1 TiB");
}

struct CausalCacheInsertArgs {
  std::uint32_t src, rows, position;
  std::uint64_t cache_ddr_bytes;
  bool value = false;
};
inline LaunchPlan causal_cache_insert(const CausalCacheInsertArgs& a) {
  detail::require(a.rows > 0 && a.rows <= 512 &&
      a.position <= causal_attention::capacity &&
      a.rows <= causal_attention::capacity - a.position,
      "D256 cache insert exceeds the admitted range");
  causal_cache_range(a.cache_ddr_bytes);
  detail::span(a.src, 2ULL * a.rows * causal_attention::dim);
  LaunchPlan p{a.value ? Entry::VcacheInsertD256 : Entry::KcacheInsertD256};
  detail::u32(p, 0, a.src);
  detail::u32(p, 2, static_cast<std::uint32_t>(a.cache_ddr_bytes >> 8));
  p.regs[4] = a.rows;
  p.regs[5] = a.position;
  p.register_count = 6;
  p.cores = causal_attention::cores;
  p.grid[0] = (a.rows + (a.value ? a.position % 16 : 0) + 15) / 16;
  return p;
}

// physical prefix buckets admitted by the resident Attention pair.
inline bool resident_prefill_bucket(std::uint32_t rows) {
  return rows >= 64 && rows <= 512 && rows % 64 == 0;
}

// Exact initial-prefill V insert that keeps the accepted DDR cache result and
// also publishes its transposed [256,M] value tile to a fixed LocalSPM
// handoff for bucket M. The following resident Attention node must run on the
// same cores.
struct CausalValueCacheInsertStagedArgs {
  std::uint32_t src, value_spm;
  std::uint64_t cache_ddr_bytes;
  std::uint32_t rows, position;
};
inline LaunchPlan causal_value_cache_insert_staged(
    const CausalValueCacheInsertStagedArgs& a) {
  detail::require(resident_prefill_bucket(a.rows) && a.position == 0,
      "staged D256 V cache insert requires a 64..512 bucket at position 0");
  const std::uint64_t tensor_bytes = 2ULL * a.rows * causal_attention::dim;
  causal_cache_range(a.cache_ddr_bytes);
  detail::span(a.src, tensor_bytes);
  detail::span(a.value_spm, tensor_bytes);
  detail::disjoint(a.src, tensor_bytes, a.value_spm, tensor_bytes);
  LaunchPlan p{Entry::VcacheInsertStagedD256};
  detail::u32(p, 0, a.src);
  detail::u32(p, 2, a.value_spm);
  detail::u32(p, 4,
      static_cast<std::uint32_t>(a.cache_ddr_bytes >> 8));
  p.regs[6] = a.rows;
  p.regs[7] = a.position;
  p.register_count = 8;
  p.cores = causal_attention::cores;
  p.grid[0] = static_cast<std::uint16_t>(a.rows / 16);
  return p;
}

struct CausalSdpaArgs {
  std::uint32_t dst, query;
  std::uint64_t key_ddr_bytes, value_ddr_bytes;
  std::uint32_t query_rows, kv_length;
};
inline LaunchPlan causal_sdpa(const CausalSdpaArgs& a) {
  detail::require(a.query_rows > 0 && a.query_rows <= 384 &&
      a.kv_length >= a.query_rows &&
      a.kv_length <= causal_attention::capacity,
      "D256 causal SDPA requires 1..384 query rows within cache capacity");
  causal_cache_range(a.key_ddr_bytes);
  causal_cache_range(a.value_ddr_bytes);
  constexpr std::uint64_t cache_bytes = causal_attention::cache_elements() * 2;
  detail::require(a.key_ddr_bytes + cache_bytes <= a.value_ddr_bytes ||
                  a.value_ddr_bytes + cache_bytes <= a.key_ddr_bytes,
      "D256 K/V cache DDR ranges must be disjoint");
  const auto bytes = 2ULL * a.query_rows * causal_attention::dim;
  detail::span(a.query, bytes);
  detail::span(a.dst, bytes);
  detail::disjoint(a.dst, bytes, a.query, bytes);
  LaunchPlan p{Entry::CausalSdpaD256};
  detail::u32(p, 0, a.dst);
  detail::u32(p, 2, a.query);
  detail::u32(p, 4, static_cast<std::uint32_t>(a.key_ddr_bytes >> 8));
  detail::u32(p, 6, static_cast<std::uint32_t>(a.value_ddr_bytes >> 8));
  p.regs[8] = a.query_rows;
  p.regs[9] = a.kv_length;
  p.register_count = 10;
  p.cores = causal_attention::cores;
  p.grid[0] = (a.query_rows + 47) / 48;
  return p;
}

// Exact initial-prefill Attention consumer for resident producer buffers of
// one bucket M. K is the unchanged compact [M,256] K-insert source and V is
// the transposed [256,M] handoff published by
// causal_value_cache_insert_staged().
struct CausalSdpaSpmArgs {
  std::uint32_t dst, query, key_spm, value_spm;
  std::uint32_t query_rows, kv_length;
};
inline LaunchPlan causal_sdpa_prefix_spm(const CausalSdpaSpmArgs& a) {
  detail::require(resident_prefill_bucket(a.query_rows) &&
      a.kv_length == a.query_rows,
      "SPM D256 causal SDPA requires one 64..512 bucket with KV == M");
  const std::uint64_t tensor_bytes =
      2ULL * a.query_rows * causal_attention::dim;
  const std::array<std::uint32_t, 4> offsets{
      a.dst, a.query, a.key_spm, a.value_spm};
  for (const auto offset : offsets) detail::span(offset, tensor_bytes);
  for (unsigned i = 0; i < offsets.size(); ++i)
    for (unsigned j = 0; j < i; ++j)
      detail::disjoint(offsets[i], tensor_bytes, offsets[j], tensor_bytes);
  detail::require(4 * tensor_bytes <= working_bytes,
      "SPM D256 causal SDPA exceeds the SPM working budget");

  LaunchPlan p{Entry::CausalSdpaPrefixSpmD256};
  detail::u32(p, 0, a.dst);
  detail::u32(p, 2, a.query);
  detail::u32(p, 4, a.key_spm);
  detail::u32(p, 6, a.value_spm);
  p.regs[8] = a.query_rows;
  p.regs[9] = a.kv_length;
  p.register_count = 10;
  p.cores = causal_attention::cores;
  p.grid[0] = 8;
  return p;
}

struct MaskedSdpaArgs {
  std::uint32_t dst, query, mask;
  std::uint64_t key_ddr_bytes, value_ddr_bytes;
  std::uint32_t kv_length;
  std::uint32_t rows = 32;
};

// Exact Action attention profile: one D256 head per core, Q16/Q32, and a
// bucketed two-dimensional FP16 mask already resident in LocalSPM.
inline LaunchPlan masked_sdpa(const MaskedSdpaArgs& a) {
  detail::compact_row_bucket_rows(a.rows);
  detail::require(a.kv_length >= a.rows + 64 && a.kv_length <= a.rows + 512 &&
                      (a.kv_length - a.rows) % 64 == 0,
                  "Action SDPA requires KV length rows+64*k, k in [1,8]");
  causal_cache_range(a.key_ddr_bytes);
  causal_cache_range(a.value_ddr_bytes);
  constexpr std::uint64_t cache_bytes =
      causal_attention::cache_elements() * 2;
  detail::require(a.key_ddr_bytes + cache_bytes <= a.value_ddr_bytes ||
                      a.value_ddr_bytes + cache_bytes <= a.key_ddr_bytes,
                  "Action K/V cache DDR ranges must be disjoint");
  const std::uint64_t query_bytes = 2ULL * a.rows * 256;
  const std::uint64_t mask_bytes = 2ULL * a.rows * a.kv_length;
  detail::span(a.query, query_bytes); detail::span(a.dst, query_bytes);
  detail::span(a.mask, mask_bytes);
  detail::disjoint(a.dst, query_bytes, a.query, query_bytes);
  detail::disjoint(a.dst, query_bytes, a.mask, mask_bytes);
  detail::disjoint(a.query, query_bytes, a.mask, mask_bytes);
  LaunchPlan p{a.rows == 16 ? Entry::SdpaMaskedM16D256F16
                             : Entry::SdpaMaskedM32D256F16};
  detail::u32(p, 0, a.dst);
  detail::u32(p, 2, a.query);
  detail::u32(p, 4,
              static_cast<std::uint32_t>(a.key_ddr_bytes >> 8));
  detail::u32(p, 6,
              static_cast<std::uint32_t>(a.value_ddr_bytes >> 8));
  detail::u32(p, 8, a.mask);
  p.regs[10] = static_cast<std::uint16_t>(a.kv_length);
  p.register_count = 11; p.grid[0] = 1;
  p.cores = causal_attention::cores;
  return p;
}
}  // namespace xinfer::rpu::abi
