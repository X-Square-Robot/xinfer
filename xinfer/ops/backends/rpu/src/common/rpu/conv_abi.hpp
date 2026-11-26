// Copyright 2026 X-Square Robot
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "common/rpu/kernel_abi.hpp"

namespace xinfer::rpu::abi {

// Offsets are local SPM byte offsets shared by a TP8 broadcast. The weight
// address is a byte address in DDR, not the 256-byte register value.
struct CausalDepthwiseConvPrefillArgs {
  std::uint32_t pad, carry, halo;
  std::uint64_t weight_ddr_bytes;
  unsigned rows;
};

inline void causal_depthwise_conv_weights(std::uint64_t address) {
  detail::require(address != 0 && address % 256 == 0 && address < (1ULL << 40) &&
                      16384ULL <= (1ULL << 40) - address,
                  "causal depthwise conv weights require 16 KiB of aligned DDR");
}

inline LaunchPlan causal_depthwise_conv1d_prefill(
    const CausalDepthwiseConvPrefillArgs& a) {
  detail::require(a.rows >= 2 && a.rows <= 512,
                  "causal depthwise conv prefill rows must be 2..512");
  const unsigned segments = a.rows >= 32 ? 8 : 1;
  const std::uint64_t pad_bytes = 2ULL * (3 + a.rows) * 256;
  constexpr std::uint64_t carry_bytes = 2ULL * 3 * 256;
  const std::uint64_t halo_bytes = 2ULL * segments * 3 * 256;
  detail::span(a.pad, pad_bytes);
  detail::span(a.carry, carry_bytes);
  detail::disjoint(a.pad, pad_bytes, a.carry, carry_bytes);
  if (segments == 8) {
    detail::span(a.halo, halo_bytes);
    detail::disjoint(a.pad, pad_bytes, a.halo, halo_bytes);
    detail::disjoint(a.carry, carry_bytes, a.halo, halo_bytes);
  }
  causal_depthwise_conv_weights(a.weight_ddr_bytes);
  LaunchPlan p{Entry::CausalDepthwiseConvPrefill};
  detail::u32(p, 0, a.pad);
  detail::u32(p, 2, a.carry);
  detail::u32(p, 4, segments == 8 ? a.halo : 0);
  detail::u32(p, 6, static_cast<std::uint32_t>(a.weight_ddr_bytes >> 8));
  p.regs[8] = static_cast<std::uint16_t>(a.rows);
  p.regs[9] = static_cast<std::uint16_t>(segments);
  p.register_count = 10;
  p.grid = {1, 1, static_cast<std::uint16_t>(segments)};
  p.cores = 8;
  return p;
}

struct CausalDepthwiseConvStepArgs {
  std::uint32_t state, input;
  std::uint64_t weight_ddr_bytes;
};

inline LaunchPlan causal_depthwise_conv1d_step(
    const CausalDepthwiseConvStepArgs& a) {
  constexpr std::uint64_t state_bytes = 2ULL * 4 * 256;
  constexpr std::uint64_t input_bytes = 2ULL * 256;
  detail::span(a.state, state_bytes);
  detail::span(a.input, input_bytes);
  detail::disjoint(a.state, state_bytes, a.input, input_bytes);
  causal_depthwise_conv_weights(a.weight_ddr_bytes);
  LaunchPlan p{Entry::CausalDepthwiseConvStep};
  detail::u32(p, 0, a.state);
  detail::u32(p, 2, a.input);
  detail::u32(p, 4, static_cast<std::uint32_t>(a.weight_ddr_bytes >> 8));
  p.register_count = 6;
  p.cores = 8;
  return p;
}

}  // namespace xinfer::rpu::abi
