// Copyright 2026 X-Square Robot
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "common/rpu/kernel_abi.hpp"
#include <array>

namespace xinfer::rpu::fusion {

struct ConvPrefillPlan {
  const char* name = "conv_silu_prefill_f16";
  std::array<std::uint16_t, 64> regs{};
  unsigned register_count = 25;
  std::array<std::uint16_t, 3> grid{1, 3, 8};
  unsigned cores = 8;
};

// LocalSPM byte offsets; DDR addresses are byte addresses and are converted
// to the launch ABI address unit (256 bytes) only here.
struct ConvPrefillArgs {
  std::array<std::uint32_t, 3> pad;
  std::uint32_t history;
  std::array<std::uint32_t, 3> out;
  std::uint32_t history_out;
  std::array<std::uint64_t, 3> weight_ddr;
  unsigned rows, valid_rows;
  bool initial;
};

inline ConvPrefillPlan conv_silu_prefill(const ConvPrefillArgs& a) {
  using namespace xinfer::rpu::abi::detail;
  require(a.rows >= 64 && a.rows <= 512 && a.rows % 64 == 0,
          "conv fusion requires a 64..512 row bucket divisible by 64");
  require(a.valid_rows >= 1 && a.valid_rows <= a.rows,
          "conv fusion valid_rows must be in [1,rows]");
  const std::uint64_t pad_bytes = 2ULL * (a.rows + 3) * 256;
  const std::uint64_t out_bytes = 2ULL * a.rows * 256;
  constexpr std::uint64_t state_bytes = 2ULL * 3 * 4 * 256;
  const std::array<std::uint32_t, 4> inputs{
      a.pad[0], a.pad[1], a.pad[2], a.history};
  const std::array<std::uint64_t, 4> input_sizes{
      pad_bytes, pad_bytes, pad_bytes, state_bytes};
  const std::array<std::uint32_t, 4> outputs{
      a.out[0], a.out[1], a.out[2], a.history_out};
  const std::array<std::uint64_t, 4> output_sizes{
      out_bytes, out_bytes, out_bytes, state_bytes};
  for (unsigned i = 0; i < 4; ++i) {
    span(inputs[i], input_sizes[i]);
    span(outputs[i], output_sizes[i]);
    for (unsigned j = 0; j < 4; ++j)
      disjoint(outputs[i], output_sizes[i], inputs[j], input_sizes[j]);
    for (unsigned j = 0; j < i; ++j)
      disjoint(outputs[i], output_sizes[i], outputs[j], output_sizes[j]);
  }
  ConvPrefillPlan p;
  auto put = [&p](unsigned reg, std::uint32_t value) {
    p.regs[reg] = static_cast<std::uint16_t>(value);
    p.regs[reg + 1] = static_cast<std::uint16_t>(value >> 16);
  };
  for (unsigned i = 0; i < 3; ++i) {
    put(2 * i, a.pad[i]);
    put(8 + 2 * i, a.out[i]);
    ddr_span(a.weight_ddr[i], 8ULL * 4 * 256 * 2);
    put(16 + 2 * i, static_cast<std::uint32_t>(a.weight_ddr[i] >> 8));
  }
  put(6, a.history);
  put(14, a.history_out);
  p.regs[22] = static_cast<std::uint16_t>(a.rows);
  p.regs[23] = static_cast<std::uint16_t>(a.valid_rows);
  p.regs[24] = a.initial ? 1 : 0;
  return p;
}

} // namespace xinfer::rpu::fusion
