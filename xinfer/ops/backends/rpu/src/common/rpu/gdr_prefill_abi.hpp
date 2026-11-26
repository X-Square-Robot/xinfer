// Copyright 2026 X-Square Robot
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "common/rpu/kernel_abi.hpp"
#include <array>
#include <cstdint>
#include <string>

namespace xinfer::rpu::fusion {
// Scratch is private per (head, 64-row chunk): two heads per core.
inline constexpr std::uint64_t gdr_scratch_bytes_per_chunk = 2ULL * 2 * 49408;
inline constexpr std::uint64_t gdr_scratch_bytes_for_rows(unsigned rows) {
  return gdr_scratch_bytes_per_chunk * (rows / 64);
}
inline constexpr std::uint64_t gdr_state_bytes = 2ULL * 2 * 128 * 128;
struct GdrPrefillArgs {
  std::uint32_t q_in, k_in, v_in, z, a, b, dt, negA, norm;
  std::uint32_t state, output, scratch;
  unsigned rows, valid_rows;
  bool initial;
  std::uint32_t raw_out = 0, norm_out = 0;
  bool debug = false;
};
struct GdrPrefillPlan {
  std::string name = "gdr_prefill_f16";
  std::array<std::uint16_t, 64> regs{};
  unsigned register_count = 33;
  std::array<std::uint16_t, 3> grid{8, 1, 1};
  std::array<unsigned,2> zero_counters{0,1};
  unsigned cores = 8;
};
inline GdrPrefillPlan gdr_prefill(const GdrPrefillArgs& a) {
  using namespace xinfer::rpu::abi::detail;
  require(a.rows >= 64 && a.rows <= 512 && a.rows % 64 == 0,
          "GDR prefill requires an exact 64..512-row bucket");
  const std::uint64_t gdr_scratch_bytes = gdr_scratch_bytes_for_rows(a.rows);
  require(a.valid_rows >= 1 && a.valid_rows <= a.rows,
          "GDR prefill valid_rows must be in [1,rows]");
  const std::uint64_t matrix = 2ULL * a.rows * 256;
  const std::uint64_t gates = 2ULL * a.rows * 16;
  const std::array<std::uint32_t,9> input{a.q_in,a.k_in,a.v_in,a.z,a.a,a.b,a.dt,a.negA,a.norm};
  const std::array<std::uint64_t,9> size{matrix,matrix,matrix,matrix,gates,gates,32,32,256};
  for(unsigned i=0;i<input.size();++i) {
    span(input[i],size[i]);
    disjoint(a.state,gdr_state_bytes,input[i],size[i]);
    disjoint(a.scratch,gdr_scratch_bytes,input[i],size[i]);
    disjoint(a.output,matrix,input[i],size[i]);
  }
  span(a.state,gdr_state_bytes); span(a.scratch,gdr_scratch_bytes); span(a.output,matrix);
  disjoint(a.state,gdr_state_bytes,a.scratch,gdr_scratch_bytes);
  disjoint(a.state,gdr_state_bytes,a.output,matrix);
  disjoint(a.scratch,gdr_scratch_bytes,a.output,matrix);
  if(a.debug) {
    span(a.raw_out,matrix); span(a.norm_out,matrix);
    for(unsigned i=0;i<input.size();++i) {
      disjoint(a.raw_out,matrix,input[i],size[i]);
      disjoint(a.norm_out,matrix,input[i],size[i]);
    }
    disjoint(a.raw_out,matrix,a.norm_out,matrix);
    disjoint(a.raw_out,matrix,a.output,matrix); disjoint(a.norm_out,matrix,a.output,matrix);
    disjoint(a.raw_out,matrix,a.state,gdr_state_bytes); disjoint(a.norm_out,matrix,a.state,gdr_state_bytes);
    disjoint(a.raw_out,matrix,a.scratch,gdr_scratch_bytes); disjoint(a.norm_out,matrix,a.scratch,gdr_scratch_bytes);
  }
  GdrPrefillPlan p;
  const auto put=[&](unsigned reg,std::uint32_t value) {
    p.regs[reg]=static_cast<std::uint16_t>(value);
    p.regs[reg+1]=static_cast<std::uint16_t>(value>>16);
  };
  for(unsigned i=0;i<input.size();++i) put(i*2,input[i]);
  put(18,a.state); put(20,a.output); put(22,a.scratch);
  p.regs[24]=a.rows; p.regs[25]=a.valid_rows; p.regs[26]=a.initial;
  put(28,a.debug ? a.raw_out : 0); put(30,a.debug ? a.norm_out : 0);
  // The public launch contract places debug at scalar slot27. The final
  // padding remains zero; keep this position stable when updating the ABI.
  p.regs[27]=a.debug;
  p.regs[32]=a.debug;
  return p;
}
} // namespace xinfer::rpu::fusion
