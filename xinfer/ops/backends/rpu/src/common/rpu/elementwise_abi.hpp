// Copyright 2026 X-Square Robot
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "common/rpu/kernel_abi.hpp"

namespace xinfer::rpu::abi {
struct AddVpbArgs {
  std::uint32_t dst, x, position, elements;
  unsigned vectors_per_block = 32;
};
inline LaunchPlan add_vpb(const AddVpbArgs& a) {
  using namespace detail;
  require(a.elements > 0 && a.elements <= 868U * 1024, "invalid vision ADD elements");
  require(a.vectors_per_block > 0 && a.vectors_per_block <= 64, "invalid ADD tile");
  const std::uint64_t bytes = 2ULL * a.elements;
  span(a.dst, bytes); span(a.x, bytes); span(a.position, bytes);
  elementwise_alias(a.dst, a.x, bytes); elementwise_alias(a.dst, a.position, bytes);
  LaunchPlan p{Entry::AddVpb};
  u32(p,0,a.dst); u32(p,2,a.x); u32(p,4,a.position); u32(p,6,a.elements);
  p.regs[8]=a.vectors_per_block;p.register_count=9;p.cores=8;
  p.grid[0]=(a.elements+256*a.vectors_per_block-1)/(256*a.vectors_per_block);
  return p;
}
struct FillArgs {
  std::uint32_t dst, elements;
  std::uint16_t value_bits = 0;
  unsigned vectors_per_block = 32;
};
inline LaunchPlan fill(const FillArgs& a) {
  using namespace detail;
  require(a.elements > 0 && 2ULL*a.elements <= working_bytes, "invalid fill extent");
  require((a.value_bits & 0x7c00U) != 0x7c00U, "fill value must be finite FP16");
  require(a.vectors_per_block > 0 && a.vectors_per_block <= 64, "invalid fill tile");
  // GDN beta/g padding begins after Lv two-element rows and may be only
  // FP16-aligned. The device kernel uses the exact element start and masks
  // the final partial 16-lane store.
  require(a.dst % 2 == 0 && a.dst < spm_bytes &&
          2ULL * a.elements <= spm_bytes - a.dst,
          "fill requires a FP16-aligned local SPM range");
  LaunchPlan p{Entry::Fill};u32(p,0,a.dst);u32(p,2,a.elements);
  p.regs[4]=a.value_bits;p.regs[5]=a.vectors_per_block;p.register_count=6;p.cores=8;
  p.grid[0]=(a.elements+256*a.vectors_per_block-1)/(256*a.vectors_per_block);
  return p;
}
struct AllGatherArgs {
  std::uint32_t dst, src, rows;
  unsigned row_pairs_per_block = 64;
};
inline LaunchPlan all_gather(const AllGatherArgs& a) {
  using namespace detail;
  require(a.rows > 0 && a.rows <= 868, "invalid all-gather rows");
  require(a.row_pairs_per_block > 0 && a.row_pairs_per_block <= 64, "invalid gather tile");
  const std::uint64_t in=2ULL*a.rows*128,out=2ULL*a.rows*1024;
  span(a.src,in);span(a.dst,out);disjoint(a.dst,out,a.src,in);
  LaunchPlan p{Entry::AllGather};u32(p,0,global_spm_base+a.dst);u32(p,2,a.src);
  p.regs[4]=a.rows;p.regs[5]=core_stride_bytes>>16;p.regs[6]=a.row_pairs_per_block;
  p.register_count=7;p.cores=8;
  p.grid[0]=(a.rows+2*a.row_pairs_per_block-1)/(2*a.row_pairs_per_block);
  return p;
}
} // namespace xinfer::rpu::abi
