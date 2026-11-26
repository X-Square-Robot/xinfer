// Copyright 2026 X-Square Robot
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "common/rpu/kernel_abi.hpp"
#include "common/rpu/attention_layout.hpp"
#include <algorithm>
#include <vector>

namespace xinfer::rpu::abi {
struct CacheInsertArgs {
  std::uint32_t src, rows, position, capacity;
  std::uint64_t cache_ddr_bytes;
  bool value = false;
};
inline void attention_ddr_range(std::uint64_t address, unsigned capacity) {
  const auto bytes = attention::cache_elements(capacity)*2;
  detail::require(address != 0 && address%256==0 && address < (1ULL<<40) && bytes <= (1ULL<<40)-address,
                  "cache DDR address must be aligned and within the 40-bit address range");
}
inline LaunchPlan cache_insert(const CacheInsertArgs& a) {
  detail::require(a.rows>0 && a.rows<=868 && a.position<=a.capacity && a.rows<=a.capacity-a.position,
                  "invalid cache insert position/length");
  attention_ddr_range(a.cache_ddr_bytes,a.capacity);
  detail::span(a.src,2ULL*a.rows*128);
  LaunchPlan p{a.value ? Entry::VcacheInsert : Entry::KcacheInsert};
  detail::u32(p,0,a.src); detail::u32(p,2,a.cache_ddr_bytes>>8);
  p.regs[4]=a.rows;p.regs[5]=a.position;p.register_count=6;p.cores=8;
  p.grid[0]=(a.rows+(a.value?a.position%16:0)+15)/16;
  return p;
}
struct AttentionCache { std::uint64_t key_ddr_bytes,value_ddr_bytes; unsigned capacity; };
struct FlashAttentionArgs {
  std::uint32_t dst,query;
  std::vector<unsigned> lengths;
  std::vector<AttentionCache> caches;
};
inline LaunchPlan flash_attention(const FlashAttentionArgs& a) {
  const auto rows=attention::packed_rows(a.lengths);
  detail::require(a.caches.size()==a.lengths.size(),"cache/image count mismatch");
  const auto bytes=2ULL*rows*128;
  detail::span(a.dst,bytes);detail::span(a.query,bytes);detail::disjoint(a.dst,bytes,a.query,bytes);
  LaunchPlan p{Entry::FlashAttentionVarlen};
  detail::u32(p,0,a.dst);detail::u32(p,2,a.query);
  for (unsigned i=0;i<a.caches.size();++i) {
    const auto& c=a.caches[i];detail::require(c.capacity>=a.lengths[i],"attention cache is too short");
    attention_ddr_range(c.key_ddr_bytes,c.capacity);attention_ddr_range(c.value_ddr_bytes,c.capacity);
    detail::u32(p,4+i*4,c.key_ddr_bytes>>8);detail::u32(p,6+i*4,c.value_ddr_bytes>>8);
    p.regs[16+i]=a.lengths[i];
  }
  p.register_count=19;p.cores=8;
  p.grid={static_cast<std::uint16_t>((*std::max_element(a.lengths.begin(),a.lengths.end())+15)/16),2,
          static_cast<std::uint16_t>(a.lengths.size())};
  return p;
}
} // namespace xinfer::rpu::abi
