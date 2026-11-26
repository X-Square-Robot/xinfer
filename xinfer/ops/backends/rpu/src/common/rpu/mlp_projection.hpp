// Copyright 2026 X-Square Robot
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "common/rpu/kernel_abi.hpp"

namespace xinfer::rpu::abi {
struct MlpProjectionShape { unsigned n, k, max_rows, tile_n; };
inline MlpProjectionShape mlp_projection_shape(Entry e) {
  switch (e) {
    case Entry::PatchEmbed: return {128, 1536, 868, 48};
    case Entry::Fc1N512K4096: return {512, 4096, 147, 16};
    case Entry::DownN2048K512: return {2048, 512, 147, 96};
    default: throw std::invalid_argument("expected patch embedding or merger projection");
  }
}
struct MlpProjectionArgs {
  Entry entry;
  std::uint32_t dst, src, bias, rows;
  std::uint64_t weight_ddr_bytes;
};
// A[M,K], C[M,N], bias[N]: compact FP16 LocalSPM, equal offsets on eight cores.
// W: [N/16][K/16][8][16 N][16 K] FP16 DDR, addressed in 256-byte units.
// FC2 returns unreduced partials. Caller supplies zero bias on cores 1..7.
// The complete plan.grid must be broadcast independently to all eight cores.
inline LaunchPlan mlp_projection(const MlpProjectionArgs& a) {
  using namespace detail;
  const auto s = mlp_projection_shape(a.entry);
  require(a.rows > 0 && a.rows <= s.max_rows, "unsupported vision projection rows");
  const auto ab = 2ULL*a.rows*s.k, cb = 2ULL*a.rows*s.n, bb = 2ULL*s.n;
  require(ab+cb+bb <= working_bytes, "vision projection exceeds SPM working budget");
  span(a.src, ab); span(a.dst, cb); span(a.bias, bb);
  disjoint(a.dst, cb, a.src, ab); disjoint(a.dst, cb, a.bias, bb);
  disjoint(a.src, ab, a.bias, bb);
  const auto wb = 2ULL*8*s.n*s.k;
  require(a.weight_ddr_bytes % 256 == 0 && a.weight_ddr_bytes < (1ULL<<40) &&
          wb <= (1ULL<<40)-a.weight_ddr_bytes,
          "vision projection needs an aligned DDR weight range below 1 TiB");
  LaunchPlan p{a.entry};
  u32(p, 0, a.src); u32(p, 2, static_cast<std::uint32_t>(a.weight_ddr_bytes>>8));
  const unsigned out_reg = 4;
  u32(p, out_reg, a.dst); u32(p, out_reg+2, a.bias);
  p.regs[out_reg+4]=a.rows;
  p.register_count=out_reg+5; p.cores=8; p.grid[0]=(s.n+s.tile_n-1)/s.tile_n;
  return p;
}
} // namespace xinfer::rpu::abi
