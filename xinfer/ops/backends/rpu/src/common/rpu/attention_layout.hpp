// Copyright 2026 X-Square Robot
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace xinfer::rpu::attention {
constexpr unsigned cores = 8, heads = 2, dim = 64, local_width = heads * dim;
inline void require(bool ok, const char* why) { if (!ok) throw std::invalid_argument(why); }
inline std::size_t cache_elements(unsigned capacity) {
  require(capacity > 0 && capacity <= 4096, "cache capacity must be 1..4096");
  return std::size_t((capacity + 15) / 16) * 2 * 4 * 8 * 16 * 16;
}
inline unsigned packed_rows(const std::vector<unsigned>& lengths) {
  require(!lengths.empty() && lengths.size() <= 3, "attention requires 1..3 images");
  unsigned sum = 0;
  for (auto n : lengths) { require(n > 0 && n <= 308, "image length must be 1..308"); sum += n; }
  return sum;
}
}  // namespace xinfer::rpu::attention
