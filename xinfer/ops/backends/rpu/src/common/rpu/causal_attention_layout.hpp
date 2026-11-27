// Copyright 2026 X-Square Robot
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>

namespace xinfer::rpu::causal_attention {
inline constexpr unsigned cores = 8, dim = 256, capacity = 780;

inline constexpr std::size_t cache_elements() {
  return std::size_t((capacity + 15) / 16) * 16 * cores * 16 * 16;
}

}  // namespace xinfer::rpu::causal_attention
