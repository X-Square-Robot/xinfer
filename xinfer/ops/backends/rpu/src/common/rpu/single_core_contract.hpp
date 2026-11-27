// Copyright 2026 X-Square Robot
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace xinfer::rpu::single_core {

inline bool supported_gemm_weight_shape(std::size_t n, std::size_t k) noexcept {
  return (n == 1024 && k == 64) || (n == 64 && k == 1024);
}

inline bool supported_gemm_shape(std::size_t m, std::size_t n,
                                 std::size_t k) noexcept {
  return (m == 16 || m == 32) && supported_gemm_weight_shape(n, k);
}

// Preparation-only conversion from ordinary contiguous FP16 W[N,K]. The
// single-core Action I/O kernels consume [N/16][K/16][16N][16K] in DDR.
// This copies bit patterns without arithmetic or implicit transposition. The
// caller owns at least N*K input elements and retains the returned allocation
// (or its device copy) across executions; never call this in the launch path.
inline std::vector<std::uint16_t> pack_action_io_weight(
    const std::uint16_t* weight, std::size_t n, std::size_t k) {
  if (!supported_gemm_weight_shape(n, k)) {
    throw std::invalid_argument(
        "weight shape must be [1024,64] or [64,1024] for single-core FP16 GEMM");
  }
  if (weight == nullptr) {
    throw std::invalid_argument("weight must point to N*K FP16 elements");
  }
  std::vector<std::uint16_t> packed(n * k);
  for (std::size_t row = 0; row < n; ++row) {
    for (std::size_t col = 0; col < k; ++col) {
      const auto index =
          (((row / 16) * (k / 16) + col / 16) * 16 + row % 16) * 16 +
          col % 16;
      packed[index] = weight[row * k + col];
    }
  }
  return packed;
}

}  // namespace xinfer::rpu::single_core
