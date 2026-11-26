// Copyright 2026 X-Square Robot
// SPDX-License-Identifier: Apache-2.0
#include "common/rpu/kernel_abi.hpp"
#include "common/rpu/single_core_contract.hpp"

#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void check(bool ok, const char* message) {
  if (!ok) throw std::runtime_error(message);
}

template <class F> void rejects(F&& fn) {
  try {
    fn();
  } catch (const std::invalid_argument&) {
    return;
  }
  throw std::runtime_error("unsupported single-core contract accepted");
}

void test_layernorm_epsilon_abi() {
  using namespace xinfer::rpu;
  const auto plan = abi::layernorm({0x1000, 0, 0x2000, 0, 1, 16, false, 1e-5F});
  check(plan.regs[14] == 0x00a8, "epsilon must retain the declared FP16 ABI");
  const auto eps = fp16_to_fp32(plan.regs[14]);
  check(eps == 1.0013580322265625e-5F && eps != 1e-5F,
        "epsilon diagnostic must report its actual device value");
}

void test_weight_layout() {
  using namespace xinfer::rpu;
  for (const auto shape : {std::array<std::size_t, 2>{1024, 64}, {64, 1024}}) {
    const auto n = shape[0], k = shape[1];
    std::vector<std::uint16_t> weight(n * k);
    // Every 16-bit word appears once: detects missing, duplicate, transposed,
    // byte-swapped, and overlapping entries without relying on float accuracy.
    for (std::size_t i = 0; i < weight.size(); ++i)
      weight[i] = static_cast<std::uint16_t>(i);
    const auto original = weight;
    const auto packed = single_core::pack_action_io_weight(weight.data(), n, k);
    check(weight == original, "weight preparation mutated its source");
    std::size_t cursor = 0;
    for (std::size_t nb = 0; nb < n; nb += 16)
      for (std::size_t kb = 0; kb < k; kb += 16)
        for (std::size_t nlane = 0; nlane < 16; ++nlane)
          for (std::size_t klane = 0; klane < 16; ++klane)
            check(packed[cursor++] == weight[(nb + nlane) * k + kb + klane],
                  "DDR 16x16 tile traversal disagrees with W[N,K]");
    check(cursor == packed.size(), "packed weight has incorrect extent");

    // A one-hot input must select exactly one ordinary W column. Simulate the
    // documented DDR tile traversal independently of the production pack index.
    for (const std::size_t selected : {std::size_t{0}, std::size_t{15},
                                       std::size_t{16}, k - 1}) {
      std::vector<std::uint16_t> selected_column(n);
      cursor = 0;
      for (std::size_t nb = 0; nb < n; nb += 16)
        for (std::size_t kb = 0; kb < k; kb += 16)
          for (std::size_t nlane = 0; nlane < 16; ++nlane)
            for (std::size_t klane = 0; klane < 16; ++klane) {
              if (kb + klane == selected) selected_column[nb + nlane] = packed[cursor];
              ++cursor;
            }
      for (std::size_t row = 0; row < n; ++row)
        check(selected_column[row] == weight[row * k + selected],
              "one-hot GEMM selects the wrong weight column");
    }
    for (const auto m : {16U, 32U})
      check(single_core::supported_gemm_shape(m, n, k), "one of the four single-core GEMMs is missing");
  }
  check(!single_core::supported_gemm_shape(1, 1024, 64) &&
            !single_core::supported_gemm_shape(16, 64, 64),
        "unsupported GEMM shape accepted");
  rejects([] { single_core::pack_action_io_weight(nullptr, 1024, 64); });
  std::uint16_t dummy = 0;
  rejects([&] { single_core::pack_action_io_weight(&dummy, 64, 64); });
}

void test_silu_mul_abi() {
  using namespace xinfer::rpu;
  auto p = abi::silu_mul({0x200000, 0x00000, 0x100000, 768 * 384, 8});
  check(std::string(abi::name(p.entry)) == "silu_mul_fp16", "entry name");
  check(p.grid[0] == 5 && p.cores == 8 && p.register_count == 8,
        "MLP maximum launch geometry");
  check(abi::silu_mul({0x00000, 0x00000, 0x40000, 768 * 64, 8}).grid[0] == 1,
        "M64 launch geometry");
  check(abi::silu_mul({0x00000, 0x00000, 0x40000, 768, 8}).grid[0] == 1,
        "decode in-place geometry");
  check(abi::silu_mul({0x40000, 0x00000, 0x40000, 768, 1}).cores == 1,
        "output aliases second operand");
  rejects([] { abi::silu_mul({0, 0x2000, 0x4000, 0, 8}); });
  rejects([] { abi::silu_mul({0, 0x2000, 0x4000, 17, 8}); });
  rejects([] { abi::silu_mul({0, 0x2000, 0x4000, 272, 8}); });
  rejects([] { abi::silu_mul({0x20, 0, 0x4000, 768, 8}); });
  rejects([] { abi::silu_mul({0x80000, 0, 0x40000, 768, 9}); });
  rejects([] { abi::silu_mul({0, 0x20, 0x40000, 768, 8}); });
}
}  // namespace

int main() try {
  test_layernorm_epsilon_abi();
  test_weight_layout();
  test_silu_mul_abi();
  std::cout << "PASS: single-core epsilon ABI, GEMM packed layout, SiLU-mul ABI\n";
  return 0;
} catch (const std::exception& e) {
  std::cerr << e.what() << '\n';
  return 1;
}
