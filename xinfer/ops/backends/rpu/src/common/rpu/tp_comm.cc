// Copyright 2026 X-Square Robot
// SPDX-License-Identifier: Apache-2.0
// Validate and lower comm requests to native launch plans.
#include "common/rpu/tp_lowering.hpp"
#include "common/rpu/tp_tensor_validation.hpp"
#include "common/rpu/tp_state_validation.hpp"

namespace xinfer::rpu::tp::detail {
namespace {
bool lower_tensor_validation(const Request& r, Lowered& out) {
  using namespace tensor_validation;
  const auto& op = r.op;
  if (op == "all_reduce_residual_layernorm") {
    const auto& partial = tensor(r, "partial");
    const auto m = rows(partial, "partial");
    shape(partial, {m, 1024}, "partial");
    distribution(partial, "partial", "partial");
    const auto& residual = matrix(r, "residual", m, 1024);
    // The fused producer retains only its balanced 256-element owner shard.
    // Both this kernel and standalone all-reduce read exactly that partition.
    if (residual.distribution == "sequence_shard")
      check_tensor(residual.shard_axis == 0, "Vision residual sequence shard requires axis 0");
    else distribution(residual, "replicated", "Vision residual");
    const auto& x = matrix(r, "x", m, 1024, "sequence_shard");
    check_tensor(x.shard_axis == 0, "sequence_shard x requires axis 0");
    const auto& norm = matrix(r, "norm", m, 1024, "replicated");
    const auto& gamma = vector(r, "gamma", 1024, "replicated");
    const auto& beta = vector(r, "beta", 1024, "replicated");
    // Fused whole-stage mode: the request may pin this launch to a
    // static launch-pool counter slot 3..50 (per-occurrence assignment matching the
    // batch runtime's descending 50..3 epochs; the vision merger re-arms
    // 50,49 after the tower). Omitted or 50 keeps the epoch-0 default, so
    // plain batch()/window callers see identical behavior.
    const auto counter = integer(r, "counter", 50);
    check_tensor(counter >= 3 && counter <= 50,
               "counter must be a counter pool id in 3..50 or omitted");
    const unsigned epoch = static_cast<unsigned>(50 - counter);
    auto p = abi::all_reduce_residual_layernorm({address(x), address(partial),
        address(residual), address(norm), address(gamma), address(beta), m,
        scalar(r, "epsilon", 1e-6), epoch});
    out.collective = true;
    out.zero_counters = {128, 130, 128 + 2 * static_cast<unsigned>(counter)};
    return done(r, out, p);
  }
  return false;
}
bool lower_state_validation(const Request& r, Lowered& result) {
  using namespace state_validation;
  const auto& op = r.op;
  if (op == "all_reduce_residual") {
    const auto& out = data(r, "out");
    const auto& input = data(r, "input");
    need(input.shape.size() == 2, "all_reduce input must be a matrix");
    same_expect_shape(out, input, "all_reduce out/input");
    expect_distribution(input, "partial", "all_reduce input");
    expect_distribution(out, "replicated", "all_reduce out");
    const auto with_residual = integer(r, "with_residual", 0, 1);
    std::uint32_t residual = 0;
    if (with_residual) {
      const auto& t = data(r, "residual");
      same_expect_shape(out, t, "all_reduce residual");
      if (t.distribution == "sequence_shard") {
        need(t.shard_axis == 0 && input.shape[1] == 1024 &&
                 input.shape[0] >= 128 && input.shape[0] <= 868 && input.shape[0] % 4 == 0,
             "Vision all_reduce residual requires an axis-0 balanced vector shard");
      } else expect_distribution(t, "replicated", "all_reduce residual");
      residual = local(t);
    }
    need(input.shape[1] == 1024 || input.shape[1] == 2048,
            "all_reduce supports columns 1024 or 2048");
    // Fused whole-stage mode: the request may pin this launch to a
    // static launch-pool counter slot 3..50 (the vision tower ends at 3 and the
    // merger re-arms 50,49 after it). Omitted or 50 keeps the epoch-0
    // default, so plain batch()/window callers see identical behavior.
    const auto counter = integer(r, "counter", 50);
    need(counter >= 3 && counter <= 50,
         "counter must be a counter pool id in 3..50 or omitted");
    result.plan = abi::all_reduce_residual({local(out), local(input), residual,
        rows(input, static_cast<unsigned>(input.shape[1]), "input"), 0,
        static_cast<unsigned>(input.shape[1]), bool(with_residual),
        static_cast<unsigned>(50 - counter)});
    // These are 32-bit SDK register IDs, not logical counter indices.
    result.zero_counters = {128, 130, 128 + 2 * counter};
    result.collective = true;
    finish_state(r, result); return true;
  }
  if (op == "all_gather_f16") {
    const auto& out = data(r, "out");
    const auto& input = data(r, "input");
    const auto m = rows(input, 128, "input");
    expect_shape(out, {m, 1024}, "out");
    expect_distribution(input, "sharded", "all_gather input");
    need(input.shard_axis == 1, "all_gather input must be sharded along columns");
    expect_distribution(out, "replicated", "all_gather out");
    result.plan = abi::all_gather({local(out), local(input), m,
                                  integer(r, "row_pairs_per_block", 64, 64)});
    result.collective = true;
    finish_state(r, result); return true;
  }
  return false;
}
}  // namespace

bool lower_comm(const Request& r, Lowered& out) {
  return lower_tensor_validation(r, out) ||
         lower_state_validation(r, out);
}
}  // namespace xinfer::rpu::tp::detail
