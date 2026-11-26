// Copyright 2026 X-Square Robot
// SPDX-License-Identifier: Apache-2.0
// Validate and lower layout requests to native launch plans.
#include "common/rpu/tp_lowering.hpp"
#include "common/rpu/tp_state_validation.hpp"

namespace xinfer::rpu::tp::detail {
namespace {
bool lower_state_validation(const Request& r, Lowered& result) {
  using namespace state_validation;
  const auto& op = r.op;
  if (op == "qkv_expand_kv_f16") {
    const auto& out = data(r, "out");
    const auto& input = data(r, "input");
    const auto m = rows(input, 640, "input");
    expect_shape(out, {m, 1024}, "out");
    // Q/Qgate remain core-local while KV changes from four-core slices to
    // four-core replication. The mixed planes do not form one column shard.
    expect_distribution(input, "local", "qkv input");
    expect_distribution(out, "local", "qkv out");
    need(input.shard_axis == -1 && out.shard_axis == -1,
            "mixed QKV tensors must not declare a uniform shard axis");
    result.plan = abi::qkv_expand_kv({local(out), local(input), m});
    result.zero_counters = {128};
    result.collective = true;
    finish_state(r, result); return true;
  }
  if (op == "slice_last_f16") {
    const auto& out = data(r, "out");
    const auto& input = data(r, "input");
    need(input.shape.size() == 2 && out.shape.size() == 2 &&
                input.shape[0] == out.shape[0], "slice expects matrices with equal rows");
    need(out.distribution == input.distribution && out.shard_axis == input.shard_axis,
            "slice changes distribution");
    result.plan = abi::slice_last_f16({local(out), local(input),
        static_cast<unsigned>(input.shape[0]), static_cast<unsigned>(input.shape[1]),
        static_cast<unsigned>(out.shape[1]), integer(r, "offset", 0)});
    finish_state(r, result); return true;
  }
  if (op == "transpose_nbc_c128_f16" || op == "transpose_nbc_c1_f16") {
    const auto& out = data(r, "out");
    const auto& input = data(r, "input");
    const auto c = op == "transpose_nbc_c128_f16" ? 128 : 1;
    need(input.shape.size() == 3 && input.shape[2] == c,
            "transpose requires matching rank-3 input");
    expect_shape(out, {input.shape[1], input.shape[0], c}, "out");
    need(out.distribution == input.distribution, "transpose changes distribution");
    const auto output_axis = input.shard_axis == 0 ? 1 : input.shard_axis == 1 ? 0 : input.shard_axis;
    need(out.shard_axis == output_axis, "transpose output shard axis does not follow the axis swap");
    result.plan = abi::transpose_nbc_f16({local(out), local(input),
        static_cast<unsigned>(input.shape[0]), static_cast<unsigned>(input.shape[1]),
        static_cast<unsigned>(c)});
    finish_state(r, result); return true;
  }
  return false;
}
}  // namespace

bool lower_layout(const Request& r, Lowered& out) {
  return lower_state_validation(r, out);
}
}  // namespace xinfer::rpu::tp::detail
