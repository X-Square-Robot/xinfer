// Copyright 2026 X-Square Robot
// SPDX-License-Identifier: Apache-2.0
// Validate and lower conv requests to native launch plans.
#include "common/rpu/tp_lowering.hpp"
#include "common/rpu/tp_state_validation.hpp"

namespace xinfer::rpu::tp::detail {
namespace {
bool lower_state_validation(const Request& r, Lowered& result) {
  using namespace state_validation;
  const auto& op = r.op;
  if (op == "causal_depthwise_conv1d_prefill_f16") {
    const auto& pad = data(r, "pad");
    const auto& carry = data(r, "carry");
    const auto& weight = data(r, "weight", "ddr");
    const auto padded_rows = rows(pad, 256, "pad");
    need(padded_rows >= 5 && padded_rows <= 515, "conv pad needs rows+3, rows in [2,512]");
    const auto m = padded_rows - 3;
    expect_shape(carry, {3, 256}, "carry"); expect_shape(weight, {8, 4, 256}, "weight");
    conv_output(pad, 1, "pad");
    conv_dependency(carry, pad, 1, "carry", false);
    std::uint32_t halo = 0;
    if (m >= 32) {
      const auto& h = data(r, "halo");
      expect_shape(h, {8, 3, 256}, "halo"); halo = local(h);
      conv_dependency(h, pad, 2, "halo");
    }
    need(carry.state_kind == "history", "carry requires a history owner");
    result.plan = abi::causal_depthwise_conv1d_prefill({local(pad), local(carry), halo,
                                                     weight.address, m});
    result.state_accesses = {{"carry", 0, 1, true}};
    finish_state(r, result); return true;
  }
  if (op == "causal_depthwise_conv1d_step_f16") {
    const auto& state = data(r, "state");
    const auto& input = data(r, "input");
    const auto& weight = data(r, "weight", "ddr");
    expect_shape(state, {4, 256}, "state"); expect_shape(input, {256}, "input");
    expect_shape(weight, {8, 4, 256}, "weight");
    conv_output(state, 1, "state");
    conv_dependency(input, state, 0, "input");
    need(state.state_kind == "history", "conv state requires a history owner");
    result.plan = abi::causal_depthwise_conv1d_step({local(state), local(input), weight.address});
    result.state_accesses = {{"state", 0, 1, false}, {"state", 0, 1, true}};
    finish_state(r, result); return true;
  }
  if (op == "conv_silu_prefill_f16") {
    fusion::ConvPrefillArgs args{};
    const auto& history = data(r, "history");
    const auto& history_out = data(r, "history_out");
    expect_shape(history, {3, 4, 256}, "history"); expect_shape(history_out, {3, 4, 256}, "history_out");
    need(history.state_kind == "history" && history_out.state_kind == "history",
            "fused conv history tensors require history owners");
    args.history = local(history); args.history_out = local(history_out);
    const std::array<std::string, 3> paths{"q", "k", "v"};
    unsigned m = 0;
    const auto& qout = data(r, "qout");
    conv_output(qout, 1, "qout");
    args.initial = bool(integer(r, "initial", -1, 1));
    if (!args.initial) conv_dependency(history, qout, 2, "history");
    bool replicated_pads = true;
    for (unsigned i = 0; i < paths.size(); ++i) {
      const auto& pad = data(r, paths[i] + "pad");
      const auto& out = data(r, paths[i] + "out");
      const auto& weight = data(r, paths[i] + "weight", "ddr");
      const auto n = rows(out, 256, paths[i] + "out");
      if (i == 0) m = n;
      need(n == m, "fused conv Q/K/V row counts differ");
      expect_shape(pad, {m + 3, 256}, paths[i] + "pad");
      expect_shape(weight, {8, 4, 256}, paths[i] + "weight");
      conv_dependency(out, qout, 1, paths[i] + "out", false);
      conv_dependency(pad, qout, 1, paths[i] + "pad");
      replicated_pads = replicated_pads && pad.distribution == "replicated";
      args.pad[i] = local(pad); args.out[i] = local(out); args.weight_ddr[i] = weight.address;
    }
    args.rows = m; args.valid_rows = integer(r, "valid_rows", -1, 512);
    if (history_out.distribution == "replicated") {
      need(history_out.shard_axis == -1 && replicated_pads &&
               (args.initial || args.valid_rows >= 3 || history.distribution == "replicated"),
           "history_out can be replicated only when pads and any retained history are replicated");
    } else {
      conv_dependency(history_out, qout, 2, "history_out", false);
    }
    const auto p = fusion::conv_silu_prefill(args);
    result.plan = abi::LaunchPlan{abi::Entry::ConvSiluPrefillFusion};
    result.plan.regs = p.regs; result.plan.register_count = p.register_count;
    result.plan.grid = p.grid; result.plan.cores = p.cores;
    if (!args.initial) result.state_accesses.push_back({"history", 0, 1, false});
    result.state_accesses.push_back({"history_out", 0, 1, true});
    finish_state(r, result); return true;
  }
  return false;
}
}  // namespace

bool lower_conv(const Request& r, Lowered& out) {
  return lower_state_validation(r, out);
}
}  // namespace xinfer::rpu::tp::detail
