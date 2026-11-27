// Copyright 2026 X-Square Robot
// SPDX-License-Identifier: Apache-2.0
// Validate and lower denoise output requests to native launch plans.
#include "common/rpu/tp_lowering.hpp"
#include "common/rpu/tp_tensor_validation.hpp"

namespace xinfer::rpu::tp::detail {
namespace {
bool lower_tensor_validation(const Request& r, Lowered& out) {
  using namespace tensor_validation;
  const auto& op = r.op;
  if (op == "residual_gate_m16h1024_f16" || op == "residual_gate_m32h1024_f16") {
    const unsigned m = op == "residual_gate_m16h1024_f16" ? 16 : 32;
    const auto& branch = matrix(r, "branch", m, 1024, "replicated");
    const auto& residual = matrix(r, "residual", m, 1024, "replicated");
    const auto& dst = matrix(r, "out", m, 1024, "replicated");
    const auto& gate = vector(r, "gate", 1024, "replicated");
    const abi::ResidualGateArgs gate_args{address(dst), address(branch),
        address(residual), address(gate), m};
    // Plain residual_gate uses hardware mul+add directly.
    return done(r, out, abi::residual_gate(gate_args));
  }
  if (op == "euler_update_m16d64_f16" || op == "euler_update_m32d64_f16") {
    const unsigned m = op == "euler_update_m16d64_f16" ? 16 : 32;
    const auto& action = matrix(r, "action", m, 64);
    const auto& velocity = matrix(r, "velocity", m, 64);
    const auto& padding = matrix(r, "padding_velocity", m, 64);
    const auto& mask = vector(r, "keep_mask", 64);
    same_distribution(action, velocity);
    same_distribution(action, padding);
    check_tensor(action.distribution != "partial", "Euler cannot consume TP partials");
    check_tensor(mask.distribution != "partial", "Euler keep_mask must be resolved, not partial");
    check_tensor(action.distribution != "replicated" || mask.distribution == "replicated",
               "Euler keep_mask must be replicated when action/velocity are replicated");
    const bool owned_action = action.distribution == "owner0" || action.distribution == "single";
    const bool owned_mask = mask.distribution == "owner0" || mask.distribution == "single";
    // Euler multiplies velocity by the mask on every core. A nonowned mask
    // needs a finite-value proof so nonowner 0*Inf/NaN cannot become NaN.
    if (owned_action && !owned_mask) r.require_values("keep_mask", -65504.0, 65504.0);
    auto p = abi::euler_update({address(action), address(velocity),
        address(mask), address(padding), scalar(r, "delta_t", 1.0), m});
    p.cores = 8;
    return done(r, out, p);
  }
  if (op == "down_allreduce_rmsnorm_m16n1024k256_acc32" ||
      op == "down_allreduce_rmsnorm_m32n1024k256_acc32") {
    const unsigned m = op == "down_allreduce_rmsnorm_m16n1024k256_acc32" ? 16 : 32;
    const auto& src = matrix(r, "src", m, 256, "sharded");
    const auto& partial = matrix(r, "partial", m, 1024, "partial");
    const auto& residual = matrix(r, "residual", m, 1024, "replicated");
    const auto& stream = matrix(r, "stream", m, 1024, "replicated");
    const auto& norm = matrix(r, "norm", m, 1024, "replicated");
    const auto& w = weight(r, 1024, 256);
    const auto rtc = integer(r, "rtc_rows", 0);
    const auto& gate = tensor(r, "gate_row", "ddr");
    const auto& next = tensor(r, "next_row", "ddr");
    const std::vector<std::int64_t> modulation_shape = rtc
        ? std::vector<std::int64_t>{2, 3, 1024} : std::vector<std::int64_t>{3, 1024};
    shape(gate, modulation_shape, "gate_row");
    shape(next, modulation_shape, "next_row");
    std::uint64_t next2 = 0;
    std::uint32_t w_out = 0;
    if (r.tensors.count("next2_row")) {
      const auto& row = tensor(r, "next2_row", "ddr");
      shape(row, modulation_shape, "next2_row");
      next2 = row.address;
      w_out = address(matrix(r, "w_out", m, 1024, "replicated"));
    }
    // Fused whole-stage mode: the request may pin this launch to a
    // static launch-pool counter slot 3..49 (per-occurrence
    // assignment). Omitted or 50 keeps the batch-assigned epoch default,
    // so plain batch()/window callers see identical behavior.
    const auto counter = integer(r, "counter", 50);
    check_tensor(counter == 50 || (counter >= 3 && counter <= 49),
               "counter must be a counter pool id in 3..49 or omitted");
    const denoise::DownAllReduceNormArgs args{address(src),
        address(partial), address(residual), address(stream), address(norm),
        w.address, gate.address, next.address, scalar(r, "epsilon", 1e-6),
        static_cast<unsigned>(counter), next2, w_out, m,
        scalar(r, "normalization_scale", 1.0), rtc};
    // Plain entry now lowers to the hardware kernel; the hw builder lays
    // out the ABI-padded prefix (slot 34 reserved for an unused LUT pointer)
    // that the calibrated window ABI requires.
    auto p = denoise::down_allreduce_rmsnorm_hw(args);
    out.collective = true;
    for (const auto id : p.zero_counters) out.zero_counters.push_back(128 + 2 * id);
    return done(r, out, convert(p, entry(p.name)));
  }
  return false;
}
}  // namespace

bool lower_denoise_output(const Request& r, Lowered& out) {
  return lower_tensor_validation(r, out);
}
}  // namespace xinfer::rpu::tp::detail
