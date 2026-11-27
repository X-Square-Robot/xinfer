// Copyright 2026 X-Square Robot
// SPDX-License-Identifier: Apache-2.0
// Validate and lower normalization requests to native launch plans.
#include "common/rpu/tp_lowering.hpp"
#include "common/rpu/tp_matrix_validation.hpp"
#include "common/rpu/tp_state_validation.hpp"

namespace xinfer::rpu::tp::detail {
namespace {
bool lower_matrix_validation(const Request& r, Lowered& out) {
  using namespace matrix_validation;
  const auto& op = r.op;
  if (op == "layernorm_f16") {
    const auto& x = r.tensor("input");
    const auto& y = r.tensor("out");
    require(x.shape.size() == 2, "LayerNorm input must be a matrix");
    fp16(x); layout(x, "row_major"); same_role(y, x);
    fp16(y); layout(y, "row_major");
    require(y.shape == x.shape, "LayerNorm output shape mismatch");
    const auto rows = dim(x.shape[0], "rows"), cols = dim(x.shape[1], "cols");
    const auto& w = local(r, "weight", {cols}); per_core(w);
    replicated_parameter(x, w, "weight");
    const bool has_bias = r.has_tensor("bias");
    std::uint32_t bias_address = 0;
    if (has_bias) {
      const auto& b = local(r, "bias", {cols}); per_core(b);
      replicated_parameter(x, b, "bias"); bias_address = spm(b);
    }
    const auto epsilon = scalar(r, "epsilon", 1.0e-5);
    require(epsilon > 0 && epsilon <= 65504.0F,
            "LayerNorm epsilon must be positive finite FP16");
    out.plan = abi::layernorm({spm(y), spm(x), spm(w), bias_address,
                               rows, cols, has_bias, epsilon});
    out.plan.cores = 8;
    finish(r, out, {"out"}); return true;
  }
  if (op == "rmsnorm_f16" || op == "rmsnorm_gdn_f16" ||
      op == "rmsnorm_coeff_h2048_f16") {
    const auto& x = r.tensor("x");
    require(x.shape.size() == 2, "RMSNorm requires [rows,cols]");
    const auto rows = dim(x.shape[0], "rows"), cols = dim(x.shape[1], "cols");
    local(r, "x", {rows, cols});
    const auto& y = local(r, "out", {rows, cols}); same_role(y, x);
    const auto& w = local(r, "weight", {cols});
    require(w.distribution == "local" || w.distribution == "replicated", "RMSNorm weight must be local or replicated");
    require(w.shard_axis == -1, "RMSNorm weight must be unsharded");
    replicated_parameter(x, w, "weight");
    if (x.distribution == "sharded") require(x.shard_axis == 0, "RMSNorm cannot normalize a column shard");
    const abi::RMSNormF16Args args{spm(y), spm(x), spm(w), rows, cols, scalar(r, "epsilon", 1.0e-6)};
    if (op == "rmsnorm_gdn_f16") {
      r.require_values("x", -1.0, 1.0);
      out.plan = abi::rmsnorm_gdn_f16(args);
    } else if (op == "rmsnorm_coeff_h2048_f16")
      out.plan = abi::rmsnorm_coeff_h2048_f16(args);
    else out.plan = abi::rmsnorm_f16(args);
    finish(r, out, {"out"}); return true;
  }

  if (op == "adaptive_rmsnorm_m16h1024_f16" || op == "adaptive_rmsnorm_m32h1024_f16") {
    const unsigned rows = op == "adaptive_rmsnorm_m16h1024_f16" ? 16 : 32;
    const auto& y = local(r, "out", {rows, 1024});
    const auto& x = local(r, "x", {rows, 1024}); same_role(y, x);
    require(x.distribution != "sharded" || x.shard_axis == 0,
            "adaptive RMSNorm cannot normalize a column shard");
    const auto& mod = local(r, "modulation", {3, 1024});
    require(mod.distribution == "replicated" || mod.distribution == "local", "modulation must be local or replicated");
    require(mod.shard_axis == -1, "modulation must be unsharded");
    replicated_parameter(x, mod, "modulation");
    const abi::AdaptiveRmsnormArgs args{spm(y), spm(x), spm(mod),
        scalar(r, "epsilon", 1.0e-6), rows, scalar(r, "normalization_scale", 1.0)};
    // Plain adaptive RMSNorm uses hardware sqrt_reciprocal + valuRepeat.
    out.plan = abi::adaptive_rmsnorm(args);
    finish(r, out, {"out"}); return true;
  }

  return false;
}
bool lower_state_validation(const Request& r, Lowered& result) {
  using namespace state_validation;
  const auto& op = r.op;
  if (op == "l2norm_f16") {
    const auto& out = data(r, "out");
    const auto& input = data(r, "input");
    local_pair(out, input);
    need(input.distribution != "partial", "L2Norm requires a resolved tensor, not a partial sum");
    const auto epsilon = r.scalar("epsilon", 1.0e-6);
    need(std::isfinite(epsilon), "epsilon must be finite");
    result.plan = abi::l2norm_f16({local(out), local(input), rows(input, 128, "input"),
                                       128, static_cast<float>(epsilon)});
    finish_state(r, result); return true;
  }
  return false;
}
}  // namespace

bool lower_normalization(const Request& r, Lowered& out) {
  return lower_matrix_validation(r, out) ||
         lower_state_validation(r, out);
}
}  // namespace xinfer::rpu::tp::detail
