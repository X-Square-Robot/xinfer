// Copyright 2026 X-Square Robot
// SPDX-License-Identifier: Apache-2.0
// Validate and lower activation requests to native launch plans.
#include "common/rpu/tp_lowering.hpp"
#include "common/rpu/tp_matrix_validation.hpp"
#include "common/rpu/tp_tensor_validation.hpp"

namespace xinfer::rpu::tp::detail {
namespace {
bool lower_matrix_validation(const Request& r, Lowered& out) {
  using namespace matrix_validation;
  const auto& op = r.op;
  if (op == "sigmoid_mul_m16d256_f16" || op == "sigmoid_mul_m32d256_f16") {
    const unsigned rows = op == "sigmoid_mul_m16d256_f16" ? 16 : 32;
    const auto& y = local(r, "out", {rows, 256});
    const auto& gate = local(r, "gate", {rows, 256});
    const auto& value = local(r, "value", {rows, 256});
    same_role(y, gate); same_role(y, value);
    const abi::SigmoidMulArgs args{spm(y), spm(gate), spm(value), rows};
    // Plain kernel uses the hardware sigmoid opcode for both m16 and m32.
    out.plan = abi::sigmoid_mul(args);
    finish(r, out, {"out"}); return true;
  }

  return false;
}
bool lower_tensor_validation(const Request& r, Lowered& out) {
  using namespace tensor_validation;
  const auto& op = r.op;
  if (op == "gelu_f16" || op == "unary_f16") {
    const auto& src = tensor(r, "src");
    const auto& dst = tensor(r, "out");
    shape(dst, src.shape, "out");
    same_distribution(dst, src, true);
    const auto count = u32(elements(src, "src"), "elements");
    const auto selection = integer(r, op == "gelu_f16" ? "mode" : "operation",
                                   op == "gelu_f16" ? 1 : 0);
    check_tensor(selection <= (op == "gelu_f16" ? 1U : 3U),
               "unsupported activation selector before enum conversion");
    const bool core0_owned = src.distribution == "owner0" || src.distribution == "single";
    check_tensor(!core0_owned || op == "gelu_f16" || selection == 2,
               "operation must preserve zero for owner0/single src/output; only SiLU or GELU is supported");
    abi::LaunchPlan p = op == "gelu_f16"
        ? abi::gelu({address(dst), address(src), count,
                     static_cast<abi::GeluMode>(selection)})
        : abi::unary_f16({address(dst), address(src), count,
                     static_cast<abi::UnaryOp>(selection)});
    // Independent local kernels can run on every member of the fixed TP8 group.
    p.cores = 8;
    return done(r, out, p);
  }
  return false;
}
}  // namespace

bool lower_activation(const Request& r, Lowered& out) {
  return lower_matrix_validation(r, out) ||
         lower_tensor_validation(r, out);
}
}  // namespace xinfer::rpu::tp::detail
