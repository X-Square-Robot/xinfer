// Copyright 2026 X-Square Robot
// SPDX-License-Identifier: Apache-2.0
#include "common/rpu/tp_lowering.hpp"

namespace xinfer::rpu::tp {
bool lower_operator(const Request& r, Lowered& out) {
  return detail::lower_activation(r, out) ||
         detail::lower_attention(r, out) ||
         detail::lower_comm(r, out) ||
         detail::lower_conv(r, out) ||
         detail::lower_denoise_output(r, out) ||
         detail::lower_elementwise(r, out) ||
         detail::lower_gated_delta_rule(r, out) ||
         detail::lower_gemm(r, out) ||
         detail::lower_layout(r, out) ||
         detail::lower_normalization(r, out);
}
}  // namespace xinfer::rpu::tp
