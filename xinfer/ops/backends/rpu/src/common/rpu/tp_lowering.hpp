// Copyright 2026 X-Square Robot
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "common/rpu/tp_contract.hpp"

// Private semantic dispatch used by the resident-tensor execution adapter.
namespace xinfer::rpu::tp::detail {
bool lower_activation(const Request&, Lowered&);
bool lower_attention(const Request&, Lowered&);
bool lower_comm(const Request&, Lowered&);
bool lower_conv(const Request&, Lowered&);
bool lower_denoise_output(const Request&, Lowered&);
bool lower_elementwise(const Request&, Lowered&);
bool lower_gated_delta_rule(const Request&, Lowered&);
bool lower_gemm(const Request&, Lowered&);
bool lower_layout(const Request&, Lowered&);
bool lower_normalization(const Request&, Lowered&);
}  // namespace xinfer::rpu::tp::detail
