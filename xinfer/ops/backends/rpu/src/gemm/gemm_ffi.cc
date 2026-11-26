// Copyright 2026 X-Square Robot
// SPDX-License-Identifier: Apache-2.0
#include "common/rpu/execution.hpp"
#include "common/rpu/ffi_tensor.hpp"

namespace xinfer::rpu::ffi {
execution::Handle prepare_weight(execution::Handle context, tvm::ffi::TensorView weight) {
  const auto host = host_tensor(weight, "weight", true);
  return execution::prepare_weight(context, host.data, host.rows, host.cols);
}
}  // namespace xinfer::rpu::ffi
TVM_FFI_DLL_EXPORT_TYPED_FUNC(rpu_prepare_weight, xinfer::rpu::ffi::prepare_weight);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(rpu_prepare_gemm, xinfer::rpu::execution::prepare_gemm);
