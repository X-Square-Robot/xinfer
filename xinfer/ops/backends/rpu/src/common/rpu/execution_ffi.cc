// Copyright 2026 X-Square Robot
// SPDX-License-Identifier: Apache-2.0
#include "common/rpu/execution.hpp"
#include "common/rpu/ffi_tensor.hpp"

namespace xinfer::rpu::ffi {
void upload(execution::Handle buffer, tvm::ffi::TensorView input) {
  const auto host = host_tensor(input, "input");
  execution::buffer_upload(buffer, host.data, host.rows, host.cols);
}
void download(tvm::ffi::TensorView output, execution::Handle buffer) {
  const auto host = host_tensor(output, "output");
  execution::buffer_download(host.data, buffer, host.rows, host.cols);
}
}  // namespace xinfer::rpu::ffi

TVM_FFI_DLL_EXPORT_TYPED_FUNC(rpu_context_create, xinfer::rpu::execution::context_create);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(rpu_release, xinfer::rpu::execution::release);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(rpu_buffer_empty, xinfer::rpu::execution::buffer_empty);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(rpu_buffer_upload, xinfer::rpu::ffi::upload);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(rpu_buffer_download, xinfer::rpu::ffi::download);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(rpu_execute, xinfer::rpu::execution::execute);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(rpu_wait, xinfer::rpu::execution::wait);
