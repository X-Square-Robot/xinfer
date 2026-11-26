// Copyright 2026 X-Square Robot
// SPDX-License-Identifier: Apache-2.0
// XInfer library metadata is available without an SDK or device context.
#include "catalog_metadata.hpp"

#include <tvm/ffi/container/array.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/string.h>

#include <cstdint>
#include <string>

namespace xinfer::rpu::ffi {
std::int64_t abi_version() { return metadata::abi_version; }
std::string catalog_sha256() { return metadata::catalog_sha256; }
std::string compiled_backends() { return "rpu"; }
bool runtime_available() {
#if XINFER_RPU_RUNTIME
  return true;
#else
  return false;
#endif
}
tvm::ffi::Array<tvm::ffi::String> kernel_names() {
  tvm::ffi::Array<tvm::ffi::String> result;
  for (const auto& asset : metadata::assets) result.push_back(tvm::ffi::String(asset.name));
  return result;
}
}  // namespace xinfer::rpu::ffi

TVM_FFI_DLL_EXPORT_TYPED_FUNC(rpu_abi_version, xinfer::rpu::ffi::abi_version);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(rpu_catalog_sha256, xinfer::rpu::ffi::catalog_sha256);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(compiled_backends, xinfer::rpu::ffi::compiled_backends);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(rpu_runtime_available, xinfer::rpu::ffi::runtime_available);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(rpu_kernel_names, xinfer::rpu::ffi::kernel_names);
