// Copyright 2026 X-Square Robot
// SPDX-License-Identifier: Apache-2.0
#include "common/rpu/tp_execution.hpp"
#include "common/rpu/ffi_tensor.hpp"
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/string.h>

namespace xinfer::rpu::ffi {
using tvm::ffi::Array;
using tvm::ffi::String;
template <typename T> std::vector<T> values(const Array<T>& input) {
  return std::vector<T>(input.begin(), input.end());
}
std::vector<std::string> strings(const Array<String>& input) {
  std::vector<std::string> result;
  for (const auto& s : input) result.emplace_back(s);
  return result;
}
execution::Handle tp_empty(execution::Handle ctx, Array<std::int64_t> dims,
    String dtype, String storage, String layout, String distribution,
    std::int64_t axis, String state) {
  return execution::tp_empty(ctx, values(dims), dtype, storage, layout, distribution, axis, state);
}
void tp_upload(execution::Handle id, tvm::ffi::TensorView input) {
  const auto host = tp_host(input); execution::tp_upload(id, host.data, host.shape, host.dtype);
}
void tp_download(tvm::ffi::TensorView output, execution::Handle id) {
  const auto host = tp_host(output); execution::tp_download(host.data, id, host.shape, host.dtype);
}
execution::Handle tp_weight(execution::Handle ctx, tvm::ffi::TensorView input, String layout) {
  const auto host = tp_host(input);
  if (host.dtype != "float16") throw std::invalid_argument("prepared weight requires float16");
  return execution::tp_weight(ctx, host.data, host.shape, layout);
}
execution::Handle tp_prepare(execution::Handle ctx, String op,
    Array<String> names, Array<std::int64_t> tensors, Array<String> ikeys,
    Array<std::int64_t> integers, Array<String> fkeys, Array<double> scalars) {
  return execution::tp_prepare(ctx, op, strings(names), values(tensors), strings(ikeys),
                               values(integers), strings(fkeys), values(scalars));
}
execution::Handle tp_batch(execution::Handle ctx, Array<std::int64_t> nodes) {
  return execution::tp_batch(ctx, values(nodes));
}
execution::Handle tp_batch_with_window(execution::Handle ctx, Array<std::int64_t> nodes,
                                     std::int64_t window) {
  return execution::tp_batch_with_window(ctx, values(nodes), window);
}
execution::Handle tp_batch_fused(execution::Handle ctx, Array<std::int64_t> nodes) {
  return execution::tp_batch_fused(ctx, values(nodes));
}
void tp_rebind(execution::Handle operation, Array<String> names,
    Array<std::int64_t> tensors, Array<String> ikeys, Array<std::int64_t> integers,
    Array<String> fkeys, Array<double> scalars) {
  execution::tp_rebind(operation, strings(names), values(tensors), strings(ikeys),
                       values(integers), strings(fkeys), values(scalars));
}
execution::Handle tp_view(execution::Handle tensor, Array<std::int64_t> dims,
    std::int64_t offset, String distribution, std::int64_t axis) {
  return execution::tp_view(tensor, values(dims), offset, distribution, axis);
}
Array<String> tp_selection(execution::Handle operation) {
  const auto values = execution::tp_selection(operation);
  Array<String> result;
  for (const auto& value : values) result.push_back(String(value));
  return result;
}
Array<std::int64_t> tp_batch_stats(execution::Handle batch) {
  const auto values = execution::tp_batch_stats(batch);
  return Array<std::int64_t>(values.begin(), values.end());
}
Array<std::int64_t> tp_context_stats(execution::Handle context) {
  const auto values = execution::tp_context_stats(context);
  return Array<std::int64_t>(values.begin(), values.end());
}
Array<std::int64_t> tp_pool_stats(execution::Handle context) {
  const auto values = execution::tp_pool_stats(context);
  return Array<std::int64_t>(values.begin(), values.end());
}
}  // namespace xinfer::rpu::ffi
TVM_FFI_DLL_EXPORT_TYPED_FUNC(rpu_tp_context_create, xinfer::rpu::execution::tp_context_create);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(rpu_tp_context_create_with_pool, xinfer::rpu::execution::tp_context_create_with_pool);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(rpu_tp_ar_window_max, xinfer::rpu::execution::tp_ar_window_max);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(rpu_tp_down_norm_window_max, xinfer::rpu::execution::tp_down_norm_window_max);
// Backward-compatible query name; new code uses the semantic DownNorm symbol.
TVM_FFI_DLL_EXPORT_TYPED_FUNC(rpu_tp_action_m16_window_max, xinfer::rpu::execution::tp_down_norm_window_max);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(rpu_tp_batch_stats, xinfer::rpu::ffi::tp_batch_stats);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(rpu_tp_context_stats, xinfer::rpu::ffi::tp_context_stats);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(rpu_tp_pool_stats, xinfer::rpu::ffi::tp_pool_stats);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(rpu_tp_empty, xinfer::rpu::ffi::tp_empty);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(rpu_tp_upload, xinfer::rpu::ffi::tp_upload);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(rpu_tp_download, xinfer::rpu::ffi::tp_download);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(rpu_tp_weight, xinfer::rpu::ffi::tp_weight);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(rpu_tp_reset, xinfer::rpu::execution::tp_reset);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(rpu_tp_invalidate, xinfer::rpu::execution::tp_invalidate);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(rpu_tp_view, xinfer::rpu::ffi::tp_view);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(rpu_tp_prepare, xinfer::rpu::ffi::tp_prepare);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(rpu_tp_batch, xinfer::rpu::ffi::tp_batch);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(rpu_tp_batch_with_window, xinfer::rpu::ffi::tp_batch_with_window);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(rpu_tp_batch_fused, xinfer::rpu::ffi::tp_batch_fused);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(rpu_tp_rebind, xinfer::rpu::ffi::tp_rebind);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(rpu_tp_selection, xinfer::rpu::ffi::tp_selection);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(rpu_tp_execute, xinfer::rpu::execution::tp_execute);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(rpu_tp_wait, xinfer::rpu::execution::tp_wait);
