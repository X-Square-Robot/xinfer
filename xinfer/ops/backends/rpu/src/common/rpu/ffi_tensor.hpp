// Copyright 2026 X-Square Robot
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/function.h>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace xinfer::rpu::ffi {
struct HostTensor {
  std::uint16_t* data;
  std::size_t rows, cols;
};
inline HostTensor host_tensor(tvm::ffi::TensorView tensor, const char* name,
                              bool matrix_only = false) {
  const auto fail = [name](const char* why) {
    throw std::invalid_argument(std::string(name) + " " + why);
  };
  if (tensor.device().device_type != kDLCPU || tensor.device().device_id != 0)
    fail("must be a CPU tensor; device transfers are explicit");
  const auto dtype = tensor.dtype();
  if (dtype.code != kDLFloat || dtype.bits != 16 || dtype.lanes != 1)
    fail("must have scalar FP16 dtype");
  if (tensor.ndim() != 2 && (matrix_only || tensor.ndim() != 1))
    fail(matrix_only ? "must have rank 2" : "must have rank 1 or 2");
  if (!tensor.shape().data()) fail("shape must not be null");
  const auto rows = tensor.ndim() == 1 ? 1 : tensor.shape()[0];
  const auto cols = tensor.shape()[tensor.ndim() - 1];
  if (rows <= 0 || rows > 7U * 1024U * 1024U / 2U ||
      cols <= 0 || cols > 7U * 1024U * 1024U / 2U ||
      static_cast<std::uint64_t>(rows) * cols > 7U * 1024U * 1024U / 2U)
    fail("shape exceeds the single-core buffer contract");
  if (!tensor.IsContiguous()) fail("must be contiguous row-major");
  const auto base = reinterpret_cast<std::uintptr_t>(tensor.data_ptr());
  const auto offset = tensor.byte_offset();
  const auto bytes = static_cast<std::uint64_t>(rows) * cols * 2;
  if (!base || offset > std::numeric_limits<std::uintptr_t>::max() - base ||
      bytes > std::numeric_limits<std::uintptr_t>::max() - base - offset ||
      (base + offset) % 2)
    fail("has null, overflowing or unaligned host storage");
  return {reinterpret_cast<std::uint16_t*>(base + offset),
          static_cast<std::size_t>(rows), static_cast<std::size_t>(cols)};
}
struct TpHost {
  void* data;
  std::vector<std::int64_t> shape;
  std::string dtype;
};
inline TpHost tp_host(tvm::ffi::TensorView tensor) {
  if (tensor.device().device_type != kDLCPU || tensor.device().device_id != 0 ||
      tensor.ndim() < 1 || tensor.ndim() > 7)
    throw std::invalid_argument("TP host tensor requires contiguous CPU rank1..7 storage");
  if (!tensor.shape().data())
    throw std::invalid_argument("TP host shape must not be null");
  const auto dtype = tensor.dtype();
  const bool half = dtype.code == kDLFloat && dtype.bits == 16;
  if (dtype.lanes != 1 || (!half && !(dtype.code == kDLInt && dtype.bits == 32)))
    throw std::invalid_argument("TP host dtype must be float16 or int32");
  TpHost host{nullptr, {}, half ? "float16" : "int32"};
  std::uint64_t bytes = half ? 2 : 4;
  for (int i = 0; i < tensor.ndim(); ++i) {
    const auto d = tensor.shape()[i];
    if (d <= 0 || std::uint64_t(d) > (1ULL << 40) / bytes)
      throw std::invalid_argument("TP host shape is empty or overflows");
    bytes *= d; host.shape.push_back(d);
  }
  // TVM's stride check multiplies signed dimensions. Validate rank and the
  // complete byte extent first, before it can index or overflow on bad input.
  if (!tensor.IsContiguous())
    throw std::invalid_argument("TP host tensor requires contiguous CPU rank1..7 storage");
  const auto base = reinterpret_cast<std::uintptr_t>(tensor.data_ptr());
  const auto offset = tensor.byte_offset();
  if (!base || offset > std::numeric_limits<std::uintptr_t>::max() - base ||
      bytes > std::numeric_limits<std::uintptr_t>::max() - base - offset ||
      (base + offset) % (half ? 2 : 4))
    throw std::invalid_argument("TP host storage is null, unaligned or overflows");
  host.data = reinterpret_cast<void*>(base + offset); return host;
}
}  // namespace xinfer::rpu::ffi
