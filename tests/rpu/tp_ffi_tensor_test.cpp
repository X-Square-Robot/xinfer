// Copyright 2026 X-Square Robot
// SPDX-License-Identifier: Apache-2.0
#include "common/rpu/ffi_tensor.hpp"

#include <array>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using xinfer::rpu::ffi::tp_host;
using xinfer::rpu::ffi::host_tensor;
unsigned accepted = 0, rejected = 0;

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

void reject(const DLTensor& tensor, const char* message) {
  try {
    (void)tp_host(tvm::ffi::TensorView(&tensor));
  } catch (const std::invalid_argument& error) {
    require(std::string(error.what()).find(message) != std::string::npos,
            "unexpected TP host rejection");
    ++rejected;
    return;
  }
  throw std::runtime_error("invalid TP host tensor was accepted");
}
}  // namespace

int main() {
  try {
    std::array<std::uint64_t, 64> storage{};
    std::int64_t dims[] = {2, 16}, strides[] = {16, 1};
    DLTensor base{};
    base.data = storage.data(); base.device = {kDLCPU, 0};
    base.ndim = 2; base.dtype = {kDLFloat, 16, 1};
    base.shape = dims; base.strides = strides; base.byte_offset = 2;
    auto host = tp_host(tvm::ffi::TensorView(&base));
    require(host.data == reinterpret_cast<unsigned char*>(storage.data()) + 2 &&
                host.shape == std::vector<std::int64_t>{2, 16} && host.dtype == "float16",
            "FP16 shape or byte offset was not preserved");
    ++accepted;

    // The single-core boundary must reject a malformed shape before indexing it, just
    // like TP. Direct FFI is not restricted to tensors created by NumPy/Torch.
    for (const int rank : {1, 2}) {
      auto no_shape = base;
      no_shape.ndim = rank; no_shape.shape = nullptr;
      bool failed = false;
      try { (void)host_tensor(tvm::ffi::TensorView(&no_shape), "input"); }
      catch (const std::invalid_argument& error) {
        failed = std::string(error.what()).find("shape must not be null") != std::string::npos;
      }
      require(failed, "single-core null shape was not rejected before indexing");
      ++rejected;
    }
    auto integer = base;
    integer.dtype = {kDLInt, 32, 1}; integer.byte_offset = 4;
    host = tp_host(tvm::ffi::TensorView(&integer));
    require(host.data == reinterpret_cast<unsigned char*>(storage.data()) + 4 &&
                host.dtype == "int32", "int32 host tensor was decoded incorrectly");
    ++accepted;
    std::int64_t seven_dims[] = {8, 1, 1, 1, 1, 1, 16};
    auto seven = base; seven.ndim = 7; seven.shape = seven_dims; seven.strides = nullptr;
    require(tp_host(tvm::ffi::TensorView(&seven)).shape.size() == 7,
            "physical-core leading dimension must permit host rank7");
    ++accepted;

    auto bad = base; bad.device = {kDLCUDA, 0}; reject(bad, "CPU rank1..7");
    bad = base; bad.device.device_id = 1; reject(bad, "CPU rank1..7");
    for (const auto rank : {-1, 0, 8}) {
      bad = base; bad.ndim = rank; bad.shape = nullptr;
      // Explicit strides must not be inspected until rank has passed.
      reject(bad, "CPU rank1..7");
    }
    bad = base; bad.shape = nullptr; reject(bad, "shape must not be null");
    bad.strides = nullptr; reject(bad, "shape must not be null");
    for (const DLDataType dtype : {DLDataType{kDLFloat, 32, 1},
                                  DLDataType{kDLInt, 16, 1},
                                  DLDataType{kDLFloat, 16, 2}}) {
      bad = base; bad.dtype = dtype; reject(bad, "dtype");
    }
    for (const std::int64_t dimension : {std::int64_t{0}, std::int64_t{-1}}) {
      std::int64_t invalid_dims[] = {dimension, 16};
      bad = base; bad.shape = invalid_dims; reject(bad, "shape");
    }
    // The old pre-validation IsContiguous call multiplied 2^32 by 2^32
    // in signed int64_t. UBSan must see no overflow in the production helper.
    std::int64_t huge_dims[] = {1LL << 32, 1LL << 32};
    std::int64_t huge_strides[] = {1LL << 32, 1};
    bad = base; bad.shape = huge_dims; bad.strides = huge_strides; reject(bad, "shape");
    std::int64_t noncontiguous[] = {17, 1};
    bad = base; bad.strides = noncontiguous; reject(bad, "contiguous");
    bad = base; bad.data = nullptr; reject(bad, "storage");
    bad = base; bad.byte_offset = 1; reject(bad, "unaligned");
    bad = integer; bad.byte_offset = 2; reject(bad, "unaligned");
    bad = base; bad.byte_offset = std::numeric_limits<std::uint64_t>::max();
    reject(bad, "overflows");
    bad = base; bad.data = reinterpret_cast<void*>(std::numeric_limits<std::uintptr_t>::max() - 1);
    bad.byte_offset = 0; reject(bad, "overflows");
    std::cout << "Host TensorView checks: " << accepted << " accepted, "
              << rejected << " rejected\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
