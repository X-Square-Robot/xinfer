#pragma once

#include <tvm/ffi/function.h>

#include <stdexcept>
#include <string>

#include "common/tvm_ffi_cuda_tensor.h"
#include "mxfp8_format_core.cuh"

namespace xinfer::mxfp8 {

inline void CheckScaleStorage(const tvm::ffi::TensorView& scales,
                              int64_t rows, int64_t k,
                              const char* name) {
  xinfer::ffi::CheckFloat8E8M0Storage(scales, name);
  if (xinfer::ffi::NumElements(scales) < RequiredScaleElements(rows, k)) {
    throw std::runtime_error(std::string(name) + " storage is too small");
  }
}

inline void CheckCommonOutput(const tvm::ffi::TensorView& output,
                              const tvm::ffi::TensorView& input,
                              const char* name) {
  xinfer::ffi::CheckCuda(output, name);
  xinfer::ffi::CheckContiguous(output, name);
  xinfer::ffi::CheckSameDevice(output, input, name);
}

}  // namespace xinfer::mxfp8
