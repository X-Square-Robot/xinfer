// cuTensorMapEncodeTiled looked up through the runtime (the library does not link libcuda).
#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <stdexcept>

namespace xinfer::ffi {

using TensorMapEncodeFn = CUresult (*)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*,
                                       const cuuint64_t*, const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave,
                                       CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill);

// The driver entry point (process-wide, the same for every device).
inline TensorMapEncodeFn TensorMapEncodeTiled() {
  static TensorMapEncodeFn f = [] {
    void* p = nullptr;
    cudaDriverEntryPointQueryResult q;
    if (cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &p, 12000, cudaEnableDefault, &q) != cudaSuccess ||
        q != cudaDriverEntryPointSuccess || p == nullptr) {
      throw std::runtime_error("cuTensorMapEncodeTiled is unavailable in this CUDA runtime");
    }
    return reinterpret_cast<TensorMapEncodeFn>(p);
  }();
  return f;
}

}  // namespace xinfer::ffi
