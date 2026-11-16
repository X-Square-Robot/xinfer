#pragma once

#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>

#include <cuda_runtime.h>

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace xinfer::ffi {

struct TensorByteSpan {
  uintptr_t begin;
  uintptr_t end;
};

inline bool IsBFloat16(const DLDataType& dtype) {
  return dtype.code == kDLBfloat && dtype.bits == 16 && dtype.lanes == 1;
}

inline bool IsFloat32(const DLDataType& dtype) {
  return dtype.code == kDLFloat && dtype.bits == 32 && dtype.lanes == 1;
}

inline bool IsFloat8E4M3FN(const DLDataType& dtype) {
  return dtype.code == kDLFloat8_e4m3fn && dtype.bits == 8 &&
         dtype.lanes == 1;
}

inline bool IsFloat8E8M0FNU(const DLDataType& dtype) {
  return dtype.code == kDLFloat8_e8m0fnu && dtype.bits == 8 &&
         dtype.lanes == 1;
}

inline bool IsUInt8(const DLDataType& dtype) {
  return dtype.code == kDLUInt && dtype.bits == 8 && dtype.lanes == 1;
}

inline bool IsInt32(const DLDataType& dtype) {
  return dtype.code == kDLInt && dtype.bits == 32 && dtype.lanes == 1;
}

inline int64_t NumElements(const tvm::ffi::TensorView& tensor) {
  int64_t count = 1;
  for (int index = 0; index < tensor.ndim(); ++index) {
    const int64_t dimension = tensor.size(index);
    if (dimension < 0) {
      throw std::runtime_error("tensor dimensions must be non-negative");
    }
    if (dimension != 0 &&
        count > std::numeric_limits<int64_t>::max() / dimension) {
      throw std::runtime_error("tensor element count overflows int64");
    }
    count *= dimension;
  }
  return count;
}

inline int64_t NumBytes(const tvm::ffi::TensorView& tensor) {
  const int bits_per_element = tensor.dtype().bits * tensor.dtype().lanes;
  if (bits_per_element <= 0 || bits_per_element % 8 != 0) {
    throw std::runtime_error("tensor dtype is not byte-addressable");
  }
  const int64_t elements = NumElements(tensor);
  const int64_t bytes_per_element = bits_per_element / 8;
  if (elements != 0 &&
      bytes_per_element > std::numeric_limits<int64_t>::max() / elements) {
    throw std::runtime_error("tensor byte count overflows int64");
  }
  return elements * bytes_per_element;
}

// Returns a conservative half-open byte span covering every element addressed
// by a non-negatively-strided tensor view. Gaps between dense rows remain part
// of the span, so overlap checks reject ambiguous aliases instead of accepting
// a view whose untouched padding happens to contain another tensor.
inline TensorByteSpan ByteSpan(const tvm::ffi::TensorView& tensor,
                               const char* name) {
  const int bits_per_element = tensor.dtype().bits * tensor.dtype().lanes;
  if (bits_per_element <= 0 || bits_per_element % 8 != 0) {
    throw std::runtime_error(std::string(name) +
                             " dtype is not byte-addressable");
  }
  const int64_t bytes_per_element = bits_per_element / 8;
  int64_t maximum_element_offset = 0;
  bool empty = false;
  for (int index = 0; index < tensor.ndim(); ++index) {
    const int64_t dimension = tensor.size(index);
    const int64_t stride = tensor.stride(index);
    if (dimension < 0) {
      throw std::runtime_error(std::string(name) +
                               " dimensions must be non-negative");
    }
    if (stride < 0) {
      throw std::runtime_error(std::string(name) +
                               " must not have negative strides");
    }
    if (dimension == 0) {
      empty = true;
      continue;
    }
    const int64_t steps = dimension - 1;
    if (steps != 0 &&
        stride > (std::numeric_limits<int64_t>::max() -
                  maximum_element_offset) /
                     steps) {
      throw std::runtime_error(std::string(name) +
                               " storage span overflows int64");
    }
    maximum_element_offset += steps * stride;
  }

  const uintptr_t begin = reinterpret_cast<uintptr_t>(tensor.data_ptr());
  if (empty) {
    return {begin, begin};
  }
  if (begin == 0) {
    throw std::runtime_error(std::string(name) + " data pointer is null");
  }
  if (maximum_element_offset >
      (std::numeric_limits<int64_t>::max() / bytes_per_element) - 1) {
    throw std::runtime_error(std::string(name) +
                             " storage span overflows int64");
  }
  const int64_t span_bytes =
      (maximum_element_offset + 1) * bytes_per_element;
  if (static_cast<uint64_t>(span_bytes) >
      std::numeric_limits<uintptr_t>::max() - begin) {
    throw std::runtime_error(std::string(name) +
                             " address range overflows uintptr_t");
  }
  return {begin, begin + static_cast<uintptr_t>(span_bytes)};
}

inline bool IsExactTensorView(const tvm::ffi::TensorView& lhs,
                              const tvm::ffi::TensorView& rhs) {
  if (lhs.data_ptr() != rhs.data_ptr() || lhs.ndim() != rhs.ndim() ||
      lhs.device().device_type != rhs.device().device_type ||
      lhs.device().device_id != rhs.device().device_id ||
      lhs.dtype().code != rhs.dtype().code ||
      lhs.dtype().bits != rhs.dtype().bits ||
      lhs.dtype().lanes != rhs.dtype().lanes) {
    return false;
  }
  for (int index = 0; index < lhs.ndim(); ++index) {
    if (lhs.size(index) != rhs.size(index) ||
        lhs.stride(index) != rhs.stride(index)) {
      return false;
    }
  }
  return true;
}

inline bool ByteSpansOverlap(const TensorByteSpan& lhs,
                             const TensorByteSpan& rhs) {
  return lhs.begin < rhs.end && rhs.begin < lhs.end;
}

inline void CheckNoOverlap(const tvm::ffi::TensorView& lhs,
                           const char* lhs_name,
                           const tvm::ffi::TensorView& rhs,
                           const char* rhs_name) {
  if (ByteSpansOverlap(ByteSpan(lhs, lhs_name), ByteSpan(rhs, rhs_name))) {
    throw std::runtime_error(std::string(lhs_name) + " must not overlap " +
                             rhs_name);
  }
}

inline void CheckExactAliasOrNoOverlap(
    const tvm::ffi::TensorView& output, const char* output_name,
    const tvm::ffi::TensorView& input, const char* input_name) {
  if (IsExactTensorView(output, input)) {
    return;
  }
  if (ByteSpansOverlap(ByteSpan(output, output_name),
                       ByteSpan(input, input_name))) {
    throw std::runtime_error(
        std::string(output_name) + " and " + input_name +
        " must be the same tensor view or have no storage overlap");
  }
}

inline void CheckCuda(const tvm::ffi::TensorView& tensor, const char* name) {
  if (tensor.device().device_type != kDLCUDA) {
    throw std::runtime_error(std::string(name) + " must be a CUDA tensor");
  }
}

inline void CheckBFloat16(const tvm::ffi::TensorView& tensor,
                          const char* name) {
  if (!IsBFloat16(tensor.dtype())) {
    throw std::runtime_error(std::string(name) + " must have BF16 dtype");
  }
}

inline void CheckFloat32(const tvm::ffi::TensorView& tensor,
                         const char* name) {
  if (!IsFloat32(tensor.dtype())) {
    throw std::runtime_error(std::string(name) + " must have FP32 dtype");
  }
}

inline void CheckFloat8E4M3FN(const tvm::ffi::TensorView& tensor,
                              const char* name) {
  if (!IsFloat8E4M3FN(tensor.dtype())) {
    throw std::runtime_error(std::string(name) +
                             " must have float8_e4m3fn dtype");
  }
}

inline void CheckFloat8E8M0FNU(const tvm::ffi::TensorView& tensor,
                               const char* name) {
  if (!IsFloat8E8M0FNU(tensor.dtype())) {
    throw std::runtime_error(std::string(name) +
                             " must have float8_e8m0fnu dtype");
  }
}

// Float8 buffers are sometimes passed as uint8 views so callers can allocate
// raw storage and expose a typed view afterwards. Accept both spellings at
// producer and consumer boundaries; their physical representation is one byte.
inline void CheckFloat8E4M3Storage(const tvm::ffi::TensorView& tensor,
                                   const char* name) {
  if (!IsFloat8E4M3FN(tensor.dtype()) && !IsUInt8(tensor.dtype())) {
    throw std::runtime_error(std::string(name) +
                             " must have float8_e4m3fn or uint8 storage");
  }
}

inline void CheckFloat8E8M0Storage(const tvm::ffi::TensorView& tensor,
                                   const char* name) {
  if (!IsFloat8E8M0FNU(tensor.dtype()) && !IsUInt8(tensor.dtype())) {
    throw std::runtime_error(std::string(name) +
                             " must have float8_e8m0fnu or uint8 storage");
  }
}

inline void CheckUInt8(const tvm::ffi::TensorView& tensor,
                       const char* name) {
  if (!IsUInt8(tensor.dtype())) {
    throw std::runtime_error(std::string(name) + " must have uint8 dtype");
  }
}

inline void CheckInt32(const tvm::ffi::TensorView& tensor,
                       const char* name) {
  if (!IsInt32(tensor.dtype())) {
    throw std::runtime_error(std::string(name) + " must have int32 dtype");
  }
}

inline void CheckContiguous(const tvm::ffi::TensorView& tensor,
                            const char* name) {
  if (!tensor.IsContiguous()) {
    throw std::runtime_error(std::string(name) + " must be contiguous");
  }
}

inline void CheckSameDevice(const tvm::ffi::TensorView& tensor,
                            const tvm::ffi::TensorView& reference,
                            const char* name) {
  if (tensor.device().device_type != reference.device().device_type ||
      tensor.device().device_id != reference.device().device_id) {
    throw std::runtime_error(std::string(name) +
                             " must be on the same device");
  }
}

inline void CheckSameShape(const tvm::ffi::TensorView& tensor,
                           const tvm::ffi::TensorView& reference,
                           const char* name) {
  if (tensor.ndim() != reference.ndim()) {
    throw std::runtime_error(std::string(name) + " shape mismatch");
  }
  for (int index = 0; index < tensor.ndim(); ++index) {
    if (tensor.size(index) != reference.size(index)) {
      throw std::runtime_error(std::string(name) + " shape mismatch");
    }
  }
}

// Compute capabilities the library is compiled for, set by src/CMakeLists.txt.
#ifndef XINFER_TARGET_ARCHS
#define XINFER_TARGET_ARCHS "unknown"
#endif

inline void CheckCudaError(cudaError_t error, const char* operation) {
  if (error != cudaSuccess) {
    std::string message = std::string(operation) + " failed: " +
                          cudaGetErrorString(error);
    if (error == cudaErrorNoKernelImageForDevice) {
      // Each build targets only its own architectures (see the root CMakeLists.txt).
      message += " (xinfer_kernels.so was built for compute capability " XINFER_TARGET_ARCHS
                 " only; rebuild on this machine with ./build.sh, or ./build.sh --arch <cc>)";
    }
    throw std::runtime_error(message);
  }
}

inline cudaDeviceProp RequireExactComputeCapability(
    const tvm::ffi::TensorView& tensor, int required_major,
    int required_minor, const char* operation) {
  CheckCuda(tensor, operation);
  cudaDeviceProp properties{};
  CheckCudaError(cudaGetDeviceProperties(&properties,
                                         tensor.device().device_id),
                 "cudaGetDeviceProperties");
  if (properties.major != required_major ||
      properties.minor != required_minor) {
    throw std::runtime_error(
        std::string(operation) + " requires CUDA compute capability " +
        std::to_string(required_major) + "." +
        std::to_string(required_minor) + ", but current device is " +
        std::to_string(properties.major) + "." +
        std::to_string(properties.minor));
  }
  return properties;
}

inline cudaStream_t Stream(const tvm::ffi::TensorView& tensor) {
  CheckCudaError(cudaSetDevice(tensor.device().device_id), "cudaSetDevice");
  return static_cast<cudaStream_t>(TVMFFIEnvGetStream(
      tensor.device().device_type, tensor.device().device_id));
}

// Architecture gate for _sm<arch> sources under src/<owner>/cuda/ (SM<arch>_SOURCES): those kernels
// exist for one architecture, so calling them on any other GPU is an error, never a silent fallback.
// cudaDeviceGetAttribute keeps it cheap enough for per-call hot paths (cudaGetDeviceProperties is not).
inline void RequireArchitecture(const tvm::ffi::TensorView& tensor, int required_major,
                                int required_minor, const char* operation) {
  CheckCuda(tensor, operation);
  const int device = tensor.device().device_id;
  int major = 0;
  int minor = 0;
  CheckCudaError(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device),
                 "cudaDeviceGetAttribute");
  CheckCudaError(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device),
                 "cudaDeviceGetAttribute");
  if (major != required_major || minor != required_minor) {
    throw std::runtime_error(std::string(operation) + " is an SM" + std::to_string(required_major) +
                             std::to_string(required_minor) + " backend; device " +
                             std::to_string(device) + " is compute capability " +
                             std::to_string(major) + "." + std::to_string(minor));
  }
}

inline void CheckLastCudaError(const char* operation) {
  CheckCudaError(cudaGetLastError(), operation);
}

template <typename T>
inline T* Data(const tvm::ffi::TensorView& tensor) {
  return reinterpret_cast<T*>(tensor.data_ptr());
}

template <typename T>
inline const T* ConstData(const tvm::ffi::TensorView& tensor) {
  return reinterpret_cast<const T*>(tensor.data_ptr());
}

}  // namespace xinfer::ffi
