// TVM FFI binding helpers of the CUTLASS entry points (nvfp4_cutlass_sm120.cu): argument checks, the device
// guard and the stream.
#pragma once

#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>

#include <cuda_runtime.h>

#include <atomic>
#include <climits>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <string>

namespace xinfer::cutlass_ffi {

inline void Require(bool ok, const std::string& message) {
  if (!ok) throw std::runtime_error(message);
}

inline bool HasDtype(const tvm::ffi::TensorView& t, int code, int bits) {
  return t.dtype().code == code && t.dtype().bits == bits && t.dtype().lanes == 1;
}

inline void CheckCuda(const tvm::ffi::TensorView& t, const char* name) {
  Require(t.device().device_type == kDLCUDA, std::string(name) + " must be a CUDA tensor");
}

inline void CheckSameDevice(const tvm::ffi::TensorView& t, const tvm::ffi::TensorView& ref,
                            const char* name) {
  CheckCuda(t, name);
  Require(t.device().device_id == ref.device().device_id,
          std::string(name) + " must be on the same CUDA device as the other tensors");
}

inline void CheckBf16(const tvm::ffi::TensorView& t, const char* name) {
  Require(HasDtype(t, kDLBfloat, 16), std::string(name) + " must be bfloat16");
}

inline void CheckF32(const tvm::ffi::TensorView& t, const char* name) {
  Require(HasDtype(t, kDLFloat, 32), std::string(name) + " must be float32");
}

// E4M3 data: float8_e4m3fn or a uint8 view of it.
inline void CheckE4m3(const tvm::ffi::TensorView& t, const char* name) {
  Require(HasDtype(t, kDLFloat8_e4m3fn, 8) || HasDtype(t, kDLUInt, 8),
          std::string(name) + " must have float8_e4m3fn or uint8 storage");
}

// E8M0 scales: float8_e8m0fnu or a uint8 view of it.
inline void CheckE8m0(const tvm::ffi::TensorView& t, const char* name) {
  Require(HasDtype(t, kDLFloat8_e8m0fnu, 8) || HasDtype(t, kDLUInt, 8),
          std::string(name) + " must have float8_e8m0fnu or uint8 storage");
}

inline void CheckUInt8(const tvm::ffi::TensorView& t, const char* name) {
  Require(HasDtype(t, kDLUInt, 8), std::string(name) + " must be uint8");
}

inline int64_t Numel(const tvm::ffi::TensorView& t) {
  int64_t n = 1;
  for (int i = 0; i < t.ndim(); ++i) n *= t.size(i);
  return n;
}

inline void CheckContiguous(const tvm::ffi::TensorView& t, const char* name) {
  int64_t expected = 1;
  for (int i = t.ndim() - 1; i >= 0; --i) {
    Require(t.size(i) == 1 || t.stride(i) == expected, std::string(name) + " must be contiguous");
    expected *= t.size(i);
  }
}

inline void CheckRank(const tvm::ffi::TensorView& t, int rank, const char* name) {
  Require(t.ndim() == rank, std::string(name) + " must be " + std::to_string(rank) + "-D");
}

inline void CheckAligned(const tvm::ffi::TensorView& t, int bytes, const char* name) {
  Require(reinterpret_cast<uintptr_t>(t.data_ptr()) % bytes == 0,
          std::string(name) + " must be " + std::to_string(bytes) + "-byte aligned");
}

inline int CheckedInt(int64_t value, const char* name) {
  Require(value >= 0 && value <= INT_MAX, std::string(name) + " is outside the int32 range");
  return static_cast<int>(value);
}

inline int64_t RoundUp(int64_t value, int64_t multiple) {
  return (value + multiple - 1) / multiple * multiple;
}

// One-byte entries of a SWIZZLE_32_4_4 scale buffer for a [rows, k] operand:
// rows padded to 128, 1x32 blocks padded to 4.
inline int64_t ScaleElements(int64_t rows, int64_t k) {
  return RoundUp(rows, 128) * RoundUp((k + 31) / 32, 4);
}

inline void CheckScaleCapacity(const tvm::ffi::TensorView& scales, int64_t rows, int64_t k,
                               const char* name) {
  Require(Numel(scales) >= ScaleElements(rows, k),
          std::string(name) + " must hold at least " + std::to_string(ScaleElements(rows, k)) +
              " E8M0 entries (rows padded to 128, 1x32 blocks padded to 4)");
}

// The objects carry sm_120a SASS only; reject other devices before launching.
inline void CheckCapability120(const tvm::ffi::TensorView& t, const char* op) {
  constexpr int kMaxDevices = 64;
  static std::atomic<int> cached[kMaxDevices];  // 0 unknown, else major * 10 + minor
  const int device = t.device().device_id;
  int cc = device >= 0 && device < kMaxDevices ? cached[device].load(std::memory_order_relaxed) : 0;
  if (cc == 0) {
    int major = 0, minor = 0;
    Require(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) == cudaSuccess &&
                cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device) == cudaSuccess,
            std::string(op) + ": cannot query the compute capability");
    cc = major * 10 + minor;
    if (device >= 0 && device < kMaxDevices) cached[device].store(cc, std::memory_order_relaxed);
  }
  Require(cc == 120, std::string(op) + " requires compute capability 12.0 (sm_120a), got " +
                         std::to_string(cc / 10) + "." + std::to_string(cc % 10));
}

// Makes the tensors' device current for the launch and restores the caller's device.
class DeviceGuard {
 public:
  explicit DeviceGuard(int device) {
    Require(cudaGetDevice(&previous_) == cudaSuccess, "cudaGetDevice failed");
    if (previous_ != device) Require(cudaSetDevice(device) == cudaSuccess, "cudaSetDevice failed");
  }
  ~DeviceGuard() {
    int current = -1;
    if (cudaGetDevice(&current) == cudaSuccess && current != previous_) cudaSetDevice(previous_);
  }
  DeviceGuard(const DeviceGuard&) = delete;
  DeviceGuard& operator=(const DeviceGuard&) = delete;

 private:
  int previous_ = 0;
};

// The caller's current stream. For calls that receive torch CUDA tensors,
// tvm-ffi sets it from torch's current stream (DLPack exchange API).
inline cudaStream_t Stream(const tvm::ffi::TensorView& t) {
  return static_cast<cudaStream_t>(TVMFFIEnvGetStream(t.device().device_type, t.device().device_id));
}

inline void CheckLaunch(const char* op) {
  cudaError_t err = cudaGetLastError();
  Require(err == cudaSuccess, std::string(op) + " launch failed: " + cudaGetErrorString(err));
}

}  // namespace xinfer::cutlass_ffi
