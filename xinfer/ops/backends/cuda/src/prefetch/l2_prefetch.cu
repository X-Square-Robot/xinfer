// L2 prefetch: issue prefetch.global.L2 for every 128-byte line of a byte range of a tensor.
//
// The caller owns stream selection and the ordering between prefetch and compute launches (typically a side stream
// forked from the compute stream); nothing here creates streams or events.

#include <tvm/ffi/function.h>

#include <cuda_runtime.h>

#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <unordered_set>

#include "common/tvm_ffi_cuda_tensor.h"

namespace xinfer::prefetch {
namespace {

using tvm::ffi::TensorView;

__global__ void __launch_bounds__(256) L2PrefetchKernel(const char* __restrict__ first, int64_t first_bytes,
                                                        const char* __restrict__ second, int64_t second_bytes) {
  const char* pointers[2] = {first, second};
  const int64_t sizes[2] = {first_bytes, second_bytes};
  const int64_t stride = int64_t{gridDim.x} * blockDim.x;
  for (int buffer = 0; buffer < 2; ++buffer) {
    const char* p = pointers[buffer];
    if (!p) continue;
    const int64_t lines = (sizes[buffer] + 127) >> 7;
    for (int64_t line = blockIdx.x * int64_t{blockDim.x} + threadIdx.x; line < lines; line += stride) {
      asm volatile("prefetch.global.L2 [%0];" ::"l"(p + (line << 7)) : "memory");
    }
  }
}

// The kernel uses no shared memory; preferring the max-shared carveout keeps it from forcing an L1/shared
// reconfiguration when it lands on an SM next to a GEMM CTA. cudaFuncSetAttribute applies to the current
// device, so each device is configured once (Stream() has already selected the tensor's device).
void ConfigureCarveout(int device) {
  static std::mutex mutex;
  static std::unordered_set<int> configured;
  std::lock_guard<std::mutex> lock(mutex);
  if (configured.count(device) != 0) return;
  xinfer::ffi::CheckCudaError(cudaFuncSetAttribute(L2PrefetchKernel, cudaFuncAttributePreferredSharedMemoryCarveout,
                                                   cudaSharedmemCarveoutMaxShared),
                              "L2 prefetch carveout configuration");
  configured.insert(device);
}

void Launch(const TensorView& t, const char* first, int64_t first_bytes, const char* second, int64_t second_bytes,
            int64_t grid, int64_t threads) {
  cudaStream_t stream = xinfer::ffi::Stream(t);
  ConfigureCarveout(t.device().device_id);
  L2PrefetchKernel<<<static_cast<unsigned>(grid), static_cast<unsigned>(threads), 0, stream>>>(
      first, first_bytes, second, second_bytes);
  xinfer::ffi::CheckLastCudaError("L2 prefetch launch");
}

}  // namespace

// Prefetch bytes [offset, offset + bytes) of t. grid >= 1, threads in [1, 256].
void L2Prefetch(TensorView t, int64_t offset, int64_t bytes, int64_t grid, int64_t threads) {
  using namespace xinfer::ffi;
  CheckCuda(t, "tensor");
  CheckContiguous(t, "tensor");
  const int64_t size = NumBytes(t);
  if (offset < 0 || bytes < 0 || offset > size || bytes > size - offset) {
    throw std::runtime_error("l2_prefetch: range [offset, offset + bytes) out of bounds");
  }
  if (grid < 1 || grid > 65535 || threads < 1 || threads > 256) {
    throw std::runtime_error("l2_prefetch: grid must be in [1, 65535] and threads in [1, 256]");
  }
  if (bytes == 0) return;
  Launch(t, static_cast<const char*>(t.data_ptr()) + offset, bytes, nullptr, 0, grid, threads);
}

}  // namespace xinfer::prefetch

TVM_FFI_DLL_EXPORT_TYPED_FUNC(l2_prefetch, xinfer::prefetch::L2Prefetch);
