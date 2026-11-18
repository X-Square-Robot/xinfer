// Programmatic dependent-launch plumbing for latency-sensitive CUDA pipelines.
//
// Callers decide whether PDL is appropriate for a particular launch. This
// header owns only the CUDA launch mechanism and device-side synchronization.
#pragma once

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace xinfer::cuda {

template <class Kernel, class... Args>
inline void pdl_launch(dim3 grid, dim3 block, size_t smem,
                       cudaStream_t stream, Kernel kernel, Args... args) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = grid;
  cfg.blockDim = block;
  cfg.dynamicSmemBytes = static_cast<unsigned>(smem);
  cfg.stream = stream;

  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = 1;

  const cudaError_t err = cudaLaunchKernelEx(&cfg, kernel, args...);
  if (err != cudaSuccess) {
    throw std::runtime_error(
        std::string("cudaLaunchKernelEx: ") + cudaGetErrorString(err));
  }
}

#if defined(__CUDACC__)
__device__ __forceinline__ void pdl_wait() {
  asm volatile("griddepcontrol.wait;" ::: "memory");
}

__device__ __forceinline__ void pdl_trigger() {
  asm volatile("griddepcontrol.launch_dependents;");
}
#endif

}  // namespace xinfer::cuda
