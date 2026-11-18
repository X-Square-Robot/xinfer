// BF16 RMSNorm for Jetson Thor (sm_110a): one warp per row, the row kept in registers.
//
// The generic rmsnorm calls it for BF16 rows of at least 256 (rmsnorm.cu, UseRMSNormSM110). The generic kernel
// launches a 1024-thread block per row with 4-byte loads: at hidden 1280 only 640 threads touch data while all
// 1024 pay two __syncthreads and a full-block reduction, and the input is read from global memory twice.
//
// This kernel: hidden % 256 == 0 so each lane owns NVEC = hidden / 256 float4 vectors (8 BF16 each). The row is
// loaded once into registers (128-bit loads), squared and accumulated in FP32, reduced with 5 warp shuffles (no
// shared memory, no __syncthreads), then normalized from registers and stored as 128-bit vectors. Each warp
// handles ceil(rows / (SMs * 32)) consecutive rows, so the launch is about one resident wave with uniform work
// per warp.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <climits>
#include <cstdint>

#include "common/tvm_ffi_cuda_tensor.h"
#include "norm/sm110/bf16_rmsnorm_sm110.h"

namespace {

constexpr int kWarp = 32;
constexpr int kVecElems = 8;  // bf16 per 16-byte float4

template <int NVEC>
__global__ void rmsnorm_warp_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ in,
    const __nv_bfloat16* __restrict__ weight,
    const float eps,
    const float inv_hidden,
    const int rows,
    const int rows_per_warp) {
  const int gwarp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31;
  constexpr int kVecsPerRow = NVEC * kWarp;
  const float4* vw = reinterpret_cast<const float4*>(weight);

  const int row_begin = gwarp * rows_per_warp;
  const int row_end = min(row_begin + rows_per_warp, rows);

  for (int row = row_begin; row < row_end; ++row) {
    const float4* vin =
        reinterpret_cast<const float4*>(in) + (long)row * kVecsPerRow;
    float4* vout = reinterpret_cast<float4*>(out) + (long)row * kVecsPerRow;

    // Load the whole row once; it stays in registers for the store pass.
    // (110 regs at NVEC=5 — deliberate: two-pass/40-reg and prefetch
    // variants both measured slower)
    float4 xv[NVEC];
#pragma unroll
    for (int i = 0; i < NVEC; ++i) xv[i] = vin[lane + kWarp * i];

    float ss = 0.f;
#pragma unroll
    for (int i = 0; i < NVEC; ++i) {
      const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&xv[i]);
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const float2 f = __bfloat1622float2(p[j]);
        ss = fmaf(f.x, f.x, fmaf(f.y, f.y, ss));
      }
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
      ss += __shfl_xor_sync(0xffffffffu, ss, off);

    const float inv_rms = rsqrtf(ss * inv_hidden + eps);

#pragma unroll
    for (int i = 0; i < NVEC; ++i) {
      const float4 wv4 = vw[lane + kWarp * i];
      const __nv_bfloat162* xp = reinterpret_cast<const __nv_bfloat162*>(&xv[i]);
      const __nv_bfloat162* wp = reinterpret_cast<const __nv_bfloat162*>(&wv4);
      float4 ov4;
      __nv_bfloat162* op = reinterpret_cast<__nv_bfloat162*>(&ov4);
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const float2 xf = __bfloat1622float2(xp[j]);
        const float2 wf = __bfloat1622float2(wp[j]);
        op[j] = __floats2bfloat162_rn(xf.x * wf.x * inv_rms,
                                      xf.y * wf.y * inv_rms);
      }
      vout[lane + kWarp * i] = ov4;
    }
  }
}

using KernelFn = void (*)(__nv_bfloat16*, const __nv_bfloat16*,
                          const __nv_bfloat16*, float, float, int, int);

KernelFn pick_kernel(int nvec) {
  switch (nvec) {
    case 1: return rmsnorm_warp_kernel<1>;
    case 2: return rmsnorm_warp_kernel<2>;
    case 3: return rmsnorm_warp_kernel<3>;
    case 4: return rmsnorm_warp_kernel<4>;
    case 5: return rmsnorm_warp_kernel<5>;   // hidden = 1280 (measured ViT)
    case 6: return rmsnorm_warp_kernel<6>;
    case 7: return rmsnorm_warp_kernel<7>;
    case 8: return rmsnorm_warp_kernel<8>;
    case 10: return rmsnorm_warp_kernel<10>;
    case 12: return rmsnorm_warp_kernel<12>;
    case 14: return rmsnorm_warp_kernel<14>;
    case 16: return rmsnorm_warp_kernel<16>;
    default: return nullptr;
  }
}

}  // namespace

bool rmsnorm_bf16_supported(int64_t hidden) {
  return hidden % (kWarp * kVecElems) == 0 &&
         pick_kernel((int)(hidden / (kWarp * kVecElems))) != nullptr;
}

namespace bf16_rmsnorm_sm110 {

bool Supported(int64_t hidden) { return rmsnorm_bf16_supported(hidden); }

void Launch(__nv_bfloat16* out, const __nv_bfloat16* input,
            const __nv_bfloat16* weight, int64_t rows, int64_t hidden,
            float epsilon, int block_size, int device_id, cudaStream_t stream) {
  KernelFn kernel = pick_kernel(static_cast<int>(hidden / (kWarp * kVecElems)));
  if (kernel == nullptr) {
    throw std::runtime_error("unsupported RMSNorm hidden size");
  }
  // Size the grid to at most one resident wave: the kernel is register-bound
  // to ~16 warps/SM, so more warps than fit resident only create a
  // partial-wave tail. Give each warp ceil(rows / capacity) consecutive rows.
  // (SMs * 32 warp capacity measured best on Thor; the exact-occupancy grid
  // measured slower.)
  int sm_count = 0;
  xinfer::ffi::CheckCudaError(
      cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device_id),
      "cudaDeviceGetAttribute");
  const int warp_capacity = sm_count * 32;
  const int rows_per_warp = (int)((rows + warp_capacity - 1) / warp_capacity);
  const int64_t warps = (rows + rows_per_warp - 1) / rows_per_warp;
  const int grid =
      (int)((warps * kWarp + block_size - 1) / block_size);

  kernel<<<grid, block_size, 0, stream>>>(
      out, input, weight, epsilon, 1.0f / (float)hidden, (int)rows,
      rows_per_warp);
  xinfer::ffi::CheckLastCudaError("bf16_rmsnorm_sm110 launch");
}

}  // namespace bf16_rmsnorm_sm110
