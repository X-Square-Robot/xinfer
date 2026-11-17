// BF16 -> NVFP4 quantization (nvfp4_format_core.cuh): packed E2M1 data [M, K / 2] and E4M3 block scales in the
// SWIZZLE_32_4_4 storage, under an optional per-tensor global scale.
//
// Block scales from the block amax (activations, and weights with amax scales), or, for weights, from the MSE search
// over the 126 positive E4M3 values. The padding scale bytes (rows up to a multiple of 128, 16-column blocks up to a
// multiple of 4) are written as zero.

#include <tvm/ffi/function.h>

#include <cfloat>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "common/tvm_ffi_cuda_tensor.h"
#include "quant/nvfp4_format_core.cuh"

namespace {

using namespace xinfer::nvfp4;

constexpr int kThreads = 256;

// Amax scales: two threads per 16-column block, eight consecutive columns each (one 16-byte load, one 4-byte
// store). Thread t of a row covers block t / 2; threads past the last block or row write the zero padding scale.
__global__ __launch_bounds__(kThreads) void QuantizeAmaxKernel(const __nv_bfloat16* __restrict__ input,
                                                               uint8_t* __restrict__ data, uint8_t* __restrict__ scales,
                                                               int64_t rows, int64_t padded_rows, int k, int k_blocks,
                                                               int scale_col_blocks, float g, bool vector_loads) {
  const int64_t per_row = int64_t{scale_col_blocks} * 8;  // 2 threads x 4 blocks per scale column block
  const int64_t total = padded_rows * per_row;
  for (int64_t t = blockIdx.x * int64_t{blockDim.x} + threadIdx.x; t < total; t += int64_t{gridDim.x} * blockDim.x) {
    const int64_t row = t / per_row;
    const int j = static_cast<int>(t - row * per_row);
    const int block = j / 2, half = j & 1;
    const bool live = row < rows && block < k_blocks;
    float v[8];
    float amax = 0.0f;
    if (live) {
      const int64_t base = row * k + block * kBlock + half * 8;
      if (vector_loads) {
        const uint4 raw = *reinterpret_cast<const uint4*>(input + base);
        const __nv_bfloat16* h = reinterpret_cast<const __nv_bfloat16*>(&raw);
#pragma unroll
        for (int i = 0; i < 8; ++i) v[i] = __bfloat162float(h[i]);
      } else {
#pragma unroll
        for (int i = 0; i < 8; ++i) v[i] = __bfloat162float(input[base + i]);
      }
#pragma unroll
      for (int i = 0; i < 8; ++i) amax = fmaxf(amax, fabsf(v[i]));
    }
    // The partner thread (j ^ 1) is the other half of the same block, in the same warp and loop iteration.
    amax = fmaxf(amax, __shfl_xor_sync(__activemask(), amax, 1));
    if (!live) {
      if (half == 0) scales[ScaleOffset(row, block, scale_col_blocks)] = 0;
      continue;
    }
    const float scale = AmaxScale(amax, g);
    if (half == 0) scales[ScaleOffset(row, block, scale_col_blocks)] = ScaleBits(scale);
    *reinterpret_cast<uint32_t*>(data + row * (k / 2) + block * 8 + half * 4) = Pack8(v, __fmul_rn(scale, g));
  }
}

// MSE scales (weights): one warp per 16-column block, lane i < 16 owning column i, as the Wall-X kernel.
__global__ __launch_bounds__(kThreads) void QuantizeMseKernel(const __nv_bfloat16* __restrict__ input,
                                                              uint8_t* __restrict__ data, uint8_t* __restrict__ scales,
                                                              int64_t rows, int64_t padded_rows, int k, int k_blocks,
                                                              int scale_col_blocks, float g) {
  const int lane = threadIdx.x & 31;
  const int64_t per_row = int64_t{scale_col_blocks} * 4;
  const int64_t total = padded_rows * per_row;
  const int64_t warps = int64_t{gridDim.x} * (blockDim.x / 32);
  for (int64_t w = blockIdx.x * int64_t{blockDim.x / 32} + threadIdx.x / 32; w < total; w += warps) {
    const int64_t row = w / per_row;
    const int block = static_cast<int>(w - row * per_row);
    if (row >= rows || block >= k_blocks) {
      if (lane == 0) scales[ScaleOffset(row, block, scale_col_blocks)] = 0;
      continue;
    }
    const bool owner = lane < kBlock;
    const float x = owner ? __bfloat162float(input[row * k + block * kBlock + lane]) : 0.0f;
    float best = FLT_MAX;
    int best_bits = 1;
    for (int bits = 1; bits <= 126; ++bits) {
      __nv_fp8_e4m3 candidate;
      candidate.__x = static_cast<__nv_fp8_storage_t>(bits);
      const float decode = __fmul_rn(static_cast<float>(candidate), g);
      float error = 0.0f;
      if (owner) {
        const float delta = __fmaf_rn(-RoundE2M1(ApproxDiv(x, decode)), decode, x);
        error = __fmul_rn(delta, delta);
      }
      // Lanes 16..31 hold zero: the sum is ((e0 + e8) + (e4 + e12)) + ..., the pairwise order of the block.
#pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1) error += __shfl_down_sync(0xffffffffu, error, offset);
      if (error < best) {  // lane 0's sum decides; a later scale must be strictly better
        best = error;
        best_bits = bits;
      }
    }
    best_bits = __shfl_sync(0xffffffffu, best_bits, 0);
    __nv_fp8_e4m3 selected;
    selected.__x = static_cast<__nv_fp8_storage_t>(best_bits);
    const float decode = __fmul_rn(static_cast<float>(selected), g);
    const float partner = __shfl_down_sync(0xffffffffu, x, 1);
    if (lane == 0) scales[ScaleOffset(row, block, scale_col_blocks)] = selected.__x;
    if (owner && (lane & 1) == 0) {
      data[row * (k / 2) + block * 8 + lane / 2] = PackPair(ApproxDiv(x, decode), ApproxDiv(partner, decode));
    }
  }
}

unsigned Grid(int64_t threads) {
  const int64_t blocks = (threads + kThreads - 1) / kThreads;
  return static_cast<unsigned>(blocks < 65535 * 8 ? (blocks > 0 ? blocks : 1) : 65535 * 8);
}

}  // namespace

namespace xinfer::nvfp4 {

// quantized (uint8 [M, K / 2]) and scales (E4M3 or uint8 storage of ScaleStorageBytes(M, K) bytes) of BF16
// input [M, K], K a multiple of 16; global_scale divides every value (1 for activations); mse selects the MSE
// scale search instead of the amax scale.
void QuantizeInto(tvm::ffi::TensorView quantized, tvm::ffi::TensorView scales, tvm::ffi::TensorView input,
                  double global_scale, bool mse) {
  using namespace xinfer::ffi;
  CheckCuda(input, "input");
  CheckBFloat16(input, "input");
  CheckContiguous(input, "input");
  CheckCuda(quantized, "quantized");
  CheckUInt8(quantized, "quantized");
  CheckContiguous(quantized, "quantized");
  CheckFloat8E4M3Storage(scales, "scales");
  CheckContiguous(scales, "scales");
  CheckSameDevice(quantized, input, "quantize_nvfp4_into: quantized");
  CheckSameDevice(scales, input, "quantize_nvfp4_into: scales");
  if (input.ndim() != 2 || quantized.ndim() != 2) {
    throw std::runtime_error("quantize_nvfp4_into: input [M, K] and quantized [M, K / 2] must be rank-2");
  }
  const int64_t rows = input.size(0), k = input.size(1);
  if (k % kBlock) throw std::runtime_error("quantize_nvfp4_into: K must be a multiple of 16");
  if (k > (int64_t{1} << 30)) throw std::runtime_error("quantize_nvfp4_into: K must be at most 2^30");
  if (quantized.size(0) != rows || quantized.size(1) != k / 2) {
    throw std::runtime_error("quantize_nvfp4_into: quantized must be uint8 [M, K / 2]");
  }
  if (NumBytes(scales) < ScaleStorageBytes(rows, k)) {
    throw std::runtime_error("quantize_nvfp4_into: scales must hold " + std::to_string(ScaleStorageBytes(rows, k)) +
                             " bytes (SWIZZLE_32_4_4)");
  }
  const float g = static_cast<float>(global_scale);
  if (!(g > 0.0f) || !std::isfinite(g)) {
    throw std::runtime_error("quantize_nvfp4_into: global_scale must be finite and positive in FP32");
  }
  if (rows == 0 || k == 0) return;
  const int k_blocks = static_cast<int>(k / kBlock);
  const int scale_col_blocks = (k_blocks + 3) / 4;
  const int64_t padded_rows = (rows + 127) / 128 * 128;
  cudaStream_t stream = Stream(input);
  const __nv_bfloat16* in = ConstData<__nv_bfloat16>(input);
  if (mse) {
    const int64_t warps = padded_rows * scale_col_blocks * 4;
    QuantizeMseKernel<<<Grid(warps * 32), kThreads, 0, stream>>>(in, Data<uint8_t>(quantized),
                                                                Data<uint8_t>(scales), rows, padded_rows,
                                                                static_cast<int>(k), k_blocks, scale_col_blocks, g);
  } else {
    const bool vector_loads = reinterpret_cast<uintptr_t>(in) % 16 == 0;
    QuantizeAmaxKernel<<<Grid(padded_rows * scale_col_blocks * 8), kThreads, 0, stream>>>(
        in, Data<uint8_t>(quantized), Data<uint8_t>(scales), rows, padded_rows, static_cast<int>(k), k_blocks,
        scale_col_blocks, g, vector_loads);
  }
  CheckLastCudaError("quantize_nvfp4_into launch");
}

}  // namespace xinfer::nvfp4

TVM_FFI_DLL_EXPORT_TYPED_FUNC(quantize_nvfp4_into, xinfer::nvfp4::QuantizeInto);
