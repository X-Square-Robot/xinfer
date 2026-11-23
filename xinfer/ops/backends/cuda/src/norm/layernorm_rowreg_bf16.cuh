// Register-resident BF16 LayerNorm: 16 threads per row, each holding K / 128 16-byte vectors of x, gamma and
// beta, all loads issued up front; mean and variance in two passes over the registers (FP32), each finished
// with 4 xor shuffles inside the 16-lane group; y = fma((x - mean) * rstd, gamma, beta), rounded once to BF16.
// No shared memory, no barrier. Two kernels share the row code:
//   layernorm_rowreg_bf16_kernel        writes y as BF16 (y may alias x exactly: a thread reads its whole row
//                                       slice before writing);
//   layernorm_quant_mxfp8_rowreg_kernel writes MXFP8 with one scale per 128 columns: vector i of a row's 16
//                                       threads is the row's 128-wide tile i, so the tile amax is 4 more shuffles.
//                                       Bit for bit the first kernel followed by quantize_mxfp8_into(.., true);
//   layernorm_quant_nvfp4_rowreg_kernel writes NVFP4 (and y when given an output): vectors v and v ^ 1 of a row
//                                       are one 16-column block, so the block amax is one more shuffle. Bit for bit
//                                       the first kernel followed by quantize_nvfp4.
//
// Shape envelope (checked by the host): BF16 x / y / gamma / beta, K % 128 == 0, 128 <= K <= 2048, packed rows,
// 16-byte aligned pointers (quantized 8-byte, scales 4-byte).
#pragma once

#include <cuda_bf16.h>

#include <cstdint>

#include "norm/bf16_vector.cuh"
#include "quant/mxfp8_format_core.cuh"
#include "quant/nvfp4_format_core.cuh"

namespace xinfer_layernorm::rowreg {

inline constexpr int kThreadsPerRow = 16;
inline constexpr int kRowsPerBlock = 2;
inline constexpr int kQuantRowsPerBlock = 4;
inline constexpr int kMaxVectors = 16;  // K <= 2048

// Loads one thread's slice of a row (vector sub + 16 i of x, gamma and beta), all loads issued up front.
template <int kVectors>
__device__ __forceinline__ void load_row_slice(uint4 (&xv)[kVectors], uint4 (&gv)[kVectors], uint4 (&bv)[kVectors],
                                               const __nv_bfloat16* input, const __nv_bfloat16* gamma,
                                               const __nv_bfloat16* beta, int row, int sub) {
  const uint4* x = reinterpret_cast<const uint4*>(input + static_cast<size_t>(row) * kVectors * 128) + sub;
  const uint4* g = reinterpret_cast<const uint4*>(gamma) + sub;
  const uint4* b = reinterpret_cast<const uint4*>(beta) + sub;
#pragma unroll
  for (int i = 0; i < kVectors; ++i) xv[i] = x[kThreadsPerRow * i];
#pragma unroll
  for (int i = 0; i < kVectors; ++i) {
    gv[i] = g[kThreadsPerRow * i];
    bv[i] = b[kThreadsPerRow * i];
  }
}

// Two-pass mean and 1 / sqrt(variance + epsilon) of the row whose 16 lanes hold xv; every lane in the mask
// `lanes` must call this (the shuffles stay inside each row's 16 lanes).
template <int kVectors>
__device__ __forceinline__ void row_statistics(const uint4 (&xv)[kVectors], uint32_t lanes, float epsilon,
                                               float& mean, float& rstd) {
  using xinfer::normalization::unpack_bf16x8;
  constexpr int kK = kVectors * 128;
  float sum = 0.0f;
#pragma unroll
  for (int i = 0; i < kVectors; ++i) {
    float f[8];
    unpack_bf16x8(xv[i], f);
#pragma unroll
    for (int j = 0; j < 8; ++j) sum += f[j];
  }
#pragma unroll
  for (int offset = kThreadsPerRow / 2; offset > 0; offset >>= 1) sum += __shfl_xor_sync(lanes, sum, offset);
  mean = sum / static_cast<float>(kK);
  float squares = 0.0f;
#pragma unroll
  for (int i = 0; i < kVectors; ++i) {
    float f[8];
    unpack_bf16x8(xv[i], f);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const float d = f[j] - mean;
      squares = fmaf(d, d, squares);
    }
  }
#pragma unroll
  for (int offset = kThreadsPerRow / 2; offset > 0; offset >>= 1) {
    squares += __shfl_xor_sync(lanes, squares, offset);
  }
  rstd = rsqrtf(squares / static_cast<float>(kK) + epsilon);
}

// One vector of y as four BF16 pairs (round to nearest even).
__device__ __forceinline__ uint4 normalize_bf16x8(const uint4& x, const uint4& gamma, const uint4& beta, float mean,
                                                  float rstd) {
  using xinfer::normalization::pack_bf16x2;
  using xinfer::normalization::unpack_bf16x8;
  float f[8], gf[8], bf[8];
  unpack_bf16x8(x, f);
  unpack_bf16x8(gamma, gf);
  unpack_bf16x8(beta, bf);
  uint4 result;
  uint32_t* words = reinterpret_cast<uint32_t*>(&result);
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    words[j] = pack_bf16x2(fmaf((f[2 * j] - mean) * rstd, gf[2 * j], bf[2 * j]),
                           fmaf((f[2 * j + 1] - mean) * rstd, gf[2 * j + 1], bf[2 * j + 1]));
  }
  return result;
}

// Rows of a block are 16-lane halves of its warps: a missing trailing row keeps its lanes in the shuffles.
template <int kVectors>
__global__ __launch_bounds__(kThreadsPerRow * kRowsPerBlock) void layernorm_rowreg_bf16_kernel(
    __nv_bfloat16* output, const __nv_bfloat16* input, const __nv_bfloat16* __restrict__ gamma,
    const __nv_bfloat16* __restrict__ beta, int rows, float epsilon) {
  const int sub = threadIdx.x % kThreadsPerRow;
  const int row = blockIdx.x * kRowsPerBlock + threadIdx.x / kThreadsPerRow;
  const bool live = row < rows;
  uint4 xv[kVectors], gv[kVectors], bv[kVectors];
  load_row_slice<kVectors>(xv, gv, bv, input, gamma, beta, live ? row : rows - 1, sub);
  float mean, rstd;
  // the mask is computed from blockDim on purpose: a literal full mask measures 4% slower here (1568 x 1024)
  row_statistics<kVectors>(xv, 0xffffffffu >> (32 - static_cast<int>(blockDim.x)), epsilon, mean, rstd);
  if (!live) return;
  uint4* y = reinterpret_cast<uint4*>(output + static_cast<size_t>(row) * kVectors * 128) + sub;
#pragma unroll
  for (int i = 0; i < kVectors; ++i) y[kThreadsPerRow * i] = normalize_bf16x8(xv[i], gv[i], bv[i], mean, rstd);
}

template <int kVectors>
__global__ __launch_bounds__(kThreadsPerRow * kQuantRowsPerBlock) void layernorm_quant_mxfp8_rowreg_kernel(
    uint8_t* __restrict__ quantized, uint32_t* __restrict__ scales, const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ gamma, const __nv_bfloat16* __restrict__ beta, int rows, float epsilon) {
  using namespace xinfer::mxfp8;
  const int sub = threadIdx.x % kThreadsPerRow;
  const int row = blockIdx.x * kQuantRowsPerBlock + threadIdx.x / kThreadsPerRow;
  const bool live = row < rows;
  uint4 xv[kVectors], gv[kVectors], bv[kVectors];
  load_row_slice<kVectors>(xv, gv, bv, input, gamma, beta, live ? row : rows - 1, sub);
  float mean, rstd;
  row_statistics<kVectors>(xv, 0xffffffffu, epsilon, mean, rstd);
  uint2* q = reinterpret_cast<uint2*>(quantized + static_cast<size_t>(row) * kVectors * 128) + sub;
#pragma unroll
  for (int i = 0; i < kVectors; ++i) {
    float y[8];
    xinfer::normalization::unpack_bf16x8(normalize_bf16x8(xv[i], gv[i], bv[i], mean, rstd), y);
    float local_max = 0.0f;
#pragma unroll
    for (int j = 0; j < 8; ++j) local_max = fmaxf(local_max, fabsf(y[j]));
    const uint8_t scale = amax_to_e8m0_ceil(half_warp_reduce_max(local_max));
    const uint2 packed = quantize_e4m3x8(y, e8m0_inv(scale));
    if (!live) continue;
    q[kThreadsPerRow * i] = packed;
    // the four 1x32 slots of tile i are four consecutive bytes
    if (sub == 0) scales[scale_offset(row, 4 * i, kVectors) / 4] = 0x01010101u * scale;
  }
}

// The NVFP4 output of layernorm_quant_nvfp4_rowreg_kernel: data uint8 [rows, K / 2], E4M3 SWIZZLE_32_4_4 scales of
// the rows padded to 128.
struct Nvfp4Out {
  uint8_t* data = nullptr;
  uint8_t* scales = nullptr;
  int padded_rows = 0;
};

// The grid covers the rows padded to 128: the threads of a padding row write its zero scale bytes. `output` may be
// null (no BF16 y written).
template <int kVectors>
__global__ __launch_bounds__(kThreadsPerRow * kRowsPerBlock) void layernorm_quant_nvfp4_rowreg_kernel(
    __nv_bfloat16* output, const __nv_bfloat16* input, const __nv_bfloat16* __restrict__ gamma,
    const __nv_bfloat16* __restrict__ beta, int rows, float epsilon, Nvfp4Out nvfp4) {
  using xinfer::normalization::unpack_bf16x8;
  constexpr int kK = kVectors * 128;
  constexpr int kScaleColBlocks = kK / 64;
  const int sub = threadIdx.x % kThreadsPerRow;
  const int row = blockIdx.x * kRowsPerBlock + threadIdx.x / kThreadsPerRow;
  const bool live = row < rows;
  uint4 xv[kVectors], gv[kVectors], bv[kVectors];
  load_row_slice<kVectors>(xv, gv, bv, input, gamma, beta, live ? row : rows - 1, sub);
  float mean, rstd;
  row_statistics<kVectors>(xv, 0xffffffffu >> (32 - static_cast<int>(blockDim.x)), epsilon, mean, rstd);
  if (!live) {
    if (row < nvfp4.padded_rows && sub % 2 == 0) {  // the 16-column blocks of vectors sub, sub + 16, ...
      for (int i = 0; i < kVectors; ++i) {
        nvfp4.scales[xinfer::nvfp4::ScaleOffset(row, (sub + kThreadsPerRow * i) / 2, kScaleColBlocks)] = 0;
      }
    }
    return;
  }
  uint4* y = nullptr;
  if (output != nullptr) y = reinterpret_cast<uint4*>(output + static_cast<size_t>(row) * kK) + sub;
#pragma unroll
  for (int i = 0; i < kVectors; ++i) {
    const uint4 result = normalize_bf16x8(xv[i], gv[i], bv[i], mean, rstd);
    const int v = sub + kThreadsPerRow * i;  // with its neighbour v ^ 1 (lane sub ^ 1, same row) one 16-column block
    float h[8];
    unpack_bf16x8(result, h);
    float amax = 0.0f;
#pragma unroll
    for (int j = 0; j < 8; ++j) amax = fmaxf(amax, fabsf(h[j]));
    amax = fmaxf(amax, __shfl_xor_sync(__activemask(), amax, 1));
    const float scale = xinfer::nvfp4::AmaxScale(amax, 1.0f);
    if (v % 2 == 0) {
      nvfp4.scales[xinfer::nvfp4::ScaleOffset(row, v / 2, kScaleColBlocks)] = xinfer::nvfp4::ScaleBits(scale);
    }
    *reinterpret_cast<uint32_t*>(nvfp4.data + static_cast<size_t>(row) * (kK / 2) + v * 4) =
        xinfer::nvfp4::Pack8(h, scale);
    if (y != nullptr) y[kThreadsPerRow * i] = result;
  }
}

}  // namespace xinfer_layernorm::rowreg
