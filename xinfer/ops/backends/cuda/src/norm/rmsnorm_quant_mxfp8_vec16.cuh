// Vectorized variant of rmsnorm_quant_mxfp8 (RMSNorm, optional residual add, MXFP8 E4M3 output, E8M0 scales in
// the SWIZZLE_32_4_4 layout of xinfer::mxfp8::scale_offset).
//
// Work split: every thread owns 16 consecutive columns (two 16-byte loads of the row, of the residual and of the
// weight, one 16-byte store of the quantized values); a row uses K / 16 threads and a block holds whole rows.
// A 32-column scale group is 2 threads and a 128-column tile 8 aligned threads, so all maxima are warp shuffles.
//
// Numerics (same formulas as the generic kernel, different summation order):
//   per-thread sum of squares over its 16 columns in column order (FFMA), xor tree over the 8 threads of a
//   128-column tile, then the tiles of the row added in column order; inverse_rms = rsqrt(sum / K + eps);
//   y = (x [+ residual]) * inverse_rms * w (w + 1 in FP32 when zero-centered; optionally rounded to BF16);
//   E8M0 from the exact amax rule;
//   q = e4m3_satfinite(max(-448, y * 2^-e)), which equals the generic clamp-then-convert for every input.
// The order depends only on K, so results are deterministic and independent of the number of rows (batch
// invariant). Against the generic kernel only the last bits of the sum differ.
//
// Weight replicas: every block reads the whole weight row at once, so with one copy all SMs hit the same L2 lines
// (measured +0.4 us at 2112 x 1280). The weight may be passed as R identical rows (R a power of two, prepared once
// by the caller); a block on SM s reads row s & (R - 1).
//
// Shape envelope: K = logical_k = row width = padded K, K % 128 == 0, 128 <= K <= 4096, any number of rows;
// 16-byte aligned input / residual / residual_output / weight rows and quantized rows (checked by the host).
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <cstdint>
#include <type_traits>

#include "quant/mxfp8_format_core.cuh"
#include "bf16_vector.cuh"

namespace xinfer::rmsnorm_quant_mxfp8_vec16 {

inline constexpr int kColumnsPerThread = 16;
inline constexpr int kThreadsPerTile = 8;   // 128 columns
inline constexpr int kMaxBlockThreads = 256;
inline constexpr int kMaxK = kMaxBlockThreads * kColumnsPerThread;  // 4096

using xinfer::normalization::pack_bf16x2;
using xinfer::normalization::unpack_bf16x8;

// e4m3(low) in the low byte, e4m3(high) in the high byte, round-to-nearest-even with saturation to +-448
// (one cvt.rn.satfinite.e4m3x2.f32 on SM89+, the cuda_fp8.h software path on older targets).
__device__ __forceinline__ uint32_t e4m3x2_satfinite(float high, float low) {
  return __nv_cvt_float2_to_fp8x2(make_float2(low, high), __NV_SATFINITE, __NV_E4M3);
}

// WeightT: __nv_bfloat16 or float (one 16-column slice is two or four 16-byte loads).
template <typename WeightT, bool kAddResidual, bool kShareScale128, bool kRoundToBf16, bool kZeroCentered>
__global__ __launch_bounds__(kMaxBlockThreads) void rmsnorm_quant_mxfp8_vec16_kernel(
    const __nv_bfloat16* input, const __nv_bfloat16* residual,
    __nv_bfloat16* residual_output,
    const WeightT* __restrict__ weight,
    uint8_t* __restrict__ quantized, uint8_t* __restrict__ scales, int rows,
    int threads_per_row, int rows_per_block, int input_stride,
    int residual_stride, int residual_output_stride, int logical_k,
    int scale_col_blocks, uint32_t weight_replica_mask, float epsilon) {
  using xinfer::mxfp8::amax_to_e8m0_ceil;
  using xinfer::mxfp8::e8m0_inv;
  using xinfer::mxfp8::kFp8Max;
  using xinfer::mxfp8::scale_offset;
  extern __shared__ float shared_tile_sum[];  // one per 8 threads

  const int thread = threadIdx.x;
  // A block of 80 threads (K = 1280) ends with a partial warp: shuffle only over the lanes that exist. Shuffles
  // never leave an aligned group of 8 threads, and every group is complete (K / 16 and blocks are multiples of 8).
  const int warp_lanes = min(32, static_cast<int>(blockDim.x) - (thread & ~31));
  const uint32_t lanes = warp_lanes == 32 ? 0xffffffffu : (1u << warp_lanes) - 1u;
  const int local_row = thread / threads_per_row;
  const int t = thread - local_row * threads_per_row;  // thread within the row
  const int row = blockIdx.x * rows_per_block + local_row;
  const bool live = row < rows;  // dead rows still join the shuffles and the barrier
  const int column = t * kColumnsPerThread;

  float value[kColumnsPerThread];
  float weight_value[kColumnsPerThread];
  {
    uint32_t sm;
    asm("mov.u32 %0, %%smid;" : "=r"(sm));
    const WeightT* w = weight + static_cast<size_t>(sm & weight_replica_mask) * logical_k + column;
    if constexpr (std::is_same_v<WeightT, float>) {
      const float4* w4 = reinterpret_cast<const float4*>(w);
#pragma unroll
      for (int i = 0; i < kColumnsPerThread / 4; ++i) {
        const float4 v = w4[i];
        weight_value[4 * i] = v.x;
        weight_value[4 * i + 1] = v.y;
        weight_value[4 * i + 2] = v.z;
        weight_value[4 * i + 3] = v.w;
      }
    } else {
      const uint4* w8 = reinterpret_cast<const uint4*>(w);
      unpack_bf16x8(w8[0], weight_value);
      unpack_bf16x8(w8[1], weight_value + 8);
    }
    if constexpr (kZeroCentered) {
#pragma unroll
      for (int i = 0; i < kColumnsPerThread; ++i) weight_value[i] += 1.0f;
    }
  }
  if (live) {
    const uint4* x = reinterpret_cast<const uint4*>(
        input + static_cast<size_t>(row) * input_stride + column);
    const uint4 x0 = x[0];
    const uint4 x1 = x[1];
    unpack_bf16x8(x0, value);
    unpack_bf16x8(x1, value + 8);
    if constexpr (kAddResidual) {
      const uint4* r = reinterpret_cast<const uint4*>(
          residual + static_cast<size_t>(row) * residual_stride + column);
      float added[kColumnsPerThread];
      unpack_bf16x8(r[0], added);
      unpack_bf16x8(r[1], added + 8);
#pragma unroll
      for (int i = 0; i < kColumnsPerThread; ++i) value[i] += added[i];
      // Every read of these columns is done, so residual_output may alias input or residual exactly.
      uint4 packed[2];
      uint32_t* words = reinterpret_cast<uint32_t*>(packed);
#pragma unroll
      for (int i = 0; i < kColumnsPerThread / 2; ++i) {
        words[i] = pack_bf16x2(value[2 * i], value[2 * i + 1]);
      }
      uint4* out = reinterpret_cast<uint4*>(
          residual_output + static_cast<size_t>(row) * residual_output_stride + column);
      out[0] = packed[0];
      out[1] = packed[1];
    }
  } else {
#pragma unroll
    for (int i = 0; i < kColumnsPerThread; ++i) value[i] = 0.0f;
  }

  float sum = 0.0f;
#pragma unroll
  for (int i = 0; i < kColumnsPerThread; ++i) sum += value[i] * value[i];
  // xor tree inside the 8 aligned threads of a 128-column tile; every lane ends with the same bits
  sum += __shfl_xor_sync(lanes, sum, 1);
  sum += __shfl_xor_sync(lanes, sum, 2);
  sum += __shfl_xor_sync(lanes, sum, 4);
  if ((thread & (kThreadsPerTile - 1)) == 0) shared_tile_sum[thread / kThreadsPerTile] = sum;
  __syncthreads();
  const int tiles = threads_per_row / kThreadsPerTile;
  const float* row_tiles = shared_tile_sum + local_row * tiles;
  float total = 0.0f;
  for (int i = 0; i < tiles; ++i) total += row_tiles[i];
  const float inverse_rms = rsqrtf(total / static_cast<float>(logical_k) + epsilon);

#pragma unroll
  for (int i = 0; i < kColumnsPerThread; ++i) {
    float y = value[i] * inverse_rms * weight_value[i];
    if constexpr (kRoundToBf16) y = __bfloat162float(__float2bfloat16(y));
    value[i] = y;
  }
  // Seeded with a real element (not 0) so an all-NaN group keeps a NaN maximum (E8M0 255), as the generic kernel.
  float amax = fabsf(value[0]);
#pragma unroll
  for (int i = 1; i < kColumnsPerThread; ++i) amax = fmaxf(amax, fabsf(value[i]));
  amax = fmaxf(amax, __shfl_xor_sync(lanes, amax, 1));  // 32-column group
  if constexpr (kShareScale128) {
    amax = fmaxf(amax, __shfl_xor_sync(lanes, amax, 2));
    amax = fmaxf(amax, __shfl_xor_sync(lanes, amax, 4));  // 128-column tile
  }
  const uint8_t scale = amax_to_e8m0_ceil(amax);
  if (!live) return;
  if ((t & 1) == 0) scales[scale_offset(row, t >> 1, scale_col_blocks)] = scale;
  const float inverse_scale = e8m0_inv(scale);
  uint4 packed;
  uint32_t* words = reinterpret_cast<uint32_t*>(&packed);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const uint32_t low = e4m3x2_satfinite(fmaxf(-kFp8Max, value[4 * i + 1] * inverse_scale),
                                          fmaxf(-kFp8Max, value[4 * i] * inverse_scale));
    const uint32_t high = e4m3x2_satfinite(fmaxf(-kFp8Max, value[4 * i + 3] * inverse_scale),
                                           fmaxf(-kFp8Max, value[4 * i + 2] * inverse_scale));
    words[i] = low | (high << 16);
  }
  *reinterpret_cast<uint4*>(quantized + static_cast<size_t>(row) * logical_k + column) = packed;
}

}  // namespace xinfer::rmsnorm_quant_mxfp8_vec16
