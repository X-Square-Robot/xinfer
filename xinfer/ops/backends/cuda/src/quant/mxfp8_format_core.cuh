#pragma once

#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <climits>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace xinfer::mxfp8 {

inline constexpr float kFp8Max = 448.0f;
inline constexpr int kProgram = 128;  // logical 1x128 block width

inline int CheckedInt(int64_t value, const char* name) {
  if (value < 0 || value > INT_MAX) {
    throw std::runtime_error(std::string(name) + " is outside int32 range");
  }
  return static_cast<int>(value);
}

inline float CheckedPositiveFloat(double value, const char* name) {
  const float narrowed = static_cast<float>(value);
  if (!std::isfinite(value) || !std::isfinite(narrowed) || narrowed <= 0.0f) {
    throw std::runtime_error(std::string(name) +
                             " must be finite, positive, and representable "
                             "in FP32");
  }
  return narrowed;
}

inline int64_t CheckedMultiply(int64_t lhs, int64_t rhs,
                               const char* name) {
  if (lhs < 0 || rhs < 0 ||
      (rhs != 0 && lhs > std::numeric_limits<int64_t>::max() / rhs)) {
    throw std::runtime_error(std::string(name) + " overflows int64");
  }
  return lhs * rhs;
}

inline int64_t RoundUp(int64_t value, int64_t alignment,
                       const char* name) {
  if (value < 0 || alignment <= 0 ||
      value > std::numeric_limits<int64_t>::max() - (alignment - 1)) {
    throw std::runtime_error(std::string(name) + " cannot be rounded safely");
  }
  return ((value + alignment - 1) / alignment) * alignment;
}

// One-byte E8M0 entries occupied by a SWIZZLE_32_4_4 scale tensor for a
// logical [rows, K] matrix. Both dimensions are padded by the native layout.
inline int64_t RequiredScaleElements(int64_t rows, int64_t k) {
  if (rows <= 0 || k <= 0) {
    throw std::runtime_error("MXFP8 scale dimensions must be positive");
  }
  const int64_t padded_rows = RoundUp(rows, 128, "MXFP8 scale rows");
  const int64_t k_blocks = k / 32 + (k % 32 != 0);
  const int64_t scale_columns =
      RoundUp(k_blocks, 4, "MXFP8 scale columns");
  return CheckedMultiply(padded_rows, scale_columns,
                         "MXFP8 scale storage");
}

// Exact finite E8M0 ceil scale for E4M3 quantization. Comparing the FP32
// mantissa with the exact 1.75 boundary remains stable under --use_fast_math.
__device__ __forceinline__ uint8_t amax_to_e8m0_ceil(float amax) {
  const uint32_t bits = __float_as_uint(amax);
  const uint32_t exponent = (bits >> 23) & 0xffu;
  const uint32_t mantissa = bits & 0x7fffffu;
  int biased = static_cast<int>(exponent) - 8 +
               (mantissa > 0x600000u);
  biased = max(0, min(254, biased));
  return exponent == 0xffu ? 255 : static_cast<uint8_t>(biased);
}

// Reciprocal of the E8M0 scale used to map values into the E4M3 range.
__device__ __forceinline__ float e8m0_inv(uint8_t biased) {
  return exp2f(127.0f - static_cast<float>(biased));
}

// value / scale clamped to the E4M3 range and rounded to nearest even (NaN clamps to -448).
__device__ __forceinline__ __nv_fp8_e4m3 quantize_e4m3(float value,
                                                      float inverse_scale) {
  return __nv_fp8_e4m3(fminf(kFp8Max, fmaxf(-kFp8Max, value * inverse_scale)));
}

// Eight values to eight E4M3 bytes, in column order (low byte first).
__device__ __forceinline__ uint2 quantize_e4m3x8(const float* values,
                                                 float inverse_scale) {
  uint32_t words[2] = {0u, 0u};
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    words[i / 4] |=
        static_cast<uint32_t>(quantize_e4m3(values[i], inverse_scale).__x)
        << (8 * (i % 4));
  }
  return make_uint2(words[0], words[1]);
}

// Maximum over the 16-lane half of a warp that holds the calling thread.
__device__ __forceinline__ float half_warp_reduce_max(float value) {
#pragma unroll
  for (int mask = 8; mask > 0; mask >>= 1) {
    value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, mask));
  }
  return value;
}

// Physical offset of scale[row, k_block] in SWIZZLE_32_4_4 layout.
__device__ __forceinline__ int64_t scale_offset(int row, int k_block,
                                                int scale_col_blocks) {
  const int row_in_block = row & 127;
  const int64_t swizzle_block =
      static_cast<int64_t>(row / 128) * scale_col_blocks + k_block / 4;
  return swizzle_block * 512 + (row_in_block & 31) * 16 +
         (row_in_block / 32) * 4 + (k_block & 3);
}

template <int Threads>
__device__ __forceinline__ float block_reduce_sum(float value) {
  __shared__ float shared[Threads];
  shared[threadIdx.x] = value;
  __syncthreads();
  for (int stride = Threads / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      shared[threadIdx.x] += shared[threadIdx.x + stride];
    }
    __syncthreads();
  }
  return shared[0];
}

__device__ __forceinline__ float warp_reduce_max32(float value) {
  constexpr unsigned kFullMask = 0xffffffffu;
  for (int offset = 16; offset > 0; offset >>= 1) {
    value = fmaxf(value, __shfl_down_sync(kFullMask, value, offset));
  }
  return value;
}

}  // namespace xinfer::mxfp8
