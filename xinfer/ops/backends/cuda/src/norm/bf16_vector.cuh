// BF16 <-> FP32 helpers for 16-byte vectorized normalization kernels.
#pragma once

#include <cuda_bf16.h>

#include <cstdint>

namespace xinfer::normalization {

// Eight BF16 values (one 16-byte vector) to FP32, in column order.
__device__ __forceinline__ void unpack_bf16x8(const uint4& packed, float* out) {
  const uint32_t words[4] = {packed.x, packed.y, packed.z, packed.w};
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    out[2 * i] = __uint_as_float(words[i] << 16);
    out[2 * i + 1] = __uint_as_float(words[i] & 0xffff0000u);
  }
}

// Two FP32 values to one BF16 pair (round to nearest even), low column in the low half.
__device__ __forceinline__ uint32_t pack_bf16x2(float low, float high) {
  const __nv_bfloat162 pair = __floats2bfloat162_rn(low, high);
  return *reinterpret_cast<const uint32_t*>(&pair);
}

}  // namespace xinfer::normalization
