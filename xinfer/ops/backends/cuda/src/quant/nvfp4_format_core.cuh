#pragma once

// NVFP4: E2M1 elements packed two per byte (column 2j in the low nibble), one E4M3 scale per 16 columns of K in
// the SWIZZLE_32_4_4 storage of MXFP8 (rows padded to 128, 16-column blocks padded to 4; padding bytes zero), and
// an optional per-tensor FP32 global scale g: a value is e2m1 * scale * g.
//
// The arithmetic is that of the Wall-X NVFP4 kernels (built with --use_fast_math) and of the QAT fake quantizer
// trained against them, written with explicit instructions so it does not depend on the compile flags:
//   - a quotient a / b is a * rcp.approx.ftz(b);
//   - amax block scale: e4m3_rn(min(448, max(2^-6, amax / (6 g))));
//   - MSE block scale: of the E4M3 codes 1..126 (increasing), the first whose reconstruction
//     q(x / (s g)) * (s g) has the smallest error sum over the block, each error fma(-q, s g, x) squared and the
//     16 squares summed pairwise (8 + 8, 4 + 4, 2 + 2, 1 + 1);
//   - element: e2m1_rn_satfinite(x / (s g)), a negative value that rounds to zero keeps its sign.

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace xinfer::nvfp4 {

inline constexpr int kBlock = 16;
inline constexpr float kE2M1Max = 6.0f;
inline constexpr float kE4M3Max = 448.0f;
inline constexpr float kMinScale = 0.015625f;  // 2^-6, the smallest normal E4M3: the floor of an amax scale

// Bytes of the swizzled E4M3 scales of a logical [rows, k] matrix (k a multiple of 16), padding included.
inline int64_t ScaleStorageBytes(int64_t rows, int64_t k) {
  const int64_t padded_rows = (rows + 127) / 128 * 128;
  const int64_t blocks4 = (k / kBlock + 3) / 4 * 4;
  return padded_rows * blocks4;
}

// Physical offset of scale[row, k_block] in the SWIZZLE_32_4_4 storage; scale_col_blocks = ceil(k_blocks / 4).
__device__ __forceinline__ int64_t ScaleOffset(int64_t row, int k_block, int scale_col_blocks) {
  const int64_t row_in_block = row & 127;
  return ((row / 128) * scale_col_blocks + k_block / 4) * 512 + (row_in_block & 31) * 16 + (row_in_block / 32) * 4 +
         (k_block & 3);
}

// a / b as the fast-math kernels compute it: a * rcp.approx.ftz(b).
__device__ __forceinline__ float ApproxDiv(float a, float b) {
  float r;
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(b));
  return __fmul_rn(a, r);
}

// E4M3 (RN, saturating) of a positive block scale, as a float.
__device__ __forceinline__ float RoundScale(float raw) { return static_cast<float>(__nv_fp8_e4m3(raw)); }

__device__ __forceinline__ uint8_t ScaleBits(float scale) { return __nv_fp8_e4m3(scale).__x; }

// The amax block scale (E4M3 value) of a block with absolute maximum `amax` under global scale g.
__device__ __forceinline__ float AmaxScale(float amax, float g) {
  return RoundScale(fminf(kE4M3Max, fmaxf(kMinScale, ApproxDiv(amax, __fmul_rn(kE2M1Max, g)))));
}

// The E2M1 code of x: the nearest of 0, 0.5, 1, 1.5, 2, 3, 4, 6 (ties to the even code, saturated to 6) in bits 0-2,
// the sign in bit 3, set for every negative x (also when it rounds to zero): what cvt.rn.satfinite.e2m1x2.f32 gives for
// finite x. Comparisons only: that instruction exists on sm_100a / sm_120a alone, and cuda_fp4.h emulates it slowly
// elsewhere (the generic sources build for plain sm_120).
__device__ __forceinline__ uint32_t E2M1Code(float x) {
  const float a = fabsf(x);
  const uint32_t code = static_cast<uint32_t>(a > 0.25f) + static_cast<uint32_t>(a >= 0.75f) +
                        static_cast<uint32_t>(a > 1.25f) + static_cast<uint32_t>(a >= 1.75f) +
                        static_cast<uint32_t>(a > 2.5f) + static_cast<uint32_t>(a >= 3.5f) +
                        static_cast<uint32_t>(a > 5.0f);
  return code | (signbit(x) ? 8u : 0u);
}

// Two values as one byte of E2M1 codes: x in the low nibble.
__device__ __forceinline__ uint8_t PackPair(float x, float y) {
  return static_cast<uint8_t>(E2M1Code(x) | (E2M1Code(y) << 4));
}

// The E2M1 value nearest x (ties to even, saturated to +-6).
__device__ __forceinline__ float RoundE2M1(float x) {
  const uint32_t code = E2M1Code(x);
  const uint32_t magnitude = code & 7u;
  // 0.5 m for codes 0, 1; 2^(e - 1) (1 + 0.5 m) for exponent e = code >> 1
  const float value = magnitude < 2u ? 0.5f * static_cast<float>(magnitude)
                                     : __uint_as_float((126u + (magnitude >> 1)) << 23) *
                                           (1.0f + 0.5f * static_cast<float>(magnitude & 1u));
  return (code & 8u) ? -value : value;
}

// Eight values (x / decode, decode = scale * g) as four bytes, element 0 in the lowest nibble.
__device__ __forceinline__ uint32_t Pack8(const float (&v)[8], float decode) {
  uint32_t word = 0;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const uint32_t byte = PackPair(ApproxDiv(v[2 * i], decode), ApproxDiv(v[2 * i + 1], decode));
    word |= byte << (8 * i);
  }
  return word;
}

}  // namespace xinfer::nvfp4
