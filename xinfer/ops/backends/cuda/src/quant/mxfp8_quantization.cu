// Pure BF16-to-MXFP8 quantization.
//
// Produces E4M3 data with E8M0 block scales in the SWIZZLE_32_4_4 layout of
// block-scaled tensor-core GEMMs. ``share_scale_128`` collapses one logical scale
// across a 128-wide K block and replicates it into four native 1x32 slots.
// Two bitwise-identical backends: generic (any shape) and vec8 (1x128 scales,
// unpadded K % 128 == 0, aligned pointers).

#include "norm/bf16_vector.cuh"
#include "mxfp8_format_utils.cuh"

namespace {

using xinfer::mxfp8::e8m0_inv;
using xinfer::mxfp8::amax_to_e8m0_ceil;
using xinfer::mxfp8::half_warp_reduce_max;
using xinfer::mxfp8::kProgram;
using xinfer::mxfp8::quantize_e4m3;
using xinfer::mxfp8::quantize_e4m3x8;
using xinfer::mxfp8::scale_offset;
using xinfer::mxfp8::warp_reduce_max32;
using xinfer::normalization::unpack_bf16x8;

// One warp handles one 128-wide K tile of one row (4 elements per lane); the grid covers exactly
// padded_rows x k_tiles tiles.
__global__ __launch_bounds__(32) void quantize_mxfp8_warp_kernel(
    const __nv_bfloat16* __restrict__ input,
    __nv_fp8_e4m3* __restrict__ quantized, uint8_t* __restrict__ scales,
    int rows, int logical_k, int padded_k, int k_tiles, bool share_scale_128) {
  int tile_id = blockIdx.x;
  int row = tile_id / k_tiles;
  int tile = tile_id - row * k_tiles;
  int base = tile * kProgram;
  int lane = threadIdx.x;
  float values[4];
  float local_max = 0.0f;
#pragma unroll
  for (int item = 0; item < 4; ++item) {
    int column = base + lane + item * 32;
    float value = 0.0f;
    if (row < rows && column < logical_k) {
      value = __bfloat162float(input[(size_t)row * logical_k + column]);
    }
    values[item] = value;
    local_max = fmaxf(local_max, fabsf(value));
  }

  uint8_t scale_values[4];
  if (share_scale_128) {
    float absolute_max = warp_reduce_max32(local_max);
    uint8_t scale = amax_to_e8m0_ceil(absolute_max);
    scale = __shfl_sync(0xffffffffu, scale, 0);
#pragma unroll
    for (int item = 0; item < 4; ++item) scale_values[item] = scale;
  } else {
#pragma unroll
    for (int item = 0; item < 4; ++item) {
      float absolute_max = warp_reduce_max32(fabsf(values[item]));
      uint8_t scale = amax_to_e8m0_ceil(absolute_max);
      scale_values[item] = __shfl_sync(0xffffffffu, scale, 0);
    }
  }

#pragma unroll
  for (int item = 0; item < 4; ++item) {
    int column = base + lane + item * 32;
    if (column < padded_k) {
      quantized[(size_t)row * padded_k + column] =
          quantize_e4m3(values[item], e8m0_inv(scale_values[item]));
    }
  }
  if (lane < 4) {
    scales[scale_offset(row, tile * 4 + lane, k_tiles)] = scale_values[lane];
  }
}

// 16 threads per 128-wide tile (8 elements each), 16 tiles per 256-thread block.
// Rows in [rows, padded_rows) are written as zeros with a zero scale, like the generic kernel.
__global__ __launch_bounds__(256) void quantize_mxfp8_vec8_kernel(
    const __nv_bfloat16* __restrict__ input, uint8_t* __restrict__ quantized,
    uint32_t* __restrict__ scales, int rows, int padded_rows, int k) {
  const int k_tiles = k / kProgram;
  const int64_t tile_id =
      static_cast<int64_t>(blockIdx.x) * 16 + (threadIdx.x >> 4);
  const bool live = tile_id < static_cast<int64_t>(padded_rows) * k_tiles;
  const int64_t id = live ? tile_id : 0;
  const int row = static_cast<int>(id / k_tiles);  // one 64-bit division
  const int tile = static_cast<int>(id - static_cast<int64_t>(row) * k_tiles);
  const size_t offset = static_cast<size_t>(row) * k + tile * kProgram +
                        (threadIdx.x & 15) * 8;
  uint4 raw = make_uint4(0u, 0u, 0u, 0u);
  if (row < rows) raw = *reinterpret_cast<const uint4*>(input + offset);
  float values[8];
  unpack_bf16x8(raw, values);
  float local_max = 0.0f;
#pragma unroll
  for (int i = 0; i < 8; ++i) local_max = fmaxf(local_max, fabsf(values[i]));
  // a tile's 16 threads are one half of a warp; inactive tiles join the shuffles
  const uint8_t scale = amax_to_e8m0_ceil(half_warp_reduce_max(local_max));
  const uint2 packed = quantize_e4m3x8(values, e8m0_inv(scale));
  if (!live) return;
  *reinterpret_cast<uint2*>(quantized + offset) = packed;
  if ((threadIdx.x & 15) == 0) {
    // the four 1x32 slots of a 128-wide tile are four consecutive bytes
    scales[scale_offset(row, tile * 4, k_tiles) / 4] = 0x01010101u * scale;
  }
}

}  // namespace

namespace xinfer_mxfp8_quantization {

constexpr int64_t kBackendAuto = 0;
constexpr int64_t kBackendGeneric = 1;
constexpr int64_t kBackendVec8 = 2;

// Resolves the requested backend (0 auto, 1 generic, 2 vec8); auto takes vec8 whenever it applies.
int64_t SelectBackend(int64_t logical_k, int64_t padded_k,
                      bool share_scale_128, bool aligned, int64_t requested) {
  const bool vec8_ok = share_scale_128 && logical_k == padded_k &&
                       logical_k % kProgram == 0 && aligned;
  if (requested == kBackendAuto) {
    return vec8_ok ? kBackendVec8 : kBackendGeneric;
  }
  if (requested == kBackendVec8 && !vec8_ok) {
    throw std::runtime_error(
        "quantize_mxfp8: vec8 backend requires share_scale_128, K == padded "
        "K, K % 128 == 0 and 16/8/4-byte aligned input/quantized/scales");
  }
  if (requested != kBackendGeneric && requested != kBackendVec8) {
    throw std::runtime_error(
        "quantize_mxfp8: backend must be 0 (auto), 1 (generic) or 2 (vec8), "
        "got " + std::to_string(requested));
  }
  return requested;
}

bool Aligned(const tvm::ffi::TensorView& tensor, uintptr_t bytes) {
  return reinterpret_cast<uintptr_t>(tensor.data_ptr()) % bytes == 0;
}

void QuantizeWithBackend(tvm::ffi::TensorView input,
                         tvm::ffi::TensorView quantized,
                         tvm::ffi::TensorView scales, bool share_scale_128,
                         int64_t backend) {
  using xinfer::mxfp8::CheckedInt;
  using xinfer::mxfp8::CheckCommonOutput;

  xinfer::ffi::CheckCuda(input, "input");
  xinfer::ffi::CheckBFloat16(input, "input");
  xinfer::ffi::CheckContiguous(input, "input");
  CheckCommonOutput(quantized, input, "quantized");
  CheckCommonOutput(scales, input, "scales");
  xinfer::ffi::CheckFloat8E4M3Storage(quantized, "quantized");
  if (input.ndim() != 2 || quantized.ndim() != 2) {
    throw std::runtime_error("input and quantized must be 2D");
  }
  const int rows = CheckedInt(input.size(0), "rows");
  const int padded_rows = CheckedInt(quantized.size(0), "padded rows");
  const int logical_k = CheckedInt(input.size(1), "logical K");
  const int padded_k = CheckedInt(quantized.size(1), "padded K");
  if (rows <= 0 || logical_k <= 0 || padded_rows < rows ||
      padded_k < logical_k || padded_k % 32 != 0) {
    throw std::runtime_error("invalid MXFP8 quantization shape or padding");
  }
  const int k_tiles = (padded_k / 32 + 3) / 4;  // 128-wide tiles = scale column blocks
  xinfer::mxfp8::CheckScaleStorage(
      scales, padded_rows, padded_k, "scales");
  const bool aligned =
      Aligned(input, 16) && Aligned(quantized, 8) && Aligned(scales, 4);
  cudaStream_t stream = xinfer::ffi::Stream(input);
  if (SelectBackend(logical_k, padded_k, share_scale_128, aligned, backend) ==
      kBackendVec8) {
    const int64_t blocks =
        (static_cast<int64_t>(padded_rows) * k_tiles + 15) / 16;
    if (blocks > INT_MAX) {
      throw std::runtime_error("quantize_mxfp8: too many 128-wide tiles");
    }
    quantize_mxfp8_vec8_kernel<<<static_cast<unsigned>(blocks), 256, 0,
                                 stream>>>(
        xinfer::ffi::ConstData<__nv_bfloat16>(input),
        xinfer::ffi::Data<uint8_t>(quantized),
        xinfer::ffi::Data<uint32_t>(scales), rows, padded_rows, padded_k);
  } else {
    quantize_mxfp8_warp_kernel<<<padded_rows * k_tiles, 32, 0, stream>>>(
        xinfer::ffi::ConstData<__nv_bfloat16>(input),
        xinfer::ffi::Data<__nv_fp8_e4m3>(quantized),
        xinfer::ffi::Data<uint8_t>(scales), rows, logical_k, padded_k,
        k_tiles, share_scale_128);
  }
  xinfer::ffi::CheckLastCudaError("quantize_mxfp8 launch");
}

void Quantize(tvm::ffi::TensorView input, tvm::ffi::TensorView quantized,
              tvm::ffi::TensorView scales, bool share_scale_128) {
  QuantizeWithBackend(input, quantized, scales, share_scale_128,
                      kBackendAuto);
}

}  // namespace xinfer_mxfp8_quantization

TVM_FFI_DLL_EXPORT_TYPED_FUNC(
    quantize_mxfp8_into, xinfer_mxfp8_quantization::Quantize);
