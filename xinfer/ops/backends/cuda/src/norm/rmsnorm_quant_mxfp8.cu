// RMSNorm (optionally after residual-add) followed by MXFP8 or row-FP8 quantization.

#include "quant/mxfp8_format_utils.cuh"
#include "rmsnorm_quant_mxfp8_vec16.cuh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <type_traits>

namespace {

using xinfer::mxfp8::block_reduce_sum;
using xinfer::mxfp8::e8m0_inv;
using xinfer::mxfp8::amax_to_e8m0_ceil;
using xinfer::mxfp8::kFp8Max;
using xinfer::mxfp8::kProgram;
using xinfer::mxfp8::scale_offset;
using xinfer::mxfp8::warp_reduce_max32;

__device__ __forceinline__ float weight_to_float(__nv_bfloat16 w) { return __bfloat162float(w); }
__device__ __forceinline__ float weight_to_float(float w) { return w; }

// One CTA processes one row. WeightT: __nv_bfloat16 or float.
template <typename WeightT>
__global__ __launch_bounds__(128) void rmsnorm_quant_mxfp8_kernel(
    const __nv_bfloat16* input, const __nv_bfloat16* residual,
    __nv_bfloat16* residual_output,
    const WeightT* __restrict__ weight,
    __nv_fp8_e4m3* __restrict__ quantized, uint8_t* __restrict__ scales,
    int rows, int row_width, int logical_k, int input_stride,
    int residual_stride, int residual_output_stride, int padded_k,
    int k_tiles, int scale_col_blocks, float epsilon, bool add_residual,
    bool share_scale_128, bool round_to_bf16, bool zero_centered) {
  constexpr int kThreads = 128;
  int row = blockIdx.x;
  float sum = 0.0f;
  for (int column = threadIdx.x; column < logical_k; column += kThreads) {
    float value =
        __bfloat162float(input[(size_t)row * input_stride + column]);
    if (add_residual) {
      value +=
          __bfloat162float(residual[(size_t)row * residual_stride + column]);
    }
    sum += value * value;
  }
  float total = block_reduce_sum<kThreads>(sum);
  float inverse_rms =
      rsqrtf(total / static_cast<float>(logical_k) + epsilon);

  __shared__ float group_absolute_max[4];
  __shared__ uint8_t shared_scales[4];
  for (int tile = 0; tile < k_tiles; ++tile) {
    int base = tile * kProgram;
    int group = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int column = base + threadIdx.x;
    float output = 0.0f;
    if (column < logical_k) {
      float value =
          __bfloat162float(input[(size_t)row * input_stride + column]);
      if (add_residual) {
        value += __bfloat162float(
            residual[(size_t)row * residual_stride + column]);
      }
      float w = weight_to_float(weight[column]);
      if (zero_centered) w += 1.0f;
      output = value * inverse_rms * w;
      if (round_to_bf16) output = __bfloat162float(__float2bfloat16(output));
    }
    float absolute_max = warp_reduce_max32(fabsf(output));
    if (lane == 0) group_absolute_max[group] = absolute_max;
    __syncthreads();
    if (threadIdx.x < 4) {
      float selected =
          share_scale_128
              ? fmaxf(fmaxf(group_absolute_max[0], group_absolute_max[1]),
                      fmaxf(group_absolute_max[2], group_absolute_max[3]))
              : group_absolute_max[threadIdx.x];
      shared_scales[threadIdx.x] = amax_to_e8m0_ceil(selected);
    }
    __syncthreads();
    uint8_t scale = share_scale_128 ? shared_scales[0]
                                    : shared_scales[group];
    float inverse_scale = e8m0_inv(scale);
    if (column < padded_k) {
      quantized[(size_t)row * padded_k + column] = __nv_fp8_e4m3(
          fminf(kFp8Max, fmaxf(-kFp8Max, output * inverse_scale)));
    }
    if (threadIdx.x < 4) {
      int k_block = tile * 4 + lane;
      if (k_block < scale_col_blocks * 4) {
        scales[scale_offset(row, k_block, scale_col_blocks)] =
            shared_scales[lane];
      }
    }
    __syncthreads();
  }

  // Write the residual result only after every quantization tile has consumed
  // the original input and residual. Downstream adapters intentionally allow
  // residual_output to alias either operand for sliced in-place updates.
  if (add_residual) {
    for (int column = threadIdx.x; column < row_width;
         column += kThreads) {
      float value =
          __bfloat162float(input[(size_t)row * input_stride + column]) +
          __bfloat162float(
              residual[(size_t)row * residual_stride + column]);
      residual_output[(size_t)row * residual_output_stride + column] =
          __float2bfloat16(value);
    }
  }
}

}  // namespace

namespace xinfer::mxfp8_rmsnorm_quant {

namespace vec16 = xinfer::rmsnorm_quant_mxfp8_vec16;

// Backend selector of the full-parameter entry rmsnorm_quant_mxfp8_into.
constexpr int64_t kBackendAuto = 0;
constexpr int64_t kBackendGeneric = 1;
constexpr int64_t kBackendVec16 = 2;

// Shape envelope of the vectorized variant (see rmsnorm_quant_mxfp8_vec16.cuh).
bool Vec16ShapeSupported(int64_t row_width, int64_t logical_k,
                         int64_t padded_k) {
  return logical_k == row_width && logical_k == padded_k &&
         logical_k % kProgram == 0 && logical_k >= kProgram &&
         logical_k <= vec16::kMaxK;
}

// Auto dispatch takes the vectorized variant for every shape inside its envelope (when the tensors are 16-byte
// aligned), on every architecture: it is plain CUDA. It is not bitwise identical to the generic kernel (its sum
// of squares is ordered for 16-byte loads); both are deterministic.
int64_t SelectBackend(int64_t row_width, int64_t logical_k, int64_t padded_k,
                      bool aligned, int64_t requested) {
  const bool shape_ok = Vec16ShapeSupported(row_width, logical_k, padded_k);
  if (requested == kBackendGeneric) return kBackendGeneric;
  if (requested == kBackendVec16) {
    if (!shape_ok || !aligned) {
      throw std::runtime_error(
          "rmsnorm_quant_mxfp8: vec16 backend requires K == row width == "
          "padded K, K % 128 == 0, 128 <= K <= 4096 and 16-byte aligned "
          "rows; got row width " +
          std::to_string(row_width) + ", K " + std::to_string(logical_k) +
          ", padded K " + std::to_string(padded_k) +
          (aligned ? "" : ", misaligned rows"));
    }
    return kBackendVec16;
  }
  if (requested != kBackendAuto) {
    throw std::runtime_error(
        "rmsnorm_quant_mxfp8: backend must be 0 (auto), 1 (generic) or 2 "
        "(vec16), got " + std::to_string(requested));
  }
  return shape_ok && aligned ? kBackendVec16 : kBackendGeneric;
}

// f(std::true_type{}) or f(std::false_type{}): turns a runtime flag into a template argument.
template <typename F>
void WithBool(bool flag, F&& f) {
  if (flag) {
    f(std::true_type{});
  } else {
    f(std::false_type{});
  }
}

bool Aligned16(const tvm::ffi::TensorView& tensor) {
  return reinterpret_cast<uintptr_t>(tensor.data_ptr()) % 16 == 0 &&
         (tensor.ndim() < 2 || tensor.stride(0) % 8 == 0);
}

// weight is [K] or [R, K] with R identical rows, R a power of two <= 64 (see the vec16 header). Returns K.
int64_t WeightWidth(const tvm::ffi::TensorView& weight) {
  if (weight.ndim() == 1) return weight.size(0);
  if (weight.ndim() == 2) {
    const int64_t replicas = weight.size(0);
    if (replicas < 1 || replicas > 64 || (replicas & (replicas - 1)) != 0) {
      throw std::runtime_error(
          "weight replicas R must be a power of two <= 64, got " +
          std::to_string(replicas));
    }
    return weight.size(1);
  }
  throw std::runtime_error("weight must be [K] or [R, K]");
}

int64_t WeightReplicas(const tvm::ffi::TensorView& weight) {
  return weight.ndim() == 2 ? weight.size(0) : 1;
}

void CheckDenseRows(const tvm::ffi::TensorView& tensor, const char* name) {
  if (tensor.ndim() != 2 || tensor.stride(1) != 1 ||
      tensor.stride(0) < tensor.size(1)) {
    throw std::runtime_error(
        std::string(name) +
        " must be 2D with a contiguous inner dimension and valid row stride");
  }
}

void CheckPointerAlignment(const tvm::ffi::TensorView& tensor,
                           uintptr_t alignment, const char* name) {
  if (reinterpret_cast<uintptr_t>(tensor.data_ptr()) % alignment != 0) {
    throw std::runtime_error(std::string(name) + " pointer is misaligned");
  }
}

void Run(tvm::ffi::TensorView input, tvm::ffi::TensorView residual,
         tvm::ffi::TensorView residual_output, tvm::ffi::TensorView weight,
         tvm::ffi::TensorView quantized, tvm::ffi::TensorView scales,
         int64_t logical_k, double epsilon, bool add_residual,
         bool share_scale_128, bool round_to_bf16, bool zero_centered,
         int64_t backend) {
  using xinfer::mxfp8::CheckedInt;
  using xinfer::mxfp8::CheckCommonOutput;

  xinfer::ffi::CheckCuda(input, "input");
  xinfer::ffi::CheckBFloat16(input, "input");
  CheckDenseRows(input, "input");
  xinfer::ffi::CheckCuda(weight, "weight");
  const bool weight_fp32 = xinfer::ffi::IsFloat32(weight.dtype());
  if (!weight_fp32) xinfer::ffi::CheckBFloat16(weight, "weight");  // BF16 or FP32
  xinfer::ffi::CheckContiguous(weight, "weight");
  xinfer::ffi::CheckSameDevice(weight, input, "weight");
  CheckCommonOutput(quantized, input, "quantized");
  CheckCommonOutput(scales, input, "scales");
  xinfer::ffi::CheckFloat8E4M3Storage(quantized, "quantized");
  if (input.ndim() != 2 || quantized.ndim() != 2 ||
      quantized.size(0) != input.size(0)) {
    throw std::runtime_error("input/quantized shape mismatch");
  }
  const int rows = CheckedInt(input.size(0), "rows");
  const int row_width = CheckedInt(input.size(1), "input width");
  const int input_stride = CheckedInt(input.stride(0), "input row stride");
  const int logical_k_int = CheckedInt(logical_k, "logical K");
  const int padded_k = CheckedInt(quantized.size(1), "padded K");
  const float epsilon_fp32 =
      xinfer::mxfp8::CheckedPositiveFloat(epsilon, "epsilon");
  if (rows <= 0 || logical_k_int <= 0 || logical_k_int > row_width ||
      WeightWidth(weight) != logical_k || padded_k < logical_k_int ||
      padded_k % 32 != 0) {
    throw std::runtime_error("invalid MXFP8 RMSNorm shape or epsilon");
  }
  int k_blocks = padded_k / 32;
  int scale_columns = ((k_blocks + 3) / 4) * 4;
  int k_tiles = scale_columns / 4;
  xinfer::mxfp8::CheckScaleStorage(scales, rows, padded_k, "scales");
  CheckPointerAlignment(input, alignof(__nv_bfloat16), "input");
  CheckPointerAlignment(weight, weight_fp32 ? alignof(float) : alignof(__nv_bfloat16), "weight");
  xinfer::ffi::CheckNoOverlap(quantized, "quantized", scales, "scales");
  xinfer::ffi::CheckNoOverlap(quantized, "quantized", input, "input");
  xinfer::ffi::CheckNoOverlap(quantized, "quantized", weight, "weight");
  xinfer::ffi::CheckNoOverlap(scales, "scales", input, "input");
  xinfer::ffi::CheckNoOverlap(scales, "scales", weight, "weight");
  const __nv_bfloat16* residual_data = nullptr;
  __nv_bfloat16* residual_output_data = nullptr;
  int residual_stride = 0;
  int residual_output_stride = 0;
  if (add_residual) {
    xinfer::ffi::CheckCuda(residual, "residual");
    xinfer::ffi::CheckBFloat16(residual, "residual");
    CheckDenseRows(residual, "residual");
    xinfer::ffi::CheckSameDevice(residual, input, "residual");
    xinfer::ffi::CheckSameShape(residual, input, "residual");
    xinfer::ffi::CheckCuda(residual_output, "residual_output");
    xinfer::ffi::CheckSameDevice(
        residual_output, input, "residual_output");
    xinfer::ffi::CheckBFloat16(residual_output, "residual_output");
    CheckDenseRows(residual_output, "residual_output");
    xinfer::ffi::CheckSameShape(residual_output, input, "residual_output");
    residual_data = xinfer::ffi::ConstData<__nv_bfloat16>(residual);
    residual_output_data =
        xinfer::ffi::Data<__nv_bfloat16>(residual_output);
    residual_stride =
        CheckedInt(residual.stride(0), "residual row stride");
    residual_output_stride = CheckedInt(
        residual_output.stride(0), "residual_output row stride");
    CheckPointerAlignment(residual, alignof(__nv_bfloat16), "residual");
    CheckPointerAlignment(residual_output, alignof(__nv_bfloat16),
                          "residual_output");
    xinfer::ffi::CheckExactAliasOrNoOverlap(
        residual_output, "residual_output", input, "input");
    xinfer::ffi::CheckExactAliasOrNoOverlap(
        residual_output, "residual_output", residual, "residual");
    xinfer::ffi::CheckNoOverlap(residual_output, "residual_output", weight,
                                  "weight");
    xinfer::ffi::CheckNoOverlap(quantized, "quantized", residual,
                                  "residual");
    xinfer::ffi::CheckNoOverlap(quantized, "quantized", residual_output,
                                  "residual_output");
    xinfer::ffi::CheckNoOverlap(scales, "scales", residual, "residual");
    xinfer::ffi::CheckNoOverlap(scales, "scales", residual_output,
                                  "residual_output");
  }
  const bool aligned =
      Aligned16(input) && Aligned16(weight) && Aligned16(quantized) &&
      (!add_residual || (Aligned16(residual) && Aligned16(residual_output)));
  const int64_t selected = SelectBackend(row_width, logical_k_int, padded_k,
                                         aligned, backend);
  cudaStream_t stream = xinfer::ffi::Stream(input);
  if (selected == kBackendVec16) {
    const int threads_per_row = logical_k_int / vec16::kColumnsPerThread;
    const int rows_per_block = std::max(1, 32 / threads_per_row);
    const int threads = threads_per_row * rows_per_block;
    const int blocks = (rows + rows_per_block - 1) / rows_per_block;
    const size_t shared_bytes =
        static_cast<size_t>(threads / vec16::kThreadsPerTile) * sizeof(float);
    const uint32_t replica_mask =
        static_cast<uint32_t>(WeightReplicas(weight) - 1);
    auto launch = [&](auto kernel, auto weight_data) {
      kernel<<<blocks, threads, shared_bytes, stream>>>(
          xinfer::ffi::ConstData<__nv_bfloat16>(input), residual_data,
          residual_output_data, weight_data,
          xinfer::ffi::Data<uint8_t>(quantized),
          xinfer::ffi::Data<uint8_t>(scales), rows, threads_per_row,
          rows_per_block, input_stride, residual_stride,
          residual_output_stride, logical_k_int, scale_columns / 4,
          replica_mask, epsilon_fp32);
    };
    // one instantiation per (weight dtype, add_residual, share_scale_128, round_to_bf16, zero_centered)
    WithBool(weight_fp32, [&](auto fp32) {
      using WeightT = std::conditional_t<decltype(fp32)::value, float, __nv_bfloat16>;
      WithBool(add_residual, [&](auto add) {
        WithBool(share_scale_128, [&](auto share) {
          WithBool(round_to_bf16, [&](auto round) {
            WithBool(zero_centered, [&](auto zc) {
              launch(vec16::rmsnorm_quant_mxfp8_vec16_kernel<WeightT, decltype(add)::value, decltype(share)::value,
                                                             decltype(round)::value, decltype(zc)::value>,
                     xinfer::ffi::ConstData<WeightT>(weight));
            });
          });
        });
      });
    });
    xinfer::ffi::CheckLastCudaError("rmsnorm_quant_mxfp8 vec16 launch");
    return;
  }
  auto launch_generic = [&](auto weight_data) {
    rmsnorm_quant_mxfp8_kernel<<<rows, 128, 0, stream>>>(
        xinfer::ffi::ConstData<__nv_bfloat16>(input), residual_data,
        residual_output_data, weight_data,
        xinfer::ffi::Data<__nv_fp8_e4m3>(quantized),
        xinfer::ffi::Data<uint8_t>(scales), rows, row_width, logical_k_int,
        input_stride, residual_stride, residual_output_stride, padded_k,
        k_tiles, scale_columns / 4, epsilon_fp32, add_residual,
        share_scale_128, round_to_bf16, zero_centered);
  };
  if (weight_fp32) {
    launch_generic(xinfer::ffi::ConstData<float>(weight));
  } else {
    launch_generic(xinfer::ffi::ConstData<__nv_bfloat16>(weight));
  }
  xinfer::ffi::CheckLastCudaError("rmsnorm_quant_mxfp8 launch");
}

}  // namespace xinfer::mxfp8_rmsnorm_quant

TVM_FFI_DLL_EXPORT_TYPED_FUNC(
    rmsnorm_quant_mxfp8_into, xinfer::mxfp8_rmsnorm_quant::Run);
