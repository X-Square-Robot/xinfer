/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/optional.h>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <cstdint>
#include <cstdlib>
#include <initializer_list>
#include <type_traits>

#include "common/tvm_ffi_cuda_tensor.h"
#include "quant/nvfp4_format_core.cuh"

#include "common/cuda_utils.h"
#ifdef XINFER_WITH_SM110
#include "norm/sm110/bf16_rmsnorm_sm110.h"
#endif

namespace tvm_ffi_rmsnorm {

struct RMSNormShape {
  int64_t num_tokens;
  int64_t hidden_size;
  int64_t row_stride;
};

RMSNormShape GetRMSNormShape(const tvm::ffi::TensorView& tensor) {
  ASSERT_CHECK(tensor.ndim() >= 1);
  const int ndim = tensor.ndim();
  const int64_t hidden_size = tensor.size(ndim - 1);
  ASSERT_CHECK(hidden_size > 0);
  ASSERT_CHECK(tensor.stride(ndim - 1) == 1);

  int64_t num_tokens = 1;
  for (int i = 0; i < ndim - 1; ++i) {
    num_tokens *= tensor.size(i);
  }
  ASSERT_CHECK(num_tokens > 0);

  int64_t row_stride = hidden_size;
  if (ndim > 1) {
    row_stride = tensor.stride(ndim - 2);
    ASSERT_CHECK(row_stride >= hidden_size);
    int64_t expected_stride = row_stride;
    for (int i = ndim - 2; i >= 0; --i) {
      ASSERT_CHECK(tensor.stride(i) == expected_stride);
      expected_stride *= tensor.size(i);
    }
  }
  return {num_tokens, hidden_size, row_stride};
}

void CheckSameShape(const tvm::ffi::TensorView& lhs,
                    const tvm::ffi::TensorView& rhs) {
  ASSERT_CHECK(lhs.ndim() == rhs.ndim());
  for (int i = 0; i < lhs.ndim(); ++i) {
    ASSERT_CHECK(lhs.size(i) == rhs.size(i));
  }
}

bool IsFloat32(const DLDataType& dtype) {
  return dtype.code == kDLFloat && dtype.bits == 32;
}

bool IsFloat16(const DLDataType& dtype) {
  return dtype.code == kDLFloat && dtype.bits == 16;
}

bool IsBFloat16(const DLDataType& dtype) {
  return dtype.code == kDLBfloat && dtype.bits == 16;
}

int64_t GetVecSize(const DLDataType& dtype) {
  if (IsFloat32(dtype)) return VecTraits<float>::vec_size;
  if (IsFloat16(dtype)) return VecTraits<__half>::vec_size;
  if (IsBFloat16(dtype)) return VecTraits<__nv_bfloat16>::vec_size;
  throw std::runtime_error("Unsupported data type for RMSNorm");
}

template <typename T>
struct dependent_false : std::false_type {};

template<typename T>
__device__ __forceinline__ float ScalarToFloat(T value) {
  if constexpr (std::is_same_v<T, float>) {
    return value;
  } else if constexpr (std::is_same_v<T, __half>) {
    return __half2float(value);
  } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
    return __bfloat162float(value);
  } else {
    static_assert(dependent_false<T>::value, "Unsupported RMSNorm scalar type");
  }
}

template<typename T>
__device__ __forceinline__ float RoundFloatToScalarType(float value) {
  if constexpr (std::is_same_v<T, float>) {
    return value;
  } else if constexpr (std::is_same_v<T, __half>) {
    return __half2float(__float2half(value));
  } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
    return __bfloat162float(__float2bfloat16(value));
  } else {
    static_assert(dependent_false<T>::value, "Unsupported RMSNorm scalar type");
  }
}

template<typename T>
__device__ __forceinline__ void VecToFloats(
    typename VecTraits<T>::Type vec, float* values) {
  if constexpr (std::is_same_v<T, float>) {
    values[0] = vec.x;
    values[1] = vec.y;
    values[2] = vec.z;
    values[3] = vec.w;
  } else if constexpr (std::is_same_v<T, __half>) {
    values[0] = __half2float(vec.x);
    values[1] = __half2float(vec.y);
  } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
    values[0] = __bfloat162float(vec.x);
    values[1] = __bfloat162float(vec.y);
  } else {
    static_assert(dependent_false<T>::value, "Unsupported RMSNorm vector type");
  }
}

template<typename T>
__device__ __forceinline__ typename VecTraits<T>::Type FloatsToVec(const float* values) {
  typename VecTraits<T>::Type vec;
  if constexpr (std::is_same_v<T, float>) {
    vec.x = values[0];
    vec.y = values[1];
    vec.z = values[2];
    vec.w = values[3];
  } else if constexpr (std::is_same_v<T, __half>) {
    vec.x = __float2half(values[0]);
    vec.y = __float2half(values[1]);
  } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
    vec.x = __float2bfloat16(values[0]);
    vec.y = __float2bfloat16(values[1]);
  } else {
    static_assert(dependent_false<T>::value, "Unsupported RMSNorm vector type");
  }
  return vec;
}

template<typename WeightDType, int VEC_SIZE>
__device__ __forceinline__ void LoadWeightFloats(
    const WeightDType* weight, int64_t offset, float* values) {
  #pragma unroll
  for (int j = 0; j < VEC_SIZE; ++j) {
    values[j] = ScalarToFloat(weight[offset + j]);
  }
}

template<typename T>
__device__ __forceinline__ uint32_t PackVec2Bits(typename VecTraits<T>::Type vec) {
  static_assert(!std::is_same_v<T, float>, "PackVec2Bits is for 16-bit RMSNorm data");
  union Bits {
    typename VecTraits<T>::Type vec;
    uint32_t raw;
  };
  Bits bits;
  bits.vec = vec;
  return bits.raw;
}

template<typename T>
__device__ __forceinline__ typename VecTraits<T>::Type UnpackVec2Bits(uint32_t raw) {
  static_assert(!std::is_same_v<T, float>, "UnpackVec2Bits is for 16-bit RMSNorm data");
  union Bits {
    uint32_t raw;
    typename VecTraits<T>::Type vec;
  };
  Bits bits;
  bits.raw = raw;
  return bits.vec;
}

// ---------------- 16-byte wide access helpers (16-bit dtypes) ----------------
// The default VecTraits for half/bf16 are 4-byte (x2) vectors, which cannot
// saturate HBM on A100/H100-class parts. The wide fast paths below move data as
// 16-byte uint4 transactions (8 x 16-bit scalars) instead.

template<typename T>
__device__ __forceinline__ void Raw8ToFloats(uint4 raw, float* values) {
  static_assert(!std::is_same_v<T, float>, "Raw8ToFloats is for 16-bit data");
  VecToFloats<T>(UnpackVec2Bits<T>(raw.x), values);
  VecToFloats<T>(UnpackVec2Bits<T>(raw.y), values + 2);
  VecToFloats<T>(UnpackVec2Bits<T>(raw.z), values + 4);
  VecToFloats<T>(UnpackVec2Bits<T>(raw.w), values + 6);
}

template<typename T>
__device__ __forceinline__ uint4 FloatsToRaw8(const float* values) {
  static_assert(!std::is_same_v<T, float>, "FloatsToRaw8 is for 16-bit data");
  uint4 raw;
  raw.x = PackVec2Bits<T>(FloatsToVec<T>(values));
  raw.y = PackVec2Bits<T>(FloatsToVec<T>(values + 2));
  raw.z = PackVec2Bits<T>(FloatsToVec<T>(values + 4));
  raw.w = PackVec2Bits<T>(FloatsToVec<T>(values + 6));
  return raw;
}

// Load 8 consecutive weight scalars as floats. fp32 weights use two float4
// transactions, 16-bit weights one uint4 transaction.
template<typename WeightDType>
__device__ __forceinline__ void LoadWeight8Floats(
    const WeightDType* weight, int64_t offset, float* values) {
  if constexpr (std::is_same_v<WeightDType, float>) {
    const float4 lo = *reinterpret_cast<const float4*>(weight + offset);
    const float4 hi = *reinterpret_cast<const float4*>(weight + offset + 4);
    values[0] = lo.x; values[1] = lo.y; values[2] = lo.z; values[3] = lo.w;
    values[4] = hi.x; values[5] = hi.y; values[6] = hi.z; values[7] = hi.w;
  } else {
    Raw8ToFloats<WeightDType>(
        *reinterpret_cast<const uint4*>(weight + offset), values);
  }
}

inline bool HostIsAligned16(const void* ptr) {
  return (reinterpret_cast<uintptr_t>(ptr) & 0xF) == 0;
}

__device__ __forceinline__ float warpReduceSum(float val) {
  for (int offset = 16; offset > 0; offset /= 2)
    val += __shfl_down_sync(0xffffffff, val, offset);
  return val;
}

__device__ __forceinline__ float blockReduceSum(float val) {
  static __shared__ float shared[32];
  int lane = threadIdx.x % 32;
  int wid = threadIdx.x / 32;
  val = warpReduceSum(val);
  if (lane == 0) shared[wid] = val;
  __syncthreads();
  int num_warps = (blockDim.x + 31) / 32;
  val = (threadIdx.x < num_warps) ? shared[lane] : 0.0f;
  if (wid == 0) val = warpReduceSum(val);
  return val;
}

// ======================== Forward ========================

template<typename T, typename WeightDType, bool ZeroCentered>
__global__ void rmsnormKernel(
    T* output, const T* input, const WeightDType* weight,
    float* rstd_output,
    const float epsilon, const int64_t hidden_size,
    const int64_t input_row_stride, const int64_t output_row_stride) {
    using Vec = typename VecTraits<T>::Type;
    constexpr int VEC_SIZE = VecTraits<T>::vec_size;
    float square_sum = 0.0f;
    __shared__ float inv_rms;
    const int64_t tid = threadIdx.x;
    const int64_t bid = blockIdx.x;
    const int64_t input_offset = bid * input_row_stride;
    const int64_t output_offset = bid * output_row_stride;
    const int64_t vec_hidden = hidden_size / VEC_SIZE;

    for (int64_t i = tid; i < vec_hidden; i += blockDim.x) {
        Vec x_vec = VecTraits<T>::load(input + input_offset + i * VEC_SIZE);
        float x_vals[VEC_SIZE];
        VecToFloats<T>(x_vec, x_vals);
        #pragma unroll
        for (int j = 0; j < VEC_SIZE; ++j) {
            square_sum += x_vals[j] * x_vals[j];
        }
    }
    square_sum = blockReduceSum(square_sum);
    if (tid == 0) {
        inv_rms = rsqrtf(square_sum / static_cast<float>(hidden_size) + epsilon);
        if (rstd_output != nullptr) {
            rstd_output[bid] = inv_rms;
        }
    }
    __syncthreads();

    for (int64_t i = tid; i < vec_hidden; i += blockDim.x) {
        Vec in_vec = VecTraits<T>::load(input + input_offset + i * VEC_SIZE);
        float x_vals[VEC_SIZE], w_vals[VEC_SIZE], out_vals[VEC_SIZE];
        VecToFloats<T>(in_vec, x_vals);
        LoadWeightFloats<WeightDType, VEC_SIZE>(weight, i * VEC_SIZE, w_vals);
        #pragma unroll
        for (int j = 0; j < VEC_SIZE; ++j) {
            if constexpr (ZeroCentered) w_vals[j] += 1.0f;
            out_vals[j] = x_vals[j] * w_vals[j] * inv_rms;
        }
        VecTraits<T>::store(output + output_offset + i * VEC_SIZE, FloatsToVec<T>(out_vals));
    }
}

template<typename T, typename WeightDType, bool ZeroCentered>
void LaunchRMSNormKernel(
    tvm::ffi::TensorView output,
    tvm::ffi::TensorView input,
    tvm::ffi::TensorView weight,
    float* rstd_output,
    float epsilon,
    int64_t hidden_size,
    int64_t num_tokens,
    int64_t input_row_stride,
    int64_t output_row_stride,
    int64_t block_size,
    cudaStream_t stream) {
  rmsnormKernel<T, WeightDType, ZeroCentered><<<num_tokens, block_size, 0, stream>>>(
      static_cast<T*>(output.data_ptr()),
      static_cast<const T*>(input.data_ptr()),
      static_cast<const WeightDType*>(weight.data_ptr()),
      rstd_output,
      epsilon, hidden_size, input_row_stride, output_row_stride);
}

// Wide (16-byte vectorized) forward path for 16-bit dtypes. One block per row;
// each thread owns up to VPT uint4 vectors (8 scalars each), the raw input bits
// are kept in registers between the reduction and the normalize pass, so each
// input element is read from global memory exactly once and written once.
template<typename T, typename WeightDType, int VPT, bool ZeroCentered>
__global__ void __launch_bounds__(1024)
rmsnormForwardWideKernel(
    T* __restrict__ output, const T* __restrict__ input,
    const WeightDType* __restrict__ weight,
    float* __restrict__ rstd_output,
    const float epsilon, const int64_t hidden_size,
    const int64_t input_row_stride, const int64_t output_row_stride) {
  constexpr int ELEMS = 8;
  static_assert(!std::is_same_v<T, float>, "wide path is for 16-bit tensors");
  const int tid = threadIdx.x;
  const int64_t row = blockIdx.x;
  const T* in_row = input + row * input_row_stride;
  T* out_row = output + row * output_row_stride;
  const int64_t vec_hidden = hidden_size / ELEMS;

  // The weight is prefetched alongside x so its (L2) latency hides behind the
  // block reduction instead of stalling the normalize pass.
  uint4 x_raw[VPT];
  float w_cache[VPT][ELEMS];
  float sq = 0.0f;
  #pragma unroll
  for (int k = 0; k < VPT; ++k) {
    const int64_t i = tid + k * static_cast<int64_t>(blockDim.x);
    if (i < vec_hidden) {
      x_raw[k] = *reinterpret_cast<const uint4*>(in_row + i * ELEMS);
      LoadWeight8Floats<WeightDType>(weight, i * ELEMS, w_cache[k]);
      float x_vals[ELEMS];
      Raw8ToFloats<T>(x_raw[k], x_vals);
      #pragma unroll
      for (int j = 0; j < ELEMS; ++j) sq += x_vals[j] * x_vals[j];
    }
  }
  sq = blockReduceSum(sq);
  __shared__ float s_inv_rms;
  if (tid == 0) {
    s_inv_rms = rsqrtf(sq / static_cast<float>(hidden_size) + epsilon);
    if (rstd_output != nullptr) {
      rstd_output[row] = s_inv_rms;
    }
  }
  __syncthreads();
  const float irms = s_inv_rms;

  #pragma unroll
  for (int k = 0; k < VPT; ++k) {
    const int64_t i = tid + k * static_cast<int64_t>(blockDim.x);
    if (i < vec_hidden) {
      float x_vals[ELEMS], out_vals[ELEMS];
      Raw8ToFloats<T>(x_raw[k], x_vals);
      #pragma unroll
      for (int j = 0; j < ELEMS; ++j) {
        float wv = w_cache[k][j];
        if constexpr (ZeroCentered) wv += 1.0f;
        out_vals[j] = x_vals[j] * wv * irms;
      }
      *reinterpret_cast<uint4*>(out_row + i * ELEMS) = FloatsToRaw8<T>(out_vals);
    }
  }
}

constexpr int kWideElems = 8;          // 16-bit scalars per 16B vector

inline int64_t WideRoundUp(int64_t x, int64_t m) { return (x + m - 1) / m * m; }

// Pick vectors-per-thread for the forward wide kernel: the per-thread register
// cache is 16B * VPT, so with __launch_bounds__(1024) up to VPT=4 fits without
// spilling. Returns 0 (-> generic fallback) for larger hidden sizes.
inline int PickWideVPTFwd(int64_t vec_hidden) {
  if (vec_hidden <= 1024) return 1;
  if (vec_hidden <= 2048) return 2;
  if (vec_hidden <= 4096) return 4;
  return 0;
}

inline int64_t WideBlockSize(int64_t vec_hidden, int vpt) {
  return std::min<int64_t>(1024, WideRoundUp((vec_hidden + vpt - 1) / vpt, 32));
}

// Whether the wide 16B path can serve this layout (dtype/VPT checked separately).
inline bool WideLayoutOK(int64_t hidden_size, std::initializer_list<int64_t> row_strides,
                         std::initializer_list<const void*> ptrs) {
  if (hidden_size % kWideElems != 0) return false;
  for (int64_t s : row_strides) {
    if (s % kWideElems != 0) return false;
  }
  for (const void* p : ptrs) {
    if (!HostIsAligned16(p)) return false;
  }
  return true;
}

template<typename T, typename WeightDType, bool ZeroCentered>
void LaunchRMSNormForwardWideKernel(
    tvm::ffi::TensorView output,
    tvm::ffi::TensorView input,
    tvm::ffi::TensorView weight,
    float* rstd_output,
    float epsilon,
    int64_t hidden_size,
    int64_t num_tokens,
    int64_t input_row_stride,
    int64_t output_row_stride,
    cudaStream_t stream) {
  const int64_t vec_hidden = hidden_size / kWideElems;
  const int vpt = PickWideVPTFwd(vec_hidden);
  const int64_t block_size = WideBlockSize(vec_hidden, vpt);
  T* out_ptr = static_cast<T*>(output.data_ptr());
  const T* in_ptr = static_cast<const T*>(input.data_ptr());
  const WeightDType* w_ptr = static_cast<const WeightDType*>(weight.data_ptr());
  switch (vpt) {
    case 1:
      rmsnormForwardWideKernel<T, WeightDType, 1, ZeroCentered>
          <<<num_tokens, block_size, 0, stream>>>(
              out_ptr, in_ptr, w_ptr, rstd_output, epsilon, hidden_size,
              input_row_stride, output_row_stride);
      break;
    case 2:
      rmsnormForwardWideKernel<T, WeightDType, 2, ZeroCentered>
          <<<num_tokens, block_size, 0, stream>>>(
              out_ptr, in_ptr, w_ptr, rstd_output, epsilon, hidden_size,
              input_row_stride, output_row_stride);
      break;
    case 4:
      rmsnormForwardWideKernel<T, WeightDType, 4, ZeroCentered>
          <<<num_tokens, block_size, 0, stream>>>(
              out_ptr, in_ptr, w_ptr, rstd_output, epsilon, hidden_size,
              input_row_stride, output_row_stride);
      break;
    default:
      throw std::runtime_error("RMSNorm wide path: unsupported VPT");
  }
}

// Try dispatching the forward wide path; returns false if the dtype/layout
// combination is not covered (caller falls back to the generic kernel).
template<bool ZeroCentered>
bool TryRMSNormForwardWide(
    tvm::ffi::TensorView output,
    tvm::ffi::TensorView input,
    tvm::ffi::TensorView weight,
    float* rstd_output,
    float epsilon,
    int64_t hidden_size,
    int64_t num_tokens,
    int64_t input_row_stride,
    int64_t output_row_stride,
    cudaStream_t stream) {
  if (!WideLayoutOK(hidden_size, {input_row_stride, output_row_stride},
                    {input.data_ptr(), output.data_ptr(), weight.data_ptr()})) {
    return false;
  }
  if (PickWideVPTFwd(hidden_size / kWideElems) == 0) return false;
  if (IsBFloat16(input.dtype()) && IsBFloat16(weight.dtype())) {
    LaunchRMSNormForwardWideKernel<__nv_bfloat16, __nv_bfloat16, ZeroCentered>(
        output, input, weight, rstd_output, epsilon, hidden_size, num_tokens,
        input_row_stride, output_row_stride, stream);
  } else if (IsBFloat16(input.dtype()) && IsFloat32(weight.dtype())) {
    LaunchRMSNormForwardWideKernel<__nv_bfloat16, float, ZeroCentered>(
        output, input, weight, rstd_output, epsilon, hidden_size, num_tokens,
        input_row_stride, output_row_stride, stream);
  } else if (IsFloat16(input.dtype()) && IsFloat16(weight.dtype())) {
    LaunchRMSNormForwardWideKernel<half, half, ZeroCentered>(
        output, input, weight, rstd_output, epsilon, hidden_size, num_tokens,
        input_row_stride, output_row_stride, stream);
  } else if (IsFloat16(input.dtype()) && IsFloat32(weight.dtype())) {
    LaunchRMSNormForwardWideKernel<half, float, ZeroCentered>(
        output, input, weight, rstd_output, epsilon, hidden_size, num_tokens,
        input_row_stride, output_row_stride, stream);
  } else if (IsFloat16(input.dtype()) && IsBFloat16(weight.dtype())) {
    LaunchRMSNormForwardWideKernel<half, __nv_bfloat16, ZeroCentered>(
        output, input, weight, rstd_output, epsilon, hidden_size, num_tokens,
        input_row_stride, output_row_stride, stream);
  } else {
    return false;
  }
  return true;
}

// Optional NVFP4 copy of the normalized output (add_rmsnorm_zero_centered_nvfp4): the BF16 value the kernel writes,
// quantized as quantize_nvfp4 does (nvfp4_format_core.cuh). The grid then covers the rows padded to 128; blocks past
// num_tokens only write their row's zero scale bytes.
struct Nvfp4Out {
  uint8_t* data = nullptr;    // packed E2M1 [rows, hidden / 2]
  uint8_t* scales = nullptr;  // E4M3 SWIZZLE_32_4_4 storage
  int scale_col_blocks = 0;   // ceil(hidden / 16 / 4)
  int64_t num_tokens = 0;
};

template<typename T, typename WeightDType, int VPT, bool ZeroCentered, bool Nvfp4 = false>
__global__ void __launch_bounds__(1024)
addRMSNormForwardWideKernel(
    T* __restrict__ sum_output, T* __restrict__ norm_output,
    const T* __restrict__ residual, const T* __restrict__ input,
    const WeightDType* __restrict__ weight,
    float* __restrict__ rstd_output,
    const float epsilon, const int64_t hidden_size,
    const int64_t residual_row_stride, const int64_t input_row_stride,
    const int64_t sum_row_stride, const int64_t norm_row_stride, const Nvfp4Out nvfp4 = {});


template<typename T, typename WeightDType, bool ZeroCentered>
void LaunchAddRMSNormForwardWideKernel(
    tvm::ffi::TensorView sum_output,
    tvm::ffi::TensorView norm_output,
    tvm::ffi::TensorView residual,
    tvm::ffi::TensorView input,
    tvm::ffi::TensorView weight,
    float* rstd_output,
    float epsilon,
    int64_t hidden_size,
    int64_t num_tokens,
    int64_t residual_row_stride,
    int64_t input_row_stride,
    int64_t sum_row_stride,
    int64_t norm_row_stride,
    cudaStream_t stream) {
  const int64_t vec_hidden = hidden_size / kWideElems;
  const int vpt = PickWideVPTFwd(vec_hidden);
  const int64_t block_size = WideBlockSize(vec_hidden, vpt);
  T* sum_ptr = static_cast<T*>(sum_output.data_ptr());
  T* norm_ptr = static_cast<T*>(norm_output.data_ptr());
  const T* res_ptr = static_cast<const T*>(residual.data_ptr());
  const T* in_ptr = static_cast<const T*>(input.data_ptr());
  const WeightDType* w_ptr = static_cast<const WeightDType*>(weight.data_ptr());
  switch (vpt) {
    case 1:
      addRMSNormForwardWideKernel<T, WeightDType, 1, ZeroCentered>
          <<<num_tokens, block_size, 0, stream>>>(
              sum_ptr, norm_ptr, res_ptr, in_ptr, w_ptr, rstd_output, epsilon,
              hidden_size, residual_row_stride, input_row_stride,
              sum_row_stride, norm_row_stride);
      break;
    case 2:
      addRMSNormForwardWideKernel<T, WeightDType, 2, ZeroCentered>
          <<<num_tokens, block_size, 0, stream>>>(
              sum_ptr, norm_ptr, res_ptr, in_ptr, w_ptr, rstd_output, epsilon,
              hidden_size, residual_row_stride, input_row_stride,
              sum_row_stride, norm_row_stride);
      break;
    case 4:
      addRMSNormForwardWideKernel<T, WeightDType, 4, ZeroCentered>
          <<<num_tokens, block_size, 0, stream>>>(
              sum_ptr, norm_ptr, res_ptr, in_ptr, w_ptr, rstd_output, epsilon,
              hidden_size, residual_row_stride, input_row_stride,
              sum_row_stride, norm_row_stride);
      break;
    default:
      throw std::runtime_error("AddRMSNorm wide path: unsupported VPT");
  }
}

template<bool ZeroCentered>
bool TryAddRMSNormForwardWide(
    tvm::ffi::TensorView sum_output,
    tvm::ffi::TensorView norm_output,
    tvm::ffi::TensorView residual,
    tvm::ffi::TensorView input,
    tvm::ffi::TensorView weight,
    float* rstd_output,
    float epsilon,
    int64_t hidden_size,
    int64_t num_tokens,
    int64_t residual_row_stride,
    int64_t input_row_stride,
    int64_t sum_row_stride,
    int64_t norm_row_stride,
    cudaStream_t stream) {
  if (!WideLayoutOK(hidden_size,
                    {residual_row_stride, input_row_stride, sum_row_stride,
                     norm_row_stride},
                    {residual.data_ptr(), input.data_ptr(),
                     sum_output.data_ptr(), norm_output.data_ptr(),
                     weight.data_ptr()})) {
    return false;
  }
  if (PickWideVPTFwd(hidden_size / kWideElems) == 0) return false;
  if (IsBFloat16(input.dtype()) && IsBFloat16(weight.dtype())) {
    LaunchAddRMSNormForwardWideKernel<__nv_bfloat16, __nv_bfloat16, ZeroCentered>(
        sum_output, norm_output, residual, input, weight, rstd_output, epsilon,
        hidden_size, num_tokens, residual_row_stride, input_row_stride,
        sum_row_stride, norm_row_stride, stream);
  } else if (IsBFloat16(input.dtype()) && IsFloat32(weight.dtype())) {
    LaunchAddRMSNormForwardWideKernel<__nv_bfloat16, float, ZeroCentered>(
        sum_output, norm_output, residual, input, weight, rstd_output, epsilon,
        hidden_size, num_tokens, residual_row_stride, input_row_stride,
        sum_row_stride, norm_row_stride, stream);
  } else if (IsFloat16(input.dtype()) && IsFloat16(weight.dtype())) {
    LaunchAddRMSNormForwardWideKernel<half, half, ZeroCentered>(
        sum_output, norm_output, residual, input, weight, rstd_output, epsilon,
        hidden_size, num_tokens, residual_row_stride, input_row_stride,
        sum_row_stride, norm_row_stride, stream);
  } else if (IsFloat16(input.dtype()) && IsFloat32(weight.dtype())) {
    LaunchAddRMSNormForwardWideKernel<half, float, ZeroCentered>(
        sum_output, norm_output, residual, input, weight, rstd_output, epsilon,
        hidden_size, num_tokens, residual_row_stride, input_row_stride,
        sum_row_stride, norm_row_stride, stream);
  } else if (IsFloat16(input.dtype()) && IsBFloat16(weight.dtype())) {
    LaunchAddRMSNormForwardWideKernel<half, __nv_bfloat16, ZeroCentered>(
        sum_output, norm_output, residual, input, weight, rstd_output, epsilon,
        hidden_size, num_tokens, residual_row_stride, input_row_stride,
        sum_row_stride, norm_row_stride, stream);
  } else {
    return false;
  }
  return true;
}

// Wide forward for the fused add+rmsnorm. Deliberately mirrors
// rmsnormForwardWideKernel's block configuration and accumulation order so the
// fused op stays BITWISE identical to the materialized path
// (sum = round(residual+input); rmsnorm(sum)) — tests assert this coupling.
template<typename T, typename WeightDType, int VPT, bool ZeroCentered, bool Nvfp4>
__global__ void __launch_bounds__(1024)
addRMSNormForwardWideKernel(
    T* __restrict__ sum_output, T* __restrict__ norm_output,
    const T* __restrict__ residual, const T* __restrict__ input,
    const WeightDType* __restrict__ weight,
    float* __restrict__ rstd_output,
    const float epsilon, const int64_t hidden_size,
    const int64_t residual_row_stride, const int64_t input_row_stride,
    const int64_t sum_row_stride, const int64_t norm_row_stride, const Nvfp4Out nvfp4) {
  constexpr int ELEMS = 8;
  static_assert(!std::is_same_v<T, float>, "wide path is for 16-bit tensors");
  const int tid = threadIdx.x;
  const int64_t row = blockIdx.x;
  if constexpr (Nvfp4) {
    if (row >= nvfp4.num_tokens) {  // a padding row of the scale storage
      for (int b = tid; b < nvfp4.scale_col_blocks * 4; b += blockDim.x) {
        nvfp4.scales[xinfer::nvfp4::ScaleOffset(row, b, nvfp4.scale_col_blocks)] = 0;
      }
      return;
    }
  }
  const T* res_row = residual + row * residual_row_stride;
  const T* in_row = input + row * input_row_stride;
  T* sum_row = sum_output + row * sum_row_stride;
  T* norm_row = nullptr;  // none when the NVFP4 call is given no norm output
  if (!Nvfp4 || norm_output != nullptr) norm_row = norm_output + row * norm_row_stride;
  const int64_t vec_hidden = hidden_size / ELEMS;

  uint4 s_raw[VPT];
  float w_cache[VPT][ELEMS];
  float sq = 0.0f;
  #pragma unroll
  for (int k = 0; k < VPT; ++k) {
    const int64_t i = tid + k * static_cast<int64_t>(blockDim.x);
    if (i < vec_hidden) {
      const uint4 r_raw = *reinterpret_cast<const uint4*>(res_row + i * ELEMS);
      const uint4 x_raw = *reinterpret_cast<const uint4*>(in_row + i * ELEMS);
      LoadWeight8Floats<WeightDType>(weight, i * ELEMS, w_cache[k]);
      float r_vals[ELEMS], x_vals[ELEMS], s_vals[ELEMS];
      Raw8ToFloats<T>(r_raw, r_vals);
      Raw8ToFloats<T>(x_raw, x_vals);
      #pragma unroll
      for (int j = 0; j < ELEMS; ++j) {
        // Match the materialized path: the bf16/fp16 add produces a rounded
        // sum tensor, and rmsnorm consumes that rounded tensor.
        s_vals[j] = RoundFloatToScalarType<T>(r_vals[j] + x_vals[j]);
        sq += s_vals[j] * s_vals[j];
      }
      s_raw[k] = FloatsToRaw8<T>(s_vals);
      *reinterpret_cast<uint4*>(sum_row + i * ELEMS) = s_raw[k];
    }
  }
  sq = blockReduceSum(sq);
  __shared__ float s_inv_rms;
  if (tid == 0) {
    s_inv_rms = rsqrtf(sq / static_cast<float>(hidden_size) + epsilon);
    if (rstd_output != nullptr) {
      rstd_output[row] = s_inv_rms;
    }
  }
  __syncthreads();
  const float irms = s_inv_rms;

  #pragma unroll
  for (int k = 0; k < VPT; ++k) {
    const int64_t i = tid + k * static_cast<int64_t>(blockDim.x);
    if (i < vec_hidden) {
      float s_vals[ELEMS], out_vals[ELEMS];
      Raw8ToFloats<T>(s_raw[k], s_vals);
      #pragma unroll
      for (int j = 0; j < ELEMS; ++j) {
        float wv = w_cache[k][j];
        if constexpr (ZeroCentered) wv += 1.0f;
        out_vals[j] = s_vals[j] * wv * irms;
      }
      const uint4 out_raw = FloatsToRaw8<T>(out_vals);
      if (!Nvfp4 || norm_row != nullptr) *reinterpret_cast<uint4*>(norm_row + i * ELEMS) = out_raw;
      if constexpr (Nvfp4) {
        // The 16-column block of vectors i and i ^ 1 spans this thread and its neighbour (blockDim is even).
        float h[ELEMS];
        Raw8ToFloats<T>(out_raw, h);
        float amax = 0.0f;
        #pragma unroll
        for (int j = 0; j < ELEMS; ++j) amax = fmaxf(amax, fabsf(h[j]));
        amax = fmaxf(amax, __shfl_xor_sync(__activemask(), amax, 1));
        const float scale = xinfer::nvfp4::AmaxScale(amax, 1.0f);
        if ((i & 1) == 0) {
          nvfp4.scales[xinfer::nvfp4::ScaleOffset(row, static_cast<int>(i / 2), nvfp4.scale_col_blocks)] =
              xinfer::nvfp4::ScaleBits(scale);
        }
        *reinterpret_cast<uint32_t*>(nvfp4.data + row * (hidden_size / 2) + i * 4) =
            xinfer::nvfp4::Pack8(h, scale);
      }
    }
  }
}

template<typename T, typename WeightDType, bool ZeroCentered>
__global__ void addRMSNormKernel(
    T* sum_output,
    T* norm_output,
    const T* residual,
    const T* input,
    const WeightDType* weight,
    float* rstd_output,
    const float epsilon,
    const int64_t hidden_size,
    const int64_t residual_row_stride,
    const int64_t input_row_stride,
    const int64_t sum_row_stride,
    const int64_t norm_row_stride) {
  using Vec = typename VecTraits<T>::Type;
  constexpr int VEC_SIZE = VecTraits<T>::vec_size;
  float square_sum = 0.0f;
  __shared__ float inv_rms;
  const int64_t tid = threadIdx.x;
  const int64_t bid = blockIdx.x;
  const int64_t residual_offset = bid * residual_row_stride;
  const int64_t input_offset = bid * input_row_stride;
  const int64_t sum_offset = bid * sum_row_stride;
  const int64_t norm_offset = bid * norm_row_stride;
  const int64_t vec_hidden = hidden_size / VEC_SIZE;

  for (int64_t i = tid; i < vec_hidden; i += blockDim.x) {
    Vec residual_vec = VecTraits<T>::load(residual + residual_offset + i * VEC_SIZE);
    Vec input_vec = VecTraits<T>::load(input + input_offset + i * VEC_SIZE);
    float residual_vals[VEC_SIZE], input_vals[VEC_SIZE], sum_vals[VEC_SIZE];
    VecToFloats<T>(residual_vec, residual_vals);
    VecToFloats<T>(input_vec, input_vals);
    #pragma unroll
    for (int j = 0; j < VEC_SIZE; ++j) {
      // Match the old path: bf16/fp16 add materializes a rounded sum tensor,
      // and RMSNorm consumes that rounded tensor.
      sum_vals[j] = RoundFloatToScalarType<T>(residual_vals[j] + input_vals[j]);
      square_sum += sum_vals[j] * sum_vals[j];
    }
    VecTraits<T>::store(sum_output + sum_offset + i * VEC_SIZE,
                        FloatsToVec<T>(sum_vals));
  }
  square_sum = blockReduceSum(square_sum);
  if (tid == 0) {
    inv_rms = rsqrtf(square_sum / static_cast<float>(hidden_size) + epsilon);
    if (rstd_output != nullptr) {
      rstd_output[bid] = inv_rms;
    }
  }
  __syncthreads();

  for (int64_t i = tid; i < vec_hidden; i += blockDim.x) {
    Vec sum_vec = VecTraits<T>::load(sum_output + sum_offset + i * VEC_SIZE);
    float sum_vals[VEC_SIZE];
    float w_vals[VEC_SIZE], out_vals[VEC_SIZE];
    VecToFloats<T>(sum_vec, sum_vals);
    LoadWeightFloats<WeightDType, VEC_SIZE>(weight, i * VEC_SIZE, w_vals);
    #pragma unroll
    for (int j = 0; j < VEC_SIZE; ++j) {
      if constexpr (ZeroCentered) w_vals[j] += 1.0f;
      out_vals[j] = sum_vals[j] * w_vals[j] * inv_rms;
    }
    VecTraits<T>::store(norm_output + norm_offset + i * VEC_SIZE,
                        FloatsToVec<T>(out_vals));
  }
}

template<typename T, typename WeightDType, bool ZeroCentered>
void LaunchAddRMSNormKernel(
    tvm::ffi::TensorView sum_output,
    tvm::ffi::TensorView norm_output,
    tvm::ffi::TensorView residual,
    tvm::ffi::TensorView input,
    tvm::ffi::TensorView weight,
    float* rstd_output,
    float epsilon,
    int64_t hidden_size,
    int64_t num_tokens,
    int64_t residual_row_stride,
    int64_t input_row_stride,
    int64_t sum_row_stride,
    int64_t norm_row_stride,
    int64_t block_size,
    cudaStream_t stream) {
  addRMSNormKernel<T, WeightDType, ZeroCentered><<<num_tokens, block_size, 0, stream>>>(
      static_cast<T*>(sum_output.data_ptr()),
      static_cast<T*>(norm_output.data_ptr()),
      static_cast<const T*>(residual.data_ptr()),
      static_cast<const T*>(input.data_ptr()),
      static_cast<const WeightDType*>(weight.data_ptr()),
      rstd_output,
      epsilon, hidden_size, residual_row_stride, input_row_stride,
      sum_row_stride, norm_row_stride);
}

#ifdef XINFER_WITH_SM110
// SM110 kernel of the BF16 forward (sm110/bf16_rmsnorm_sm110.cu): one warp per row, row kept in registers. Measured
// on Thor against the wide kernel: 1.2-1.5x faster from 256 rows up, slower below (2112 x 1280: 16.8 -> 11.8 us;
// 20 x 1024: 2.1 vs 2.4 us), hence the row threshold.
constexpr int64_t kSm110RmsnormMinRows = 256;

bool UseRMSNormSM110(const tvm::ffi::TensorView& output, const tvm::ffi::TensorView& input,
                     const tvm::ffi::TensorView& weight, const RMSNormShape& shape,
                     const RMSNormShape& output_shape) {
  if (!IsBFloat16(input.dtype()) || !IsBFloat16(weight.dtype()) || !IsBFloat16(output.dtype()) ||
      shape.num_tokens < kSm110RmsnormMinRows || shape.row_stride != shape.hidden_size ||
      output_shape.row_stride != shape.hidden_size || !bf16_rmsnorm_sm110::Supported(shape.hidden_size)) {
    return false;
  }
  for (const void* pointer : {output.data_ptr(), input.data_ptr(), weight.data_ptr()}) {
    if (reinterpret_cast<uintptr_t>(pointer) % 16 != 0) return false;
  }
  int major = 0;
  int minor = 0;
  const int device = input.device().device_id;
  cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
  cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device);
  return major == 11 && minor == 0;
}
#endif

template<bool ZeroCentered>
void RMSNormImpl(tvm::ffi::TensorView output, tvm::ffi::TensorView input,
                 tvm::ffi::TensorView weight, double epsilon,
                 bool use_wide, float* rstd_ptr = nullptr) {
  ASSERT_CHECK(weight.ndim() == 1);
  ASSERT_CHECK(weight.IsContiguous());
  CheckSameShape(output, input);
  const RMSNormShape shape = GetRMSNormShape(input);
  const RMSNormShape output_shape = GetRMSNormShape(output);
  const int64_t hidden_size = shape.hidden_size;
  const int64_t num_tokens = shape.num_tokens;
  ASSERT_CHECK(output_shape.hidden_size == hidden_size);
  ASSERT_CHECK(output_shape.num_tokens == num_tokens);
  ASSERT_CHECK(hidden_size == weight.size(0));
  ASSERT_CHECK(hidden_size % GetVecSize(input.dtype()) == 0);
  ASSERT_CHECK(shape.row_stride % GetVecSize(input.dtype()) == 0);
  ASSERT_CHECK(output_shape.row_stride % GetVecSize(input.dtype()) == 0);
  cudaStream_t stream = static_cast<cudaStream_t>(
      TVMFFIEnvGetStream(input.device().device_type, input.device().device_id));
#ifdef XINFER_WITH_SM110
  if constexpr (!ZeroCentered) {
    if (use_wide && rstd_ptr == nullptr && UseRMSNormSM110(output, input, weight, shape, output_shape)) {
      bf16_rmsnorm_sm110::Launch(static_cast<__nv_bfloat16*>(output.data_ptr()),
                                 static_cast<const __nv_bfloat16*>(input.data_ptr()),
                                 static_cast<const __nv_bfloat16*>(weight.data_ptr()), num_tokens, hidden_size,
                                 static_cast<float>(epsilon), 256, input.device().device_id, stream);
      return;
    }
  }
#endif
  if (use_wide && TryRMSNormForwardWide<ZeroCentered>(
          output, input, weight, rstd_ptr, static_cast<float>(epsilon),
          hidden_size, num_tokens, shape.row_stride, output_shape.row_stride,
          stream)) {
    sync_check_cuda_error();
    return;
  }
  const int64_t block_size = std::min(hidden_size, 1024L);
  if (IsFloat32(input.dtype()) && IsFloat32(weight.dtype()))
    LaunchRMSNormKernel<float, float, ZeroCentered>(
        output, input, weight, rstd_ptr, static_cast<float>(epsilon), hidden_size, num_tokens,
        shape.row_stride, output_shape.row_stride, block_size, stream);
  else if (IsFloat16(input.dtype()) && IsFloat16(weight.dtype()))
    LaunchRMSNormKernel<half, half, ZeroCentered>(
        output, input, weight, rstd_ptr, static_cast<float>(epsilon), hidden_size, num_tokens,
        shape.row_stride, output_shape.row_stride, block_size, stream);
  else if (IsFloat16(input.dtype()) && IsFloat32(weight.dtype()))
    LaunchRMSNormKernel<half, float, ZeroCentered>(
        output, input, weight, rstd_ptr, static_cast<float>(epsilon), hidden_size, num_tokens,
        shape.row_stride, output_shape.row_stride, block_size, stream);
  else if (IsFloat16(input.dtype()) && IsBFloat16(weight.dtype()))
    LaunchRMSNormKernel<half, __nv_bfloat16, ZeroCentered>(
        output, input, weight, rstd_ptr, static_cast<float>(epsilon), hidden_size, num_tokens,
        shape.row_stride, output_shape.row_stride, block_size, stream);
  else if (IsBFloat16(input.dtype()) && IsBFloat16(weight.dtype()))
    LaunchRMSNormKernel<__nv_bfloat16, __nv_bfloat16, ZeroCentered>(
        output, input, weight, rstd_ptr, static_cast<float>(epsilon), hidden_size, num_tokens,
        shape.row_stride, output_shape.row_stride, block_size, stream);
  else if (IsBFloat16(input.dtype()) && IsFloat32(weight.dtype()))
    LaunchRMSNormKernel<__nv_bfloat16, float, ZeroCentered>(
        output, input, weight, rstd_ptr, static_cast<float>(epsilon), hidden_size, num_tokens,
        shape.row_stride, output_shape.row_stride, block_size, stream);
  else throw std::runtime_error("Unsupported data type for RMSNorm");
  sync_check_cuda_error();
}

template<bool ZeroCentered>
void AddRMSNormImpl(tvm::ffi::TensorView sum_output,
                    tvm::ffi::TensorView norm_output,
                    tvm::ffi::TensorView residual,
                    tvm::ffi::TensorView input,
                    tvm::ffi::TensorView weight,
                    double epsilon) {
  ASSERT_CHECK(weight.ndim() == 1);
  ASSERT_CHECK(weight.IsContiguous());
  CheckSameShape(sum_output, input);
  CheckSameShape(norm_output, input);
  CheckSameShape(residual, input);
  const RMSNormShape input_shape = GetRMSNormShape(input);
  const RMSNormShape residual_shape = GetRMSNormShape(residual);
  const RMSNormShape sum_shape = GetRMSNormShape(sum_output);
  const RMSNormShape norm_shape = GetRMSNormShape(norm_output);
  const int64_t hidden_size = input_shape.hidden_size;
  const int64_t num_tokens = input_shape.num_tokens;
  ASSERT_CHECK(residual_shape.hidden_size == hidden_size);
  ASSERT_CHECK(residual_shape.num_tokens == num_tokens);
  ASSERT_CHECK(sum_shape.hidden_size == hidden_size);
  ASSERT_CHECK(sum_shape.num_tokens == num_tokens);
  ASSERT_CHECK(norm_shape.hidden_size == hidden_size);
  ASSERT_CHECK(norm_shape.num_tokens == num_tokens);
  ASSERT_CHECK(hidden_size == weight.size(0));
  ASSERT_CHECK(input.dtype() == residual.dtype());
  ASSERT_CHECK(input.dtype() == sum_output.dtype());
  ASSERT_CHECK(input.dtype() == norm_output.dtype());
  ASSERT_CHECK(hidden_size % GetVecSize(input.dtype()) == 0);
  ASSERT_CHECK(input_shape.row_stride % GetVecSize(input.dtype()) == 0);
  ASSERT_CHECK(residual_shape.row_stride % GetVecSize(input.dtype()) == 0);
  ASSERT_CHECK(sum_shape.row_stride % GetVecSize(input.dtype()) == 0);
  ASSERT_CHECK(norm_shape.row_stride % GetVecSize(input.dtype()) == 0);

  cudaStream_t stream = static_cast<cudaStream_t>(
      TVMFFIEnvGetStream(input.device().device_type, input.device().device_id));
  if (TryAddRMSNormForwardWide<ZeroCentered>(
          sum_output, norm_output, residual, input, weight, nullptr,
          static_cast<float>(epsilon), hidden_size, num_tokens,
          residual_shape.row_stride, input_shape.row_stride,
          sum_shape.row_stride, norm_shape.row_stride, stream)) {
    sync_check_cuda_error();
    return;
  }
  const int64_t block_size = std::min(hidden_size, 1024L);
  if (IsFloat32(input.dtype()) && IsFloat32(weight.dtype()))
    LaunchAddRMSNormKernel<float, float, ZeroCentered>(
        sum_output, norm_output, residual, input, weight, nullptr, static_cast<float>(epsilon),
        hidden_size, num_tokens, residual_shape.row_stride, input_shape.row_stride,
        sum_shape.row_stride, norm_shape.row_stride, block_size, stream);
  else if (IsFloat16(input.dtype()) && IsFloat16(weight.dtype()))
    LaunchAddRMSNormKernel<half, half, ZeroCentered>(
        sum_output, norm_output, residual, input, weight, nullptr, static_cast<float>(epsilon),
        hidden_size, num_tokens, residual_shape.row_stride, input_shape.row_stride,
        sum_shape.row_stride, norm_shape.row_stride, block_size, stream);
  else if (IsFloat16(input.dtype()) && IsFloat32(weight.dtype()))
    LaunchAddRMSNormKernel<half, float, ZeroCentered>(
        sum_output, norm_output, residual, input, weight, nullptr, static_cast<float>(epsilon),
        hidden_size, num_tokens, residual_shape.row_stride, input_shape.row_stride,
        sum_shape.row_stride, norm_shape.row_stride, block_size, stream);
  else if (IsFloat16(input.dtype()) && IsBFloat16(weight.dtype()))
    LaunchAddRMSNormKernel<half, __nv_bfloat16, ZeroCentered>(
        sum_output, norm_output, residual, input, weight, nullptr, static_cast<float>(epsilon),
        hidden_size, num_tokens, residual_shape.row_stride, input_shape.row_stride,
        sum_shape.row_stride, norm_shape.row_stride, block_size, stream);
  else if (IsBFloat16(input.dtype()) && IsBFloat16(weight.dtype()))
    LaunchAddRMSNormKernel<__nv_bfloat16, __nv_bfloat16, ZeroCentered>(
        sum_output, norm_output, residual, input, weight, nullptr, static_cast<float>(epsilon),
        hidden_size, num_tokens, residual_shape.row_stride, input_shape.row_stride,
        sum_shape.row_stride, norm_shape.row_stride, block_size, stream);
  else if (IsBFloat16(input.dtype()) && IsFloat32(weight.dtype()))
    LaunchAddRMSNormKernel<__nv_bfloat16, float, ZeroCentered>(
        sum_output, norm_output, residual, input, weight, nullptr, static_cast<float>(epsilon),
        hidden_size, num_tokens, residual_shape.row_stride, input_shape.row_stride,
        sum_shape.row_stride, norm_shape.row_stride, block_size, stream);
  else throw std::runtime_error("Unsupported data type for AddRMSNorm");
  sync_check_cuda_error();
}

void RMSNorm(tvm::ffi::TensorView output, tvm::ffi::TensorView input,
             tvm::ffi::TensorView weight, double epsilon, bool use_wide) {
  RMSNormImpl<false>(output, input, weight, epsilon, use_wide);
}

void RMSNormZeroCentered(tvm::ffi::TensorView output, tvm::ffi::TensorView input,
                         tvm::ffi::TensorView weight, double epsilon,
                         bool use_wide) {
  RMSNormImpl<true>(output, input, weight, epsilon, use_wide);
}

void AddRMSNorm(tvm::ffi::TensorView sum_output,
                tvm::ffi::TensorView norm_output,
                tvm::ffi::TensorView residual,
                tvm::ffi::TensorView input,
                tvm::ffi::TensorView weight,
                double epsilon) {
  AddRMSNormImpl<false>(sum_output, norm_output, residual, input, weight, epsilon);
}

void AddRMSNormZeroCentered(tvm::ffi::TensorView sum_output,
                            tvm::ffi::TensorView norm_output,
                            tvm::ffi::TensorView residual,
                            tvm::ffi::TensorView input,
                            tvm::ffi::TensorView weight,
                            double epsilon) {
  AddRMSNormImpl<true>(sum_output, norm_output, residual, input, weight, epsilon);
}

template<typename T, typename WeightDType>
void LaunchAddRMSNormNvfp4(T* sum_ptr, T* norm_ptr, const T* res_ptr, const T* in_ptr, const WeightDType* w_ptr,
                           float epsilon, int64_t hidden_size, int64_t num_tokens, Nvfp4Out nvfp4,
                           cudaStream_t stream) {
  const int64_t vec_hidden = hidden_size / kWideElems;
  const int vpt = PickWideVPTFwd(vec_hidden);
  const int64_t block_size = WideBlockSize(vec_hidden, vpt);
  const int64_t padded_rows = (num_tokens + 127) / 128 * 128;
#define XINFER_ADD_RMSNORM_NVFP4(VPT)                                                                      \
  addRMSNormForwardWideKernel<T, WeightDType, VPT, true, true><<<padded_rows, block_size, 0, stream>>>(      \
      sum_ptr, norm_ptr, res_ptr, in_ptr, w_ptr, nullptr, epsilon, hidden_size, hidden_size, hidden_size, \
      hidden_size, hidden_size, nvfp4)
  switch (vpt) {
    case 1: XINFER_ADD_RMSNORM_NVFP4(1); break;
    case 2: XINFER_ADD_RMSNORM_NVFP4(2); break;
    case 4: XINFER_ADD_RMSNORM_NVFP4(4); break;
    default: throw std::runtime_error("add_rmsnorm_zero_centered_nvfp4: hidden size too large for the wide kernel");
  }
#undef XINFER_ADD_RMSNORM_NVFP4
}

// sum = bf16(residual + input), norm = the BF16 zero-centered RMSNorm of sum, and (data, scales) the NVFP4 of norm:
// the wide kernel of add_rmsnorm_zero_centered (its sum and norm bit for bit) writing quantize_nvfp4 of the norm too;
// the BF16 norm is written only into a given norm_output. BF16 contiguous [M, K] tensors, K a multiple of 64, 16-byte
// aligned; weight [K] BF16 or FP32; data uint8 [M, K / 2] (4-byte aligned), scales E4M3 (or uint8) storage of the rows
// padded to 128.
void AddRMSNormZeroCenteredNvfp4(tvm::ffi::TensorView sum_output, tvm::ffi::Optional<tvm::ffi::TensorView> norm_output,
                                 tvm::ffi::TensorView data, tvm::ffi::TensorView scales,
                                 tvm::ffi::TensorView residual, tvm::ffi::TensorView input,
                                 tvm::ffi::TensorView weight, double epsilon) {
  namespace ffi = xinfer::ffi;
  std::vector<const tvm::ffi::TensorView*> rows = {&sum_output, &residual, &input};
  if (norm_output.has_value()) rows.push_back(&norm_output.value());
  for (const tvm::ffi::TensorView* t : rows) {
    ffi::CheckCuda(*t, "add_rmsnorm_zero_centered_nvfp4: tensors");
    ffi::CheckBFloat16(*t, "add_rmsnorm_zero_centered_nvfp4: tensors");
    ffi::CheckContiguous(*t, "add_rmsnorm_zero_centered_nvfp4: tensors");
    ffi::CheckSameShape(*t, input, "add_rmsnorm_zero_centered_nvfp4: sum/norm/residual");
    ffi::CheckSameDevice(*t, input, "add_rmsnorm_zero_centered_nvfp4: sum/norm/residual");
  }
  ffi::CheckUInt8(data, "data");
  ffi::CheckContiguous(data, "data");
  ffi::CheckFloat8E4M3Storage(scales, "scales");
  ffi::CheckContiguous(scales, "scales");
  ffi::CheckContiguous(weight, "weight");
  for (const tvm::ffi::TensorView* t : {&data, &scales, &weight}) {
    ffi::CheckSameDevice(*t, input, "add_rmsnorm_zero_centered_nvfp4: data/scales/weight");
  }
  const RMSNormShape shape = GetRMSNormShape(input);
  const int64_t k = shape.hidden_size, m = shape.num_tokens;
  if (k % 64 || PickWideVPTFwd(k / kWideElems) == 0 || weight.ndim() != 1 || weight.size(0) != k) {
    throw std::runtime_error("add_rmsnorm_zero_centered_nvfp4: hidden must be a multiple of 64 within the wide kernel, "
                             "weight [hidden]");
  }
  if (data.ndim() != 2 || data.size(0) != m || data.size(1) != k / 2 ||
      ffi::NumBytes(scales) < xinfer::nvfp4::ScaleStorageBytes(m, k)) {
    throw std::runtime_error("add_rmsnorm_zero_centered_nvfp4: data must be uint8 [M, K / 2] and scales hold the "
                             "SWIZZLE_32_4_4 storage");
  }
  for (const void* ptr : {sum_output.data_ptr(), norm_output.has_value() ? norm_output.value().data_ptr() : nullptr,
                          residual.data_ptr(), input.data_ptr(), weight.data_ptr()}) {
    if (reinterpret_cast<uintptr_t>(ptr) % 16) throw std::runtime_error("add_rmsnorm_zero_centered_nvfp4: unaligned");
  }
  if (reinterpret_cast<uintptr_t>(data.data_ptr()) % 4) {
    throw std::runtime_error("add_rmsnorm_zero_centered_nvfp4: data must be 4-byte aligned (four bytes per store)");
  }
  if (m == 0) return;
  const Nvfp4Out nvfp4{static_cast<uint8_t*>(data.data_ptr()), static_cast<uint8_t*>(scales.data_ptr()),
                       static_cast<int>(k / 64), m};
  cudaStream_t stream = static_cast<cudaStream_t>(
      TVMFFIEnvGetStream(input.device().device_type, input.device().device_id));
  auto* sum_ptr = static_cast<__nv_bfloat16*>(sum_output.data_ptr());
  auto* norm_ptr = norm_output.has_value() ? static_cast<__nv_bfloat16*>(norm_output.value().data_ptr()) : nullptr;
  const auto* res_ptr = static_cast<const __nv_bfloat16*>(residual.data_ptr());
  const auto* in_ptr = static_cast<const __nv_bfloat16*>(input.data_ptr());
  if (IsFloat32(weight.dtype())) {
    LaunchAddRMSNormNvfp4(sum_ptr, norm_ptr, res_ptr, in_ptr, static_cast<const float*>(weight.data_ptr()),
                          static_cast<float>(epsilon), k, m, nvfp4, stream);
  } else if (IsBFloat16(weight.dtype())) {
    LaunchAddRMSNormNvfp4(sum_ptr, norm_ptr, res_ptr, in_ptr, static_cast<const __nv_bfloat16*>(weight.data_ptr()),
                          static_cast<float>(epsilon), k, m, nvfp4, stream);
  } else {
    throw std::runtime_error("add_rmsnorm_zero_centered_nvfp4: weight must be FP32 or BF16");
  }
  sync_check_cuda_error();
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(rmsnorm, tvm_ffi_rmsnorm::RMSNorm);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(rmsnorm_zero_centered, tvm_ffi_rmsnorm::RMSNormZeroCentered);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(add_rmsnorm, tvm_ffi_rmsnorm::AddRMSNorm);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(add_rmsnorm_zero_centered, tvm_ffi_rmsnorm::AddRMSNormZeroCentered);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(add_rmsnorm_zero_centered_nvfp4, tvm_ffi_rmsnorm::AddRMSNormZeroCenteredNvfp4);

// ======================== Backward: Persistent Kernel + Partial Reduction ========================
//
// Launch num_blocks = min(num_tokens, SM_count) persistent blocks.
// Each block loops over assigned rows computing dx, accumulating dw in registers.
// After the loop, each block writes its partial dw to partial_dw[block_id, hidden_size].
// A lightweight reduction kernel sums partial_dw into final dw.
//
// Advantages:
//   - No atomicAdd contention (lock-free)
//   - Weight W loaded once into L1 cache, reused across all rows
//   - X and dY read exactly once per row, producing both dX and partial dW
//

// ======================== Gated RMSNorm ========================

// ======================== Adaptive RMSNorm ========================

struct AdaRMSNormShape {
  int64_t num_tokens;
  int64_t hidden_size;
  int64_t num_mod_rows;
  int64_t tokens_per_mod;
};

bool SameDType(const DLDataType& lhs, const DLDataType& rhs) {
  return lhs.code == rhs.code && lhs.bits == rhs.bits && lhs.lanes == rhs.lanes;
}

AdaRMSNormShape GetAdaRMSNormShape(const tvm::ffi::TensorView& input,
                                   const tvm::ffi::TensorView& modulation) {
  const RMSNormShape input_shape = GetRMSNormShape(input);
  const int64_t hidden_size = input_shape.hidden_size;
  ASSERT_CHECK(modulation.ndim() >= 1);
  ASSERT_CHECK(modulation.IsContiguous());
  const int mod_ndim = modulation.ndim();
  ASSERT_CHECK(modulation.size(mod_ndim - 1) == hidden_size * 3);

  if (mod_ndim == input.ndim()) {
    bool same_prefix = true;
    for (int i = 0; i < input.ndim() - 1; ++i) {
      if (modulation.size(i) != input.size(i)) {
        same_prefix = false;
        break;
      }
    }
    if (same_prefix) {
      return {input_shape.num_tokens, hidden_size, input_shape.num_tokens, 1};
    }

    if (input.ndim() == 3 && modulation.size(0) == input.size(0) &&
        modulation.size(1) == 1) {
      return {input_shape.num_tokens, hidden_size, input.size(0), input.size(1)};
    }
  }

  if (input.ndim() == 3 && mod_ndim == 2 && modulation.size(0) == input.size(0)) {
    return {input_shape.num_tokens, hidden_size, input.size(0), input.size(1)};
  }

  throw std::runtime_error(
      "Unsupported AdaRMSNorm shapes: modulation must match input prefix or be "
      "[batch, 3 * hidden] for a [batch, seq, hidden] input");
}

void CheckAdaGateShape(const tvm::ffi::TensorView& gate,
                       const AdaRMSNormShape& shape) {
  ASSERT_CHECK(gate.ndim() >= 1);
  ASSERT_CHECK(gate.IsContiguous());
  ASSERT_CHECK(gate.size(gate.ndim() - 1) == shape.hidden_size);
  int64_t gate_rows = 1;
  for (int i = 0; i < gate.ndim() - 1; ++i) {
    gate_rows *= gate.size(i);
  }
  ASSERT_CHECK(gate_rows == shape.num_mod_rows);
}

// Optional row split of a shared modulation (RTC denoise): in batch row b the
// first pinned_rows[b] tokens use pinned_modulation, laid out like the shared
// [batch, 3 * hidden] modulation, and the rest keep modulation. With a split
// every token owns a gate row, so gates are [num_tokens, hidden].
struct AdaPinned {
  const void* modulation = nullptr;
  const int* rows = nullptr;
};

AdaPinned GetAdaPinned(
    const tvm::ffi::Optional<tvm::ffi::TensorView>& pinned_modulation,
    const tvm::ffi::Optional<tvm::ffi::TensorView>& pinned_rows,
    const tvm::ffi::TensorView& modulation,
    const AdaRMSNormShape& shape) {
  if (pinned_modulation.has_value() != pinned_rows.has_value()) {
    throw std::runtime_error(
        "AdaRMSNorm: pinned_modulation and pinned_rows must be given together");
  }
  if (!pinned_modulation.has_value()) {
    return {};
  }
  const tvm::ffi::TensorView pinned = pinned_modulation.value();
  const tvm::ffi::TensorView rows = pinned_rows.value();
  const auto device = modulation.device();
  if (pinned.device().device_type != device.device_type ||
      pinned.device().device_id != device.device_id ||
      rows.device().device_type != device.device_type ||
      rows.device().device_id != device.device_id) {
    throw std::runtime_error(
        "AdaRMSNorm: pinned_modulation and pinned_rows must be on the "
        "modulation device");
  }
  if (!SameDType(pinned.dtype(), modulation.dtype()) || !pinned.IsContiguous()) {
    throw std::runtime_error(
        "AdaRMSNorm: pinned_modulation must be contiguous with the modulation dtype");
  }
  int64_t pinned_elements = 1;
  for (int i = 0; i < pinned.ndim(); ++i) pinned_elements *= pinned.size(i);
  if (shape.num_mod_rows * shape.tokens_per_mod != shape.num_tokens ||
      pinned_elements != shape.num_mod_rows * shape.hidden_size * 3) {
    throw std::runtime_error(
        "AdaRMSNorm: pinned_modulation needs a shared [batch, 3 * hidden] "
        "modulation and the same shape");
  }
  if (rows.dtype().code != kDLInt || rows.dtype().bits != 32 ||
      rows.dtype().lanes != 1 || !rows.IsContiguous() || rows.ndim() != 1 ||
      rows.size(0) != shape.num_mod_rows) {
    throw std::runtime_error("AdaRMSNorm: pinned_rows must be contiguous int32 [batch]");
  }
  return {pinned.data_ptr(), static_cast<const int*>(rows.data_ptr())};
}

void CheckAdaGateRows(const tvm::ffi::TensorView& gate, int64_t rows,
                      int64_t hidden_size, const char* name) {
  int64_t gate_rows = 1;
  for (int i = 0; i + 1 < gate.ndim(); ++i) gate_rows *= gate.size(i);
  if (gate.ndim() < 1 || !gate.IsContiguous() ||
      gate.size(gate.ndim() - 1) != hidden_size || gate_rows != rows) {
    throw std::runtime_error(
        std::string("AdaRMSNorm: ") + name + " must be contiguous with " +
        std::to_string(rows) + " rows of hidden_size");
  }
}

template<typename T, typename ModDType, bool kPinned = false>
// Launched with up to 1024 threads (block = min(hidden, 1024)): the bound caps registers at 64 so every
// instantiation stays launchable when compiled natively (several need 66-72 registers unbounded on sm_110).
__global__ void __launch_bounds__(1024) adarmsnormKernel(
    T* __restrict__ output,
    T* __restrict__ gate,
    const T* __restrict__ input,
    const ModDType* __restrict__ modulation,
    const float epsilon,
    const int64_t hidden_size,
    const int64_t tokens_per_mod,
    const int64_t input_row_stride,
    const int64_t output_row_stride,
    const ModDType* __restrict__ pinned_modulation,
    const int* __restrict__ pinned_rows) {
    using Vec = typename VecTraits<T>::Type;
    constexpr int VEC_SIZE = VecTraits<T>::vec_size;

    float square_sum = 0.0f;
    __shared__ float inv_rms;
    const int64_t tid = threadIdx.x;
    const int64_t row = blockIdx.x;
    const int64_t input_offset = row * input_row_stride;
    const int64_t output_offset = row * output_row_stride;
    const int64_t mod_row = row / tokens_per_mod;
    const int64_t mod_base = mod_row * hidden_size * 3;
    const int64_t vec_hidden = hidden_size / VEC_SIZE;
    const ModDType* mod = modulation;
    int64_t gate_offset = mod_row * hidden_size;
    bool writes_gate = (row % tokens_per_mod) == 0;
    if constexpr (kPinned) {
        if (row % tokens_per_mod < pinned_rows[mod_row]) mod = pinned_modulation;
        gate_offset = row * hidden_size;
        writes_gate = true;
    }

    for (int64_t i = tid; i < vec_hidden; i += blockDim.x) {
        Vec x_vec = VecTraits<T>::load(input + input_offset + i * VEC_SIZE);
        float x_vals[VEC_SIZE];
        VecToFloats<T>(x_vec, x_vals);
#pragma unroll
        for (int j = 0; j < VEC_SIZE; ++j) {
            square_sum += x_vals[j] * x_vals[j];
        }
    }
    square_sum = blockReduceSum(square_sum);
    if (tid == 0) inv_rms = rsqrtf(square_sum / static_cast<float>(hidden_size) + epsilon);
    __syncthreads();

    for (int64_t i = tid; i < vec_hidden; i += blockDim.x) {
        Vec in_vec = VecTraits<T>::load(input + input_offset + i * VEC_SIZE);
        float x_vals[VEC_SIZE], scale_vals[VEC_SIZE], shift_vals[VEC_SIZE], out_vals[VEC_SIZE];
        VecToFloats<T>(in_vec, x_vals);
        LoadWeightFloats<ModDType, VEC_SIZE>(mod, mod_base + i * VEC_SIZE, scale_vals);
        LoadWeightFloats<ModDType, VEC_SIZE>(
            mod, mod_base + hidden_size + i * VEC_SIZE, shift_vals);
#pragma unroll
        for (int j = 0; j < VEC_SIZE; ++j) {
            out_vals[j] = x_vals[j] * inv_rms * (1.0f + scale_vals[j]) + shift_vals[j];
        }
        VecTraits<T>::store(output + output_offset + i * VEC_SIZE, FloatsToVec<T>(out_vals));
    }

    if (writes_gate) {
        for (int64_t i = tid; i < vec_hidden; i += blockDim.x) {
            float gate_vals[VEC_SIZE];
            LoadWeightFloats<ModDType, VEC_SIZE>(
                mod, mod_base + hidden_size * 2 + i * VEC_SIZE, gate_vals);
            VecTraits<T>::store(gate + gate_offset + i * VEC_SIZE, FloatsToVec<T>(gate_vals));
        }
    }
}

template<typename T, typename ModDType>
void LaunchAdaRMSNormKernel(
    tvm::ffi::TensorView output,
    tvm::ffi::TensorView gate,
    tvm::ffi::TensorView input,
    tvm::ffi::TensorView modulation,
    float epsilon,
    int64_t hidden_size,
    int64_t num_tokens,
    int64_t tokens_per_mod,
    int64_t input_row_stride,
    int64_t output_row_stride,
    int64_t block_size,
    const AdaPinned& pinned,
    cudaStream_t stream) {
  auto launch = [&](auto kernel) {
    kernel<<<num_tokens, block_size, 0, stream>>>(
        static_cast<T*>(output.data_ptr()), static_cast<T*>(gate.data_ptr()),
        static_cast<const T*>(input.data_ptr()),
        static_cast<const ModDType*>(modulation.data_ptr()), epsilon, hidden_size,
        tokens_per_mod, input_row_stride, output_row_stride,
        static_cast<const ModDType*>(pinned.modulation), pinned.rows);
  };
  if (pinned.rows != nullptr) {
    launch(adarmsnormKernel<T, ModDType, true>);
  } else {
    launch(adarmsnormKernel<T, ModDType, false>);
  }
}

template<typename T>
void DispatchAdaRMSNormKernel(
    tvm::ffi::TensorView output,
    tvm::ffi::TensorView gate,
    tvm::ffi::TensorView input,
    tvm::ffi::TensorView modulation,
    float epsilon,
    int64_t hidden_size,
    int64_t num_tokens,
    int64_t tokens_per_mod,
    int64_t input_row_stride,
    int64_t output_row_stride,
    int64_t block_size,
    const AdaPinned& pinned,
    cudaStream_t stream) {
  if (IsFloat32(modulation.dtype()))
    LaunchAdaRMSNormKernel<T, float>(
        output, gate, input, modulation, epsilon, hidden_size, num_tokens,
        tokens_per_mod, input_row_stride, output_row_stride, block_size, pinned,
        stream);
  else if (IsFloat16(modulation.dtype()))
    LaunchAdaRMSNormKernel<T, half>(
        output, gate, input, modulation, epsilon, hidden_size, num_tokens,
        tokens_per_mod, input_row_stride, output_row_stride, block_size, pinned,
        stream);
  else if (IsBFloat16(modulation.dtype()))
    LaunchAdaRMSNormKernel<T, __nv_bfloat16>(
        output, gate, input, modulation, epsilon, hidden_size, num_tokens,
        tokens_per_mod, input_row_stride, output_row_stride, block_size, pinned,
        stream);
  else throw std::runtime_error("Unsupported modulation data type for AdaRMSNorm");
}

void AdaRMSNorm(tvm::ffi::TensorView output, tvm::ffi::TensorView gate,
                tvm::ffi::TensorView input, tvm::ffi::TensorView modulation,
                tvm::ffi::Optional<tvm::ffi::TensorView> pinned_modulation,
                tvm::ffi::Optional<tvm::ffi::TensorView> pinned_rows,
                double epsilon) {
  ASSERT_CHECK(SameDType(output.dtype(), input.dtype()));
  ASSERT_CHECK(SameDType(gate.dtype(), input.dtype()));
  CheckSameShape(output, input);
  const AdaRMSNormShape shape = GetAdaRMSNormShape(input, modulation);
  const RMSNormShape input_shape = GetRMSNormShape(input);
  const RMSNormShape output_shape = GetRMSNormShape(output);
  ASSERT_CHECK(output_shape.hidden_size == input_shape.hidden_size);
  ASSERT_CHECK(output_shape.num_tokens == input_shape.num_tokens);
  const AdaPinned pinned =
      GetAdaPinned(pinned_modulation, pinned_rows, modulation, shape);
  if (pinned.rows != nullptr) {
    CheckAdaGateRows(gate, shape.num_tokens, shape.hidden_size, "gate");
  } else {
    CheckAdaGateShape(gate, shape);
  }
  ASSERT_CHECK(shape.hidden_size % GetVecSize(input.dtype()) == 0);
  ASSERT_CHECK(input_shape.row_stride % GetVecSize(input.dtype()) == 0);
  ASSERT_CHECK(output_shape.row_stride % GetVecSize(input.dtype()) == 0);
  cudaStream_t stream = static_cast<cudaStream_t>(
      TVMFFIEnvGetStream(input.device().device_type, input.device().device_id));
  const int64_t block_size = std::min(shape.hidden_size, 1024L);

  if (IsFloat32(input.dtype()))
    DispatchAdaRMSNormKernel<float>(
        output, gate, input, modulation, static_cast<float>(epsilon),
        shape.hidden_size, shape.num_tokens, shape.tokens_per_mod,
        input_shape.row_stride, output_shape.row_stride, block_size, pinned,
        stream);
  else if (IsFloat16(input.dtype()))
    DispatchAdaRMSNormKernel<half>(
        output, gate, input, modulation, static_cast<float>(epsilon),
        shape.hidden_size, shape.num_tokens, shape.tokens_per_mod,
        input_shape.row_stride, output_shape.row_stride, block_size, pinned,
        stream);
  else if (IsBFloat16(input.dtype()))
    DispatchAdaRMSNormKernel<__nv_bfloat16>(
        output, gate, input, modulation, static_cast<float>(epsilon),
        shape.hidden_size, shape.num_tokens, shape.tokens_per_mod,
        input_shape.row_stride, output_shape.row_stride, block_size, pinned,
        stream);
  else throw std::runtime_error("Unsupported data type for AdaRMSNorm");
  sync_check_cuda_error();
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(adarmsnorm, tvm_ffi_rmsnorm::AdaRMSNorm);

// Compute
//   residual_output = residual + round_to_T(branch * branch_gate)
//   output, gate = AdaRMSNorm(residual_output, modulation)
// in one launch.  The explicit round trip through T preserves the two BF16
// rounding points of the original eager multiply followed by add.
template<typename T, typename ModDType, bool kPinned = false>
// Launched with up to 1024 threads (block = min(hidden, 1024)): the bound caps registers at 64 so every
// instantiation stays launchable when compiled natively (several need 66-72 registers unbounded on sm_110).
__global__ void __launch_bounds__(1024) gatedResidualAdaRMSNormKernel(
    T* __restrict__ output,
    T* __restrict__ gate,
    T* __restrict__ residual_output,
    const T* __restrict__ residual,
    const T* __restrict__ branch,
    const T* __restrict__ branch_gate,
    const ModDType* __restrict__ modulation,
    const float epsilon,
    const int64_t hidden_size,
    const int64_t tokens_per_mod,
    const ModDType* __restrict__ pinned_modulation,
    const int* __restrict__ pinned_rows) {
  using Vec = typename VecTraits<T>::Type;
  constexpr int VEC_SIZE = VecTraits<T>::vec_size;

  float square_sum = 0.0f;
  __shared__ float inv_rms;
  const int64_t tid = threadIdx.x;
  const int64_t row = blockIdx.x;
  const int64_t row_offset = row * hidden_size;
  const int64_t mod_row = row / tokens_per_mod;
  const int64_t mod_base = mod_row * hidden_size * 3;
  const int64_t vec_hidden = hidden_size / VEC_SIZE;
  // branch_gate and gate share this row: [batch] rows, or [tokens] when split.
  const ModDType* mod = modulation;
  int64_t gate_offset = mod_row * hidden_size;
  bool writes_gate = (row % tokens_per_mod) == 0;
  if constexpr (kPinned) {
    if (row % tokens_per_mod < pinned_rows[mod_row]) mod = pinned_modulation;
    gate_offset = row_offset;
    writes_gate = true;
  }

  for (int64_t i = tid; i < vec_hidden; i += blockDim.x) {
    Vec residual_vec =
        VecTraits<T>::load(residual + row_offset + i * VEC_SIZE);
    Vec branch_vec = VecTraits<T>::load(branch + row_offset + i * VEC_SIZE);
    Vec branch_gate_vec =
        VecTraits<T>::load(branch_gate + gate_offset + i * VEC_SIZE);
    float residual_vals[VEC_SIZE], branch_vals[VEC_SIZE];
    float branch_gate_vals[VEC_SIZE], product_vals[VEC_SIZE];
    float rounded_product_vals[VEC_SIZE], residual_output_vals[VEC_SIZE];
    VecToFloats<T>(residual_vec, residual_vals);
    VecToFloats<T>(branch_vec, branch_vals);
    VecToFloats<T>(branch_gate_vec, branch_gate_vals);
#pragma unroll
    for (int j = 0; j < VEC_SIZE; ++j) {
      product_vals[j] = branch_vals[j] * branch_gate_vals[j];
    }
    Vec rounded_product_vec = FloatsToVec<T>(product_vals);
    VecToFloats<T>(rounded_product_vec, rounded_product_vals);
#pragma unroll
    for (int j = 0; j < VEC_SIZE; ++j) {
      residual_output_vals[j] = residual_vals[j] + rounded_product_vals[j];
    }
    Vec residual_output_vec = FloatsToVec<T>(residual_output_vals);
    VecTraits<T>::store(
        residual_output + row_offset + i * VEC_SIZE, residual_output_vec);
    // RMSNorm consumes the rounded residual output, not the FP32 accumulator.
    VecToFloats<T>(residual_output_vec, residual_output_vals);
#pragma unroll
    for (int j = 0; j < VEC_SIZE; ++j) {
      square_sum += residual_output_vals[j] * residual_output_vals[j];
    }
  }

  square_sum = blockReduceSum(square_sum);
  if (tid == 0) {
    inv_rms = rsqrtf(
        square_sum / static_cast<float>(hidden_size) + epsilon);
  }
  __syncthreads();

  for (int64_t i = tid; i < vec_hidden; i += blockDim.x) {
    Vec x_vec =
        VecTraits<T>::load(residual_output + row_offset + i * VEC_SIZE);
    float x_vals[VEC_SIZE], scale_vals[VEC_SIZE], shift_vals[VEC_SIZE];
    float out_vals[VEC_SIZE];
    VecToFloats<T>(x_vec, x_vals);
    LoadWeightFloats<ModDType, VEC_SIZE>(
        mod, mod_base + i * VEC_SIZE, scale_vals);
    LoadWeightFloats<ModDType, VEC_SIZE>(
        mod, mod_base + hidden_size + i * VEC_SIZE, shift_vals);
#pragma unroll
    for (int j = 0; j < VEC_SIZE; ++j) {
      out_vals[j] =
          x_vals[j] * inv_rms * (1.0f + scale_vals[j]) + shift_vals[j];
    }
    VecTraits<T>::store(
        output + row_offset + i * VEC_SIZE, FloatsToVec<T>(out_vals));
  }

  if (writes_gate) {
    for (int64_t i = tid; i < vec_hidden; i += blockDim.x) {
      float gate_vals[VEC_SIZE];
      LoadWeightFloats<ModDType, VEC_SIZE>(
          mod, mod_base + hidden_size * 2 + i * VEC_SIZE, gate_vals);
      VecTraits<T>::store(
          gate + gate_offset + i * VEC_SIZE, FloatsToVec<T>(gate_vals));
    }
  }
}

template<typename T, typename ModDType>
void LaunchGatedResidualAdaRMSNormKernel(
    tvm::ffi::TensorView output,
    tvm::ffi::TensorView gate,
    tvm::ffi::TensorView residual_output,
    tvm::ffi::TensorView residual,
    tvm::ffi::TensorView branch,
    tvm::ffi::TensorView branch_gate,
    tvm::ffi::TensorView modulation,
    float epsilon,
    int64_t hidden_size,
    int64_t num_tokens,
    int64_t tokens_per_mod,
    int64_t block_size,
    const AdaPinned& pinned,
    cudaStream_t stream) {
  auto launch = [&](auto kernel) {
    kernel<<<num_tokens, block_size, 0, stream>>>(
        static_cast<T*>(output.data_ptr()),
        static_cast<T*>(gate.data_ptr()),
        static_cast<T*>(residual_output.data_ptr()),
        static_cast<const T*>(residual.data_ptr()),
        static_cast<const T*>(branch.data_ptr()),
        static_cast<const T*>(branch_gate.data_ptr()),
        static_cast<const ModDType*>(modulation.data_ptr()), epsilon,
        hidden_size, tokens_per_mod,
        static_cast<const ModDType*>(pinned.modulation), pinned.rows);
  };
  if (pinned.rows != nullptr) {
    launch(gatedResidualAdaRMSNormKernel<T, ModDType, true>);
  } else {
    launch(gatedResidualAdaRMSNormKernel<T, ModDType, false>);
  }
}

template<typename T>
void DispatchGatedResidualAdaRMSNormKernel(
    tvm::ffi::TensorView output,
    tvm::ffi::TensorView gate,
    tvm::ffi::TensorView residual_output,
    tvm::ffi::TensorView residual,
    tvm::ffi::TensorView branch,
    tvm::ffi::TensorView branch_gate,
    tvm::ffi::TensorView modulation,
    float epsilon,
    int64_t hidden_size,
    int64_t num_tokens,
    int64_t tokens_per_mod,
    int64_t block_size,
    const AdaPinned& pinned,
    cudaStream_t stream) {
  if (IsFloat32(modulation.dtype())) {
    LaunchGatedResidualAdaRMSNormKernel<T, float>(
        output, gate, residual_output, residual, branch, branch_gate,
        modulation, epsilon, hidden_size, num_tokens, tokens_per_mod,
        block_size, pinned, stream);
  } else if (IsFloat16(modulation.dtype())) {
    LaunchGatedResidualAdaRMSNormKernel<T, half>(
        output, gate, residual_output, residual, branch, branch_gate,
        modulation, epsilon, hidden_size, num_tokens, tokens_per_mod,
        block_size, pinned, stream);
  } else if (IsBFloat16(modulation.dtype())) {
    LaunchGatedResidualAdaRMSNormKernel<T, __nv_bfloat16>(
        output, gate, residual_output, residual, branch, branch_gate,
        modulation, epsilon, hidden_size, num_tokens, tokens_per_mod,
        block_size, pinned, stream);
  } else {
    throw std::runtime_error(
        "Unsupported modulation data type for GatedResidualAdaRMSNorm");
  }
}

void GatedResidualAdaRMSNormImpl(
    tvm::ffi::TensorView output,
    tvm::ffi::TensorView gate,
    tvm::ffi::TensorView residual_output,
    tvm::ffi::TensorView residual,
    tvm::ffi::TensorView branch,
    tvm::ffi::TensorView branch_gate,
    tvm::ffi::TensorView modulation,
    const tvm::ffi::Optional<tvm::ffi::TensorView>& pinned_modulation,
    const tvm::ffi::Optional<tvm::ffi::TensorView>& pinned_rows,
    double epsilon) {
  xinfer::ffi::CheckCuda(residual, "residual");
  const std::pair<const tvm::ffi::TensorView*, const char*> same_as_residual[] = {
      {&output, "output"}, {&gate, "gate"}, {&residual_output, "residual_output"},
      {&branch, "branch"}, {&branch_gate, "branch_gate"}, {&modulation, "modulation"}};
  for (const auto& [tensor, name] : same_as_residual) {
    xinfer::ffi::CheckSameDevice(*tensor, residual, name);
    if (tensor != &modulation && !SameDType(tensor->dtype(), residual.dtype())) {
      throw std::runtime_error(std::string("gated_residual_adarmsnorm: ") + name +
                               " must have the residual dtype");
    }
  }
  for (const auto& [tensor, name] : {std::pair{&output, "output"},
                                     std::pair{&residual_output, "residual_output"},
                                     std::pair{&branch, "branch"}}) {
    xinfer::ffi::CheckSameShape(*tensor, residual, name);
    xinfer::ffi::CheckContiguous(*tensor, name);
  }
  xinfer::ffi::CheckContiguous(residual, "residual");
  // residual/branch are read before output/residual_output are written, but
  // per element; overlapping buffers would still race across the row.
  xinfer::ffi::CheckNoOverlap(output, "output", residual, "residual");
  xinfer::ffi::CheckNoOverlap(residual_output, "residual_output", branch, "branch");
  const AdaRMSNormShape shape = GetAdaRMSNormShape(residual, modulation);
  const AdaPinned pinned =
      GetAdaPinned(pinned_modulation, pinned_rows, modulation, shape);
  if (pinned.rows != nullptr) {
    CheckAdaGateRows(gate, shape.num_tokens, shape.hidden_size, "gate");
    CheckAdaGateRows(branch_gate, shape.num_tokens, shape.hidden_size, "branch_gate");
  } else {
    CheckAdaGateShape(gate, shape);
    CheckAdaGateShape(branch_gate, shape);
  }
  if (shape.hidden_size % GetVecSize(residual.dtype()) != 0) {
    throw std::runtime_error(
        "gated_residual_adarmsnorm: hidden size must be a multiple of the vector width");
  }
  cudaStream_t stream = xinfer::ffi::Stream(residual);
  const int64_t block_size = std::min(shape.hidden_size, 1024L);

  if (IsFloat32(residual.dtype())) {
    DispatchGatedResidualAdaRMSNormKernel<float>(
        output, gate, residual_output, residual, branch, branch_gate,
        modulation, static_cast<float>(epsilon), shape.hidden_size,
        shape.num_tokens, shape.tokens_per_mod, block_size, pinned,
        stream);
  } else if (IsFloat16(residual.dtype())) {
    DispatchGatedResidualAdaRMSNormKernel<half>(
        output, gate, residual_output, residual, branch, branch_gate,
        modulation, static_cast<float>(epsilon), shape.hidden_size,
        shape.num_tokens, shape.tokens_per_mod, block_size, pinned,
        stream);
  } else if (IsBFloat16(residual.dtype())) {
    DispatchGatedResidualAdaRMSNormKernel<__nv_bfloat16>(
        output, gate, residual_output, residual, branch, branch_gate,
        modulation, static_cast<float>(epsilon), shape.hidden_size,
        shape.num_tokens, shape.tokens_per_mod, block_size, pinned,
        stream);
  } else {
    throw std::runtime_error(
        "Unsupported data type for GatedResidualAdaRMSNorm");
  }
  sync_check_cuda_error();
}

void GatedResidualAdaRMSNorm(
    tvm::ffi::TensorView output,
    tvm::ffi::TensorView gate,
    tvm::ffi::TensorView residual_output,
    tvm::ffi::TensorView residual,
    tvm::ffi::TensorView branch,
    tvm::ffi::TensorView branch_gate,
    tvm::ffi::TensorView modulation,
    tvm::ffi::Optional<tvm::ffi::TensorView> pinned_modulation,
    tvm::ffi::Optional<tvm::ffi::TensorView> pinned_rows,
    double epsilon) {
  GatedResidualAdaRMSNormImpl(
      output, gate, residual_output, residual, branch, branch_gate,
      modulation, pinned_modulation, pinned_rows, epsilon);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(
    gated_residual_adarmsnorm,
    tvm_ffi_rmsnorm::GatedResidualAdaRMSNorm);

}  // namespace tvm_ffi_rmsnorm
