#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "common/cuda_utils.h"

namespace tvm_ffi_double_adarmsnorm {
namespace {

constexpr int kHidden = 1024;
constexpr int kThreads = 512;

__device__ __forceinline__ float WarpReduceSum(float value) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffff, value, offset);
  }
  return value;
}

// Keep the exact reduction tree used by the AdaRMSNorm kernel (rmsnorm.cu).
template <typename T>
__device__ __forceinline__ float ModValue(T value) {
  return static_cast<float>(value);
}

template <>
__device__ __forceinline__ float ModValue<__nv_bfloat16>(__nv_bfloat16 value) {
  return __bfloat162float(value);
}

__device__ __forceinline__ float BlockReduceSum(float value) {
  static __shared__ float shared[32];
  const int lane = threadIdx.x % 32;
  const int warp = threadIdx.x / 32;
  value = WarpReduceSum(value);
  if (lane == 0) shared[warp] = value;
  __syncthreads();
  const int num_warps = (blockDim.x + 31) / 32;
  value = threadIdx.x < num_warps ? shared[lane] : 0.0f;
  if (warp == 0) value = WarpReduceSum(value);
  return value;
}

// Fuses two strict-BF16 state transitions around an identity attention
// layer.  The BF16 materialization boundaries are reproduced in registers so
// the result is bitwise identical to the two AdaRMSNorm kernels.
// kPinned: the RTC row split of adarmsnorm (see rmsnorm.cu). The first
// pinned_rows[b] tokens of batch row b read the pinned modulations, and gates
// are per token.
template <typename ModT, bool kPinned = false>
__global__ __launch_bounds__(kThreads)
void DoubleAdaRMSNormBf16H1024Kernel(
    __nv_bfloat16* __restrict__ residual,
    __nv_bfloat16* __restrict__ branch,
    __nv_bfloat16* __restrict__ next_gate,
    const __nv_bfloat16* __restrict__ previous_gate,
    const ModT* __restrict__ modulation1_shared,
    const ModT* __restrict__ modulation2_shared,
    float epsilon,
    int tokens_per_mod,
    const ModT* __restrict__ pinned_modulation1,
    const ModT* __restrict__ pinned_modulation2,
    const int* __restrict__ pinned_rows) {
  __shared__ float inverse_rms1;
  __shared__ float inverse_rms2;

  const int row = blockIdx.x;
  const int lane_word = threadIdx.x;
  const int scalar0 = 2 * lane_word;
  const int scalar1 = scalar0 + 1;
  const int row_base = row * kHidden;
  const int mod_row = row / tokens_per_mod;
  const int mod_base = mod_row * (3 * kHidden);
  const ModT* modulation1 = modulation1_shared;
  const ModT* modulation2 = modulation2_shared;
  int gate_base = mod_row * kHidden;
  bool writes_gate = (row % tokens_per_mod) == 0;
  if constexpr (kPinned) {
    if (row % tokens_per_mod < pinned_rows[mod_row]) {
      modulation1 = pinned_modulation1;
      modulation2 = pinned_modulation2;
    }
    gate_base = row_base;
    writes_gate = true;
  }

  const float branch0 = __bfloat162float(branch[row_base + scalar0]);
  const float branch1 = __bfloat162float(branch[row_base + scalar1]);
  const float previous_gate0 =
      __bfloat162float(previous_gate[gate_base + scalar0]);
  const float previous_gate1 =
      __bfloat162float(previous_gate[gate_base + scalar1]);
  const float residual0 = __bfloat162float(residual[row_base + scalar0]);
  const float residual1 = __bfloat162float(residual[row_base + scalar1]);

  const float product1_0 = __bfloat162float(
      __float2bfloat16(branch0 * previous_gate0));
  const float product1_1 = __bfloat162float(
      __float2bfloat16(branch1 * previous_gate1));
  const __nv_bfloat16 r1_bf16_0 = __float2bfloat16(residual0 + product1_0);
  const __nv_bfloat16 r1_bf16_1 = __float2bfloat16(residual1 + product1_1);
  const float r1_0 = __bfloat162float(r1_bf16_0);
  const float r1_1 = __bfloat162float(r1_bf16_1);

  float square_sum1 = fmaf(r1_0, r1_0, 0.0f);
  square_sum1 = fmaf(r1_1, r1_1, square_sum1);
  square_sum1 = BlockReduceSum(square_sum1);
  if (lane_word == 0) {
    inverse_rms1 = rsqrtf(square_sum1 / float(kHidden) + epsilon);
  }
  __syncthreads();

  const __nv_bfloat16 normalized1_bf16_0 = __float2bfloat16(
      r1_0 * inverse_rms1 * (1.0f + ModValue(modulation1[mod_base + scalar0])) +
      ModValue(modulation1[mod_base + kHidden + scalar0]));
  const __nv_bfloat16 normalized1_bf16_1 = __float2bfloat16(
      r1_1 * inverse_rms1 * (1.0f + ModValue(modulation1[mod_base + scalar1])) +
      ModValue(modulation1[mod_base + kHidden + scalar1]));
  const float normalized1_0 = __bfloat162float(normalized1_bf16_0);
  const float normalized1_1 = __bfloat162float(normalized1_bf16_1);
  const float gate1_0 = __bfloat162float(
      __float2bfloat16(ModValue(modulation1[mod_base + 2 * kHidden + scalar0])));
  const float gate1_1 = __bfloat162float(
      __float2bfloat16(ModValue(modulation1[mod_base + 2 * kHidden + scalar1])));

  const float product2_0 = __bfloat162float(
      __float2bfloat16(normalized1_0 * gate1_0));
  const float product2_1 = __bfloat162float(
      __float2bfloat16(normalized1_1 * gate1_1));
  const __nv_bfloat16 r2_bf16_0 = __float2bfloat16(r1_0 + product2_0);
  const __nv_bfloat16 r2_bf16_1 = __float2bfloat16(r1_1 + product2_1);
  const float r2_0 = __bfloat162float(r2_bf16_0);
  const float r2_1 = __bfloat162float(r2_bf16_1);

  float square_sum2 = fmaf(r2_0, r2_0, 0.0f);
  square_sum2 = fmaf(r2_1, r2_1, square_sum2);
  square_sum2 = BlockReduceSum(square_sum2);
  if (lane_word == 0) {
    inverse_rms2 = rsqrtf(square_sum2 / float(kHidden) + epsilon);
  }
  __syncthreads();

  residual[row_base + scalar0] = r2_bf16_0;
  residual[row_base + scalar1] = r2_bf16_1;
  branch[row_base + scalar0] = __float2bfloat16(
      r2_0 * inverse_rms2 * (1.0f + ModValue(modulation2[mod_base + scalar0])) +
      ModValue(modulation2[mod_base + kHidden + scalar0]));
  branch[row_base + scalar1] = __float2bfloat16(
      r2_1 * inverse_rms2 * (1.0f + ModValue(modulation2[mod_base + scalar1])) +
      ModValue(modulation2[mod_base + kHidden + scalar1]));

  if (writes_gate) {
    next_gate[gate_base + scalar0] =
        __float2bfloat16(ModValue(modulation2[mod_base + 2 * kHidden + scalar0]));
    next_gate[gate_base + scalar1] =
        __float2bfloat16(ModValue(modulation2[mod_base + 2 * kHidden + scalar1]));
  }
}

bool SameDType(const DLDataType& lhs, const DLDataType& rhs) {
  return lhs.code == rhs.code && lhs.bits == rhs.bits &&
      lhs.lanes == rhs.lanes;
}

bool IsBFloat16(const DLDataType& dtype) {
  return dtype.code == kDLBfloat && dtype.bits == 16 && dtype.lanes == 1;
}

bool IsFloat32(const DLDataType& dtype) {
  return dtype.code == kDLFloat && dtype.bits == 32 && dtype.lanes == 1;
}

int64_t NumElements(const tvm::ffi::TensorView& tensor) {
  int64_t result = 1;
  for (int index = 0; index < tensor.ndim(); ++index) {
    result *= tensor.size(index);
  }
  return result;
}

void CheckCudaContiguous(
    const tvm::ffi::TensorView& tensor,
    const char* name,
    int device_id,
    DLDataType expected_dtype) {
  if (tensor.device().device_type != kDLCUDA) {
    throw std::runtime_error(std::string(name) + " must be CUDA");
  }
  if (tensor.device().device_id != device_id) {
    throw std::runtime_error(std::string(name) + " is on a different device");
  }
  if (!SameDType(tensor.dtype(), expected_dtype)) {
    throw std::runtime_error(std::string(name) + " has an unexpected dtype");
  }
  if (!tensor.IsContiguous()) {
    throw std::runtime_error(std::string(name) + " must be contiguous");
  }
}

struct Shape {
  int64_t batch;
  int64_t sequence;
  int64_t tokens;
};

Shape CheckActivation(
    const tvm::ffi::TensorView& tensor, const char* name) {
  if (!IsBFloat16(tensor.dtype())) {
    throw std::runtime_error(std::string(name) + " must be BF16");
  }
  CheckCudaContiguous(
      tensor, name, tensor.device().device_id, tensor.dtype());
  if (tensor.ndim() != 3 || tensor.size(2) != kHidden ||
      tensor.size(0) <= 0 || tensor.size(1) <= 0) {
    throw std::runtime_error(
        std::string(name) + " must have shape [B, S, 1024]");
  }
  const int64_t tokens = tensor.size(0) * tensor.size(1);
  // The kernels index activations as int (row * 1024 + col) and modulations
  // as batch * 3072: bound the largest of those, not just the row count.
  if (tokens * 3 * kHidden > std::numeric_limits<int>::max()) {
    throw std::runtime_error("activation is too large for this kernel");
  }
  return {tensor.size(0), tensor.size(1), tokens};
}

void CheckSameActivation(
    const tvm::ffi::TensorView& tensor,
    const char* name,
    const tvm::ffi::TensorView& reference,
    const Shape& shape) {
  CheckCudaContiguous(
      tensor, name, reference.device().device_id, reference.dtype());
  if (tensor.ndim() != 3 || tensor.size(0) != shape.batch ||
      tensor.size(1) != shape.sequence || tensor.size(2) != kHidden) {
    throw std::runtime_error(std::string(name) + " shape mismatch");
  }
}

// Gates hold one row per batch row, or one per token under a pinned split.
void CheckGate(
    const tvm::ffi::TensorView& tensor,
    const char* name,
    const tvm::ffi::TensorView& reference,
    const Shape& shape,
    bool per_token = false) {
  CheckCudaContiguous(
      tensor, name, reference.device().device_id, reference.dtype());
  const int64_t rows = per_token ? shape.tokens : shape.batch;
  if (NumElements(tensor) != rows * kHidden) {
    throw std::runtime_error(std::string(name) + " shape mismatch");
  }
}

void CheckModulation(
    const tvm::ffi::TensorView& tensor,
    const char* name,
    const tvm::ffi::TensorView& reference,
    const Shape& shape) {
  if (!IsFloat32(tensor.dtype()) && !IsBFloat16(tensor.dtype())) {
    throw std::runtime_error(std::string(name) + " must be FP32 or BF16");
  }
  CheckCudaContiguous(
      tensor, name, reference.device().device_id, tensor.dtype());
  if (NumElements(tensor) != shape.batch * 3 * kHidden) {
    throw std::runtime_error(std::string(name) + " shape mismatch");
  }
}

// Optional pinned split: both pinned modulations and pinned_rows, or none.
struct Pinned {
  const void* first = nullptr;
  const void* second = nullptr;
  const int* rows = nullptr;
};

Pinned CheckPinned(
    const tvm::ffi::Optional<tvm::ffi::TensorView>& first,
    const tvm::ffi::Optional<tvm::ffi::TensorView>& second,
    const tvm::ffi::Optional<tvm::ffi::TensorView>& rows,
    const tvm::ffi::TensorView& modulation,
    const tvm::ffi::TensorView& reference,
    const Shape& shape) {
  const int given = first.has_value() + second.has_value() + rows.has_value();
  if (given == 0) {
    return {};
  }
  if (given != 3) {
    throw std::runtime_error(
        "pinned modulations and pinned_rows must be given together");
  }
  const tvm::ffi::TensorView first_view = first.value();
  const tvm::ffi::TensorView second_view = second.value();
  const tvm::ffi::TensorView rows_view = rows.value();
  CheckModulation(first_view, "pinned first modulation", reference, shape);
  CheckModulation(second_view, "pinned second modulation", reference, shape);
  if (!SameDType(first_view.dtype(), modulation.dtype()) ||
      !SameDType(second_view.dtype(), modulation.dtype())) {
    throw std::runtime_error("pinned modulations must match the modulation dtype");
  }
  const DLDataType int32{kDLInt, 32, 1};
  CheckCudaContiguous(rows_view, "pinned_rows", reference.device().device_id, int32);
  if (rows_view.ndim() != 1 || rows_view.size(0) != shape.batch) {
    throw std::runtime_error("pinned_rows must be int32 [batch]");
  }
  return {first_view.data_ptr(), second_view.data_ptr(),
          static_cast<const int*>(rows_view.data_ptr())};
}

}  // namespace

void GatedResidualIdentityDoubleAdaRMSNormBf16Inplace(
    tvm::ffi::TensorView residual,
    tvm::ffi::TensorView branch,
    tvm::ffi::TensorView next_gate,
    tvm::ffi::TensorView previous_gate,
    tvm::ffi::TensorView input_modulation,
    tvm::ffi::TensorView post_modulation,
    tvm::ffi::Optional<tvm::ffi::TensorView> pinned_input_modulation,
    tvm::ffi::Optional<tvm::ffi::TensorView> pinned_post_modulation,
    tvm::ffi::Optional<tvm::ffi::TensorView> pinned_rows,
    double epsilon) {
  const Shape shape = CheckActivation(residual, "residual");
  CheckSameActivation(branch, "branch", residual, shape);
  CheckModulation(input_modulation, "input_modulation", residual, shape);
  CheckModulation(post_modulation, "post_modulation", residual, shape);
  const Pinned pinned = CheckPinned(
      pinned_input_modulation, pinned_post_modulation, pinned_rows,
      input_modulation, residual, shape);
  const bool per_token = pinned.rows != nullptr;
  CheckGate(next_gate, "next_gate", residual, shape, per_token);
  CheckGate(previous_gate, "previous_gate", residual, shape, per_token);
  if (input_modulation.dtype().code != post_modulation.dtype().code ||
      input_modulation.dtype().bits != post_modulation.dtype().bits) {
    throw std::runtime_error("input_modulation and post_modulation must have the same dtype");
  }
  if (residual.data_ptr() == branch.data_ptr() ||
      next_gate.data_ptr() == previous_gate.data_ptr()) {
    throw std::runtime_error("fused AdaRMSNorm inputs must not alias");
  }
  if (!(epsilon > 0.0)) {
    throw std::runtime_error("epsilon must be positive");
  }
  cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(
      residual.device().device_type, residual.device().device_id));
  auto launch = [&](auto kernel, auto* mod) {
    using ModT = std::remove_const_t<std::remove_pointer_t<decltype(mod)>>;
    kernel<<<shape.tokens, kThreads, 0, stream>>>(
        static_cast<__nv_bfloat16*>(residual.data_ptr()),
        static_cast<__nv_bfloat16*>(branch.data_ptr()),
        static_cast<__nv_bfloat16*>(next_gate.data_ptr()),
        static_cast<const __nv_bfloat16*>(previous_gate.data_ptr()),
        static_cast<const ModT*>(input_modulation.data_ptr()),
        static_cast<const ModT*>(post_modulation.data_ptr()),
        static_cast<float>(epsilon), static_cast<int>(shape.sequence),
        static_cast<const ModT*>(pinned.first),
        static_cast<const ModT*>(pinned.second), pinned.rows);
  };
  if (IsFloat32(input_modulation.dtype())) {
    if (per_token) {
      launch(DoubleAdaRMSNormBf16H1024Kernel<float, true>, static_cast<float*>(nullptr));
    } else {
      launch(DoubleAdaRMSNormBf16H1024Kernel<float, false>, static_cast<float*>(nullptr));
    }
  } else if (per_token) {
    launch(DoubleAdaRMSNormBf16H1024Kernel<__nv_bfloat16, true>,
           static_cast<__nv_bfloat16*>(nullptr));
  } else {
    launch(DoubleAdaRMSNormBf16H1024Kernel<__nv_bfloat16, false>,
           static_cast<__nv_bfloat16*>(nullptr));
  }
  sync_check_cuda_error();
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(
    gated_residual_double_adarmsnorm_inplace,
    tvm_ffi_double_adarmsnorm::GatedResidualIdentityDoubleAdaRMSNormBf16Inplace);

}  // namespace tvm_ffi_double_adarmsnorm
