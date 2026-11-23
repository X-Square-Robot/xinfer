#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/optional.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "common/cuda_utils.h"
#include "common/reduction.cuh"
#include "common/tvm_ffi_cuda_tensor.h"
#include "quant/mxfp8_format_utils.cuh"
#include "layernorm_rowreg_bf16.cuh"

namespace xinfer_layernorm {

constexpr int WARP_SIZE = 32;
constexpr int NUM_THREADS = 256;
constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;

struct RowShape {
  int64_t num_tokens;
  int64_t hidden_size;
};

// LayerNorm normalizes over the LAST dim; every leading dim is a token dim.
// Reading size(0)/size(1) directly would normalize any 3D+ input ([B, S, H])
// over S with B blocks -- silently garbage. Derive the shape
// rank-agnostically and require dense row-major layout.
inline RowShape GetRowShape(const tvm::ffi::TensorView& t) {
  const int ndim = t.ndim();
  ASSERT_CHECK(ndim >= 1);
  const int64_t hidden_size = t.size(ndim - 1);
  ASSERT_CHECK(hidden_size > 0);
  ASSERT_CHECK(hidden_size % 2 == 0);  // T2-vectorized kernel
  int64_t num_tokens = 1;
  for (int i = 0; i < ndim - 1; ++i) num_tokens *= t.size(i);
  ASSERT_CHECK(num_tokens > 0);
  // dense row-major check
  int64_t expect = 1;
  for (int i = ndim - 1; i >= 0; --i) {
    ASSERT_CHECK(t.stride(i) == expect);
    expect *= t.size(i);
  }
  return {num_tokens, hidden_size};
}

template<typename T>
__global__ void layernormKernel(
    T* out,
    const T* input,
    const T* __restrict__ weight,
    const T* __restrict__ bias,
    const float epsilon,
    const int64_t num_tokens,
    const int64_t hidden_size
) {
    typedef std::conditional_t<std::is_same<T, half>::value, half2,
            std::conditional_t<std::is_same<T, __nv_bfloat16>::value, __nv_bfloat162, float2>> T2;

    extern __shared__ float shared_mem[];
    T2* input_buf = (T2*)shared_mem;

    __shared__ float s_mean, s_variance;
    float mean = 0.0, variance = 0.0;

    #pragma unroll 4
    for (int64_t idx = threadIdx.x; idx < hidden_size/2; idx += blockDim.x) {
        T2 elem = ((T2*)input)[blockIdx.x * hidden_size/2 + idx];
        input_buf[idx] = elem;
        const float x = (float)elem.x;
        const float y = (float)elem.y;
        mean += x+y;
        variance += x*x + y*y;
    }

    // Reduce within warp
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        mean += __shfl_down_sync(0xffffffff, mean, offset);
        variance += __shfl_down_sync(0xffffffff, variance, offset);
    }
    static __shared__ float reduction_wksp[2][NUM_WARPS];
    if ((threadIdx.x & 31) == 0) {
        reduction_wksp[0][threadIdx.x >> 5] = mean;
        reduction_wksp[1][threadIdx.x >> 5] = variance;
    }
    __syncthreads();

    // Reduce within block
    if (threadIdx.x < NUM_WARPS) {
        mean = reduction_wksp[0][threadIdx.x];
        variance = reduction_wksp[1][threadIdx.x];
    }
    #pragma unroll
    for (int offset = NUM_WARPS/2; offset > 0; offset /= 2) {
        mean += __shfl_down_sync(0xffffffff, mean, offset);
        variance += __shfl_down_sync(0xffffffff, variance, offset);
    }

    if (threadIdx.x == 0) {
        float hidden_size_fp = (float)hidden_size;
        s_mean = mean / hidden_size_fp;
        s_variance = rsqrtf(variance / hidden_size_fp - s_mean * s_mean + epsilon);
    }
    __syncthreads();

    // Normalize in fp32 and round once at the store. The previous version
    // cast mean/rstd to T (bf16) and did the whole affine in bf16, costing
    // ~4x the rounding error of aten's fp32-internal layer_norm.
    const float final_mean = s_mean;
    const float final_rstd = s_variance;
    #pragma unroll 4
    for (int64_t idx = threadIdx.x; idx < hidden_size/2; idx += blockDim.x) {
        T2 x = input_buf[idx];
        T2 weight_elem = ((T2*)weight)[idx];
        T2 bias_elem = ((T2*)bias)[idx];
        T2 result;
        result.x = (decltype(result.x))(
            ((float)x.x - final_mean) * final_rstd * (float)weight_elem.x
            + (float)bias_elem.x);
        result.y = (decltype(result.y))(
            ((float)x.y - final_mean) * final_rstd * (float)weight_elem.y
            + (float)bias_elem.y);
        ((T2*)out)[blockIdx.x * hidden_size/2 + idx] = result;
    }
}

void Validate(tvm::ffi::TensorView output, tvm::ffi::TensorView input,
              tvm::ffi::TensorView weight, tvm::ffi::TensorView bias,
              double epsilon);

bool SameType(DLDataType a, DLDataType b) {
    return a.code == b.code && a.bits == b.bits && a.lanes == b.lanes;
}

int64_t CheckDense(tvm::ffi::TensorView t, const char* name, int device,
                   DLDataType dtype) {
    auto fail = [name](const char* reason) {
        throw std::runtime_error(std::string("layernorm: ") + name + reason);
    };
    if (t.device().device_type != kDLCUDA || t.device().device_id != device)
        fail(" must be on the same current CUDA device");
    if (!SameType(t.dtype(), dtype)) fail(" dtype mismatch");
    if (t.ndim() < 1) fail(" must have rank >= 1");
    int64_t size = 1;
    for (int i = t.ndim() - 1; i >= 0; --i) {
        if (t.size(i) <= 0 || t.stride(i) != size) fail(" must be nonempty dense row-major");
        if (t.size(i) > std::numeric_limits<int64_t>::max() / size)
            fail(" size overflow");
        size *= t.size(i);
    }
    if (reinterpret_cast<uintptr_t>(t.data_ptr()) % (2 * dtype.bits / 8))
        fail(" requires two-element alignment");
    return size;
}

bool Overlaps(tvm::ffi::TensorView a, int64_t na,
              tvm::ffi::TensorView b, int64_t nb) {
    const uintptr_t pa = reinterpret_cast<uintptr_t>(a.data_ptr());
    const uintptr_t pb = reinterpret_cast<uintptr_t>(b.data_ptr());
    const uint64_t bytes = a.dtype().bits / 8;
    return pa <= pb ? (pb - pa) / bytes < static_cast<uint64_t>(na)
                    : (pa - pb) / bytes < static_cast<uint64_t>(nb);
}

void Validate(tvm::ffi::TensorView output, tvm::ffi::TensorView input,
              tvm::ffi::TensorView weight, tvm::ffi::TensorView bias,
              double epsilon) {
    const auto dtype = input.dtype();
    if (dtype.lanes != 1 || !((dtype.code == kDLFloat &&
        (dtype.bits == 16 || dtype.bits == 32)) ||
        (dtype.code == kDLBfloat && dtype.bits == 16)))
        throw std::runtime_error("layernorm: expected FP32/FP16/BF16");
    int device = -1;
    if (cudaGetDevice(&device) != cudaSuccess)
        throw std::runtime_error("layernorm: cannot get CUDA device");
    const auto n = CheckDense(input, "input", device, dtype);
    const auto no = CheckDense(output, "output", device, dtype);
    const auto nw = CheckDense(weight, "weight", device, dtype);
    const auto nb = CheckDense(bias, "bias", device, dtype);
    const auto h = input.size(input.ndim()-1);
    if (h % 2 || h > 8192 || n / h > 2147483647)
        throw std::runtime_error("layernorm: even hidden <=8192 and rows <=INT_MAX required");
    if (output.ndim() != input.ndim() || n != no)
        throw std::runtime_error("layernorm: output shape mismatch");
    for (int i = 0; i < input.ndim(); ++i)
        if (input.size(i) != output.size(i))
            throw std::runtime_error("layernorm: output shape mismatch");
    if (weight.ndim() != 1 || bias.ndim() != 1 || nw != h || nb != h)
        throw std::runtime_error("layernorm: weight/bias must be [hidden]");
    if (!std::isfinite(epsilon) || epsilon <= 0 || epsilon > std::numeric_limits<float>::max())
        throw std::runtime_error("layernorm: epsilon must be finite positive FP32");
    if ((output.data_ptr() != input.data_ptr() && Overlaps(output, n, input, n)) ||
        Overlaps(output, n, weight, nw) || Overlaps(output, n, bias, nb))
        throw std::runtime_error("layernorm: only exact input/output alias is supported");
}

bool Aligned(const tvm::ffi::TensorView& tensor, uintptr_t bytes) {
    return reinterpret_cast<uintptr_t>(tensor.data_ptr()) % bytes == 0;
}

// Register-resident BF16 fast path (layernorm_rowreg_bf16.cuh): BF16 tensors, K % 128 == 0, K <= 2048, 16-byte
// aligned. Two-pass statistics, so it is also more accurate than the single-pass E[x^2] - mean^2 kernel below.
bool UseRowReg(const tvm::ffi::TensorView& output, const tvm::ffi::TensorView& input,
               const tvm::ffi::TensorView& weight, const tvm::ffi::TensorView& bias, int64_t hidden_size) {
    if (output.dtype().code != kDLBfloat || output.dtype().bits != 16 || hidden_size % 128 != 0 ||
        hidden_size > 128 * rowreg::kMaxVectors) {
        return false;
    }
    return Aligned(output, 16) && Aligned(input, 16) && Aligned(weight, 16) && Aligned(bias, 16);
}

template <int kVectors>
void LaunchRowReg(int vectors, __nv_bfloat16* output, const __nv_bfloat16* input, const __nv_bfloat16* gamma,
                  const __nv_bfloat16* beta, int rows, float epsilon, cudaStream_t stream) {
    if (vectors != kVectors) {
        if constexpr (kVectors < rowreg::kMaxVectors) {
            LaunchRowReg<kVectors + 1>(vectors, output, input, gamma, beta, rows, epsilon, stream);
        }
        return;
    }
    const int blocks = (rows + rowreg::kRowsPerBlock - 1) / rowreg::kRowsPerBlock;
    rowreg::layernorm_rowreg_bf16_kernel<kVectors>
        <<<blocks, rowreg::kThreadsPerRow * rowreg::kRowsPerBlock, 0, stream>>>(
            output, input, gamma, beta, rows, epsilon);
}

void Layernorm(
    tvm::ffi::TensorView output,
    tvm::ffi::TensorView input,
    tvm::ffi::TensorView weight,
    tvm::ffi::TensorView bias,
    double epsilon
) {
    Validate(output, input, weight, bias, epsilon);
    const RowShape shape = GetRowShape(output);
    const RowShape in_shape = GetRowShape(input);
    ASSERT_CHECK(in_shape.num_tokens == shape.num_tokens);
    ASSERT_CHECK(in_shape.hidden_size == shape.hidden_size);
    const int64_t num_tokens = shape.num_tokens;
    const int64_t hidden_size = shape.hidden_size;
    ASSERT_CHECK(weight.size(weight.ndim() - 1) == hidden_size);
    ASSERT_CHECK(bias.size(bias.ndim() - 1) == hidden_size);

    dim3 grid(num_tokens);
    dim3 block(NUM_THREADS);

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(output.device().device_type, output.device().device_id));

    if (UseRowReg(output, input, weight, bias, hidden_size)) {
        LaunchRowReg<1>(static_cast<int>(hidden_size / 128),
                        static_cast<__nv_bfloat16*>(output.data_ptr()),
                        static_cast<const __nv_bfloat16*>(input.data_ptr()),
                        static_cast<const __nv_bfloat16*>(weight.data_ptr()),
                        static_cast<const __nv_bfloat16*>(bias.data_ptr()),
                        static_cast<int>(num_tokens), static_cast<float>(epsilon), stream);
        sync_check_cuda_error();
        return;
    }

    if (output.dtype().code == kDLFloat && output.dtype().bits == 32) {
        layernormKernel<float><<<grid, block, hidden_size*sizeof(float), stream>>>(
            static_cast<float*>(output.data_ptr()),
            static_cast<const float*>(input.data_ptr()),
            static_cast<const float*>(weight.data_ptr()),
            static_cast<const float*>(bias.data_ptr()),
            (float)epsilon, num_tokens, hidden_size);
    } else if (output.dtype().code == kDLFloat && output.dtype().bits == 16) {
        layernormKernel<half><<<grid, block, hidden_size*sizeof(half), stream>>>(
            static_cast<half*>(output.data_ptr()),
            static_cast<const half*>(input.data_ptr()),
            static_cast<const half*>(weight.data_ptr()),
            static_cast<const half*>(bias.data_ptr()),
            (float)epsilon, num_tokens, hidden_size);
    } else if (output.dtype().code == kDLBfloat && output.dtype().bits == 16) {
        layernormKernel<__nv_bfloat16><<<grid, block, hidden_size*sizeof(__nv_bfloat16), stream>>>(
            static_cast<__nv_bfloat16*>(output.data_ptr()),
            static_cast<const __nv_bfloat16*>(input.data_ptr()),
            static_cast<const __nv_bfloat16*>(weight.data_ptr()),
            static_cast<const __nv_bfloat16*>(bias.data_ptr()),
            (float)epsilon, num_tokens, hidden_size);
    } else {
        throw std::runtime_error("Unsupported data type for layernorm");
    }

    sync_check_cuda_error();
}

// Envelope of the fused LayerNorm + MXFP8 kernel: the register-resident kernel's (BF16 [rows, K] packed rows,
// K % 128 == 0, K <= 2048, 16-byte aligned input / weight / bias).
bool LayernormQuantSupported(const tvm::ffi::TensorView& input, const tvm::ffi::TensorView& weight,
                             const tvm::ffi::TensorView& bias) {
    const auto dtype = input.dtype();
    if (dtype.code != kDLBfloat || dtype.bits != 16 || dtype.lanes != 1 || input.ndim() != 2) return false;
    const int64_t hidden_size = input.size(1);
    return input.size(0) > 0 && hidden_size >= 128 && hidden_size % 128 == 0 &&
           hidden_size <= 128 * rowreg::kMaxVectors && input.IsContiguous() &&
           Aligned(input, 16) && Aligned(weight, 16) && Aligned(bias, 16);
}

// Whether layernorm(.., quant="mxfp8", scale_block=128) runs the fused kernel (true) or layernorm followed by
// quantize_mxfp8_into (false; the same bits). The fused kernel is taken where it measured at least
// 5% faster than the two kernels: above K = 1280 its per-thread instruction
// chain is longer than two launches unless there are enough rows.
bool LayernormQuantFusedSelected(tvm::ffi::TensorView input, tvm::ffi::TensorView weight,
                                 tvm::ffi::TensorView bias) {
    return LayernormQuantSupported(input, weight, bias) && (input.size(1) <= 1280 || input.size(0) >= 384);
}

template <int kVectors>
void LaunchRowRegQuant(int vectors, uint8_t* quantized, uint32_t* scales, const __nv_bfloat16* input,
                       const __nv_bfloat16* gamma, const __nv_bfloat16* beta, int rows, float epsilon,
                       cudaStream_t stream) {
    if (vectors != kVectors) {
        if constexpr (kVectors < rowreg::kMaxVectors) {
            LaunchRowRegQuant<kVectors + 1>(vectors, quantized, scales, input, gamma, beta, rows, epsilon, stream);
        }
        return;
    }
    const int blocks = (rows + rowreg::kQuantRowsPerBlock - 1) / rowreg::kQuantRowsPerBlock;
    rowreg::layernorm_quant_mxfp8_rowreg_kernel<kVectors>
        <<<blocks, rowreg::kThreadsPerRow * rowreg::kQuantRowsPerBlock, 0, stream>>>(
            quantized, scales, input, gamma, beta, rows, epsilon);
}

// LayerNorm with MXFP8 output, one scale per 128 columns: bit for bit layernorm (BF16) followed by
// quantize_mxfp8_into(.., share_scale_128 = true), in one kernel. quantized is [rows, K] E4M3 storage, scales the
// SWIZZLE_32_4_4 storage of [rows, K].
void LayernormQuant(tvm::ffi::TensorView quantized, tvm::ffi::TensorView scales, tvm::ffi::TensorView input,
                    tvm::ffi::TensorView weight, tvm::ffi::TensorView bias, double epsilon) {
    xinfer::ffi::CheckCuda(input, "input");
    if (!LayernormQuantSupported(input, weight, bias))
        throw std::runtime_error(
            "layernorm_quant_mxfp8: requires BF16 [rows, K] packed rows with K % 128 == 0, 128 <= K <= 2048 and "
            "16-byte aligned input / weight / bias");
    const int64_t rows = input.size(0), hidden_size = input.size(1);
    const auto dtype = input.dtype();
    int device = -1;
    if (cudaGetDevice(&device) != cudaSuccess)
        throw std::runtime_error("layernorm_quant_mxfp8: cannot get CUDA device");
    CheckDense(input, "input", device, dtype);
    if (weight.ndim() != 1 || bias.ndim() != 1 || CheckDense(weight, "weight", device, dtype) != hidden_size ||
        CheckDense(bias, "bias", device, dtype) != hidden_size)
        throw std::runtime_error("layernorm_quant_mxfp8: weight/bias must be [hidden]");
    if (!std::isfinite(epsilon) || epsilon <= 0 || epsilon > std::numeric_limits<float>::max())
        throw std::runtime_error("layernorm_quant_mxfp8: epsilon must be finite positive FP32");
    xinfer::mxfp8::CheckCommonOutput(quantized, input, "quantized");
    xinfer::mxfp8::CheckCommonOutput(scales, input, "scales");
    xinfer::ffi::CheckFloat8E4M3Storage(quantized, "quantized");
    if (quantized.ndim() != 2 || quantized.size(0) != rows || quantized.size(1) != hidden_size ||
        rows > std::numeric_limits<int>::max())
        throw std::runtime_error("layernorm_quant_mxfp8: quantized must be [rows, K]");
    xinfer::mxfp8::CheckScaleStorage(scales, rows, hidden_size, "scales");
    if (!Aligned(quantized, 8) || !Aligned(scales, 4))
        throw std::runtime_error("layernorm_quant_mxfp8: quantized / scales must be 8 / 4-byte aligned");
    cudaStream_t stream = xinfer::ffi::Stream(input);
    LaunchRowRegQuant<1>(static_cast<int>(hidden_size / 128), xinfer::ffi::Data<uint8_t>(quantized),
                         xinfer::ffi::Data<uint32_t>(scales), xinfer::ffi::ConstData<__nv_bfloat16>(input),
                         xinfer::ffi::ConstData<__nv_bfloat16>(weight),
                         xinfer::ffi::ConstData<__nv_bfloat16>(bias), static_cast<int>(rows),
                         static_cast<float>(epsilon), stream);
    xinfer::ffi::CheckLastCudaError("layernorm_quant_mxfp8 launch");
}

template <int kVectors>
void LaunchRowRegNvfp4(int vectors, __nv_bfloat16* output, const __nv_bfloat16* input, const __nv_bfloat16* gamma,
                       const __nv_bfloat16* beta, int rows, float epsilon, rowreg::Nvfp4Out nvfp4,
                       cudaStream_t stream) {
    if (vectors != kVectors) {
        if constexpr (kVectors < rowreg::kMaxVectors) {
            LaunchRowRegNvfp4<kVectors + 1>(vectors, output, input, gamma, beta, rows, epsilon, nvfp4, stream);
        }
        return;
    }
    const int blocks = (nvfp4.padded_rows + rowreg::kRowsPerBlock - 1) / rowreg::kRowsPerBlock;
    rowreg::layernorm_quant_nvfp4_rowreg_kernel<kVectors>
        <<<blocks, rowreg::kThreadsPerRow * rowreg::kRowsPerBlock, 0, stream>>>(
            output, input, gamma, beta, rows, epsilon, nvfp4);
}

// The NVFP4 (data uint8 [M, K / 2], E4M3 SWIZZLE_32_4_4 scales) of the BF16 LayerNorm y, bit for bit quantize_nvfp4 of
// the y that layernorm writes, and y itself into output when given. The register-resident kernel's envelope: BF16,
// K % 128 == 0, K <= 2048, contiguous rows, 16-byte aligned.
void LayernormQuantNvfp4(tvm::ffi::TensorView data, tvm::ffi::TensorView scales, tvm::ffi::Optional<tvm::ffi::TensorView> output,
                         tvm::ffi::TensorView input, tvm::ffi::TensorView weight, tvm::ffi::TensorView bias, double epsilon) {
    namespace ffi = xinfer::ffi;
    ffi::CheckUInt8(data, "data");
    ffi::CheckContiguous(data, "data");
    ffi::CheckFloat8E4M3Storage(scales, "scales");
    ffi::CheckContiguous(scales, "scales");
    if (output.has_value()) {
        Validate(output.value(), input, weight, bias, epsilon);
    } else {  // Validate's checks of the inputs
        int device = -1;
        if (cudaGetDevice(&device) != cudaSuccess) throw std::runtime_error("layernorm: cannot get CUDA device");
        CheckDense(input, "input", device, input.dtype());
        CheckDense(weight, "weight", device, input.dtype());
        CheckDense(bias, "bias", device, input.dtype());
        if (!std::isfinite(epsilon) || epsilon <= 0 || epsilon > std::numeric_limits<float>::max())
            throw std::runtime_error("layernorm: epsilon must be finite positive FP32");
    }
    const RowShape shape = GetRowShape(input);
    const int64_t m = shape.num_tokens, k = shape.hidden_size;
    if (input.dtype().code != kDLBfloat || input.dtype().bits != 16 || weight.dtype().code != kDLBfloat ||
        bias.dtype().code != kDLBfloat || k % 128 != 0 || k > 128 * rowreg::kMaxVectors ||
        weight.ndim() != 1 || weight.size(0) != k || bias.ndim() != 1 || bias.size(0) != k) {
        throw std::runtime_error("layernorm_quant_nvfp4: BF16 [M, K] with K % 128 == 0 and K <= 2048, weight/bias [K]");
    }
    for (const void* pointer : {input.data_ptr(), weight.data_ptr(), bias.data_ptr(),
                                output.has_value() ? output.value().data_ptr() : nullptr}) {
        if (reinterpret_cast<uintptr_t>(pointer) % 16 != 0) throw std::runtime_error("layernorm_quant_nvfp4: unaligned");
    }
    if (reinterpret_cast<uintptr_t>(data.data_ptr()) % 4 != 0) {
        throw std::runtime_error("layernorm_quant_nvfp4: data must be 4-byte aligned (four bytes per store)");
    }
    if (data.ndim() != 2 || data.size(0) != m || data.size(1) != k / 2 ||
        ffi::NumBytes(scales) < xinfer::nvfp4::ScaleStorageBytes(m, k)) {
        throw std::runtime_error("layernorm_quant_nvfp4: data must be uint8 [M, K / 2] and scales hold the "
                                 "SWIZZLE_32_4_4 storage");
    }
    for (const tvm::ffi::TensorView* t : {&data, &scales, &weight, &bias}) {
        ffi::CheckSameDevice(*t, input, "layernorm_quant_nvfp4: data/scales/weight/bias");
    }
    if (m == 0) return;
    const rowreg::Nvfp4Out nvfp4{static_cast<uint8_t*>(data.data_ptr()), static_cast<uint8_t*>(scales.data_ptr()),
                                 static_cast<int>((m + 127) / 128 * 128)};
    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(input.device().device_type, input.device().device_id));
    LaunchRowRegNvfp4<1>(static_cast<int>(k / 128),
                         output.has_value() ? static_cast<__nv_bfloat16*>(output.value().data_ptr()) : nullptr,
                         static_cast<const __nv_bfloat16*>(input.data_ptr()),
                         static_cast<const __nv_bfloat16*>(weight.data_ptr()),
                         static_cast<const __nv_bfloat16*>(bias.data_ptr()), static_cast<int>(m),
                         static_cast<float>(epsilon), nvfp4, stream);
    sync_check_cuda_error();
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(layernorm, xinfer_layernorm::Layernorm);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(layernorm_quant_mxfp8, xinfer_layernorm::LayernormQuant);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(layernorm_quant_mxfp8_fused_selected,
                              xinfer_layernorm::LayernormQuantFusedSelected);

TVM_FFI_DLL_EXPORT_TYPED_FUNC(layernorm_quant_nvfp4, xinfer_layernorm::LayernormQuantNvfp4);

}  // namespace xinfer_layernorm
