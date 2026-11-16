#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>

#include "common/cuda_utils.h"

namespace xinfer_qkv_norm_split {

constexpr int kThreads = 256;
constexpr int kWarpSize = 32;
constexpr int kHeadsPerGroupedBlock = kThreads / kWarpSize;

static void check_same_device(
    tvm::ffi::TensorView tensor,
    const char* name,
    int device_type,
    int device_id) {
    if (tensor.device().device_type != device_type || tensor.device().device_id != device_id) {
        throw std::runtime_error(std::string(name) + ": expected tensor on the same device");
    }
}

static void check_i64_1d(
    tvm::ffi::TensorView tensor,
    const char* name,
    int64_t expected_size,
    int device_type,
    int device_id) {
    if (tensor.ndim() != 1 || tensor.size(0) != expected_size) {
        throw std::runtime_error(
            std::string(name) + ": expected shape [" + std::to_string(expected_size) + "]");
    }
    if (!tensor.IsContiguous()) {
        throw std::runtime_error(std::string(name) + ": expected contiguous tensor");
    }
    if (!(tensor.dtype().code == kDLInt && tensor.dtype().bits == 64)) {
        throw std::runtime_error(std::string(name) + ": expected int64 tensor");
    }
    check_same_device(tensor, name, device_type, device_id);
}

static void check_weight(
    tvm::ffi::TensorView tensor,
    const char* name,
    int64_t head_dim,
    int device_type,
    int device_id) {
    if (tensor.ndim() != 1 || tensor.size(0) != head_dim) {
        throw std::runtime_error(
            std::string(name) + ": expected shape [" + std::to_string(head_dim) + "]");
    }
    if (!tensor.IsContiguous()) {
        throw std::runtime_error(std::string(name) + ": expected contiguous tensor");
    }
    const bool supported_dtype =
        (tensor.dtype().code == kDLFloat && tensor.dtype().bits == 32) ||
        (tensor.dtype().code == kDLFloat && tensor.dtype().bits == 16) ||
        (tensor.dtype().code == kDLBfloat && tensor.dtype().bits == 16);
    if (!supported_dtype) {
        throw std::runtime_error(std::string(name) + ": expected float32, float16, or bfloat16 tensor");
    }
    check_same_device(tensor, name, device_type, device_id);
}

static bool same_dtype(tvm::ffi::TensorView lhs, tvm::ffi::TensorView rhs) {
    return lhs.dtype().code == rhs.dtype().code && lhs.dtype().bits == rhs.dtype().bits;
}

template <typename T>
__device__ __forceinline__ float to_float(T value) {
    return static_cast<float>(value);
}

template <>
__device__ __forceinline__ float to_float<__half>(__half value) {
    return __half2float(value);
}

template <>
__device__ __forceinline__ float to_float<__nv_bfloat16>(__nv_bfloat16 value) {
    return __bfloat162float(value);
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float block_reduce_sum(float val) {
    static __shared__ float shared[32];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    val = warp_reduce_sum(val);
    if (lane == 0) {
        shared[warp] = val;
    }
    __syncthreads();
    const int num_warps = (blockDim.x + 31) >> 5;
    val = threadIdx.x < num_warps ? shared[lane] : 0.0f;
    if (warp == 0) {
        val = warp_reduce_sum(val);
    }
    // Trailing barrier: `shared` is reused across calls, so without it a fast warp
    // entering the next call can overwrite shared[warp] while a slow warp is still
    // reading shared[lane] above (WAR race on back-to-back reductions).
    __syncthreads();
    return val;
}

template <typename T>
__device__ __forceinline__ void vec_to_floats(typename VecTraits<T>::Type vec, float* values) {
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
    }
}

template <typename T>
__device__ __forceinline__ typename VecTraits<T>::Type floats_to_vec(const float* values) {
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
    }
    return vec;
}

template <typename W, int VEC_SIZE>
__device__ __forceinline__ void load_weight_floats(
    const W* __restrict__ weight,
    int64_t offset,
    float* values) {
    #pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
        values[i] = to_float<W>(weight[offset + i]);
    }
}

__device__ __forceinline__ bool is_action_token(
    int64_t seq,
    int64_t action_start,
    int64_t action_end) {
    return seq >= action_start && seq < action_end;
}

__device__ __forceinline__ int64_t q_offset(int64_t head, int64_t dim, int64_t head_dim) {
    return head * 2 * head_dim + dim;
}

__device__ __forceinline__ int64_t gate_offset(int64_t head, int64_t dim, int64_t head_dim) {
    return head * 2 * head_dim + head_dim + dim;
}

__device__ __forceinline__ int64_t k_offset(
    int64_t num_heads,
    int64_t head,
    int64_t dim,
    int64_t head_dim) {
    return num_heads * 2 * head_dim + head * head_dim + dim;
}

__device__ __forceinline__ int64_t v_offset(
    int64_t num_heads,
    int64_t num_key_value_heads,
    int64_t head,
    int64_t dim,
    int64_t head_dim) {
    return (num_heads * 2 + num_key_value_heads) * head_dim + head * head_dim + dim;
}

template <typename T, typename W>
__global__ void qkvNormReplaceForwardKernel(
    T* __restrict__ query,
    T* __restrict__ key,
    T* __restrict__ value,
    T* __restrict__ gate,
    const T* __restrict__ text_qkv,
    const T* __restrict__ action_qkv,
    const int64_t* __restrict__ action_starts,
    const int64_t* __restrict__ action_ends,
    const int64_t* __restrict__ action_offsets,
    const W* __restrict__ q_weight,
    const W* __restrict__ k_weight,
    const W* __restrict__ q_action_weight,
    const W* __restrict__ k_action_weight,
    int64_t batch_size,
    int64_t seq_length,
    int64_t action_length,
    int64_t qkv_dim,
    int64_t num_heads,
    int64_t num_key_value_heads,
    int64_t head_dim,
    float epsilon) {
    const int64_t row = blockIdx.x;
    const int64_t head = blockIdx.y;
    const int field = blockIdx.z;  // 0: q+gate, 1: k+value
    if (row >= batch_size * seq_length) {
        return;
    }
    if ((field == 0 && head >= num_heads) || (field == 1 && head >= num_key_value_heads)) {
        return;
    }

    const int64_t batch = row / seq_length;
    const int64_t seq = row - batch * seq_length;
    const int64_t action_start = action_starts[batch];
    const int64_t action_end = action_ends[batch];
    const bool use_action = is_action_token(seq, action_start, action_end);
    const int64_t source_row = use_action
        ? (action_offsets[batch] + seq - action_start)
        : row;
    const T* source = use_action
        ? action_qkv + source_row * qkv_dim
        : text_qkv + source_row * qkv_dim;

    using Vec = typename VecTraits<T>::Type;
    constexpr int VEC_SIZE = VecTraits<T>::vec_size;
    const int64_t vec_hidden = head_dim / VEC_SIZE;
    float sum_sq = 0.0f;
    const int64_t base_offset = (field == 0)
        ? q_offset(head, 0, head_dim)
        : k_offset(num_heads, head, 0, head_dim);

    for (int64_t idx = threadIdx.x; idx < vec_hidden; idx += blockDim.x) {
        const Vec x_vec = VecTraits<T>::load(source + base_offset + idx * VEC_SIZE);
        float x_vals[VEC_SIZE];
        vec_to_floats<T>(x_vec, x_vals);
        #pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
            sum_sq += x_vals[i] * x_vals[i];
        }
    }
    sum_sq = block_reduce_sum(sum_sq);
    __shared__ float inv_rms;
    if (threadIdx.x == 0) {
        inv_rms = rsqrtf(sum_sq / static_cast<float>(head_dim) + epsilon);
    }
    __syncthreads();

    const W* weight = nullptr;
    if (field == 0) {
        weight = use_action ? q_action_weight : q_weight;
        const int64_t out_base = (row * num_heads + head) * head_dim;
        const int64_t gate_base = row * num_heads * head_dim + head * head_dim;
        for (int64_t idx = threadIdx.x; idx < vec_hidden; idx += blockDim.x) {
            const int64_t dim = idx * VEC_SIZE;
            const Vec x_vec = VecTraits<T>::load(source + q_offset(head, dim, head_dim));
            float x_vals[VEC_SIZE], w_vals[VEC_SIZE], out_vals[VEC_SIZE];
            vec_to_floats<T>(x_vec, x_vals);
            load_weight_floats<W, VEC_SIZE>(weight, dim, w_vals);
            #pragma unroll
            for (int i = 0; i < VEC_SIZE; ++i) {
                out_vals[i] = x_vals[i] * (w_vals[i] + 1.0f) * inv_rms;
            }
            VecTraits<T>::store(query + out_base + dim, floats_to_vec<T>(out_vals));
            const Vec gate_vec = VecTraits<T>::load(source + gate_offset(head, dim, head_dim));
            VecTraits<T>::store(gate + gate_base + dim, gate_vec);
        }
    } else {
        weight = use_action ? k_action_weight : k_weight;
        const int64_t out_base = (row * num_key_value_heads + head) * head_dim;
        for (int64_t idx = threadIdx.x; idx < vec_hidden; idx += blockDim.x) {
            const int64_t dim = idx * VEC_SIZE;
            const Vec x_vec = VecTraits<T>::load(source + k_offset(num_heads, head, dim, head_dim));
            float x_vals[VEC_SIZE], w_vals[VEC_SIZE], out_vals[VEC_SIZE];
            vec_to_floats<T>(x_vec, x_vals);
            load_weight_floats<W, VEC_SIZE>(weight, dim, w_vals);
            #pragma unroll
            for (int i = 0; i < VEC_SIZE; ++i) {
                out_vals[i] = x_vals[i] * (w_vals[i] + 1.0f) * inv_rms;
            }
            VecTraits<T>::store(key + out_base + dim, floats_to_vec<T>(out_vals));
            const Vec value_vec = VecTraits<T>::load(
                source + v_offset(num_heads, num_key_value_heads, head, dim, head_dim));
            VecTraits<T>::store(value + out_base + dim, value_vec);
        }
    }
}

template <typename T, typename W>
__global__ void qkvNormReplaceForwardGroupedKernel(
    T* __restrict__ query,
    T* __restrict__ key,
    T* __restrict__ value,
    T* __restrict__ gate,
    const T* __restrict__ text_qkv,
    const T* __restrict__ action_qkv,
    const int64_t* __restrict__ action_starts,
    const int64_t* __restrict__ action_ends,
    const int64_t* __restrict__ action_offsets,
    const W* __restrict__ q_weight,
    const W* __restrict__ k_weight,
    const W* __restrict__ q_action_weight,
    const W* __restrict__ k_action_weight,
    int64_t batch_size,
    int64_t seq_length,
    int64_t action_length,
    int64_t qkv_dim,
    int64_t num_heads,
    int64_t num_key_value_heads,
    int64_t head_dim,
    float epsilon) {
    const int64_t row = blockIdx.x;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int64_t logical_head = blockIdx.y * kHeadsPerGroupedBlock + warp;
    const int64_t total_heads = num_heads + num_key_value_heads;
    if (row >= batch_size * seq_length || logical_head >= total_heads) {
        return;
    }

    const bool is_query = logical_head < num_heads;
    const int64_t head = is_query ? logical_head : logical_head - num_heads;
    const int64_t batch = row / seq_length;
    const int64_t seq = row - batch * seq_length;
    const int64_t action_start = action_starts[batch];
    const int64_t action_end = action_ends[batch];
    const bool use_action = is_action_token(seq, action_start, action_end);
    const int64_t source_row = use_action
        ? (action_offsets[batch] + seq - action_start)
        : row;
    const T* source = use_action
        ? action_qkv + source_row * qkv_dim
        : text_qkv + source_row * qkv_dim;

    using Vec = typename VecTraits<T>::Type;
    constexpr int VEC_SIZE = VecTraits<T>::vec_size;
    const int64_t vec_hidden = head_dim / VEC_SIZE;
    const int64_t base_offset = is_query
        ? q_offset(head, 0, head_dim)
        : k_offset(num_heads, head, 0, head_dim);

    float sum_sq = 0.0f;
    for (int64_t idx = lane; idx < vec_hidden; idx += kWarpSize) {
        const Vec x_vec = VecTraits<T>::load(source + base_offset + idx * VEC_SIZE);
        float x_vals[VEC_SIZE];
        vec_to_floats<T>(x_vec, x_vals);
        #pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
            sum_sq += x_vals[i] * x_vals[i];
        }
    }
    sum_sq = warp_reduce_sum(sum_sq);
    const float inv_rms = rsqrtf(
        __shfl_sync(0xffffffff, sum_sq, 0) / static_cast<float>(head_dim) + epsilon);

    const W* weight = is_query
        ? (use_action ? q_action_weight : q_weight)
        : (use_action ? k_action_weight : k_weight);

    if (is_query) {
        const int64_t out_base = (row * num_heads + head) * head_dim;
        const int64_t gate_base = row * num_heads * head_dim + head * head_dim;
        for (int64_t idx = lane; idx < vec_hidden; idx += kWarpSize) {
            const int64_t dim = idx * VEC_SIZE;
            const Vec x_vec = VecTraits<T>::load(source + q_offset(head, dim, head_dim));
            float x_vals[VEC_SIZE], w_vals[VEC_SIZE], out_vals[VEC_SIZE];
            vec_to_floats<T>(x_vec, x_vals);
            load_weight_floats<W, VEC_SIZE>(weight, dim, w_vals);
            #pragma unroll
            for (int i = 0; i < VEC_SIZE; ++i) {
                out_vals[i] = x_vals[i] * (w_vals[i] + 1.0f) * inv_rms;
            }
            VecTraits<T>::store(query + out_base + dim, floats_to_vec<T>(out_vals));
            const Vec gate_vec = VecTraits<T>::load(source + gate_offset(head, dim, head_dim));
            VecTraits<T>::store(gate + gate_base + dim, gate_vec);
        }
    } else {
        const int64_t out_base = (row * num_key_value_heads + head) * head_dim;
        for (int64_t idx = lane; idx < vec_hidden; idx += kWarpSize) {
            const int64_t dim = idx * VEC_SIZE;
            const Vec x_vec = VecTraits<T>::load(source + k_offset(num_heads, head, dim, head_dim));
            float x_vals[VEC_SIZE], w_vals[VEC_SIZE], out_vals[VEC_SIZE];
            vec_to_floats<T>(x_vec, x_vals);
            load_weight_floats<W, VEC_SIZE>(weight, dim, w_vals);
            #pragma unroll
            for (int i = 0; i < VEC_SIZE; ++i) {
                out_vals[i] = x_vals[i] * (w_vals[i] + 1.0f) * inv_rms;
            }
            VecTraits<T>::store(key + out_base + dim, floats_to_vec<T>(out_vals));
            const Vec value_vec = VecTraits<T>::load(
                source + v_offset(num_heads, num_key_value_heads, head, dim, head_dim));
            VecTraits<T>::store(value + out_base + dim, value_vec);
        }
    }
}

template <typename T, typename W>
void launch_forward(
    tvm::ffi::TensorView query,
    tvm::ffi::TensorView key,
    tvm::ffi::TensorView value,
    tvm::ffi::TensorView gate,
    tvm::ffi::TensorView text_qkv,
    tvm::ffi::TensorView action_qkv,
    tvm::ffi::TensorView action_starts,
    tvm::ffi::TensorView action_ends,
    tvm::ffi::TensorView action_offsets,
    tvm::ffi::TensorView q_weight,
    tvm::ffi::TensorView k_weight,
    tvm::ffi::TensorView q_action_weight,
    tvm::ffi::TensorView k_action_weight,
    int64_t num_heads,
    int64_t num_key_value_heads,
    int64_t head_dim,
    float epsilon,
    cudaStream_t stream) {
    const int64_t batch_size = text_qkv.size(0);
    const int64_t seq_length = text_qkv.size(1);
    const int64_t qkv_dim = text_qkv.size(2);
    const int64_t action_length = action_qkv.size(1);
    // Every kernel here strides in VEC_SIZE units (vec_hidden = head_dim / VEC_SIZE
    // truncates): a non-divisible head_dim would leave the tail dims unwritten and
    // compute inv_rms from a partial sum. Reject it explicitly.
    ASSERT_CHECK(head_dim % VecTraits<T>::vec_size == 0);
    if (head_dim % VecTraits<T>::vec_size == 0 && head_dim >= 128) {
        const int64_t total_heads = num_heads + num_key_value_heads;
        const dim3 grid(
            batch_size * seq_length,
            (total_heads + kHeadsPerGroupedBlock - 1) / kHeadsPerGroupedBlock);
        qkvNormReplaceForwardGroupedKernel<T, W><<<grid, kThreads, 0, stream>>>(
            static_cast<T*>(query.data_ptr()),
            static_cast<T*>(key.data_ptr()),
            static_cast<T*>(value.data_ptr()),
            static_cast<T*>(gate.data_ptr()),
            static_cast<const T*>(text_qkv.data_ptr()),
            static_cast<const T*>(action_qkv.data_ptr()),
            static_cast<const int64_t*>(action_starts.data_ptr()),
            static_cast<const int64_t*>(action_ends.data_ptr()),
            static_cast<const int64_t*>(action_offsets.data_ptr()),
            static_cast<const W*>(q_weight.data_ptr()),
            static_cast<const W*>(k_weight.data_ptr()),
            static_cast<const W*>(q_action_weight.data_ptr()),
            static_cast<const W*>(k_action_weight.data_ptr()),
            batch_size,
            seq_length,
            action_length,
            qkv_dim,
            num_heads,
            num_key_value_heads,
            head_dim,
            epsilon);
    } else {
        const int64_t max_heads = num_heads > num_key_value_heads ? num_heads : num_key_value_heads;
        const dim3 grid(batch_size * seq_length, max_heads, 2);
        qkvNormReplaceForwardKernel<T, W><<<grid, kThreads, 0, stream>>>(
            static_cast<T*>(query.data_ptr()),
            static_cast<T*>(key.data_ptr()),
            static_cast<T*>(value.data_ptr()),
            static_cast<T*>(gate.data_ptr()),
            static_cast<const T*>(text_qkv.data_ptr()),
            static_cast<const T*>(action_qkv.data_ptr()),
            static_cast<const int64_t*>(action_starts.data_ptr()),
            static_cast<const int64_t*>(action_ends.data_ptr()),
            static_cast<const int64_t*>(action_offsets.data_ptr()),
            static_cast<const W*>(q_weight.data_ptr()),
            static_cast<const W*>(k_weight.data_ptr()),
            static_cast<const W*>(q_action_weight.data_ptr()),
            static_cast<const W*>(k_action_weight.data_ptr()),
            batch_size,
            seq_length,
            action_length,
            qkv_dim,
            num_heads,
            num_key_value_heads,
            head_dim,
            epsilon);
    }
}

template <typename T>
void launch_forward_for_weight_dtype(
    tvm::ffi::TensorView query,
    tvm::ffi::TensorView key,
    tvm::ffi::TensorView value,
    tvm::ffi::TensorView gate,
    tvm::ffi::TensorView text_qkv,
    tvm::ffi::TensorView action_qkv,
    tvm::ffi::TensorView action_starts,
    tvm::ffi::TensorView action_ends,
    tvm::ffi::TensorView action_offsets,
    tvm::ffi::TensorView q_weight,
    tvm::ffi::TensorView k_weight,
    tvm::ffi::TensorView q_action_weight,
    tvm::ffi::TensorView k_action_weight,
    int64_t num_heads,
    int64_t num_key_value_heads,
    int64_t head_dim,
    float epsilon,
    cudaStream_t stream) {
    if (q_weight.dtype().code == kDLFloat && q_weight.dtype().bits == 32) {
        launch_forward<T, float>(
            query,
            key,
            value,
            gate,
            text_qkv,
            action_qkv,
            action_starts,
            action_ends,
            action_offsets,
            q_weight,
            k_weight,
            q_action_weight,
            k_action_weight,
            num_heads,
            num_key_value_heads,
            head_dim,
            epsilon,
            stream);
    } else if (q_weight.dtype().code == kDLFloat && q_weight.dtype().bits == 16) {
        launch_forward<T, __half>(
            query,
            key,
            value,
            gate,
            text_qkv,
            action_qkv,
            action_starts,
            action_ends,
            action_offsets,
            q_weight,
            k_weight,
            q_action_weight,
            k_action_weight,
            num_heads,
            num_key_value_heads,
            head_dim,
            epsilon,
            stream);
    } else if (q_weight.dtype().code == kDLBfloat && q_weight.dtype().bits == 16) {
        launch_forward<T, __nv_bfloat16>(
            query,
            key,
            value,
            gate,
            text_qkv,
            action_qkv,
            action_starts,
            action_ends,
            action_offsets,
            q_weight,
            k_weight,
            q_action_weight,
            k_action_weight,
            num_heads,
            num_key_value_heads,
            head_dim,
            epsilon,
            stream);
    } else {
        throw std::runtime_error("qkv_norm_split: unsupported weight dtype");
    }
}

void validate_common(
    tvm::ffi::TensorView text_qkv,
    tvm::ffi::TensorView action_qkv,
    tvm::ffi::TensorView action_starts,
    tvm::ffi::TensorView action_ends,
    tvm::ffi::TensorView action_offsets,
    tvm::ffi::TensorView q_weight,
    tvm::ffi::TensorView k_weight,
    tvm::ffi::TensorView q_action_weight,
    tvm::ffi::TensorView k_action_weight,
    int64_t num_heads,
    int64_t num_key_value_heads,
    int64_t head_dim) {
    ASSERT_CHECK(text_qkv.ndim() == 3);
    ASSERT_CHECK(action_qkv.ndim() == 3);
    ASSERT_CHECK(text_qkv.IsContiguous());
    ASSERT_CHECK(action_qkv.IsContiguous());
    // action_qkv is a packed action buffer indexed via action_offsets (flat
    // source_row = action_offsets[batch] + seq - action_start), NOT by its
    // batch dim, so it may hold only the action-bearing samples (B_a <= B) for
    // mixed action+multimodal batches. Only require it not exceed the batch.
    ASSERT_CHECK(action_qkv.size(0) <= text_qkv.size(0));
    const int64_t expected_qkv = (num_heads * 2 + num_key_value_heads * 2) * head_dim;
    ASSERT_CHECK(text_qkv.size(2) == expected_qkv);
    ASSERT_CHECK(action_qkv.size(2) == expected_qkv);
    ASSERT_CHECK(text_qkv.dtype().code == action_qkv.dtype().code);
    ASSERT_CHECK(text_qkv.dtype().bits == action_qkv.dtype().bits);
    const int device_type = text_qkv.device().device_type;
    const int device_id = text_qkv.device().device_id;
    check_same_device(action_qkv, "action_qkv", device_type, device_id);
    check_i64_1d(action_starts, "action_starts", text_qkv.size(0), device_type, device_id);
    check_i64_1d(action_ends, "action_ends", text_qkv.size(0), device_type, device_id);
    check_i64_1d(action_offsets, "action_offsets", text_qkv.size(0) + 1, device_type, device_id);
    check_weight(q_weight, "q_weight", head_dim, device_type, device_id);
    check_weight(k_weight, "k_weight", head_dim, device_type, device_id);
    check_weight(q_action_weight, "q_action_weight", head_dim, device_type, device_id);
    check_weight(k_action_weight, "k_action_weight", head_dim, device_type, device_id);
    if (!same_dtype(q_weight, k_weight) ||
        !same_dtype(q_weight, q_action_weight) ||
        !same_dtype(q_weight, k_action_weight)) {
        throw std::runtime_error("qkv_norm_split: all norm weights must have the same dtype");
    }
}

void QkvNormSplit(
    tvm::ffi::TensorView query,
    tvm::ffi::TensorView key,
    tvm::ffi::TensorView value,
    tvm::ffi::TensorView gate,
    tvm::ffi::TensorView text_qkv,
    tvm::ffi::TensorView action_qkv,
    tvm::ffi::TensorView action_starts,
    tvm::ffi::TensorView action_ends,
    tvm::ffi::TensorView action_offsets,
    tvm::ffi::TensorView q_weight,
    tvm::ffi::TensorView k_weight,
    tvm::ffi::TensorView q_action_weight,
    tvm::ffi::TensorView k_action_weight,
    int64_t num_heads,
    int64_t num_key_value_heads,
    int64_t head_dim,
    double epsilon) {
    validate_common(
        text_qkv,
        action_qkv,
        action_starts,
        action_ends,
        action_offsets,
        q_weight,
        k_weight,
        q_action_weight,
        k_action_weight,
        num_heads,
        num_key_value_heads,
        head_dim);
    const int64_t batch_size = text_qkv.size(0);
    const int64_t seq_length = text_qkv.size(1);
    ASSERT_CHECK(query.ndim() == 4 && query.size(0) == batch_size && query.size(1) == seq_length);
    ASSERT_CHECK(query.size(2) == num_heads && query.size(3) == head_dim);
    ASSERT_CHECK(key.ndim() == 4 && key.size(0) == batch_size && key.size(1) == seq_length);
    ASSERT_CHECK(key.size(2) == num_key_value_heads && key.size(3) == head_dim);
    ASSERT_CHECK(value.ndim() == 4 && value.size(0) == batch_size && value.size(1) == seq_length);
    ASSERT_CHECK(value.size(2) == num_key_value_heads && value.size(3) == head_dim);
    ASSERT_CHECK(gate.ndim() == 3 && gate.size(0) == batch_size && gate.size(1) == seq_length);
    ASSERT_CHECK(gate.size(2) == num_heads * head_dim);
    ASSERT_CHECK(query.IsContiguous() && key.IsContiguous() && value.IsContiguous() && gate.IsContiguous());

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(text_qkv.device().device_type, text_qkv.device().device_id));

    if (text_qkv.dtype().code == kDLFloat && text_qkv.dtype().bits == 32) {
        launch_forward_for_weight_dtype<float>(
            query,
            key,
            value,
            gate,
            text_qkv,
            action_qkv,
            action_starts,
            action_ends,
            action_offsets,
            q_weight,
            k_weight,
            q_action_weight,
            k_action_weight,
            num_heads,
            num_key_value_heads,
            head_dim,
            static_cast<float>(epsilon),
            stream);
    } else if (text_qkv.dtype().code == kDLFloat && text_qkv.dtype().bits == 16) {
        launch_forward_for_weight_dtype<__half>(
            query,
            key,
            value,
            gate,
            text_qkv,
            action_qkv,
            action_starts,
            action_ends,
            action_offsets,
            q_weight,
            k_weight,
            q_action_weight,
            k_action_weight,
            num_heads,
            num_key_value_heads,
            head_dim,
            static_cast<float>(epsilon),
            stream);
    } else if (text_qkv.dtype().code == kDLBfloat && text_qkv.dtype().bits == 16) {
        launch_forward_for_weight_dtype<__nv_bfloat16>(
            query,
            key,
            value,
            gate,
            text_qkv,
            action_qkv,
            action_starts,
            action_ends,
            action_offsets,
            q_weight,
            k_weight,
            q_action_weight,
            k_action_weight,
            num_heads,
            num_key_value_heads,
            head_dim,
            static_cast<float>(epsilon),
            stream);
    } else {
        throw std::runtime_error("qkv_norm_split: unsupported dtype");
    }
    sync_check_cuda_error();
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(
    qkv_norm_split,
    xinfer_qkv_norm_split::QkvNormSplit);

}  // namespace xinfer_qkv_norm_split
