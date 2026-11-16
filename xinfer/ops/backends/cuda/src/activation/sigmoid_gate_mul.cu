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

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <string>
#include <type_traits>

#include "common/cuda_utils.h"
#include "common/tvm_ffi_cuda_tensor.h"

namespace xinfer_silu_and_mul {

// ============================================================
// Activation tuning notes (H100 sm90 / A800 sm80, bf16, shapes
// [21504, 2x9216] and [26624, 2x5120]):
//
//  * The dominant win is vector width. The original kernels used
//    VecTraits (bfloat162 = 4 B per access); 16 B accesses
//    (PackVec<bf16,8>) raise achieved HBM bandwidth from ~70-75% to
//    ~90% of peak on H100 and from ~62-68% to ~86% on A800.
//  * Grid shape: a plain non-persistent 2-D grid (grid.x covers the
//    packed column dimension, grid.y = one block-row per tensor row)
//    beats every persistent variant tried (grid.y = k*sm_count for
//    k in {1,2,4,8}) by 2-6% on both architectures. The row loop is
//    kept only for the num_rows > 65535 case.
//  * BLOCK_X = 128 is best-or-tied on both architectures and divides
//    the packed width of both shapes exactly.
//  * __ldcs/__stcs streaming hints and manual row unrolling were both
//    measured slower; float math (vs half2 intrinsics) is not a
//    bottleneck at ~90% of peak bandwidth. Do not "optimize" these
//    back in without re-measuring.
//
// The pack width is chosen at runtime per call: the widest of
// {16B, 8B, 4B, scalar} compatible with the last-dim length, every
// outer stride and every base pointer. Aligned tensors take
// the 16 B path; odd shapes (odd intermediate size, misaligned slice
// bases) degrade gracefully instead of asserting.
// ============================================================

// Returns the stride (in elements) of dimension d.
// When strides is null (fully contiguous, row-major), computes from shape.
// When strides is non-null, reads the stored value directly.
static int64_t safe_stride(const tvm::ffi::TensorView& t, int d) {
    if (t.IsContiguous()) {
        int64_t s = 1;
        for (int i = t.ndim() - 1; i > d; --i) s *= t.size(i);
        return s;
    }
    return t.stride(d);
}

constexpr int MAX_OUTER_DIMS = 7;  // supports tensors up to 8-D
constexpr int BLOCK_X = 128;       // see tuning notes above
constexpr int MAX_GRID_Y = 65535;  // hardware limit on gridDim.y

struct SiluMulInplaceArgs {
    int64_t strides1[MAX_OUTER_DIMS];  // outer-dim strides of inout1,  in elements
    int64_t strides2[MAX_OUTER_DIMS];  // outer-dim strides of input2,  in elements
    int64_t shapes[MAX_OUTER_DIMS];    // outer-dim sizes (same for both tensors)
    int outer_ndim;                    // ndim - 1  (0 for 1-D tensors)
};

// ============================================================
// Scalar helpers
// ============================================================

template<typename T>
__device__ __forceinline__ float to_float(T x);

template<>
__device__ __forceinline__ float to_float<float>(float x) {
    return x;
}

template<>
__device__ __forceinline__ float to_float<__half>(__half x) {
    return __half2float(x);
}

template<>
__device__ __forceinline__ float to_float<__nv_bfloat16>(__nv_bfloat16 x) {
    return __bfloat162float(x);
}

template<typename T>
__device__ __forceinline__ T from_float(float x);

template<>
__device__ __forceinline__ float from_float<float>(float x) {
    return x;
}

template<>
__device__ __forceinline__ __half from_float<__half>(float x) {
    return __float2half(x);
}

template<>
__device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float x) {
    return __float2bfloat16(x);
}

// ============================================================
// Packed vector type: PACK elements of T, loaded/stored as one
// naturally-aligned transaction (up to 16 B).
// ============================================================

template<typename T, int PACK>
struct alignas(sizeof(T) * PACK) PackVec {
    T d[PACK];
};

template<typename T>
constexpr int max_pack() { return 16 / (int)sizeof(T); }

// Widest pack (elements) such that
//   * every length in `lens` is divisible by it, and
//   * every pointer in `ptrs` is aligned to pack*sizeof(T).
// Falls back to 1 (scalar) in the worst case.
template<typename T>
static int choose_pack(std::initializer_list<const void*> ptrs,
                       const int64_t* lens, int n_lens) {
    int pack = max_pack<T>();
    auto incompatible = [&](int p) {
        const uintptr_t align = (uintptr_t)p * sizeof(T);
        for (const void* q : ptrs) {
            if (reinterpret_cast<uintptr_t>(q) % align != 0) return true;
        }
        for (int i = 0; i < n_lens; ++i) {
            if (lens[i] % p != 0) return true;
        }
        return false;
    };
    while (pack > 1 && incompatible(pack)) pack >>= 1;
    return pack;
}

// Decompose a flat row index into per-dimension indices and accumulate
// per-tensor element offsets. NUM_TENSORS stride arrays are walked in lockstep.
template<int NUM_TENSORS>
__device__ __forceinline__ void row_offsets(
    int64_t row,
    const int64_t* shapes,
    int outer_ndim,
    const int64_t* const (&strides)[NUM_TENSORS],
    int64_t (&offsets)[NUM_TENSORS])
{
    #pragma unroll
    for (int t = 0; t < NUM_TENSORS; ++t) offsets[t] = 0;

    if (outer_ndim == 1) {
        #pragma unroll
        for (int t = 0; t < NUM_TENSORS; ++t) offsets[t] = row * strides[t][0];
        return;
    }
    if (outer_ndim > 1) {
        int64_t r = row;
        #pragma unroll MAX_OUTER_DIMS
        for (int d = MAX_OUTER_DIMS - 1; d >= 0; --d) {
            if (d < outer_ndim) {
                const int64_t idx = r % shapes[d];
                r /= shapes[d];
                #pragma unroll
                for (int t = 0; t < NUM_TENSORS; ++t) offsets[t] += idx * strides[t][d];
            }
        }
    }
}

static inline dim3 make_grid_2d(int64_t col_packs, int64_t num_rows) {
    const int grid_x = (int)((col_packs + BLOCK_X - 1) / BLOCK_X);
    const int grid_y = (int)std::min<int64_t>(num_rows, MAX_GRID_Y);
    return dim3(grid_x, grid_y);
}

// ============================================================
// Kernel 3b: in-place attention gate: output *= sigmoid(gate).
// The gate is the strided tail of a packed [Q|K|V|gate] GEMM output, so only
// the last dimension must be contiguous; no contiguous copy is materialized.
// ============================================================
template<typename T, int PACK, bool ROUND_SIGMOID = false>
__global__ void sigmoidGateMulInplaceKernel(
    T* __restrict__ output,
    const T* __restrict__ gate,
    SiluMulInplaceArgs args,
    int64_t num_rows,
    int64_t last_packs)
{
    using P = PackVec<T, PACK>;
    const int64_t col = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= last_packs) return;

    for (int64_t row = (int64_t)blockIdx.y; row < num_rows; row += (int64_t)gridDim.y) {
        const int64_t* strides[2] = {args.strides1, args.strides2};
        int64_t off[2];
        row_offsets<2>(row, args.shapes, args.outer_ndim, strides, off);

        P* p1 = reinterpret_cast<P*>(output + off[0]) + col;
        const P* p2 = reinterpret_cast<const P*>(gate + off[1]) + col;

        P x = *p1;
        P g = *p2;
        P out;
        #pragma unroll
        for (int j = 0; j < PACK; ++j) {
            const float xv = to_float<T>(x.d[j]);
            const float gv = to_float<T>(g.d[j]);
            if constexpr (ROUND_SIGMOID) {
                // Preserve the materialized sigmoid dtype before multiplying.
                const T sg = from_float<T>(1.0f / (1.0f + __expf(-gv)));
                out.d[j] = from_float<T>(xv * to_float<T>(sg));
            } else {
                out.d[j] = from_float<T>(xv / (1.0f + __expf(-gv)));
            }
        }
        *p1 = out;
    }
}

// ============================================================
// Pack-width runtime dispatch
// ============================================================

// Invokes launcher.template run<PACK>() with the widest supported pack.
template<typename T, typename Launcher>
static void dispatch_pack(int pack, Launcher&& launcher) {
    switch (pack) {
        case 8:
            if constexpr (max_pack<T>() >= 8) { launcher.template run<8>(); return; }
            break;
        case 4:
            if constexpr (max_pack<T>() >= 4) { launcher.template run<4>(); return; }
            break;
        case 2:
            launcher.template run<2>();
            return;
        default:
            break;
    }
    launcher.template run<1>();
}

// Invokes Fn<T>(args...) for the runtime dtype of `dt`.
#define DISPATCH_FLOAT_DTYPE(dt, op_name, ...)                              \
    do {                                                                     \
        if ((dt).code == kDLFloat && (dt).bits == 32) {                      \
            using scalar_t = float;                                          \
            __VA_ARGS__;                                                     \
        } else if ((dt).code == kDLFloat && (dt).bits == 16) {               \
            using scalar_t = __half;                                         \
            __VA_ARGS__;                                                     \
        } else if ((dt).code == kDLBfloat && (dt).bits == 16) {              \
            using scalar_t = __nv_bfloat16;                                  \
            __VA_ARGS__;                                                     \
        } else {                                                             \
            throw std::runtime_error(op_name ": unsupported dtype");         \
        }                                                                    \
    } while (0)

// ============================================================
// Helpers
// ============================================================

template<typename T, bool ROUND_SIGMOID>
struct SigmoidGateMulLauncher {
    T* output; const T* gate;
    const SiluMulInplaceArgs* args;
    int64_t num_rows; int64_t last_dim; cudaStream_t stream;
    template<int PACK> void run() {
        const int64_t last_packs = last_dim / PACK;
        sigmoidGateMulInplaceKernel<T, PACK, ROUND_SIGMOID>
            <<<make_grid_2d(last_packs, num_rows), BLOCK_X, 0, stream>>>(
                output, gate, *args, num_rows, last_packs);
    }
};

// Collect the strides (of dims with extent > 1) that constrain the pack
// width. Size-1 dims are skipped: their strides are never multiplied by a
// nonzero index, and torch frequently reports arbitrary values for them.
static void collect_outer_strides(
    const int64_t* strides, const int64_t* shapes, int outer_ndim,
    int64_t* out, int& count)
{
    for (int d = 0; d < outer_ndim; ++d) {
        if (shapes[d] > 1) out[count++] = strides[d];
    }
}

// ============================================================
// Public functions
// ============================================================

// output *= sigmoid(gate) in place. `gate` may be a strided view (the tail of
// a packed [Q|K|V|gate] GEMM output); only the last dim of either operand has
// to be dense. round_sigmoid rounds sigmoid(gate) to the dtype before the
// multiply, matching a materialized `output * torch.sigmoid(gate)`.
void SigmoidGateMulInplace(
    tvm::ffi::TensorView output,
    tvm::ffi::TensorView gate,
    bool round_sigmoid)
{
    const int ndim = output.ndim();
    xinfer::ffi::CheckCuda(output, "output");
    xinfer::ffi::CheckSameDevice(gate, output, "gate");
    if (ndim < 1 || ndim != gate.ndim() || ndim - 1 > MAX_OUTER_DIMS) {
        throw std::runtime_error(
            "sigmoid_gate_mul_inplace: output and gate must have the same rank in [1, " +
            std::to_string(MAX_OUTER_DIMS + 1) + "]");
    }
    if (output.dtype().code != gate.dtype().code ||
        output.dtype().bits != gate.dtype().bits ||
        output.dtype().lanes != gate.dtype().lanes) {
        throw std::runtime_error("sigmoid_gate_mul_inplace: gate must have the output dtype");
    }
    for (int i = 0; i < ndim; ++i) {
        if (output.size(i) != gate.size(i)) {
            throw std::runtime_error("sigmoid_gate_mul_inplace: gate must have the output shape");
        }
    }
    if (safe_stride(output, ndim - 1) != 1 || safe_stride(gate, ndim - 1) != 1) {
        throw std::runtime_error(
            "sigmoid_gate_mul_inplace: output and gate need a contiguous last dim");
    }
    xinfer::ffi::CheckNoOverlap(output, "output", gate, "gate");

    const int64_t last_dim = output.size(ndim - 1);
    const int outer_ndim = ndim - 1;

    SiluMulInplaceArgs args{};
    args.outer_ndim = outer_ndim;
    int64_t num_rows = 1;
    for (int d = 0; d < outer_ndim; ++d) {
        args.strides1[d] = safe_stride(output, d);
        args.strides2[d] = safe_stride(gate, d);
        args.shapes[d]   = output.size(d);
        num_rows *= output.size(d);
    }
    if (num_rows == 0 || last_dim == 0) return;

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(output.device().device_type, output.device().device_id));

    int64_t stride_lens[2 * MAX_OUTER_DIMS + 1];
    int n_lens = 0;
    stride_lens[n_lens++] = last_dim;
    collect_outer_strides(args.strides1, args.shapes, outer_ndim, stride_lens, n_lens);
    collect_outer_strides(args.strides2, args.shapes, outer_ndim, stride_lens, n_lens);

    DISPATCH_FLOAT_DTYPE(output.dtype(), "sigmoid_gate_mul_inplace", {
        const int pack = choose_pack<scalar_t>(
            {output.data_ptr(), gate.data_ptr()},
            stride_lens, n_lens);
        auto launch = [&](auto round) {
            SigmoidGateMulLauncher<scalar_t, decltype(round)::value> launcher{
                static_cast<scalar_t*>(output.data_ptr()),
                static_cast<const scalar_t*>(gate.data_ptr()),
                &args, num_rows, last_dim, stream};
            dispatch_pack<scalar_t>(pack, launcher);
        };
        if (round_sigmoid) {
            launch(std::true_type{});
        } else {
            launch(std::false_type{});
        }
    });
    sync_check_cuda_error();
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(sigmoid_gate_mul_inplace, xinfer_silu_and_mul::SigmoidGateMulInplace);

}  // namespace xinfer_silu_and_mul
