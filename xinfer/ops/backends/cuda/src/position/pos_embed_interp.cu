/**
 * Fused position embedding interpolation kernel for ViT.
 *
 * Fuses bilinear interpolation + spatial merge permutation into a single
 * CUDA kernel. Each thread computes one (output_token, hidden_dim_chunk)
 * independently:
 *   1. Find which image this output token belongs to (binary search on cumsum)
 *   2. Reverse spatial merge permutation to get source (row, col) in image grid
 *   3. Bilinear interpolation: map (row, col) to continuous coord in pos_embed table
 *   4. Read 4 corners from pos_embed_weight, weighted sum → write output
 *
 * This replaces ~20 PyTorch kernel launches with 1 fused kernel.
 */

#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cassert>
#include <limits>
#include <utility>

#include "common/cuda_utils.h"
#include "common/tvm_ffi_cuda_tensor.h"

namespace xinfer_pos_embed_interp {

template <typename T>
__device__ __forceinline__ float to_float(T x) {
    return static_cast<float>(x);
}

template <>
__device__ __forceinline__ float to_float<__half>(__half x) {
    return __half2float(x);
}

template <>
__device__ __forceinline__ float to_float<__nv_bfloat16>(__nv_bfloat16 x) {
    return __bfloat162float(x);
}

template <typename T>
__device__ __forceinline__ T from_float(float x) {
    return static_cast<T>(x);
}

template <>
__device__ __forceinline__ __half from_float<__half>(float x) {
    return __float2half_rn(x);
}

template <>
__device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float x) {
    return __float2bfloat16_rn(x);
}

// Bit-exact replica of torch.linspace(0, N-1, steps) in fp32 (ATen fill):
// the step is rounded to fp32 first, values fill from both ends toward the
// middle, and each value is one rounding of an exact double product /
// difference. Verified bitwise against CPU torch.linspace for the grid sizes
// used by the vision tower.
//
// steps == 1 returns 0.0f, matching torch.linspace(0, N-1, 1) == [0.].
// steps == 0 (torch would return an empty tensor) is unreachable here: callers
// derive steps from the interpolation grid h/w, and the kernel launch grid is
// sized by h*w, so no thread ever executes with steps == 0.
__device__ __forceinline__ float linspace_coord(
    int32_t i, int32_t steps, int32_t N
) {
    if (steps <= 1) return 0.0f;
    const float step = (float)(N - 1) / (float)(steps - 1);
    if (i < steps / 2) {
        return (float)((double)i * (double)step);
    }
    return (float)((double)(N - 1) - (double)(steps - 1 - i) * (double)step);
}

// Helper: binary search to find which image a token belongs to
__device__ __forceinline__ int find_image_idx(
    const int32_t* cumsum, int num_images, int32_t token_idx
) {
    int lo = 0, hi = num_images;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (cumsum[mid + 1] <= token_idx) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

/**
 * Each thread handles one element of the output: (token_idx, dim_idx).
 * We process VEC_SIZE elements per thread along the hidden dim for coalescing.
 */
template <typename T, int VEC_SIZE>
__global__ void fused_pos_embed_interp_kernel(
    const T* __restrict__ pos_embed_weight,      // [N*N, hidden_dim]
    const int32_t* __restrict__ grid_thw,    // [num_images, 3]
    const int32_t* __restrict__ out_cumsum,  // [num_images+1] cumsum of t*h*w
    const int32_t* __restrict__ hw_cumsum,   // [num_images+1] cumsum of h*w (pre-temporal-repeat)
    // add_input may alias output (in-place add): no __restrict__ on either.
    const T* add_input,                      // optional [total_output, hidden_dim]
    T* output,                               // [total_output, hidden_dim]
    const int num_grid_per_side,             // N
    const int spatial_merge_size,            // m
    const int hidden_dim,                    // D
    const int num_images,
    const int total_output
) {
    // Each thread processes VEC_SIZE hidden dims for one token
    const int token_idx = blockIdx.x;
    const int dim_start = (blockIdx.y * blockDim.x + threadIdx.x) * VEC_SIZE;

    if (token_idx >= total_output || dim_start >= hidden_dim) return;

    const int N = num_grid_per_side;
    const int m = spatial_merge_size;
    const int m2 = m * m;

    // 1. Find which image this output token belongs to
    int img = find_image_idx(out_cumsum, num_images, token_idx);

    int32_t h = grid_thw[img * 3 + 1];
    int32_t w = grid_thw[img * 3 + 2];
    int32_t hw = h * w;

    // Local index within this image's output
    int32_t local_idx = token_idx - out_cumsum[img];

    // Frame index (temporal) and position within frame
    int32_t pos_in_frame = local_idx % hw;

    // 2. Reverse spatial merge permutation
    // Output layout: (h/m, w/m, m, m) flattened
    // Source layout: (h, w) row-major = (h/m, m, w/m, m) flattened
    int32_t w_div_m = w / m;
    int32_t block_idx = pos_in_frame / m2;
    int32_t inner_idx = pos_in_frame % m2;
    int32_t block_row = block_idx / w_div_m;
    int32_t block_col = block_idx % w_div_m;
    int32_t inner_row = inner_idx / m;
    int32_t inner_col = inner_idx % m;

    int32_t src_row = block_row * m + inner_row;
    int32_t src_col = block_col * m + inner_col;

    // 3. Bilinear interpolation coordinates — bit-exact torch.linspace replica
    const float h_coord = linspace_coord(src_row, h, N);
    const float w_coord = linspace_coord(src_col, w, N);

    int h_floor = (int)h_coord;
    int w_floor = (int)w_coord;
    int h_ceil = min(h_floor + 1, N - 1);
    int w_ceil = min(w_floor + 1, N - 1);

    float dh = h_coord - (float)h_floor;
    float dw = w_coord - (float)w_floor;

    const float w_ff = (1.0f - dh) * (1.0f - dw);
    const float w_fc = (1.0f - dh) * dw;
    const float w_cf = dh * (1.0f - dw);
    const float w_cc = dh * dw;

    // 4-corner flat indices into pos_embed_weight [N*N, D]
    int idx_ff = h_floor * N + w_floor;
    int idx_fc = h_floor * N + w_ceil;
    int idx_cf = h_ceil * N + w_floor;
    int idx_cc = h_ceil * N + w_ceil;

    // 4. Read 4 corners and weighted sum → write output
    // Process VEC_SIZE elements for memory coalescing
    const T* row_ff = pos_embed_weight + idx_ff * hidden_dim;
    const T* row_fc = pos_embed_weight + idx_fc * hidden_dim;
    const T* row_cf = pos_embed_weight + idx_cf * hidden_dim;
    const T* row_cc = pos_embed_weight + idx_cc * hidden_dim;
    T* out_row = output + (int64_t)token_idx * hidden_dim;

    #pragma unroll
    for (int v = 0; v < VEC_SIZE; v++) {
        int d = dim_start + v;
        if (d < hidden_dim) {
            float val = w_ff * to_float(row_ff[d]);
            val = val + w_fc * to_float(row_fc[d]);
            val = val + w_cf * to_float(row_cf[d]);
            val = val + w_cc * to_float(row_cc[d]);
            if (add_input != nullptr) {
                // Preserve the legacy two-kernel BF16/FP16 rounding contract:
                // interpolate -> round to T -> add residual -> round to T.
                const T interpolated = from_float<T>(val);
                val = to_float(interpolated) +
                    to_float(add_input[(int64_t)token_idx * hidden_dim + d]);
            }
            out_row[d] = from_float<T>(val);
        }
    }
}

/**
 * Backward for pos_embed_weight.
 *
 * Forward computes:
 *   output[token, d] = sum_i weight_i(token) * pos_embed_weight[idx_i(token), d]
 *
 * Therefore:
 *   grad_weight[idx_i(token), d] = sum_{token} grad_output[token, d] * weight_i(token)
 *
 * ---------------------------------------------------------------------------
 * Two implementations live here:
 *
 *  1. `..._backward_atomic_kernel`  (legacy, scatter + atomicAdd)
 *     Each thread owns one (token, dim) and atomically adds its four weighted
 *     contributions into the table. Correct, but the accumulation order inside
 *     each table cell is decided by hardware arbitration, so the result is
 *     NOT reproducible run to run: an identical process that merely reached
 *     this kernel through a different history (fresh start vs. resumed from a
 *     checkpoint) can land the same addends in a different order and produce a
 *     different last-bit result. Kept only as a benchmark baseline and to let
 *     tests demonstrate the difference.
 *
 *  2. `..._backward_det_kernel`  (default, gather + fixed-order reduction)
 *     Inverts the loop: one block owns one table cell (i, j) and *gathers*
 *     every contribution destined for it. No atomics at all, and the addition
 *     order is a pure function of (grid_thw, N, m) -- it does not depend on
 *     scheduling, occupancy, allocator state, or anything else about the
 *     process. Identical inputs therefore give bit-identical output, always.
 *
 *     The gather set is found analytically instead of by sorting. Because
 *     `linspace_coord(r, h, N)` is monotonically increasing in r, the set of
 *     source rows whose floor lands on a given table row is a contiguous
 *     interval, so:
 *
 *        { r : h_floor(r) == i  or  h_ceil(r) == i }
 *      = { r : h_floor(r) in {i-1, i} }
 *      = [ lower_bound(i-1), lower_bound(i+1) )
 *
 *     one contiguous range, located with a binary search. Same for columns.
 *     Each block therefore only has to walk a small (rows x cols) rectangle
 *     per image -- ~1.4 source patches per image for the shape it was tuned on.
 *
 *     The block first materialises its contribution list -- (token, weight)
 *     pairs, in canonical (image, frame, row, col) order -- into shared
 *     memory, one thread per entry, and only then streams gradient rows. That
 *     keeps the coordinate/permutation arithmetic off the hot path: otherwise
 *     all eight warps of the block would redundantly recompute the same
 *     `linspace_coord` doubles and integer divisions, which measured as ~24us
 *     of the 62us runtime at that shape.
 *
 *     The clamp `h_ceil = min(h_floor+1, N-1)` is handled by deriving the row
 *     weight as
 *
 *        wr = (h_floor == i ? 1-dh : 0) + (h_ceil == i ? dh : 0)
 *
 *     which reproduces the legacy kernel's per-contribution weights bit for
 *     bit (in the doubly-counted corner case dh is exactly 0, so wr is
 *     exactly 1). Only the summation *order* differs, and that is the point.
 */

void PosEmbedInterp(
    tvm::ffi::TensorView pos_embed_weight,  // [N*N, hidden_dim]
    tvm::ffi::TensorView grid_thw,          // [num_images, 3], int32
    tvm::ffi::TensorView out_cumsum,        // [num_images+1], int32
    tvm::ffi::TensorView hw_cumsum,         // [num_images+1], int32
    tvm::ffi::TensorView output,            // [total_output, hidden_dim]
    int num_grid_per_side,
    int spatial_merge_size
) {
    ASSERT_CHECK(pos_embed_weight.ndim() == 2);
    ASSERT_CHECK(pos_embed_weight.dtype().code == output.dtype().code);
    ASSERT_CHECK(pos_embed_weight.dtype().bits == output.dtype().bits);
    ASSERT_CHECK(grid_thw.ndim() == 2 && grid_thw.size(1) == 3);
    ASSERT_CHECK(output.ndim() == 2);
    ASSERT_CHECK(pos_embed_weight.size(1) == output.size(1));

    const int num_images = grid_thw.size(0);
    const int hidden_dim = output.size(1);
    const int total_output = output.size(0);

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(output.device().device_type, output.device().device_id));

    constexpr int VEC_SIZE = 4;
    const int threads = 256;
    const int dim_blocks = (hidden_dim + threads * VEC_SIZE - 1) / (threads * VEC_SIZE);
    dim3 grid(total_output, dim_blocks);

    const int32_t* grid_ptr = static_cast<int32_t*>(grid_thw.data_ptr());
    const int32_t* out_cum_ptr = static_cast<int32_t*>(out_cumsum.data_ptr());
    const int32_t* hw_cum_ptr = static_cast<int32_t*>(hw_cumsum.data_ptr());

    if (output.dtype().code == kDLFloat && output.dtype().bits == 32) {
        fused_pos_embed_interp_kernel<float, VEC_SIZE><<<grid, threads, 0, stream>>>(
            static_cast<const float*>(pos_embed_weight.data_ptr()),
            grid_ptr, out_cum_ptr, hw_cum_ptr,
            nullptr, static_cast<float*>(output.data_ptr()),
            num_grid_per_side, spatial_merge_size, hidden_dim, num_images, total_output);
    } else if (output.dtype().code == kDLFloat && output.dtype().bits == 16) {
        fused_pos_embed_interp_kernel<__half, VEC_SIZE><<<grid, threads, 0, stream>>>(
            static_cast<const __half*>(pos_embed_weight.data_ptr()),
            grid_ptr, out_cum_ptr, hw_cum_ptr,
            nullptr, static_cast<__half*>(output.data_ptr()),
            num_grid_per_side, spatial_merge_size, hidden_dim, num_images, total_output);
    } else if (output.dtype().code == kDLBfloat && output.dtype().bits == 16) {
        fused_pos_embed_interp_kernel<__nv_bfloat16, VEC_SIZE><<<grid, threads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(pos_embed_weight.data_ptr()),
            grid_ptr, out_cum_ptr, hw_cum_ptr,
            nullptr, static_cast<__nv_bfloat16*>(output.data_ptr()),
            num_grid_per_side, spatial_merge_size, hidden_dim, num_images, total_output);
    } else {
        ASSERT_CHECK(false);
    }

    sync_check_cuda_error();
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(pos_embed_interp, xinfer_pos_embed_interp::PosEmbedInterp);

} // namespace xinfer_pos_embed_interp
