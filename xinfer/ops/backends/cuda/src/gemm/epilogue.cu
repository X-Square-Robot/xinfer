// Epilogue kernels of the generic GEMM path: whatever the cuBLASLt GEMM of generic.cc cannot fuse runs here as
// a separate elementwise pass (FP32 math, one simple pass).
#include <cuda_bf16.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/optional.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

#include "common/cuda_pdl.cuh"
#include "quant/mxfp8_format_core.cuh"
#include "common/tvm_ffi_cuda_tensor.h"

namespace xinfer::gemm {
namespace {

using tvm::ffi::Optional;
using tvm::ffi::TensorView;

constexpr int kThreads = 256;

template <class Kernel, class... Args>
void Launch(dim3 grid, cudaStream_t stream, Kernel kernel, Args... args) {
  kernel<<<grid, kThreads, 0, stream>>>(args...);
  xinfer::ffi::CheckLastCudaError("gemm epilogue launch");
}

unsigned Blocks(int64_t work) {
  const int64_t b = (work + kThreads - 1) / kThreads;
  return static_cast<unsigned>(b < 65535 * 8 ? (b > 0 ? b : 1) : 65535 * 8);
}

// A kernel launched as a programmatic dependent waits on its predecessor here; a plain launch passes through.
__device__ __forceinline__ void Enter() {
  xinfer::cuda::pdl_wait();
  xinfer::cuda::pdl_trigger();
}

__device__ float BlockSum(float v, float* shared);

__device__ __forceinline__ float Load(const __nv_bfloat16* p, int64_t i) { return __bfloat162float(p[i]); }
__device__ __forceinline__ float Load(const float* p, int64_t i) { return p[i]; }

// out = residual_scale * residual + gate * out, gate broadcast over rows when it has N elements.
template <typename R>
__global__ void ResidualKernel(__nv_bfloat16* out, const R* residual, const __nv_bfloat16* gate,
                               int64_t gate_period, float residual_scale, int64_t total) {
  Enter();
  for (int64_t i = blockIdx.x * int64_t{blockDim.x} + threadIdx.x; i < total; i += int64_t{gridDim.x} * blockDim.x) {
    float y = __bfloat162float(out[i]);
    if (gate) y *= __bfloat162float(gate[i % gate_period]);
    out[i] = __float2bfloat16(residual_scale * Load(residual, i) + y);
  }
}

// act 0: silu, 1: gelu (tanh approximation), 2: gelu (erf).
__device__ __forceinline__ float Act(float g, int act) {
  if (act == 0) return g / (1.0f + expf(-g));
  if (act == 2) return 0.5f * g * (1.0f + erff(g * 0.7071067811865476f));
  const float u = 0.7978845608028654f * (g + 0.044715f * g * g * g);
  return 0.5f * g * (1.0f + tanhf(u));
}

__global__ void ActKernel(__nv_bfloat16* out, int act, int64_t total) {
  Enter();
  for (int64_t i = blockIdx.x * int64_t{blockDim.x} + threadIdx.x; i < total; i += int64_t{gridDim.x} * blockDim.x) {
    out[i] = __float2bfloat16(Act(__bfloat162float(out[i]), act));
  }
}

// out[i, :] *= row_scale[i] (FP32 product, one rounding to BF16).
__global__ void RowScaleKernel(__nv_bfloat16* out, const float* row_scale, int64_t n, int64_t total) {
  Enter();
  for (int64_t i = blockIdx.x * int64_t{blockDim.x} + threadIdx.x; i < total; i += int64_t{gridDim.x} * blockDim.x) {
    out[i] = __float2bfloat16(__bfloat162float(out[i]) * row_scale[i / n]);
  }
}

// Gate/up order of a GEMM row of width 2n, by group width g: blocks of g gate columns, then the g matching up
// columns. g = n: [gate | up]; g = 1: [g0, u0, g1, u1, ...]; g = 16: 16-column blocks.
__device__ __forceinline__ int64_t GateIndex(int64_t row, int64_t col, int64_t n, int64_t g) {
  return row * 2 * n + (col / g) * 2 * g + col % g;
}

// out[m, j] = act(gate_j) * up_j.
__global__ void GluKernel(__nv_bfloat16* out, const __nv_bfloat16* y, int64_t n, int act, int64_t group,
                          int64_t total) {
  Enter();
  for (int64_t i = blockIdx.x * int64_t{blockDim.x} + threadIdx.x; i < total; i += int64_t{gridDim.x} * blockDim.x) {
    const int64_t row = i / n, col = i - row * n;
    const int64_t gi = GateIndex(row, col, n, group);
    const int64_t ui = gi + group;
    const float g = __bfloat162float(y[gi]);
    const float u = __bfloat162float(y[ui]);
    out[i] = __float2bfloat16(Act(g, act) * u);
  }
}

// GluKernel for one row per grid row and eight consecutive columns per thread (16-byte loads and stores, int32
// column arithmetic): the same arithmetic per element, so the same bits. Needs N and the group to be multiples of 8
// and 16-byte aligned rows (EpilogueGlu checks).
__global__ void GluKernel8(__nv_bfloat16* out, const __nv_bfloat16* y, int n, int act, int group) {
  Enter();
  const int col = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
  if (col >= n) return;
  const int64_t row = blockIdx.y;
  const __nv_bfloat16* gate = y + row * 2 * n + (col / group) * 2 * group + col % group;
  const uint4 gv = *reinterpret_cast<const uint4*>(gate);
  const uint4 uv = *reinterpret_cast<const uint4*>(gate + group);
  const __nv_bfloat16* g = reinterpret_cast<const __nv_bfloat16*>(&gv);
  const __nv_bfloat16* u = reinterpret_cast<const __nv_bfloat16*>(&uv);
  uint4 ov;
  __nv_bfloat16* o = reinterpret_cast<__nv_bfloat16*>(&ov);
#pragma unroll
  for (int j = 0; j < 8; ++j) o[j] = __float2bfloat16(Act(__bfloat162float(g[j]), act) * __bfloat162float(u[j]));
  *reinterpret_cast<uint4*>(out + row * n + col) = ov;
}

// Per-head RMSNorm in place on the first `heads` heads of each row: x * rsqrt(mean(x^2) + eps) * w, with
// w = 1 + weight when zero_centered. Q heads use q_weight, the following heads k_weight. One block per
// (row, head).
__global__ void QkNormKernel(__nv_bfloat16* qkv, const __nv_bfloat16* q_weight, const __nv_bfloat16* k_weight,
                             int64_t row_stride, int q_heads, int heads, int head_dim, float eps,
                             bool zero_centered) {
  Enter();
  __shared__ float shared[kThreads / 32];
  const int64_t row = blockIdx.x / heads;
  const int head = static_cast<int>(blockIdx.x % heads);
  __nv_bfloat16* x = qkv + row * row_stride + int64_t{head} * head_dim;
  const __nv_bfloat16* w = head < q_heads ? q_weight : k_weight;
  float ss = 0.0f;
  for (int j = threadIdx.x; j < head_dim; j += kThreads) {
    const float v = __bfloat162float(x[j]);
    ss += v * v;
  }
  const float inv = rsqrtf(BlockSum(ss, shared) / static_cast<float>(head_dim) + eps);
  for (int j = threadIdx.x; j < head_dim; j += kThreads) {
    const float g = __bfloat162float(w[j]) + (zero_centered ? 1.0f : 0.0f);
    x[j] = __float2bfloat16(__bfloat162float(x[j]) * inv * g);
  }
}

// The RoPE pair arithmetic all RoPE kernels share. PairAngle: cos and sin of pair j of `row`, from the [M, half]
// tables or the [half / 2, M, 4] pair table (table_pairs4). PairFirst / PairSecond: offsets within a head of the
// pair's two elements, [j, j + half] (rotate-half layout) or [2j, 2j + 1] (pair layout). Rotate: o1 = x1 c - x2 s
// and o2 = x2 c + x1 s, spelled out with the roundings the kernels had when the compiler contracted them (x1 c and
// x1 s fused into the FMA, x2 s and x2 c rounded first), so that every kernel gets those bits whatever the compiler
// would contract.
__device__ __forceinline__ void PairAngle(const float* cos, const float* sin, int64_t rows, int64_t row, int j,
                                          int half, bool table_pairs4, float& c, float& s) {
  if (table_pairs4) {
    const float* e = cos + ((int64_t{j >> 1} * rows + row) << 2) + 2 * (j & 1);
    c = e[0];
    s = e[1];
  } else {
    c = cos[row * half + j];
    s = sin[row * half + j];
  }
}
__device__ __forceinline__ int PairFirst(int j, bool pairs) { return pairs ? 2 * j : j; }
__device__ __forceinline__ int PairSecond(int j, int half, bool pairs) { return pairs ? 2 * j + 1 : j + half; }
__device__ __forceinline__ void Rotate(float x1, float x2, float c, float s, __nv_bfloat16& o1, __nv_bfloat16& o2) {
  o1 = __float2bfloat16(__fmaf_rn(x1, c, -__fmul_rn(x2, s)));
  o2 = __float2bfloat16(__fmaf_rn(x1, s, __fmul_rn(x2, c)));
}

// Rotate-half RoPE on the first rope_heads heads of each row: for pair j < rotary_dim / 2 with
// x1 = x[j], x2 = x[j + rotary_dim / 2] (standard layout) or x1 = x[2j], x2 = x[2j + 1] (pair layout):
// x1' = x1 cos_j - x2 sin_j, x2' = x2 cos_j + x1 sin_j. Reads y in in_pairs layout, writes out in out_pairs
// layout (out == y only when the layouts match). cos/sin are [M, rotary_dim / 2], or with table_pairs4 one
// float4 table [rotary_dim / 4, M, 4] = {cos_2i, sin_2i, cos_2i+1, sin_2i+1}.
__global__ void RopeKernel(__nv_bfloat16* out, const __nv_bfloat16* y, const float* cos, const float* sin,
                           int64_t rows, int64_t row_stride, int rope_heads, int head_dim, int half, bool in_pairs,
                           bool out_pairs, bool table_pairs4, int64_t total) {
  Enter();
  for (int64_t t = blockIdx.x * int64_t{blockDim.x} + threadIdx.x; t < total; t += int64_t{gridDim.x} * blockDim.x) {
    const int j = static_cast<int>(t % half);
    const int64_t rh = t / half;
    const int head = static_cast<int>(rh % rope_heads);
    const int64_t row = rh / rope_heads;
    const int64_t base = row * row_stride + int64_t{head} * head_dim;
    float c, s;
    PairAngle(cos, sin, rows, row, j, half, table_pairs4, c, s);
    const float x1 = __bfloat162float(y[base + PairFirst(j, in_pairs)]);
    const float x2 = __bfloat162float(y[base + PairSecond(j, half, in_pairs)]);
    Rotate(x1, x2, c, s, out[base + PairFirst(j, out_pairs)], out[base + PairSecond(j, half, out_pairs)]);
  }
}

// RopeKernel for one row per grid row (int32 index arithmetic within the row): the same arithmetic per pair.
__global__ void RopeRowKernel(__nv_bfloat16* out, const __nv_bfloat16* y, const float* cos, const float* sin,
                              int64_t rows, int64_t row_stride, int rope_heads, int head_dim, int half, bool in_pairs,
                              bool out_pairs, bool table_pairs4) {
  Enter();
  const int64_t row = blockIdx.y;
  const int pairs = rope_heads * half;
  for (int t = blockIdx.x * blockDim.x + threadIdx.x; t < pairs; t += gridDim.x * blockDim.x) {
    const int j = t % half;
    const int head = t / half;
    const int64_t base = row * row_stride + int64_t{head} * head_dim;
    float c, s;
    PairAngle(cos, sin, rows, row, j, half, table_pairs4, c, s);
    const float x1 = __bfloat162float(y[base + PairFirst(j, in_pairs)]);
    const float x2 = __bfloat162float(y[base + PairSecond(j, half, in_pairs)]);
    Rotate(x1, x2, c, s, out[base + PairFirst(j, out_pairs)], out[base + PairSecond(j, half, out_pairs)]);
  }
}

// Where RopeKvRowKernel writes: Q heads and the extra columns to q_out [M, H D + extra]; K heads to k and V to v,
// element (head h, row r, d) at h * head_stride + r * row_stride + d (head-major [H_kv, rows, D] or row-major
// [rows, H_kv D] caches), rows start + i.
struct QkvDst {
  __nv_bfloat16* q;
  __nv_bfloat16* k;
  __nv_bfloat16* v;
  int64_t q_stride, k_head, k_row, v_head, v_row, start;
};

// The generic qkv_proj_rope path with a KV cache in one pass, rows of y [M, (H + 2 H_kv) D + extra] over the grid's
// rows (grid row r takes rows r, r + gridDim.y, ...): the RoPE pairs of the H + H_kv rotated heads with the shared
// pair arithmetic, then every other element copied unchanged (the non-rotated tail of those heads, V and the extra
// columns).
__global__ void RopeKvRowKernel(QkvDst dst, const __nv_bfloat16* y, int64_t y_stride, const float* cos,
                                const float* sin, int64_t rows, int heads, int kv_heads, int head_dim, int half,
                                int extra, bool in_pairs, bool out_pairs, bool table_pairs4) {
  Enter();
  const int rope_heads = heads + kv_heads, tail = head_dim - 2 * half;
  const int pairs = rope_heads * half, tails = rope_heads * tail, rest = kv_heads * head_dim + extra;
  for (int64_t row = blockIdx.y; row < rows; row += gridDim.y) {
    const int64_t cache_row = dst.start + row;
    const __nv_bfloat16* src = y + row * y_stride;
    auto rotated = [&](int head, int o) -> __nv_bfloat16& {
      return head < heads ? dst.q[row * dst.q_stride + head * head_dim + o]
                          : dst.k[(head - heads) * dst.k_head + cache_row * dst.k_row + o];
    };
    for (int t = blockIdx.x * blockDim.x + threadIdx.x; t < pairs + tails + rest; t += gridDim.x * blockDim.x) {
      if (t < pairs) {
        const int j = t % half;
        const int head = t / half;
        float c, s;
        PairAngle(cos, sin, rows, row, j, half, table_pairs4, c, s);
        const float x1 = __bfloat162float(src[head * head_dim + PairFirst(j, in_pairs)]);
        const float x2 = __bfloat162float(src[head * head_dim + PairSecond(j, half, in_pairs)]);
        Rotate(x1, x2, c, s, rotated(head, PairFirst(j, out_pairs)), rotated(head, PairSecond(j, half, out_pairs)));
      } else if (t < pairs + tails) {
        const int head = (t - pairs) / tail, o = 2 * half + (t - pairs) % tail;
        rotated(head, o) = src[head * head_dim + o];
      } else {
        const int r = t - pairs - tails;
        const __nv_bfloat16 value = src[rope_heads * head_dim + r];
        if (r < kv_heads * head_dim) {
          dst.v[(r / head_dim) * dst.v_head + cache_row * dst.v_row + r % head_dim] = value;
        } else {
          dst.q[row * dst.q_stride + heads * head_dim + (r - kv_heads * head_dim)] = value;
        }
      }
    }
  }
}

// GLU + MXFP8 quantization: q[m, j] = e4m3(act(gate_j) * up_j / s), one E8M0 scale per 32 (or, shared, per
// 128) output columns written to SWIZZLE_32_4_4 storage (a 1x128 scale fills its four 1x32 slots). One warp
// per (row, 128-column group); each lane owns 4 consecutive columns.
__global__ void GluQuantKernel(uint8_t* q, uint8_t* scales, const __nv_bfloat16* y, int64_t m, int64_t n,
                               int64_t q_stride, int act, int64_t group, bool share_128, int scale_cols4) {
  Enter();
  const int lane = threadIdx.x % 32;
  const int64_t blocks128 = (n + 127) / 128;
  const int64_t warps = int64_t{gridDim.x} * (blockDim.x / 32);
  for (int64_t w = blockIdx.x * int64_t{blockDim.x / 32} + threadIdx.x / 32; w < m * blocks128; w += warps) {
    const int64_t row = w / blocks128, blk = w % blocks128;
    float v[4];
    float amax = 0.0f;
    for (int e = 0; e < 4; ++e) {
      const int64_t col = blk * 128 + lane * 4 + e;
      v[e] = 0.0f;
      if (col < n) {
        const int64_t gi = GateIndex(row, col, n, group);
        v[e] = Act(__bfloat162float(y[gi]), act) * __bfloat162float(y[gi + group]);
      }
      amax = fmaxf(amax, fabsf(v[e]));
    }
    // 1x32: lanes 8b..8b+7 share a block; 1x128: the whole warp.
    for (int o = share_128 ? 16 : 4; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    const uint8_t e8 = xinfer::mxfp8::amax_to_e8m0_ceil(amax);  // ceil(log2(amax / 448)), exact
    const float inv = xinfer::mxfp8::e8m0_inv(e8);
    for (int e = 0; e < 4; ++e) {
      const int64_t col = blk * 128 + lane * 4 + e;
      if (col < n) {
        const __nv_fp8_e4m3 qv(v[e] * inv);
        q[row * q_stride + col] = *reinterpret_cast<const uint8_t*>(&qv);
      }
    }
    if (lane % 8 == 0) {
      const int block = static_cast<int>(blk * 4 + lane / 8);
      if (int64_t{block} * 32 < n) {
        scales[xinfer::mxfp8::scale_offset(static_cast<int>(row), block, scale_cols4)] = e8;
      }
    }
  }
}

__device__ float BlockSum(float v, float* shared) {
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  __syncthreads();
  if (lane == 0) shared[warp] = v;
  __syncthreads();
  v = lane < kThreads / 32 ? shared[lane] : 0.0f;
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

void CheckBf16Matrix(const TensorView& t, const char* name) {
  using namespace xinfer::ffi;
  CheckCuda(t, name);
  CheckBFloat16(t, name);
  CheckContiguous(t, name);
  if (t.ndim() != 2) throw std::runtime_error(std::string(name) + " must be rank-2");
}

}  // namespace

void EpilogueResidual(TensorView out, TensorView residual, Optional<TensorView> gate, double residual_scale) {
  using namespace xinfer::ffi;
  CheckBf16Matrix(out, "out");
  CheckCuda(residual, "residual");
  CheckContiguous(residual, "residual");
  CheckSameShape(residual, out, "residual");
  const int64_t total = NumElements(out), n = out.size(1);
  int64_t period = total;
  const __nv_bfloat16* gate_ptr = nullptr;
  if (gate.has_value()) {
    CheckBFloat16(gate.value(), "residual_gate");
    CheckContiguous(gate.value(), "residual_gate");
    period = NumElements(gate.value());
    if (period != n && period != total) {
      throw std::runtime_error("residual_gate must have N or M*N elements");
    }
    gate_ptr = ConstData<__nv_bfloat16>(gate.value());
  }
  if (total == 0) return;
  cudaStream_t stream = Stream(out);
  const float scale = static_cast<float>(residual_scale);
  if (IsFloat32(residual.dtype())) {
    Launch(Blocks(total), stream, ResidualKernel<float>, Data<__nv_bfloat16>(out), ConstData<float>(residual),
           gate_ptr, period, scale, total);
  } else {
    CheckBFloat16(residual, "residual");
    Launch(Blocks(total), stream, ResidualKernel<__nv_bfloat16>, Data<__nv_bfloat16>(out),
           ConstData<__nv_bfloat16>(residual), gate_ptr, period, scale, total);
  }
}

void EpilogueAct(TensorView out, int64_t act) {
  using namespace xinfer::ffi;
  CheckBf16Matrix(out, "out");
  if (act < 0 || act > 2) throw std::runtime_error("gemm_epilogue_act: act must be 0 (silu), 1 (gelu_tanh) or 2 (gelu)");
  const int64_t total = NumElements(out);
  if (total == 0) return;
  Launch(Blocks(total), Stream(out), ActKernel, Data<__nv_bfloat16>(out), static_cast<int>(act), total);
}

void EpilogueRowScale(TensorView out, TensorView row_scale) {
  using namespace xinfer::ffi;
  CheckBf16Matrix(out, "out");
  CheckCuda(row_scale, "row_scale");
  CheckFloat32(row_scale, "row_scale");
  CheckContiguous(row_scale, "row_scale");
  if (NumElements(row_scale) != out.size(0)) throw std::runtime_error("gemm_epilogue_row_scale: row_scale must have M elements");
  const int64_t total = NumElements(out);
  if (total == 0) return;
  Launch(Blocks(total), Stream(out), RowScaleKernel, Data<__nv_bfloat16>(out), ConstData<float>(row_scale), out.size(1), total);
}

void EpilogueGlu(TensorView out, TensorView y, int64_t act, int64_t group) {
  using namespace xinfer::ffi;
  CheckBf16Matrix(out, "out");
  CheckBf16Matrix(y, "y");
  const int64_t m = out.size(0), n = out.size(1);
  if (y.size(0) != m || y.size(1) != 2 * n) throw std::runtime_error("gemm_epilogue_glu: y must be [M, 2N]");
  if (act != 0 && act != 1) throw std::runtime_error("gemm_epilogue_glu: act must be 0 (silu) or 1 (gelu_tanh)");
  if (group <= 0 || n % group) throw std::runtime_error("gemm_epilogue_glu: group must divide N");
  if (m * n == 0) return;
  __nv_bfloat16* out_ptr = Data<__nv_bfloat16>(out);
  const __nv_bfloat16* y_ptr = ConstData<__nv_bfloat16>(y);
  if (n % 8 == 0 && group % 8 == 0 && 2 * n <= INT32_MAX && m <= 65535 &&
      reinterpret_cast<uintptr_t>(out_ptr) % 16 == 0 && reinterpret_cast<uintptr_t>(y_ptr) % 16 == 0) {
    const dim3 grid(static_cast<unsigned>((n / 8 + kThreads - 1) / kThreads), static_cast<unsigned>(m));
    Launch(grid, Stream(out), GluKernel8, out_ptr, y_ptr, static_cast<int>(n), static_cast<int>(act),
           static_cast<int>(group));
    return;
  }
  Launch(Blocks(m * n), Stream(out), GluKernel, out_ptr, y_ptr, n, static_cast<int>(act), group, m * n);
}

namespace {

// RoPE tables: cos / sin [M, rotary_dim / 2], or one [rotary_dim / 4, M, 4] table (table_pairs4).
const float* RopeSin(const TensorView& cos, const Optional<TensorView>& sin, int64_t m, int64_t rotary_dim,
                     bool table_pairs4, const char* op) {
  using namespace xinfer::ffi;
  CheckCuda(cos, "cos");
  CheckFloat32(cos, "cos");
  CheckContiguous(cos, "cos");
  if (table_pairs4) {
    if (NumElements(cos) != m * rotary_dim) {
      throw std::runtime_error(std::string(op) + ": the pair table must be [rotary_dim / 4, M, 4]");
    }
    return nullptr;
  }
  if (!sin.has_value()) throw std::runtime_error(std::string(op) + ": sin is required");
  CheckFloat32(sin.value(), "sin");
  CheckContiguous(sin.value(), "sin");
  if (NumElements(cos) != m * (rotary_dim / 2) || NumElements(sin.value()) != m * (rotary_dim / 2)) {
    throw std::runtime_error(std::string(op) + ": cos/sin must be [M, rotary_dim / 2]");
  }
  return ConstData<float>(sin.value());
}

// One cache of the KV write: BF16, head-major [H_kv, rows, D] or row-major [rows, H_kv D] with a unit last stride
// and at least start + m rows. Returns {head stride, row stride} in elements.
std::pair<int64_t, int64_t> CacheStrides(const TensorView& cache, int64_t kv_heads, int64_t head_dim, int64_t start,
                                         int64_t m, const char* name) {
  using namespace xinfer::ffi;
  CheckCuda(cache, name);
  CheckBFloat16(cache, name);
  const bool head_major = cache.ndim() == 3 && cache.size(0) == kv_heads && cache.size(2) == head_dim;
  const bool row_major = cache.ndim() == 2 && cache.size(1) == kv_heads * head_dim;
  const int64_t rows = head_major ? cache.size(1) : cache.size(0);
  if ((!head_major && !row_major) || cache.stride(cache.ndim() - 1) != 1 || start < 0 || rows < start + m) {
    throw std::runtime_error(std::string("gemm_epilogue_rope_kvcache: ") + name +
                             " must be BF16 [H_kv, >= start + M, D] or [>= start + M, H_kv D] with a unit last stride");
  }
  return head_major ? std::make_pair(cache.stride(0), cache.stride(1)) : std::make_pair(head_dim, cache.stride(0));
}

}  // namespace

void EpilogueRope(TensorView out, TensorView y, TensorView cos, Optional<TensorView> sin, int64_t rope_heads,
                  int64_t head_dim, int64_t rotary_dim, bool in_pairs, bool out_pairs, bool table_pairs4) {
  using namespace xinfer::ffi;
  CheckBf16Matrix(out, "out");
  CheckBf16Matrix(y, "y");
  CheckSameShape(out, y, "out");
  const int64_t m = out.size(0);
  if (rotary_dim <= 0 || rotary_dim % (table_pairs4 ? 4 : 2) || rotary_dim > head_dim || rope_heads <= 0 ||
      rope_heads * head_dim > out.size(1)) {
    throw std::runtime_error("gemm_epilogue_rope: bad rope_heads / head_dim / rotary_dim");
  }
  const bool aliased = out.data_ptr() == y.data_ptr();
  if (aliased && in_pairs != out_pairs) {
    throw std::runtime_error("gemm_epilogue_rope: changing the layout needs out != y");
  }
  const int64_t half = rotary_dim / 2;
  const float* sin_ptr = RopeSin(cos, sin, m, rotary_dim, table_pairs4, "gemm_epilogue_rope");
  cudaStream_t stream = Stream(out);
  if (m == 0) return;
  if (!aliased) {
    CheckCudaError(cudaMemcpyAsync(out.data_ptr(), y.data_ptr(), NumBytes(out), cudaMemcpyDeviceToDevice, stream),
                   "gemm_epilogue_rope copy");
  }
  const int64_t total = m * rope_heads * half;
  if (m <= 65535 && rope_heads * half <= INT32_MAX) {
    const int64_t pairs = rope_heads * half;
    const dim3 grid(static_cast<unsigned>(std::min<int64_t>((pairs + kThreads - 1) / kThreads, 64)),
                    static_cast<unsigned>(m));
    Launch(grid, stream, RopeRowKernel, Data<__nv_bfloat16>(out), ConstData<__nv_bfloat16>(y), ConstData<float>(cos),
           sin_ptr, m, out.size(1), static_cast<int>(rope_heads), static_cast<int>(head_dim),
           static_cast<int>(half), in_pairs, out_pairs, table_pairs4);
    return;
  }
  Launch(Blocks(total), stream, RopeKernel, Data<__nv_bfloat16>(out), ConstData<__nv_bfloat16>(y),
         ConstData<float>(cos), sin_ptr, m, out.size(1), static_cast<int>(rope_heads), static_cast<int>(head_dim),
         static_cast<int>(half), in_pairs, out_pairs, table_pairs4, total);
}

// y = [Q | K | V | extra] of the GEMM ([M, (H + 2 H_kv) D + extra]); RoPE on the Q and K heads (as
// gemm_epilogue_rope), Q and the extra columns to q_out [M, H D + extra], K and V to rows start .. start + M of the
// caches. Equal to gemm_epilogue_rope on y followed by the copies, in one kernel.
void EpilogueRopeKvcache(TensorView q_out, TensorView y, TensorView cos, Optional<TensorView> sin, TensorView k_cache,
                         TensorView v_cache, int64_t start, int64_t num_heads, int64_t num_kv_heads, int64_t head_dim,
                         int64_t rotary_dim, bool in_pairs, bool out_pairs, bool table_pairs4) {
  using namespace xinfer::ffi;
  CheckBf16Matrix(q_out, "q_out");
  CheckBf16Matrix(y, "y");
  const int64_t m = y.size(0);
  const int64_t extra = y.size(1) - (num_heads + 2 * num_kv_heads) * head_dim;
  if (num_heads <= 0 || num_kv_heads <= 0 || head_dim <= 0 || extra < 0 || rotary_dim <= 0 ||
      rotary_dim % (table_pairs4 ? 4 : 2) || rotary_dim > head_dim || y.size(1) > INT32_MAX) {
    throw std::runtime_error("gemm_epilogue_rope_kvcache: bad heads / head_dim / rotary_dim for y");
  }
  if (q_out.size(0) != m || q_out.size(1) != num_heads * head_dim + extra || q_out.data_ptr() == y.data_ptr()) {
    throw std::runtime_error("gemm_epilogue_rope_kvcache: q_out must be a separate [M, H * head_dim + extra columns]");
  }
  const float* sin_ptr = RopeSin(cos, sin, m, rotary_dim, table_pairs4, "gemm_epilogue_rope_kvcache");
  const auto k = CacheStrides(k_cache, num_kv_heads, head_dim, start, m, "k_cache");
  const auto v = CacheStrides(v_cache, num_kv_heads, head_dim, start, m, "v_cache");
  if (m == 0) return;
  const QkvDst dst{Data<__nv_bfloat16>(q_out), Data<__nv_bfloat16>(k_cache), Data<__nv_bfloat16>(v_cache),
                   q_out.stride(0), k.first, k.second, v.first, v.second, start};
  const int64_t work = y.size(1) - (num_heads + num_kv_heads) * (rotary_dim / 2);  // one thread per pair
  const dim3 grid(static_cast<unsigned>(std::min<int64_t>((work + kThreads - 1) / kThreads, 64)),
                  static_cast<unsigned>(std::min<int64_t>(m, 65535)));  // grid rows loop over the rows beyond
  Launch(grid, Stream(y), RopeKvRowKernel, dst, ConstData<__nv_bfloat16>(y), y.stride(0), ConstData<float>(cos),
         sin_ptr, m, static_cast<int>(num_heads), static_cast<int>(num_kv_heads), static_cast<int>(head_dim),
         static_cast<int>(rotary_dim / 2), static_cast<int>(extra), in_pairs, out_pairs, table_pairs4);
}

void EpilogueGluQuant(TensorView q, TensorView scales, TensorView y, int64_t act, int64_t group, bool share_128) {
  using namespace xinfer::ffi;
  CheckBf16Matrix(y, "y");
  CheckCuda(q, "q");
  CheckFloat8E4M3Storage(q, "q");
  CheckContiguous(q, "q");
  CheckFloat8E8M0Storage(scales, "scales");
  if (q.ndim() != 2) throw std::runtime_error("gemm_epilogue_glu_quant: q must be rank-2");
  const int64_t m = y.size(0), n = y.size(1) / 2;
  if (y.size(1) % 2 || q.size(0) != m || q.size(1) < n) {
    throw std::runtime_error("gemm_epilogue_glu_quant: y [M, 2N], q [M, >= N]");
  }
  const int scale_cols4 = static_cast<int>((((q.size(1) + 31) / 32) + 3) / 4);
  if (NumBytes(scales) < ((m + 127) / 128) * 128 * scale_cols4 * 4) {
    throw std::runtime_error("gemm_epilogue_glu_quant: scale storage too small (SWIZZLE_32_4_4)");
  }
  if (act != 0 && act != 1) throw std::runtime_error("gemm_epilogue_glu_quant: act must be 0 (silu) or 1 (gelu_tanh)");
  if (group <= 0 || n % group) throw std::runtime_error("gemm_epilogue_glu_quant: group must divide N");
  if (m * n == 0) return;
  const int64_t warps = m * ((n + 127) / 128);
  Launch(Blocks(warps * 32), Stream(y), GluQuantKernel, Data<uint8_t>(q), Data<uint8_t>(scales),
         ConstData<__nv_bfloat16>(y), m, n, q.size(1), static_cast<int>(act), group, share_128, scale_cols4);
}

void EpilogueQkNorm(TensorView qkv, TensorView q_weight, TensorView k_weight, int64_t q_heads, int64_t kv_heads,
                    int64_t head_dim, double eps, bool zero_centered) {
  using namespace xinfer::ffi;
  CheckBf16Matrix(qkv, "qkv");
  for (const TensorView* t : {&q_weight, &k_weight}) {
    CheckCuda(*t, "q/k weight");
    CheckBFloat16(*t, "q/k weight");
    CheckContiguous(*t, "q/k weight");
    if (NumElements(*t) != head_dim) throw std::runtime_error("gemm_epilogue_qk_norm: q/k weight must be [head_dim]");
  }
  if (q_heads <= 0 || kv_heads <= 0 || head_dim <= 0 || (q_heads + kv_heads) * head_dim > qkv.size(1)) {
    throw std::runtime_error("gemm_epilogue_qk_norm: bad head layout");
  }
  const int64_t blocks = qkv.size(0) * (q_heads + kv_heads);
  if (blocks == 0) return;
  Launch(dim3(static_cast<unsigned>(blocks)), Stream(qkv), QkNormKernel, Data<__nv_bfloat16>(qkv),
         ConstData<__nv_bfloat16>(q_weight), ConstData<__nv_bfloat16>(k_weight), qkv.size(1),
         static_cast<int>(q_heads), static_cast<int>(q_heads + kv_heads), static_cast<int>(head_dim),
         static_cast<float>(eps), zero_centered);
}

}  // namespace xinfer::gemm

TVM_FFI_DLL_EXPORT_TYPED_FUNC(gemm_epilogue_residual, xinfer::gemm::EpilogueResidual);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(gemm_epilogue_act, xinfer::gemm::EpilogueAct);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(gemm_epilogue_row_scale, xinfer::gemm::EpilogueRowScale);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(gemm_epilogue_glu, xinfer::gemm::EpilogueGlu);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(gemm_epilogue_rope, xinfer::gemm::EpilogueRope);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(gemm_epilogue_rope_kvcache, xinfer::gemm::EpilogueRopeKvcache);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(gemm_epilogue_glu_quant, xinfer::gemm::EpilogueGluQuant);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(gemm_epilogue_qk_norm, xinfer::gemm::EpilogueQkNorm);
