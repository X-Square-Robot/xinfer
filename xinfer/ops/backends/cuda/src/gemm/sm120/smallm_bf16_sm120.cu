// Small-M BF16 GEMM on [K, N] weights (backend smallm_bf16_sm120): out[M, N] = x[M, K] @ W[K, N] (+ bias), M <= 32.
//
// The weight read dominates (M <= 32), so the kernel spreads it over many CTAs and keeps the arithmetic on the
// tensor cores (mma.sync m16n8k16, FP32 accumulation):
//   * grid.x = N / BN column tiles, grid.y = split K ranges of Kc = K / split rows, grid.z = ceil(M / MT) row tiles
//     of MT = 16 or 32 rows; the split CTAs of a column tile form one thread-block cluster;
//   * each of the 4 warps of a CTA owns Kc / 4 consecutive k rows and copies its weight rows and its activation
//     columns into shared memory with cp.async in 32-row chunks (one commit group each, 16-byte weight chunks
//     XOR-swizzled by k row); it runs the MMAs of chunk c (ldmatrix.trans for B, ldmatrix for the activations) while
//     the later chunks are still in flight;
//   * each CTA sums its warps' FP32 accumulators in shared memory in warp order; CTA r of a cluster owns columns
//     [r, r + 1) * BN / split and adds the split CTAs' sums of them in split order, plus the bias, and stores BF16.
//     The sums travel through distributed shared memory: each CTA loads its slice from the peers' shared memory,
//     all loads issued before use (on SM120 the fastest of the distributed shared memory paths; remote stores and
//     bulk copies are slower). No global partials, counters or workspace.
// Rows are independent in every step and the summation order (MMA k steps, warps, splits) is fixed by (K, N); the
// column tile and rows per CTA may change with M but only regroup rows and columns between CTAs. So a row's result
// does not depend on M: results are bitwise the same for any batch of rows.
#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/optional.h>

#include <cstdint>
#include <stdexcept>
#include <string>

#include "common/tvm_ffi_cuda_tensor.h"

namespace xinfer::gemm::smallm_sm120 {
namespace {

constexpr int kWarps = 4;

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void cp_async16(void* dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(smem_u32(dst)), "l"(src));
}
__device__ __forceinline__ void ldsm_x4(uint32_t (&r)[4], const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(smem_u32(p)));
}
__device__ __forceinline__ void ldsm_x4_t(uint32_t (&r)[4], const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(smem_u32(p)));
}
__device__ __forceinline__ void mma16816(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void cp_async_wait_pending(int n) {
  switch (n) {  // wait_group takes an immediate
    case 0: asm volatile("cp.async.wait_group 0;\n" ::); break;
    case 1: asm volatile("cp.async.wait_group 1;\n" ::); break;
    case 2: asm volatile("cp.async.wait_group 2;\n" ::); break;
    case 3: asm volatile("cp.async.wait_group 3;\n" ::); break;
    case 4: asm volatile("cp.async.wait_group 4;\n" ::); break;
    case 5: asm volatile("cp.async.wait_group 5;\n" ::); break;
    case 6: asm volatile("cp.async.wait_group 6;\n" ::); break;
    default: asm volatile("cp.async.wait_group 7;\n" ::); break;
  }
}

constexpr int kChunk = 32;  // k rows per cp.async group of a warp

// MT rows per CTA (16 or 32), BN tile columns (32 or 64); KW = k rows per warp (multiple of 32); S = K split = cluster
// size.
template <int MT, int BN, int KW, int S>
__global__ void __launch_bounds__(kWarps * 32) smallm_kernel(const __nv_bfloat16* __restrict__ x,
                                                             const __nv_bfloat16* __restrict__ w,
                                                             const __nv_bfloat16* __restrict__ bias,
                                                             __nv_bfloat16* __restrict__ out, int M, int N, int K) {
  static_assert(S > 1, "split");
  constexpr int Kc = KW * kWarps;          // k rows of this CTA
  constexpr int XLD = Kc + 8;              // activation row stride (elements): +16 bytes breaks bank conflicts
  constexpr int CH = BN / 8;               // 16-byte chunks per weight row
  constexpr int MA = MT / 16, NA = BN / 8; // MMA atoms
  constexpr int NC = KW / kChunk;
  static_assert(KW % kChunk == 0, "chunks");
  extern __shared__ __align__(16) unsigned char smem_raw[];
  __nv_bfloat16* xs = reinterpret_cast<__nv_bfloat16*>(smem_raw);                 // [MT][XLD]
  __nv_bfloat16* wsm = xs + MT * XLD;                                             // [kWarps][KW][BN] swizzled
  // After the MMA loop the activation and weight buffers are reused (when large enough) for the
  // [kWarps][MT][RLD] warp sums (row stride BN + 8 floats: conflict-free accumulator stores).
  constexpr int RLD = BN + 8;
  constexpr bool kRedAliased = kWarps * MT * RLD * 4 <= MT * XLD * 2 + kWarps * KW * BN * 2;
  float* red = reinterpret_cast<float*>(kRedAliased ? smem_raw : reinterpret_cast<unsigned char*>(wsm + kWarps * KW * BN));
  const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
  const int n0 = blockIdx.x * BN, k0 = blockIdx.y * Kc;
  // this CTA's rows
  x += size_t(blockIdx.z) * MT * K;
  out += size_t(blockIdx.z) * MT * N;
  M = min(MT, M - int(blockIdx.z) * MT);

  // this warp's [KW, BN] weight rows (chunk (r, ch) stored at chunk ch ^ (r & 7)) and its KW activation columns of
  // all MT rows (rows >= M are zero), one commit group per 32 k rows
  __nv_bfloat16* wsw = wsm + warp * KW * BN;
  const __nv_bfloat16* wg = w + size_t(k0 + warp * KW) * N + n0;
  const int xc0 = warp * KW;
#pragma unroll
  for (int c = 0; c < NC; ++c) {
    for (int i = lane; i < kChunk * CH; i += 32) {
      const int r = c * kChunk + i / CH, ch = i % CH;
      cp_async16(wsw + r * BN + ((ch ^ (r & 7)) % CH) * 8, wg + size_t(r) * N + ch * 8);
    }
    for (int i = lane; i < MT * (kChunk / 8); i += 32) {
      const int m = i / (kChunk / 8), col = xc0 + c * kChunk + (i % (kChunk / 8)) * 8;
      if (m < M) {
        cp_async16(xs + m * XLD + col, x + size_t(m) * K + k0 + col);
      } else {
        *reinterpret_cast<uint4*>(xs + m * XLD + col) = make_uint4(0, 0, 0, 0);
      }
    }
    asm volatile("cp.async.commit_group;\n" ::);
  }

  float acc[MA][NA][4];
#pragma unroll
  for (int a = 0; a < MA; ++a)
#pragma unroll
    for (int b = 0; b < NA; ++b)
#pragma unroll
      for (int e = 0; e < 4; ++e) acc[a][b][e] = 0.f;
#pragma unroll
  for (int c = 0; c < NC; ++c) {
    cp_async_wait_pending(NC - 1 - c);  // chunk c of this thread landed
    __syncwarp();                       // ... and of the other lanes
#pragma unroll
    for (int kk = c * kChunk; kk < (c + 1) * kChunk; kk += 16) {
      uint32_t af[MA][4];
#pragma unroll
      for (int a = 0; a < MA; ++a)
        ldsm_x4(af[a], xs + (a * 16 + lane % 16) * XLD + xc0 + kk + (lane / 16) * 8);
#pragma unroll
      for (int b = 0; b < NA; b += 2) {
        // 16 k x 16 n: lanes 0-7 k 0-7 / n b*8, 8-15 k 8-15 / n b*8, 16-23 k 0-7 / n b*8+8, 24-31 k 8-15 / n b*8+8
        const int r = kk + (lane % 16), ch = b + lane / 16;
        uint32_t bf[4];
        ldsm_x4_t(bf, wsw + r * BN + ((ch ^ (r & 7)) % CH) * 8);
#pragma unroll
        for (int a = 0; a < MA; ++a) {
          mma16816(acc[a][b], af[a], bf[0], bf[1]);
          mma16816(acc[a][b + 1], af[a], bf[2], bf[3]);
        }
      }
    }
  }

  // warp accumulators to shared memory
  if constexpr (kRedAliased) __syncthreads();  // every warp is done reading its activations and weights
  float* rw = red + warp * MT * RLD;
#pragma unroll
  for (int a = 0; a < MA; ++a)
#pragma unroll
    for (int b = 0; b < NA; ++b) {
      const int row = a * 16 + lane / 4, col = b * 8 + (lane % 4) * 2;
      *reinterpret_cast<float2*>(rw + row * RLD + col) = make_float2(acc[a][b][0], acc[a][b][1]);
      *reinterpret_cast<float2*>(rw + (row + 8) * RLD + col) = make_float2(acc[a][b][2], acc[a][b][3]);
    }
  __syncthreads();
  // warp sum in warp order, in place into warp 0's slot (each element is read and written by one thread)
  for (int i = tid * 4; i < M * BN; i += kWarps * 32 * 4) {
    const int e = (i / BN) * RLD + i % BN;
    float4 v = *reinterpret_cast<const float4*>(red + e);
#pragma unroll
    for (int q = 1; q < kWarps; ++q) {
      const float4 u = *reinterpret_cast<const float4*>(red + q * MT * RLD + e);
      v.x += u.x, v.y += u.y, v.z += u.z, v.w += u.w;
    }
    *reinterpret_cast<float4*>(red + e) = v;
  }
  // this CTA's output columns [c0, c0 + CW): the S split sums of them, added in split order
  constexpr int CW = BN / S;
  static_assert(CW % 4 == 0, "float4 columns");
  constexpr int NIT = (MT * CW + kWarps * 32 * 4 - 1) / (kWarps * 32 * 4);
  const int c0 = int(blockIdx.y) * CW;
  float4 v[NIT];
  {
    // every split CTA's sums are written (cluster-scope release / acquire)
    asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;\n" ::: "memory");
    namespace cg = cooperative_groups;
    cg::cluster_group cluster = cg::this_cluster();
    float4 u[NIT][S];
#pragma unroll
    for (int it = 0; it < NIT; ++it) {
      const int i = (it * kWarps * 32 + tid) * 4, m = i / CW, cl = i % CW;
      if (m < M) {
#pragma unroll
        for (int q = 0; q < S; ++q) {  // all loads issued before use
          u[it][q] = *reinterpret_cast<const float4*>(cluster.map_shared_rank(red + m * RLD + c0 + cl, q));
        }
      }
    }
#pragma unroll
    for (int it = 0; it < NIT; ++it) {
      v[it] = u[it][0];
#pragma unroll
      for (int q = 1; q < S; ++q) {  // split order
        v[it].x += u[it][q].x, v[it].y += u[it][q].y, v[it].z += u[it][q].z, v[it].w += u[it][q].w;
      }
    }
    // done reading the peers' shared memory (the values are consumed): relaxed, so it does not wait for this CTA's
    // output stores
    asm volatile("barrier.cluster.arrive.relaxed.aligned;\n" ::: "memory");
  }
#pragma unroll
  for (int it = 0; it < NIT; ++it) {
    const int i = (it * kWarps * 32 + tid) * 4, m = i / CW, col = c0 + i % CW;
    if (m >= M) continue;
    if (bias != nullptr) {
      const float2 b01 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(bias + n0 + col));
      const float2 b23 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(bias + n0 + col + 2));
      v[it].x += b01.x, v[it].y += b01.y, v[it].z += b23.x, v[it].w += b23.y;
    }
    __nv_bfloat162 lo = __floats2bfloat162_rn(v[it].x, v[it].y), hi = __floats2bfloat162_rn(v[it].z, v[it].w);
    uint2 o;
    o.x = *reinterpret_cast<uint32_t*>(&lo);
    o.y = *reinterpret_cast<uint32_t*>(&hi);
    *reinterpret_cast<uint2*>(out + size_t(m) * N + n0 + col) = o;
  }
  asm volatile("barrier.cluster.wait.aligned;\n" ::: "memory");  // peers done reading
}

struct Config {
  int bn, split, mt;  // column tile, K split (k rows per warp = K / split / 4), rows per CTA
};

// The K split (it fixes the summation order) depends on (K, N) only, never on M. The column tile and rows per CTA
// may depend on M: they only regroup rows / columns between CTAs. Swept on an RTX 5090 (2026-11-03, CUDA graph,
// weights in L2) over BN {16, 32, 64} x MT {16, 32} at M 1-32 for (K, N) = (1024, 2560 / 1024 / 2048 / 4096),
// (2048, 1024).
Config ConfigFor(int64_t k, int64_t n, int64_t m) {
  if (n % 32 != 0) return {0, 0, 0};
  const int split = k >= 2048 ? 4 : 2;
  int bn = 32;
  if (n % 64 == 0) {
    if (n <= 1024) {
      bn = m > (k >= 2048 ? 16 : 20) ? 64 : 32;
    } else {
      bn = n == 2048 && m == 1 ? 32 : 64;
    }
  }
  const int mt = m > 16 && n >= 2560 ? 32 : 16;
  return {bn, split, mt};
}

int KwOf(int64_t k, const Config& cf) { return int(k / cf.split / kWarps); }

// Shared memory of smallm_kernel<MT, BN, KW, ...>: activations, weights, warp sums (aliasing both if they fit).
size_t SmemBytes(int mt, int bn, int kw) {
  const size_t xbytes = size_t(mt) * (kw * kWarps + 8) * 2, wbytes = size_t(kWarps) * kw * bn * 2,
               rbytes = size_t(kWarps) * mt * (bn + 8) * 4;
  return rbytes <= xbytes + wbytes ? xbytes + wbytes : xbytes + wbytes + rbytes;
}

template <int MT, int BN, int KW, int S>
void Launch(const __nv_bfloat16* x, const __nv_bfloat16* w, const __nv_bfloat16* bias, __nv_bfloat16* out, int m,
            int n, int k, cudaStream_t st) {
  const size_t smem = SmemBytes(MT, BN, KW);
  auto kernel = smallm_kernel<MT, BN, KW, S>;
  cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, int(smem));
  cudaLaunchConfig_t lc = {};
  lc.gridDim = dim3(n / BN, S, (m + MT - 1) / MT);
  lc.blockDim = dim3(kWarps * 32);
  lc.dynamicSmemBytes = smem;
  lc.stream = st;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeClusterDimension;
  attr[0].val.clusterDim.x = 1;
  attr[0].val.clusterDim.y = S;
  attr[0].val.clusterDim.z = 1;
  lc.attrs = attr;
  lc.numAttrs = 1;
  cudaLaunchKernelEx(&lc, kernel, x, w, bias, out, m, n, k);
}

template <int MT, int BN, int KW>
void LaunchS(const Config& cf, const __nv_bfloat16* x, const __nv_bfloat16* w, const __nv_bfloat16* bias,
             __nv_bfloat16* out, int m, int n, int k, cudaStream_t st) {
  if (cf.split == 2) return Launch<MT, BN, KW, 2>(x, w, bias, out, m, n, k, st);
  return Launch<MT, BN, KW, 4>(x, w, bias, out, m, n, k, st);
}

template <int BN, int KW>
void LaunchRows(const Config& cf, const __nv_bfloat16* x, const __nv_bfloat16* w, const __nv_bfloat16* bias,
                __nv_bfloat16* out, int m, int n, int k, cudaStream_t st) {
  // row tiles side by side (grid.z); a row's arithmetic does not depend on MT
  if (cf.mt == 32) return LaunchS<32, BN, KW>(cf, x, w, bias, out, m, n, k, st);
  return LaunchS<16, BN, KW>(cf, x, w, bias, out, m, n, k, st);
}

template <int BN>
bool LaunchKw(int kw, const Config& cf, const __nv_bfloat16* x, const __nv_bfloat16* w, const __nv_bfloat16* bias,
              __nv_bfloat16* out, int m, int n, int k, cudaStream_t st) {
  switch (kw) {
    case 32: LaunchRows<BN, 32>(cf, x, w, bias, out, m, n, k, st); return true;
    case 64: LaunchRows<BN, 64>(cf, x, w, bias, out, m, n, k, st); return true;
    case 128: LaunchRows<BN, 128>(cf, x, w, bias, out, m, n, k, st); return true;
    case 256: LaunchRows<BN, 256>(cf, x, w, bias, out, m, n, k, st); return true;
    default: return false;
  }
}

}  // namespace

bool Supports(int64_t m, int64_t n, int64_t k) {
  const Config cf = ConfigFor(k, n, m);
  if (cf.bn == 0 || (cf.split != 2 && cf.split != 4) || (cf.mt != 16 && cf.mt != 32) || m < 1 || m > 32 ||
      k % (cf.split * kWarps) != 0) {
    return false;
  }
  const int kw = KwOf(k, cf);
  if (kw != 32 && kw != 64 && kw != 128 && kw != 256) return false;
  return SmemBytes(cf.mt, cf.bn, kw) <= 99 * 1024;
}

void Run(tvm::ffi::TensorView out, tvm::ffi::TensorView x, tvm::ffi::TensorView w,
         tvm::ffi::Optional<tvm::ffi::TensorView> bias) {
  using namespace xinfer::ffi;
  RequireExactComputeCapability(x, 12, 0, "smallm_bf16_sm120");
  for (auto* t : {&out, &x, &w}) {
    CheckCuda(*t, "tensor");
    CheckBFloat16(*t, "tensor");
    CheckContiguous(*t, "tensor");
  }
  if (x.ndim() != 2 || w.ndim() != 2 || out.ndim() != 2) {
    throw std::runtime_error("smallm_bf16_sm120: x [M, K], w [K, N], out [M, N] must be 2-D");
  }
  const int64_t m = x.size(0), k = x.size(1), n = w.size(1);
  if (w.size(0) != k || out.size(0) != m || out.size(1) != n) {
    throw std::runtime_error("smallm_bf16_sm120: x [M, K], w [K, N], out [M, N]");
  }
  if (!Supports(m, n, k)) {
    throw std::runtime_error("smallm_bf16_sm120: needs 1 <= M <= 32 and a configured (K, N); got M=" +
                             std::to_string(m) + " N=" + std::to_string(n) + " K=" + std::to_string(k));
  }
  const __nv_bfloat16* b = nullptr;
  if (bias.has_value()) {
    CheckCuda(bias.value(), "bias");
    CheckBFloat16(bias.value(), "bias");
    CheckContiguous(bias.value(), "bias");
    if (NumElements(bias.value()) != n) throw std::runtime_error("smallm_bf16_sm120: bias must have N elements");
    b = static_cast<const __nv_bfloat16*>(bias.value().data_ptr());
  }
  const Config cf = ConfigFor(k, n, m);
  auto* xp = static_cast<const __nv_bfloat16*>(x.data_ptr());
  auto* wp = static_cast<const __nv_bfloat16*>(w.data_ptr());
  auto* op = static_cast<__nv_bfloat16*>(out.data_ptr());
  cudaStream_t st = Stream(x);
  const int kw = KwOf(k, cf);
  if (cf.bn == 64) {
    LaunchKw<64>(kw, cf, xp, wp, b, op, int(m), int(n), int(k), st);
  } else {
    LaunchKw<32>(kw, cf, xp, wp, b, op, int(m), int(n), int(k), st);
  }
  const cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) throw std::runtime_error(std::string("smallm_bf16_sm120: ") + cudaGetErrorString(e));
}

}  // namespace xinfer::gemm::smallm_sm120

TVM_FFI_DLL_EXPORT_TYPED_FUNC(smallm_bf16_sm120, xinfer::gemm::smallm_sm120::Run);
