// Data-parallel BF16 GEMM + bias + exact GELU on [N, K] weights (backend tma_tile_bf16_sm120):
// out[M, N] = gelu_erf(bf16(x[M, K] @ W[N, K]^T + bias)), rounded to BF16.
//
// One 80 x 128 output tile per CTA (grid = ceil(M / 80) * N / 128), 8 MMA warps (1 x 8, each 80 x 16) + 1 TMA
// producer warp, BK = 32, 4 stages, 64B swizzle, mma.sync m16n8k16 with FP32 accumulation. The epilogue rounds
// acc + bias to BF16 first, applies GELU(erf) in FP32 and rounds again (the same two roundings as a GEMM with a
// BF16 output followed by a separate GELU), writes the tile into shared memory (128B swizzle) and stores it with
// TMA. CTAs never wait for each other: no workspace, no flags, run-to-run deterministic, and a row's result does
// not depend on M.
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <tvm/ffi/function.h>

#include <cstdint>
#include <stdexcept>
#include <string>

#include "common/tvm_ffi_cuda_tensor.h"
#include "common/tensor_map.h"

namespace xinfer::gemm::tma_tile_sm120 {
namespace {

constexpr int BM = 80, BN = 128, BK = 32, KK = BK / 16, STAGES = 4;
constexpr int MMA_WARPS = 8, WM = 1, WN = MMA_WARPS / WM, WTM = BM / WM, WTN = BN / WN;
constexpr int NB = BN / 64;  // TMA boxes of the output tile (64 columns = one 128B row each)
constexpr int MI = WTM / 16, NI = WTN / 8;
static_assert(WTM % 16 == 0 && NI == 2, "layout");
constexpr int THREADS = (MMA_WARPS + 1) * 32;
constexpr int A_STAGE = BM * BK * 2, B_STAGE = BN * BK * 2;  // 64B swizzle
static_assert(A_STAGE % 512 == 0 && B_STAGE % 512 == 0, "swizzle alignment");
constexpr int STAGE_BYTES = A_STAGE + B_STAGE;
constexpr int OUT_BYTES = BM * BN * 2;  // output tile: [box][BM rows][128B], 128B swizzle
static_assert(OUT_BYTES % 1024 == 0, "out tile alignment");

__device__ __forceinline__ uint32_t smem_u32(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }
__device__ __forceinline__ void mbar_init(uint64_t* b, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_u32(b)), "r"(count));
}
__device__ __forceinline__ void mbar_expect_tx(uint64_t* b, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem_u32(b)), "r"(bytes));
}
__device__ __forceinline__ void mbar_arrive(uint64_t* b) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(smem_u32(b))); }
__device__ __forceinline__ void mbar_wait(uint64_t* b, uint32_t parity) {
  asm volatile(
      "{\n .reg .pred p;\n"
      "WAIT_%=:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
      " @!p bra WAIT_%=;\n}\n" ::"r"(smem_u32(b)),
      "r"(parity));
}
__device__ __forceinline__ void tma_load_2d(void* dst, const CUtensorMap* map, uint64_t* bar, int x, int y) {
  // .shared::cta: on SM120 the .shared::cluster form adds a driver-call fallback that raises the context stack limit.
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];" ::"r"(
          smem_u32(dst)),
      "l"(map), "r"(smem_u32(bar)), "r"(x), "r"(y)
      : "memory");
}
__device__ __forceinline__ void ldsm_x4(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"(addr));
}
__device__ __forceinline__ void mma16816(float* c, const uint32_t* a, const uint32_t* b) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

struct FragK {
  uint32_t a[MI][4];
  uint32_t b[NI][2];
};
struct Smem {
  alignas(1024) uint8_t a[STAGES][A_STAGE];
  alignas(1024) uint8_t b[STAGES][B_STAGE];
  alignas(1024) uint8_t out[OUT_BYTES];  // result tile, stored with TMA
  alignas(8) uint64_t full[STAGES];
  alignas(8) uint64_t empty[STAGES];
};
struct Params {
  const __nv_bfloat16* bias;
  int tiles_n, kt;
};

__global__ void __launch_bounds__(THREADS, 1)
    tma_tile_kernel(const __grid_constant__ CUtensorMap mapA, const __grid_constant__ CUtensorMap mapB,
                    const __grid_constant__ CUtensorMap mapD, const Params p) {
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  Smem& s = *reinterpret_cast<Smem*>((reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t(1023));
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int tm = blockIdx.x / p.tiles_n, tn = blockIdx.x % p.tiles_n;
  if (threadIdx.x == 0) {
    for (int i = 0; i < STAGES; ++i) {
      mbar_init(&s.full[i], 1);
      mbar_init(&s.empty[i], MMA_WARPS);
    }
    asm volatile("fence.mbarrier_init.release.cluster;");
  }
  __syncthreads();

  if (warp == MMA_WARPS) {  // TMA producer
    if (lane == 0) {
      asm volatile("prefetch.tensormap [%0];" ::"l"(&mapA));
      asm volatile("prefetch.tensormap [%0];" ::"l"(&mapB));
      for (int gp = 0; gp < p.kt; ++gp) {
        const int st = gp % STAGES;
        if (gp >= STAGES) mbar_wait(&s.empty[st], ((gp / STAGES) - 1) & 1);
        mbar_expect_tx(&s.full[st], STAGE_BYTES);
        tma_load_2d(s.a[st], &mapA, &s.full[st], gp * BK, tm * BM);
        tma_load_2d(s.b[st], &mapB, &s.full[st], gp * BK, tn * BN);
      }
    }
    return;
  }

  const int wn = warp % WN, tid = threadIdx.x;
  float acc[MI][NI][4];
#pragma unroll
  for (int mi = 0; mi < MI; ++mi)
#pragma unroll
    for (int ni = 0; ni < NI; ++ni)
#pragma unroll
      for (int r = 0; r < 4; ++r) acc[mi][ni][r] = 0.f;
  uint32_t bia[NI];  // issued now, used by the epilogue
#pragma unroll
  for (int ni = 0; ni < NI; ++ni) {
    const int col = tn * BN + wn * WTN + ni * 8 + (lane & 3) * 2;
    bia[ni] = __ldg(reinterpret_cast<const unsigned int*>(&p.bias[col]));
  }
  const int rowA = lane & 15, rowB = wn * WTN + (lane & 7) + ((lane >> 4) << 3);
  uint32_t xa[KK], xb[KK];
#pragma unroll
  for (int kk = 0; kk < KK; ++kk) {
    xa[kk] = rowA * 64 + (((kk * 2 + (lane >> 4)) ^ ((rowA >> 1) & 3)) << 4);
    xb[kk] = rowB * 64 + (((kk * 2 + ((lane >> 3) & 1)) ^ ((rowB >> 1) & 3)) << 4);
  }
  const uint32_t sa0 = smem_u32(s.a[0]), sb0 = smem_u32(s.b[0]);
  FragK f0, f1;
  auto load_k = [&](FragK& f, int st, int kk) {
    const uint32_t a = sa0 + st * A_STAGE + xa[kk], b = sb0 + st * B_STAGE + xb[kk];
#pragma unroll
    for (int mi = 0; mi < MI; ++mi) ldsm_x4(a + mi * 16 * 64, f.a[mi][0], f.a[mi][1], f.a[mi][2], f.a[mi][3]);
    ldsm_x4(b, f.b[0][0], f.b[0][1], f.b[1][0], f.b[1][1]);
  };
  mbar_wait(&s.full[0], 0);
  load_k(f0, 0, 0);
#pragma unroll 1
  for (int g = 0; g < p.kt; ++g) {
    const int st = g % STAGES;
    const bool has_next = g + 1 < p.kt;
#pragma unroll
    for (int kk = 0; kk < KK; ++kk) {
      FragK& cur = (kk & 1) ? f1 : f0;
      FragK& nxt = (kk & 1) ? f0 : f1;
      if (kk + 1 < KK) {
        load_k(nxt, st, kk + 1);
      } else if (has_next) {
        const int st2 = (g + 1) % STAGES;
        mbar_wait(&s.full[st2], ((g + 1) / STAGES) & 1);
        load_k(nxt, st2, 0);
      }
#pragma unroll
      for (int mi = 0; mi < MI; ++mi)
#pragma unroll
        for (int ni = 0; ni < NI; ++ni) mma16816(acc[mi][ni], cur.a[mi], cur.b[ni]);
    }
    if (lane == 0) mbar_arrive(&s.empty[st]);
  }
  // epilogue: write the tile into shared memory (128B swizzle: 16B chunk ^ (row & 7)), then one TMA store
#pragma unroll
  for (int mi = 0; mi < MI; ++mi)
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int r = mi * 16 + h * 8 + (lane >> 2);
#pragma unroll
      for (int ni = 0; ni < NI; ++ni) {
        const int c = wn * WTN + ni * 8 + (lane & 3) * 2;  // column within the tile
        const int cc = c & 63;
        const int off = (c >> 6) * (BM * 128) + r * 128 + ((((cc * 2) >> 4) ^ (r & 7)) << 4) + ((cc * 2) & 15);
        const __nv_bfloat162 bb = *reinterpret_cast<const __nv_bfloat162*>(&bia[ni]);
        const float y0 = __bfloat162float(__float2bfloat16(acc[mi][ni][2 * h] + __bfloat162float(bb.x)));
        const float y1 = __bfloat162float(__float2bfloat16(acc[mi][ni][2 * h + 1] + __bfloat162float(bb.y)));
        const float z0 = 0.5f * y0 * (1.f + erff(y0 * 0.70710678118654752f));
        const float z1 = 0.5f * y1 * (1.f + erff(y1 * 0.70710678118654752f));
        *reinterpret_cast<__nv_bfloat162*>(s.out + off) = __floats2bfloat162_rn(z0, z1);
      }
    }
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  asm volatile("bar.sync 1, %0;" ::"n"(MMA_WARPS * 32));
  if (tid == 0) {
    for (int bx = 0; bx < NB; ++bx)
      asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%1, %2}], [%3];" ::"l"(&mapD),
                   "r"(tn * BN + bx * 64), "r"(tm * BM), "r"(smem_u32(s.out + bx * BM * 128))
                   : "memory");
    asm volatile("cp.async.bulk.commit_group;" ::: "memory");
    asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
  }
}


CUtensorMap Map2d(const void* ptr, int64_t rows, int64_t cols, int box_cols, int box_rows, CUtensorMapSwizzle sw,
                  CUtensorMapL2promotion promo) {
  CUtensorMap m;
  cuuint64_t dims[2] = {(cuuint64_t)cols, (cuuint64_t)rows};
  cuuint64_t strides[1] = {(cuuint64_t)cols * 2};
  cuuint32_t box[2] = {(cuuint32_t)box_cols, (cuuint32_t)box_rows};
  cuuint32_t elem[2] = {1, 1};
  if (xinfer::ffi::TensorMapEncodeTiled()(&m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<void*>(ptr), dims, strides, box, elem,
                CU_TENSOR_MAP_INTERLEAVE_NONE, sw, promo, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) != CUDA_SUCCESS) {
    throw std::runtime_error("tma_tile_bf16_sm120: cuTensorMapEncodeTiled failed");
  }
  return m;
}

}  // namespace

bool Supports(int64_t m, int64_t n, int64_t k) {
  return m >= 1 && n >= BN && k >= BK && n % BN == 0 && k % BK == 0 && m <= (int64_t(1) << 24) && n <= 65536 &&
         k <= 65536 && ((m + BM - 1) / BM) * (n / BN) < (int64_t(1) << 31);
}

void Run(tvm::ffi::TensorView out, tvm::ffi::TensorView x, tvm::ffi::TensorView w, tvm::ffi::TensorView bias) {
  using namespace xinfer::ffi;
  const char* op = "tma_tile_bf16_sm120";
  RequireArchitecture(x, 12, 0, op);
  for (auto* t : {&out, &x, &w, &bias}) {
    CheckCuda(*t, op);
    CheckBFloat16(*t, op);
    CheckContiguous(*t, op);
  }
  if (x.ndim() != 2 || w.ndim() != 2 || out.ndim() != 2) throw std::runtime_error("tma_tile_bf16_sm120: rank-2 only");
  const int64_t m = x.size(0), k = x.size(1), n = w.size(0);
  if (w.size(1) != k || out.size(0) != m || out.size(1) != n) {
    throw std::runtime_error("tma_tile_bf16_sm120: expected x [M, K], w [N, K], out [M, N]");
  }
  if (NumElements(bias) != n) throw std::runtime_error("tma_tile_bf16_sm120: bias must have N elements");
  if (!Supports(m, n, k)) {
    throw std::runtime_error("tma_tile_bf16_sm120: needs N % 128 == 0 and K % 32 == 0 (M=" + std::to_string(m) +
                             " N=" + std::to_string(n) + " K=" + std::to_string(k) + ")");
  }
  for (auto* t : {&x, &w, &out, &bias}) {
    if (reinterpret_cast<uintptr_t>(t->data_ptr()) % 16) throw std::runtime_error("tma_tile_bf16_sm120: 16-byte aligned tensors only");
  }
  Params p{};
  p.bias = static_cast<const __nv_bfloat16*>(bias.data_ptr());
  p.tiles_n = int(n / BN), p.kt = int(k / BK);
  const CUtensorMap mA = Map2d(x.data_ptr(), m, k, BK, BM, CU_TENSOR_MAP_SWIZZLE_64B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B);
  const CUtensorMap mB = Map2d(w.data_ptr(), n, k, BK, BN, CU_TENSOR_MAP_SWIZZLE_64B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B);
  const CUtensorMap mD = Map2d(out.data_ptr(), m, n, 64, BM, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE);
  const int smem = int(sizeof(Smem)) + 1024;
  const int device = x.device().device_id;
  // Stream() also makes x's device current: the function attribute below is per device.
  const cudaStream_t stream = Stream(x);
  static bool attr[64] = {};
  if (device < 0 || device >= 64) throw std::runtime_error("tma_tile_bf16_sm120: device index");
  if (!attr[device]) {
    CheckCudaError(cudaFuncSetAttribute(tma_tile_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem), "cudaFuncSetAttribute");
    attr[device] = true;
  }
  const int tiles = int(((m + BM - 1) / BM) * (n / BN));
  tma_tile_kernel<<<tiles, THREADS, smem, stream>>>(mA, mB, mD, p);
  CheckLastCudaError(op);
}

}  // namespace xinfer::gemm::tma_tile_sm120

TVM_FFI_DLL_EXPORT_TYPED_FUNC(tma_tile_bf16_sm120, xinfer::gemm::tma_tile_sm120::Run);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(tma_tile_bf16_sm120_supports, xinfer::gemm::tma_tile_sm120::Supports);
