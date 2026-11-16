// Persistent gate/up BF16 GEMM + SiLU * up on raw [2N, K] weights (gemm_glu backend tma_streamk_bf16_sm120):
// out[M, N] = bf16(silu(bf16(x @ Wg^T)) * bf16(x @ Wu^T)), W = [Wg; Wu] (gate rows first).
//
// One CTA per SM (G = SM count), 8 MMA warps (2 x 4, 80 rows x 16 output columns each) + 1 TMA producer warp. Tiles of
// 160 rows x 64 output columns (one B stage = 64 gate rows + 64 up rows), BK = 32, 4 stages. Scheduling: R = tiles / G - 1
// data-parallel rounds (round r: CTA c runs tile r * G + c, M-major tile order), then the k-steps of the remaining
// G..2G-1 tiles are cut into G equal ranges (stream-K: every CTA covers >= K/32 iterations, so a tile spans at most
// two CTAs). Cross-CTA merge of a split tile (bounded wait + wait-free fallback):
//   * the tail half (first segment of CTA c+1) writes its FP32 partial sum (workspace) and adds 1 to the per-warp flag;
//   * the head half (last segment of CTA c) waits up to XC_WAIT_NS, then adds head + tail and runs the epilogue;
//   * fallback (the tail CTA cannot get an SM): the head publishes its sum and adds 2; whichever side sees both halves
//     merges (head + tail in that order) and stores. Both orders of arrival give the same sum: bitwise deterministic.
// Epilogue: SiLU * up in registers (gate / up rounded to BF16 first, as the generic path's BF16 GEMM output), written
// per warp to shared memory and stored with TMA.
// Workspace: FP32 partials (2 x stream-K tiles x 8 warps x 8 KB); flags: stream-K tiles x 8 int32, zeroed once per
// (device, stream) and left zeroed by the kernel.
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <tvm/ffi/function.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "common/tvm_ffi_cuda_tensor.h"
#include "common/tensor_map.h"

#ifndef XC_WAIT_NS
#define XC_WAIT_NS 20000  // bounded cross-CTA wait (ns)
#endif
#define PP_STAGES 4

namespace xinfer::gemm::tma_streamk_glu_sm120 {
namespace {

constexpr int BM = 160, BNO = 64, BN = 2 * BNO, BK = 32, KK = BK / 16, STAGES = PP_STAGES;
constexpr int MMA_WARPS = 8, WM = 2, WN = 4;
constexpr int WTM = BM / WM, MI = WTM / 16, NI = 4;  // 80 rows per warp; n8 blocks 0, 1 = gate, 2, 3 = up
constexpr int THREADS = (MMA_WARPS + 1) * 32;
constexpr int A_STAGE = BM * BK * 2, B_STAGE = BN * BK * 2;
static_assert(A_STAGE % 1024 == 0 && B_STAGE % 1024 == 0, "alignment");
constexpr int STAGE_BYTES = A_STAGE + B_STAGE;
constexpr int OUT_BYTES = BM * BNO * 2;
constexpr int PART4 = MI * NI;  // partial-sum float4s per thread

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
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];" ::"r"(
          smem_u32(dst)),
      "l"(map), "r"(smem_u32(bar)), "r"(x), "r"(y)
      : "memory");
}
__device__ __forceinline__ void tma_store_2d(const CUtensorMap* map, const void* src, int x, int y) {
  asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%2, %3}], [%1];" ::"l"(map), "r"(smem_u32(src)), "r"(x), "r"(y)
               : "memory");
  asm volatile("cp.async.bulk.commit_group;" ::: "memory");
}
__device__ __forceinline__ void ldsm_x4(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
__device__ __forceinline__ void mma16816(float* c, const uint32_t* a, const uint32_t* b) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ int ld_acquire(const int* a) {
  int v;
  asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(v) : "l"(a) : "memory");
  return v;
}
// Relaxed atomics (the caller runs __threadfence() first: fence + relaxed write = release); an acq_rel atomic compiles
// to ATOMG plus a CCTL.IVALL that waits for its return on SM120. Relaxed load: only decides whether to merge (an acquire
// load right after a TMA store adds ~2.7 us to each CTA's end); a merge fences first (relaxed load + fence = acquire).
__device__ __forceinline__ int ld_relaxed(const int* a) {
  int v;
  asm volatile("ld.relaxed.gpu.global.b32 %0, [%1];" : "=r"(v) : "l"(a) : "memory");
  return v;
}
// Bounded wait: lane 0 polls the counter until (value & need) == need or XC_WAIT_NS ns pass. Normally (and under light
// preemption) the partner arrives within the window; if it cannot get an SM the caller takes the wait-free fallback, so correctness never depends on co-residency.
__device__ __forceinline__ unsigned long long gtimer() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t));
  return t;
}
__device__ __forceinline__ int wait_bits(const int* f, int need) {
  int v = ld_acquire(f);
  if ((v & need) != need) {
    const unsigned long long t0 = gtimer();
    do {
      __nanosleep(64);
      v = ld_acquire(f);
    } while ((v & need) != need && gtimer() - t0 < XC_WAIT_NS);
  }
  return v;
}
__device__ __forceinline__ void red_add_relaxed(int* a, int v) { asm volatile("red.relaxed.gpu.global.add.s32 [%0], %1;" ::"l"(a), "r"(v) : "memory"); }
__device__ __forceinline__ int atom_add_acq_rel(int* a, int v) {
  int old;
  asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], %2;" : "=r"(old) : "l"(a), "r"(v) : "memory");
  return old;
}
__device__ __forceinline__ float rb(float v) { return __bfloat162float(__float2bfloat16(v)); }
// x / (1 + exp(-x)) * up, the division through the MUFU reciprocal (<= 2 FP32 ulp from IEEE division, invisible after BF16 rounding)
__device__ __forceinline__ float silu_mul(float g, float u) { return __fdividef(g, 1.f + __expf(-g)) * u; }

struct FragK {
  uint32_t a[MI][4];
  uint32_t b[NI][2];
};
struct Smem {
  alignas(1024) uint8_t a[STAGES][A_STAGE];
  alignas(1024) uint8_t b[STAGES][B_STAGE];
  alignas(1024) uint8_t out[OUT_BYTES];  // output tile [160][64] BF16 (per-warp 80 x 16 blocks, 32B swizzle, TMA store)
  alignas(8) uint64_t full[STAGES];
  alignas(8) uint64_t empty[STAGES];
};
struct Params {
  float4* part;  // [0 tail half / 1 head half][stream-K tile][warp][PART4][32] (the head half only on the fallback)
  int* flags;    // [stream-K tile][warp]: tail half +1, head half (fallback) +2; whoever merges resets it to 0
  int M, F, K, tiles_m, tiles, kt, G;
  int R, tail0, T;  // data-parallel rounds, first stream-K tile, stream-K iterations
  int ntail;        // stream-K tiles (offset of the head-half partial slots)
};
// Segment of this CTA's v-th iteration: tile t (stream-K index lt, -1 in the data-parallel rounds), k range [kb, ke)
__device__ __forceinline__ void seg_of(const Params& p, int it0, int it1, int v, int& t, int& lt, int& kb, int& ke) {
  const int VA = p.R * p.kt;
  if (v < VA) {
    t = (v / p.kt) * p.G + blockIdx.x, lt = -1, kb = 0, ke = p.kt;
  } else {
    const int it = it0 + (v - VA);
    lt = it / p.kt, kb = it - lt * p.kt, t = p.tail0 + lt;
    ke = min((lt + 1) * p.kt, it1) - lt * p.kt;
  }
}

__global__ void __launch_bounds__(THREADS, 1) tma_streamk_glu_kernel(const __grid_constant__ CUtensorMap mapA, const __grid_constant__ CUtensorMap mapB,
                                                         const __grid_constant__ CUtensorMap mapD, const Params p) {
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  Smem& s = *reinterpret_cast<Smem*>((reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t(1023));
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int it0 = (int)((long long)blockIdx.x * p.T / p.G), it1 = (int)((long long)(blockIdx.x + 1) * p.T / p.G);
  const int total = p.R * p.kt + (it1 - it0);  // iterations of this CTA
  if (threadIdx.x == 0) {
    for (int i = 0; i < STAGES; ++i) mbar_init(&s.full[i], 1), mbar_init(&s.empty[i], MMA_WARPS);
    asm volatile("fence.mbarrier_init.release.cluster;");
  }
  __syncthreads();
  if (warp == MMA_WARPS) {  // TMA producer: one A box, two B boxes (gate / up)
    if (lane == 0) {
      asm volatile("prefetch.tensormap [%0];" ::"l"(&mapA));
      asm volatile("prefetch.tensormap [%0];" ::"l"(&mapB));
      int g = 0;
      for (int v = 0; v < total;) {
        int t, lt, kb, ke;
        seg_of(p, it0, it1, v, t, lt, kb, ke);
        const int tm = t % p.tiles_m, tn = t / p.tiles_m;
        for (; kb < ke; ++kb, ++g, ++v) {
          const int st = g % STAGES;
          if (g >= STAGES) mbar_wait(&s.empty[st], ((g / STAGES) - 1) & 1);
          mbar_expect_tx(&s.full[st], STAGE_BYTES);
          tma_load_2d(s.a[st], &mapA, &s.full[st], kb * BK, tm * BM);
          tma_load_2d(s.b[st], &mapB, &s.full[st], kb * BK, tn * BNO);
          tma_load_2d(s.b[st] + BNO * BK * 2, &mapB, &s.full[st], kb * BK, p.F + tn * BNO);
        }
      }
    }
    return;
  }
  const int wm = warp / WN, wn = warp % WN;
  float acc[MI][NI][4];
  int g = 0;
  // 64B swizzle (16B chunk ^ (row>>1)&3); A row = wm*80 + mi*16 + (lane&15); B: gate row wn*16 + (lane&7) + 8*(lane>>4), up row = gate row + 64
  const int rowA = wm * WTM + (lane & 15), rowB = wn * 16 + (lane & 7) + ((lane >> 4) << 3);
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
    ldsm_x4(b + BNO * 64, f.b[2][0], f.b[2][1], f.b[3][0], f.b[3][1]);
  };
  auto step = [&](bool has_next) {
    const int st = g % STAGES;
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
    ++g;
  };
  const uint32_t so = smem_u32(s.out);
  // per-warp output block [80][16] BF16 (32B swizzle: 16B chunk ^ (row>>2)&1), overwritten after the previous store has been read
  auto epilogue = [&](int tm, int tn) {
      const uint32_t wo = so + warp * (WTM * 32);
      if (lane == 0) asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
      __syncwarp();
#pragma unroll
      for (int mi = 0; mi < MI; ++mi)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int r = mi * 16 + h * 8 + (lane >> 2);
#pragma unroll
          for (int j = 0; j < 2; ++j) {
            const float y0 = silu_mul(rb(acc[mi][j][2 * h]), rb(acc[mi][2 + j][2 * h]));
            const float y1 = silu_mul(rb(acc[mi][j][2 * h + 1]), rb(acc[mi][2 + j][2 * h + 1]));
            const __nv_bfloat162 o = __floats2bfloat162_rn(y0, y1);
            const uint32_t addr = wo + r * 32 + ((j ^ ((r >> 2) & 1)) << 4) + (lane & 3) * 4;
            asm volatile("st.shared.b32 [%0], %1;" ::"r"(addr), "r"(*reinterpret_cast<const uint32_t*>(&o)) : "memory");
          }
        }
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
      __syncwarp();
      if (lane == 0) tma_store_2d(&mapD, s.out + warp * (WTM * 32), tn * BNO + wn * 16, tm * BM + wm * WTM);
  };

  for (int v = 0; v < total;) {
    int t, lt, kb, ke;
    seg_of(p, it0, it1, v, t, lt, kb, ke);
    const int tm = t % p.tiles_m, tn = t / p.tiles_m;
#pragma unroll
    for (int mi = 0; mi < MI; ++mi)
#pragma unroll
      for (int ni = 0; ni < NI; ++ni)
#pragma unroll
        for (int r = 0; r < 4; ++r) acc[mi][ni][r] = 0.f;
    const int n = ke - kb;
    {
      const int st = g % STAGES;
      mbar_wait(&s.full[st], (g / STAGES) & 1);
      load_k(f0, st, 0);
#pragma unroll 1
      for (int i = 0; i < n; ++i) step(i + 1 < n);
    }
    v += n;
    if (kb > 0) {  // tail half (first segment of CTA c+1): write the partial sum, then +1
      float4* dst = p.part + (size_t)(lt * MMA_WARPS + warp) * (PART4 * 32) + lane;
#pragma unroll
      for (int mi = 0; mi < MI; ++mi)
#pragma unroll
        for (int ni = 0; ni < NI; ++ni) __stcg(dst + (mi * NI + ni) * 32, make_float4(acc[mi][ni][0], acc[mi][ni][1], acc[mi][ni][2], acc[mi][ni][3]));
      __threadfence();
      __syncwarp();
      if (lane == 0) red_add_relaxed(&p.flags[lt * MMA_WARPS + warp], 1);
      continue;
    }
    if (ke < p.kt) {  // head half (last segment of this CTA): add the partial sum of CTA c+1
      int* fl = &p.flags[lt * MMA_WARPS + warp];
      const int ready = lane == 0 ? wait_bits(fl, 1) & 1 : 0;   // bounded wait for the tail half (normally long done)
      const bool got = __shfl_sync(0xffffffffu, ready, 0) != 0;
      __syncwarp();                                     // lane 0's acquire reaches the other lanes through the warp sync
      if (!got) {                                       // fallback: tail half not there -> publish the head half, +2
        float4* mine = p.part + ((size_t)(p.ntail + lt) * MMA_WARPS + warp) * (PART4 * 32) + lane;
#pragma unroll
        for (int mi = 0; mi < MI; ++mi)
#pragma unroll
          for (int ni = 0; ni < NI; ++ni) __stcg(mine + (mi * NI + ni) * 32, make_float4(acc[mi][ni][0], acc[mi][ni][1], acc[mi][ni][2], acc[mi][ni][3]));
        __threadfence();
        __syncwarp();
        const int old = lane == 0 ? atom_add_acq_rel(fl, 2) : 0;
        if ((__shfl_sync(0xffffffffu, old, 0) & 1) == 0) continue;   // tail half still not there: it merges
        __syncwarp();
      }
      const float4* src = p.part + (size_t)(lt * MMA_WARPS + warp) * (PART4 * 32) + lane;
#pragma unroll
      for (int mi = 0; mi < MI; ++mi)
#pragma unroll
        for (int ni = 0; ni < NI; ++ni) {
          const float4 q = __ldcg(src + (mi * NI + ni) * 32);
          acc[mi][ni][0] += q.x, acc[mi][ni][1] += q.y, acc[mi][ni][2] += q.z, acc[mi][ni][3] += q.w;
        }
      __syncwarp();
      if (lane == 0) *fl = 0;
    }
    epilogue(tm, tn);
  }
  if (it1 > it0 && it0 % p.kt) {   // first segment was a tail half: if the head half already left (flag 3), merge here
    const int lt = it0 / p.kt;
    int* fl = &p.flags[lt * MMA_WARPS + warp];
    const int seen = lane == 0 ? ld_relaxed(fl) : 0;   // lane 0 only (relaxed: a decision; other lanes' loads are not ordered after lane 0's +1)
    if (__shfl_sync(0xffffffffu, seen, 0) == 3) {
      __threadfence();                                  // relaxed load + fence = acquire
      __syncwarp();                                     // lane 0's acquire reaches the other lanes, which then read the published head half
      const float4* a = p.part + ((size_t)(p.ntail + lt) * MMA_WARPS + warp) * (PART4 * 32) + lane;
      const float4* b = p.part + ((size_t)lt * MMA_WARPS + warp) * (PART4 * 32) + lane;
#pragma unroll
      for (int mi = 0; mi < MI; ++mi)
#pragma unroll
        for (int ni = 0; ni < NI; ++ni) {
          const float4 x = __ldcg(a + (mi * NI + ni) * 32), y = __ldcg(b + (mi * NI + ni) * 32);
          acc[mi][ni][0] = x.x + y.x, acc[mi][ni][1] = x.y + y.y, acc[mi][ni][2] = x.z + y.z, acc[mi][ni][3] = x.w + y.w;
        }
      __syncwarp();
      if (lane == 0) *fl = 0;
      const int t = p.tail0 + lt;
      epilogue(t % p.tiles_m, t / p.tiles_m);
    }
  }
  if (lane == 0) asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
}



CUtensorMap Map2d(const void* ptr, int64_t rows, int64_t cols, int box_cols, int box_rows, CUtensorMapSwizzle sw) {
  CUtensorMap m;
  cuuint64_t dims[2] = {(cuuint64_t)cols, (cuuint64_t)rows};
  cuuint64_t strides[1] = {(cuuint64_t)cols * 2};
  cuuint32_t box[2] = {(cuuint32_t)box_cols, (cuuint32_t)box_rows};
  cuuint32_t elem[2] = {1, 1};
  if (xinfer::ffi::TensorMapEncodeTiled()(&m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<void*>(ptr), dims, strides, box, elem,
                CU_TENSOR_MAP_INTERLEAVE_NONE, sw, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) != CUDA_SUCCESS) {
    throw std::runtime_error("tma_streamk_glu_bf16_sm120: cuTensorMapEncodeTiled failed");
  }
  return m;
}

int SmCount(int device) {
  static int cache[64] = {};
  if (device < 0 || device >= 64) throw std::runtime_error("tma_streamk_glu_bf16_sm120: device index");
  if (cache[device] == 0) {
    ffi::CheckCudaError(cudaDeviceGetAttribute(&cache[device], cudaDevAttrMultiProcessorCount, device),
                        "cudaDeviceGetAttribute");
  }
  return cache[device];
}

// Tile schedule of (M, N, K) on `sms` CTAs (the fields of Params the host fills).
struct Plan {
  int tiles_m, tiles, kt, R, tail0, T, ntail;
};
Plan PlanOf(int64_t m, int64_t n, int64_t k, int sms) {
  Plan p;
  p.kt = int(k / BK), p.tiles_m = int((m + BM - 1) / BM), p.tiles = p.tiles_m * int(n / BNO);
  p.R = std::max(0, p.tiles / sms - 1), p.tail0 = p.R * sms, p.T = (p.tiles - p.tail0) * p.kt;
  p.ntail = std::max(p.tiles - p.tail0, 1);
  return p;
}
bool SupportsOn(int64_t m, int64_t n, int64_t k, int sms) {
  if (m < 1 || n < BNO || k < BK || n % BNO || k % BK || m > (int64_t(1) << 24) || n > 65536 || k > 65536) return false;
  const Plan p = PlanOf(m, n, k, sms);
  return p.T > 0 && p.T / sms >= p.kt;  // >= one tile of k-steps per CTA: a tile spans at most two CTAs
}
constexpr size_t kPartBytesPerTile = size_t(MMA_WARPS) * PART4 * 32 * 16;

}  // namespace

int64_t WorkspaceBytes(int64_t m, int64_t n, int64_t k, int64_t device) {
  return int64_t(2) * PlanOf(m, n, k, SmCount(int(device))).ntail * kPartBytesPerTile;
}
int64_t FlagBytes(int64_t m, int64_t n, int64_t k, int64_t device) {
  return int64_t(PlanOf(m, n, k, SmCount(int(device))).ntail) * MMA_WARPS * 4;
}
bool Supports(int64_t m, int64_t n, int64_t k, int64_t device) { return SupportsOn(m, n, k, SmCount(int(device))); }

// out [M, N], x [M, K], w [2N, K] (gate rows, then up rows).
void Run(tvm::ffi::TensorView out, tvm::ffi::TensorView x, tvm::ffi::TensorView w, tvm::ffi::TensorView workspace,
         tvm::ffi::TensorView flags) {
  using namespace xinfer::ffi;
  const char* op = "tma_streamk_glu_bf16_sm120";
  RequireArchitecture(x, 12, 0, op);
  for (auto* t : {&out, &x, &w}) {
    CheckCuda(*t, op);
    CheckBFloat16(*t, op);
    CheckContiguous(*t, op);
  }
  CheckUInt8(workspace, "workspace");
  CheckUInt8(flags, "flags");
  if (x.ndim() != 2 || w.ndim() != 2 || out.ndim() != 2) throw std::runtime_error("tma_streamk_glu_bf16_sm120: rank-2 only");
  const int64_t m = x.size(0), k = x.size(1), n = w.size(0) / 2;
  if (w.size(0) % 2 || w.size(1) != k || out.size(0) != m || out.size(1) != n) {
    throw std::runtime_error("tma_streamk_glu_bf16_sm120: expected x [M, K], w [2N, K], out [M, N]");
  }
  const int device = x.device().device_id;
  const int sms = SmCount(device);
  if (!SupportsOn(m, n, k, sms)) {
    throw std::runtime_error("tma_streamk_glu_bf16_sm120: needs N % 64 == 0, K % 32 == 0 and ceil(M/160) * N/64 >= " +
                             std::to_string(sms) + " tiles (M=" + std::to_string(m) + " N=" + std::to_string(n) +
                             " K=" + std::to_string(k) + ")");
  }
  const Plan pl = PlanOf(m, n, k, sms);
  if (NumBytes(workspace) < int64_t(2) * pl.ntail * int64_t(kPartBytesPerTile)) {
    throw std::runtime_error("tma_streamk_glu_bf16_sm120: workspace too small");
  }
  if (NumBytes(flags) < int64_t(pl.ntail) * MMA_WARPS * 4) throw std::runtime_error("tma_streamk_glu_bf16_sm120: flags too small");
  for (auto* t : {&x, &w, &out, &workspace}) {
    if (reinterpret_cast<uintptr_t>(t->data_ptr()) % 16) throw std::runtime_error("tma_streamk_glu_bf16_sm120: 16-byte aligned tensors only");
  }
  Params p{};
  p.part = static_cast<float4*>(workspace.data_ptr());
  p.flags = static_cast<int*>(flags.data_ptr());
  p.M = int(m), p.F = int(n), p.K = int(k), p.tiles_m = pl.tiles_m, p.tiles = pl.tiles, p.kt = pl.kt, p.G = sms;
  p.R = pl.R, p.tail0 = pl.tail0, p.T = pl.T, p.ntail = pl.ntail;
  const CUtensorMap mA = Map2d(x.data_ptr(), m, k, BK, BM, CU_TENSOR_MAP_SWIZZLE_64B),
                    mB = Map2d(w.data_ptr(), 2 * n, k, BK, BNO, CU_TENSOR_MAP_SWIZZLE_64B),
                    mD = Map2d(out.data_ptr(), m, n, 16, WTM, CU_TENSOR_MAP_SWIZZLE_32B);
  const int smem = int(sizeof(Smem)) + 1024;
  // Stream() also makes x's device current: the function attribute below is per device.
  const cudaStream_t stream = Stream(x);
  static bool attr[64] = {};
  if (!attr[device]) {
    CheckCudaError(cudaFuncSetAttribute(tma_streamk_glu_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem),
                   "cudaFuncSetAttribute");
    attr[device] = true;
  }
  tma_streamk_glu_kernel<<<sms, THREADS, smem, stream>>>(mA, mB, mD, p);
  CheckLastCudaError(op);
}

}  // namespace xinfer::gemm::tma_streamk_glu_sm120

TVM_FFI_DLL_EXPORT_TYPED_FUNC(tma_streamk_glu_bf16_sm120, xinfer::gemm::tma_streamk_glu_sm120::Run);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(tma_streamk_glu_bf16_sm120_workspace_bytes, xinfer::gemm::tma_streamk_glu_sm120::WorkspaceBytes);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(tma_streamk_glu_bf16_sm120_flag_bytes, xinfer::gemm::tma_streamk_glu_sm120::FlagBytes);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(tma_streamk_glu_bf16_sm120_supports, xinfer::gemm::tma_streamk_glu_sm120::Supports);
