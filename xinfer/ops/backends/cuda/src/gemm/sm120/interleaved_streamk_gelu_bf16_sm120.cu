// out[M, N] = gelu_tanh(x[M, K] @ W[N, K]^T + bias), BF16 in / out, FP32 accumulation.
//
// Persistent stream-K BF16 GEMM on raw [N, K] weights (backend interleaved_streamk_bf16_sm120): one CTA per SM, 8 MMA
// warps (1 x 8, each BM x 16) + 1 TMA producer warp; 112 x 128 tiles, BK 32, 4 stages, 64B swizzle, mma.sync m16n8k16.
// * Work split: all (tile, k-step) iterations in tile order are cut into one contiguous range per CTA by a host cost
//   model (plan_bounds: per-iteration cost plus the epilogue overhead the range's tail implies), at least K/32
//   iterations per CTA, so a tile spans at most two CTAs: owner (tile head) and participant (tile tail, next CTA).
// * Deferred epilogue: at a segment end the warps round acc + bias to BF16 into a per-warp shared-memory stash; the
//   activation / RoPE and the global stores are software-pipelined into the first k-steps of the next segment (the
//   two warp groups staggered). The last tile of a CTA is written straight from registers.
// * Cross-CTA merge (waits on a later CTA, bounded, with a fallback): the
//   participant writes its FP32 partial sum (workspace) and adds 1 to the (tile, warp) flag a few iterations later;
//   the owner waits at most XC_WAIT_NS and folds it into its accumulators near the end of its segment. If it times
//   out (the participant cannot get an SM), the owner publishes its own sum and adds 2; whichever side sees both
//   halves merges and writes the tile. The common path is run-to-run deterministic; the fallback adds the halves at
//   a different point (rounding-level difference), so the selection table marks the backend nondeterministic.
// * Flags: tiles x 8 int32, zeroed once per (device, stream), left zeroed by the kernel. Plan bounds: device array
//   per (device, M, N, K), built on the first call of the shape.
// GELU uses tanh.approx.f32 on the BF16-rounded (acc + bias).
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <tvm/ffi/function.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "common/tvm_ffi_cuda_tensor.h"
#include "common/tensor_map.h"

#define PK_EPI 1
#define PK_TMASTORE 0

namespace xinfer::gemm::interleaved_streamk_gelu_sm120 {
namespace {

constexpr const char* kName = "interleaved_streamk_bf16_sm120_gelu_tanh";

#ifndef PK_BM
#define PK_BM 112
#endif
#ifndef PK_STAGES
#define PK_STAGES 4
#endif
#ifndef PK_EPI
#define PK_EPI 1
#endif
#ifndef PK_STAGGER
#define PK_STAGGER 1
#endif
#ifndef PK_PFOLD
#define PK_PFOLD 1
#endif
#ifndef PK_FPI
#define PK_FPI 1
#endif
#ifndef PK_CH
#define PK_CH 4
#endif
constexpr int BM = PK_BM, BN = 128, BK = 32, KK = BK / 16, STAGES = PK_STAGES;
constexpr int MMA_WARPS = 8, WTN = BN / MMA_WARPS;
constexpr int MI = BM / 16, NI = WTN / 8, ACC4 = MI * NI;
static_assert(BM % 16 == 0 && NI == 2 && ACC4 % 2 == 0, "layout");
constexpr int THREADS = (MMA_WARPS + 1) * 32;
constexpr int A_STAGE = BM * BK * 2, B_STAGE = BN * BK * 2;
static_assert(A_STAGE % 512 == 0 && B_STAGE % 512 == 0, "swizzle alignment");
constexpr int STAGE_BYTES = A_STAGE + B_STAGE;
constexpr int NBLK = MI * NI;
constexpr int STASH_WORDS = NBLK * 2;

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
__device__ __forceinline__ float tanh_approx(float x) {
  float y;
  asm("tanh.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
__device__ __forceinline__ unsigned long long gtimer() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t));
  return t;
}
__device__ __forceinline__ unsigned long long gclock() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %clock64;" : "=l"(t));
  return t;
}
__device__ __forceinline__ float4 ldcg_v(const float4* ptr) {
  float4 v;
  asm volatile("ld.global.cg.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "l"(ptr));
  return v;
}
#ifndef XC_WAIT_NS
#define XC_WAIT_NS 20000
#endif
#ifndef XC_FORCE_RARE
#define XC_FORCE_RARE 0
#endif
__device__ __forceinline__ int ld_acquire(const int* f) {
  int v;
  asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(v) : "l"(f));
  return v;
}
__device__ __forceinline__ int ld_relaxed(const int* f) {
  int v;
  asm volatile("ld.relaxed.gpu.global.b32 %0, [%1];" : "=r"(v) : "l"(f) : "memory");
  return v;
}
__device__ __forceinline__ int wait_one(const int* f) {
  int v = ld_acquire(f);
  if (!(v & 1)) {
    const unsigned long long t0 = gtimer();
    do {
      __nanosleep(64);
      v = ld_acquire(f);
    } while (!(v & 1) && gtimer() - t0 < XC_WAIT_NS);
  }
  return v & 1;
}
__device__ __forceinline__ int atom_add_acq_rel(int* a, int v) {
  int old;
  asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], %2;" : "=r"(old) : "l"(a), "r"(v) : "memory");
  return old;
}

struct FragK {
  uint32_t a[MI][4];
  uint32_t b[NI][2];
};
struct Smem {
  alignas(1024) uint8_t a[STAGES][A_STAGE];
  alignas(1024) uint8_t b[STAGES][B_STAGE];
  uint32_t stash[MMA_WARPS][STASH_WORDS][32];
  alignas(8) uint64_t full[STAGES];
  alignas(8) uint64_t empty[STAGES];
};

struct Params {
  __nv_bfloat16* D;
  const __nv_bfloat16* bias;
  float* partials;
  int* flags;
  unsigned long long* ev;
  unsigned long long* tl;
  const int* bnd;
  int li;
  int M, N, K, tiles_n, tiles, kt, mode;
  int T, G;
};
__device__ __forceinline__ int seg_begin(const Params& p, int c) { return p.bnd ? p.bnd[c] : c * p.T / p.G; }

#define EV(code) \
  if (p.ev && threadIdx.x == 0 && nev < 63) p.ev[blockIdx.x * 64 + nev++] = ((unsigned long long)(code) << 56) | (gclock() - clk0);

__global__ void __launch_bounds__(THREADS, 1)
    pk6_kernel(const __grid_constant__ CUtensorMap mapA, const __grid_constant__ CUtensorMap mapB, const Params p) {
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  Smem& s = *reinterpret_cast<Smem*>((reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t(1023));
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int it0 = seg_begin(p, blockIdx.x), it1 = seg_begin(p, blockIdx.x + 1);
  const unsigned long long clk0 = gclock();
  int nev = 0;
  if (p.tl && threadIdx.x == 0) atomicMin(&p.tl[p.li * 3], gtimer());
  if (p.ev && threadIdx.x == 0) {
    unsigned smid;
    asm volatile("mov.u32 %0, %smid;" : "=r"(smid));
    p.ev[blockIdx.x * 64 + nev++] = (15ull << 56) | smid;
  }
  if (threadIdx.x == 0) {
    for (int i = 0; i < STAGES; ++i) {
      mbar_init(&s.full[i], 1);
      mbar_init(&s.empty[i], MMA_WARPS);
    }
    asm volatile("fence.mbarrier_init.release.cluster;");
  }
  __syncthreads();
  const int total = it1 - it0;

  if (warp == MMA_WARPS) {
    if (lane == 0) {
      asm volatile("prefetch.tensormap [%0];" ::"l"(&mapA));
      asm volatile("prefetch.tensormap [%0];" ::"l"(&mapB));
      int is_t = it0 / p.kt, is_k = it0 % p.kt;
      int is_tm = is_t / p.tiles_n, is_tn = is_t % p.tiles_n;
      for (int gp = 0; gp < total; ++gp) {
        if (p.mode == 1 && gp >= STAGES) break;
        const int st = gp % STAGES;
        if (gp >= STAGES) mbar_wait(&s.empty[st], ((gp / STAGES) - 1) & 1);
        mbar_expect_tx(&s.full[st], STAGE_BYTES);
        tma_load_2d(s.a[st], &mapA, &s.full[st], is_k * BK, is_tm * BM);
        tma_load_2d(s.b[st], &mapB, &s.full[st], is_k * BK, is_tn * BN);
        if (++is_k == p.kt) {
          is_k = 0;
          if (++is_tn == p.tiles_n) is_tn = 0, ++is_tm;
        }
      }
    }
    return;
  }

  const int wn = warp, tid = threadIdx.x;
  float acc[MI][NI][4];
  float4* acc4 = reinterpret_cast<float4*>(&acc[0][0][0]);
  int g = 0;

  const int rowA = lane & 15, rowB = wn * 16 + (lane & 7) + ((lane >> 4) << 3);
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
  auto stepx = [&](bool has_next, auto&& extra) {
    const int st = g % STAGES;
#pragma unroll
    for (int kk = 0; kk < KK; ++kk) {
      FragK& cur = (kk & 1) ? f1 : f0;
      FragK& nxt = (kk & 1) ? f0 : f1;
      if (kk + 1 < KK) {
        load_k(nxt, st, kk + 1);
      } else if (has_next) {
        const int st2 = (g + 1) % STAGES;
        if (p.mode == 0 || g + 1 < STAGES) mbar_wait(&s.full[st2], ((g + 1) / STAGES) & 1);
        load_k(nxt, st2, 0);
      }
#pragma unroll
      for (int mi = 0; mi < MI; ++mi)
#pragma unroll
        for (int ni = 0; ni < NI; ++ni) mma16816(acc[mi][ni], cur.a[mi], cur.b[ni]);
      extra(kk);
    }
    if (p.mode == 0 && lane == 0) mbar_arrive(&s.empty[st]);
    ++g;
  };
  auto step = [&](bool has_next) { stepx(has_next, [](int) {}); };
  auto run = [&](int cnt, bool more_after) {
    for (int i = 0; i < cnt; ++i) step(i + 1 < cnt || more_after);
  };

  int* sig_flag = nullptr;
  auto signal = [&] {
    __threadfence();
    __syncwarp();
    if (XC_FORCE_RARE != 1 && lane == 0) asm volatile("red.release.gpu.global.add.s32 [%0], 1;" ::"l"(sig_flag) : "memory");
    sig_flag = nullptr;
  };
  auto release_flag = [&](int t) {
    if (lane == 0) p.flags[t * MMA_WARPS + warp] = 0;
  };
  auto wait_flag = [&](int t) {
    const int r = lane == 0 ? (XC_FORCE_RARE ? 0 : wait_one(&p.flags[t * MMA_WARPS + warp])) : 0;
    const bool ok = __shfl_sync(0xffffffffu, r, 0) != 0;
    __syncwarp();
    return ok;
  };
  auto fold_rest = [&](int t, int from) {
    const float4* src = reinterpret_cast<const float4*>(p.partials + (size_t)t * (BM * BN));
#pragma unroll
    for (int j0 = 0; j0 < ACC4; j0 += 4) {
      float4 v[4];
#pragma unroll
      for (int q = 0; q < 4; ++q)
        if (j0 + q < ACC4 && j0 + q >= from) v[q] = __ldcg(&src[(j0 + q) * (MMA_WARPS * 32) + tid]);
#pragma unroll
      for (int q = 0; q < 4; ++q)
        if (j0 + q < ACC4 && j0 + q >= from) {
          float4& a4 = acc4[j0 + q];
          a4.x += v[q].x, a4.y += v[q].y, a4.z += v[q].z, a4.w += v[q].w;
        }
    }
    if (from < ACC4) release_flag(t);
  };

  uint32_t* stw = &s.stash[warp][0][lane];
  int pend = 0;
  int ptm = 0, ptn = 0;
  long long pbase = 0;
  int prows = 0;
  uint32_t bia[NI];
  float hx[4], th[4];
  auto epiA = [&](int jb) {
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const uint32_t u = stw[(jb * 2 + h) * 32];
      const __nv_bfloat162 b2 = *reinterpret_cast<const __nv_bfloat162*>(&u);
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        const float v = e ? __bfloat162float(b2.y) : __bfloat162float(b2.x);
#if PK_EPI == 1
        th[h * 2 + e] = tanh_approx(v * fmaf(0.0356774081f, v * v, 0.7978845608f));
        hx[h * 2 + e] = 0.5f * v;
#else
        th[h * 2 + e] = 0.f;
        hx[h * 2 + e] = v;
#endif
      }
    }
  };
  auto epiB = [&](int jb) {
    const int mi = jb / NI, ni = jb % NI;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int roff = mi * 16 + h * 8;
      if (roff < prows) {
        const __nv_bfloat162 o = __floats2bfloat162_rn(fmaf(hx[2 * h], th[2 * h], hx[2 * h]), fmaf(hx[2 * h + 1], th[2 * h + 1], hx[2 * h + 1]));
        *reinterpret_cast<__nv_bfloat162*>(p.D + pbase + (long long)roff * p.N + ni * 8) = o;
      }
    }
  };
  auto epi_step = [&](int sidx) {
    if (sidx >= 1 && sidx <= NBLK) epiB(sidx - 1);
    if (sidx < NBLK) epiA(sidx);
  };
  constexpr int EPI_ITERS = (NBLK + 1 + KK - 1) / KK;
  auto flush = [&] {
    if (!pend) return;
#pragma unroll 1
    for (int mi = 0; mi < MI; ++mi) {
      float x[NI][4];
#pragma unroll
      for (int ni = 0; ni < NI; ++ni)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const uint32_t u = stw[((mi * NI + ni) * 2 + h) * 32];
          const __nv_bfloat162 b2 = *reinterpret_cast<const __nv_bfloat162*>(&u);
          x[ni][2 * h] = __bfloat162float(b2.x), x[ni][2 * h + 1] = __bfloat162float(b2.y);
        }
#pragma unroll
      for (int ni = 0; ni < NI; ++ni)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int roff = mi * 16 + h * 8;
          float y[2];
#pragma unroll
          for (int e = 0; e < 2; ++e) {
            const float v = x[ni][2 * h + e];
#if PK_EPI == 1
            const float hv = 0.5f * v;
            y[e] = fmaf(hv, tanh_approx(v * fmaf(0.0356774081f, v * v, 0.7978845608f)), hv);
#else
            y[e] = v;
#endif
          }
          if (roff < prows) *reinterpret_cast<__nv_bfloat162*>(p.D + pbase + (long long)roff * p.N + ni * 8) = __floats2bfloat162_rn(y[0], y[1]);
        }
    }
    pend = 0;
  };
  auto stash_tile = [&](int t) {
#pragma unroll
    for (int mi = 0; mi < MI; ++mi)
#pragma unroll
      for (int ni = 0; ni < NI; ++ni) {
        const __nv_bfloat162 bb = *reinterpret_cast<const __nv_bfloat162*>(&bia[ni]);
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const __nv_bfloat162 o = __floats2bfloat162_rn(acc[mi][ni][2 * h] + __bfloat162float(bb.x), acc[mi][ni][2 * h + 1] + __bfloat162float(bb.y));
          stw[((mi * NI + ni) * 2 + h) * 32] = *reinterpret_cast<const uint32_t*>(&o);
        }
      }
    ptm = t / p.tiles_n, ptn = t % p.tiles_n;
    const int row0 = ptm * BM + (lane >> 2);
    pbase = (long long)row0 * p.N + ptn * BN + wn * WTN + (lane & 3) * 2;
    prows = p.M - row0;
    pend = 1;
  };

  for (int it = it0; it < it1;) {
    const int t = it / p.kt, kb = it % p.kt;
    const int tile_end = (t + 1) * p.kt;
    const int ke = (tile_end < it1 ? tile_end : it1) - t * p.kt;
    const int n = ke - kb;
    const bool is_owner = kb == 0;
    const bool split_head = is_owner && ke < p.kt;
    int folded = 0;
    EV(1)
    if (is_owner) {
#pragma unroll
      for (int ni = 0; ni < NI; ++ni) {
        const int col = (t % p.tiles_n) * BN + wn * WTN + ni * 8 + (lane & 3) * 2;
        bia[ni] = p.bias ? __ldg(reinterpret_cast<const unsigned int*>(&p.bias[col])) : 0u;
      }
    }
#pragma unroll
    for (int mi = 0; mi < MI; ++mi)
#pragma unroll
      for (int ni = 0; ni < NI; ++ni)
#pragma unroll
        for (int r = 0; r < 4; ++r) acc[mi][ni][r] = 0.f;
    {
      const int st = g % STAGES;
      if (p.mode == 0 || g < STAGES) mbar_wait(&s.full[st], (g / STAGES) & 1);
      EV(2)
      load_k(f0, st, 0);
      int i = 0;
      const bool g0 = warp < 4;
      if (pend && PK_EPI != 9) {
        if (PK_STAGGER && n >= 2 * EPI_ITERS + 1) {
          if (!g0) run(EPI_ITERS, true), i = EPI_ITERS;
#pragma unroll
          for (int e = 0; e < EPI_ITERS; ++e) stepx(true, [&](int kk) { epi_step(e * KK + kk); });
          i += EPI_ITERS;
          pend = 0;
          EV(8)
        } else if (n >= EPI_ITERS + 1) {
#pragma unroll
          for (int e = 0; e < EPI_ITERS; ++e) stepx(true, [&](int kk) { epi_step(e * KK + kk); });
          i = EPI_ITERS;
          pend = 0;
          EV(8)
        }
      }
      pend = PK_EPI == 9 ? 0 : pend;
      folded = 0;
      if (PK_PFOLD && split_head && !pend && n - i >= 2 && wait_flag(t)) {
        const float4* src = reinterpret_cast<const float4*>(p.partials + (size_t)t * (BM * BN));
        const uint32_t sbase = smem_u32(&s.stash[warp][0][0]) + lane * 16;
        constexpr int HALF = ACC4 / 2;
        static_assert(HALF * 16 * 32 <= STASH_WORDS * 4 * 32, "stash too small");
#pragma unroll
        for (int hf = 0; hf < 2; ++hf) {
#pragma unroll
          for (int q = 0; q < HALF; ++q)
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(sbase + q * 512), "l"(&src[(hf * HALF + q) * (MMA_WARPS * 32) + tid]) : "memory");
          asm volatile("cp.async.commit_group;" ::: "memory");
          step(i + 1 < n);
          ++i;
          asm volatile("cp.async.wait_group 0;" ::: "memory");
#pragma unroll
          for (int q = 0; q < HALF; ++q) {
            float4 v;
            asm volatile("ld.shared.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "r"(sbase + q * 512));
            float4& a4 = acc4[hf * HALF + q];
            a4.x += v.x, a4.y += v.y, a4.z += v.z, a4.w += v.w;
          }
        }
        folded = ACC4;
        release_flag(t);
      }
      if (sig_flag && n - i > PK_CH) {
        run(PK_CH, true);
        i += PK_CH;
        signal();
      }
      run(n - i, false);
    }
    it += n;
    EV(3)
    if (!is_owner) {
      if (sig_flag) signal();
      float4* dst = reinterpret_cast<float4*>(p.partials + (size_t)t * (BM * BN));
#pragma unroll
      for (int j = 0; j < ACC4; ++j) __stcg(&dst[j * (MMA_WARPS * 32) + tid], acc4[j]);
      sig_flag = &p.flags[t * MMA_WARPS + warp];
      if (it >= it1) signal();
      EV(4)
      continue;
    }
    bool skip = false;
    if (split_head) {
      if (sig_flag) signal();
      if (folded == 0 && !wait_flag(t)) {
        float4* mine = reinterpret_cast<float4*>(p.partials + (size_t)(p.tiles + t) * (BM * BN));
#pragma unroll
        for (int j = 0; j < ACC4; ++j) __stcg(&mine[j * (MMA_WARPS * 32) + tid], acc4[j]);
        __threadfence();
        __syncwarp();
        const int old = lane == 0 ? atom_add_acq_rel(&p.flags[t * MMA_WARPS + warp], 2) : 0;
        skip = (__shfl_sync(0xffffffffu, old, 0) & 1) == 0;
        __syncwarp();
      }
      if (!skip) fold_rest(t, folded);
      EV(5)
    }
    flush();
    if (skip) {
      EV(6)
      continue;
    }
    if (PK_EPI != 9 && it >= it1) {
      const int ptn2 = t % p.tiles_n, row0 = (t / p.tiles_n) * BM + (lane >> 2);
      const long long base = (long long)row0 * p.N + ptn2 * BN + wn * WTN + (lane & 3) * 2;
      const int rows = p.M - row0;
#pragma unroll
      for (int mi = 0; mi < MI; ++mi)
#pragma unroll
        for (int ni = 0; ni < NI; ++ni) {
          const __nv_bfloat162 bb = *reinterpret_cast<const __nv_bfloat162*>(&bia[ni]);
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            float y[2];
#pragma unroll
            for (int e = 0; e < 2; ++e) {
              const float v = __bfloat162float(__float2bfloat16(acc[mi][ni][2 * h + e] + __bfloat162float(e ? bb.y : bb.x)));
#if PK_EPI == 1
              const float hv = 0.5f * v;
              y[e] = fmaf(hv, tanh_approx(v * fmaf(0.0356774081f, v * v, 0.7978845608f)), hv);
#else
              y[e] = v;
#endif
            }
            if (mi * 16 + h * 8 < rows)
              *reinterpret_cast<__nv_bfloat162*>(p.D + base + (long long)(mi * 16 + h * 8) * p.N + ni * 8) = __floats2bfloat162_rn(y[0], y[1]);
          }
        }
    } else if (PK_EPI != 9) {
      stash_tile(t);
    }
    EV(6)
  }
  if (sig_flag) signal();
  flush();
  {
    const int b0 = seg_begin(p, blockIdx.x);
    if (b0 % p.kt && seg_begin(p, blockIdx.x + 1) > b0) {
      const int t = b0 / p.kt;
      int* fl = &p.flags[t * MMA_WARPS + warp];
      if (XC_FORCE_RARE == 1 && lane == 0) asm volatile("red.release.gpu.global.add.s32 [%0], 1;" ::"l"(fl) : "memory");
      const int seen = __shfl_sync(0xffffffffu, lane == 0 ? ld_relaxed(fl) : 0, 0);
      if (seen == 3) {
        __threadfence();
        __syncwarp();
        __nv_bfloat16* Dp = p.D;
        const __nv_bfloat16* bp = p.bias;
        int Mr = p.M, Nr = p.N;
        asm volatile("mov.b64 %0, %0;" : "+l"(Dp));
        asm volatile("mov.b64 %0, %0;" : "+l"(bp));
        asm volatile("mov.b32 %0, %0;" : "+r"(Mr));
        asm volatile("mov.b32 %0, %0;" : "+r"(Nr));
        const float4* own = reinterpret_cast<const float4*>(p.partials + (size_t)(p.tiles + t) * (BM * BN));
        const float4* par = reinterpret_cast<const float4*>(p.partials + (size_t)t * (BM * BN));
        const int ptn2 = t % p.tiles_n, row0 = (t / p.tiles_n) * BM + (lane >> 2);
        const long long base = (long long)row0 * Nr + ptn2 * BN + wn * WTN + (lane & 3) * 2;
        const int rows = Mr - row0;
#pragma unroll 1
        for (int j = 0; j < ACC4; ++j) {
          const float4 x = __ldcg(&own[j * (MMA_WARPS * 32) + tid]), y4 = __ldcg(&par[j * (MMA_WARPS * 32) + tid]);
          const float a[4] = {x.x + y4.x, x.y + y4.y, x.z + y4.z, x.w + y4.w};
          const int mi = j / NI, ni = j % NI;
          const int col = ptn2 * BN + wn * WTN + ni * 8 + (lane & 3) * 2;
          const unsigned int bw = bp ? __ldg(reinterpret_cast<const unsigned int*>(&bp[col])) : 0u;
          const __nv_bfloat162 bb = *reinterpret_cast<const __nv_bfloat162*>(&bw);
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            float y[2];
#pragma unroll
            for (int e = 0; e < 2; ++e) {
              const float v = __bfloat162float(__float2bfloat16(a[2 * h + e] + __bfloat162float(e ? bb.y : bb.x)));
#if PK_EPI == 1
              const float hv = 0.5f * v;
              y[e] = fmaf(hv, tanh_approx(v * fmaf(0.0356774081f, v * v, 0.7978845608f)), hv);
#else
              y[e] = v;
#endif
            }
            if (mi * 16 + h * 8 < rows)
              *reinterpret_cast<__nv_bfloat162*>(Dp + base + (long long)(mi * 16 + h * 8) * Nr + ni * 8) = __floats2bfloat162_rn(y[0], y[1]);
          }
        }
        __syncwarp();
        if (lane == 0) *fl = 0;
      }
    }
  }
  if (PK_EPI == 9) {
    float sum = 0;
#pragma unroll
    for (int j = 0; j < ACC4; ++j) sum += acc4[j].x;
    if (sum == 1234.5f) p.D[tid] = __float2bfloat16(sum);
  }
  if (p.tl && tid == 0) {
    const unsigned long long te = gtimer();
    atomicMax(&p.tl[p.li * 3 + 1], te);
    atomicMin(&p.tl[p.li * 3 + 2], te);
  }
  EV(7)
}


static double cta_cost(long long b0, long long b1, int kt) {
  if (b1 <= b0) return 0;
  const long long h = b1 % kt;
  const double oh = h == 0 ? 2900 : h <= 8 ? 8000 : h <= 16 ? 4500 : 3800;
  int segs = 0;
  for (long long it = b0; it < b1;) { const long long e = std::min((it / kt + 1) * kt, b1); ++segs; it = e; }
  return 1830.0 * (b1 - b0) + oh + (b0 % kt ? 700 : 0) + 300.0 * segs;
}
static std::vector<int> plan_bounds(int T, int G, int kt) {
  auto fits = [&](double cap, std::vector<int>* out) {
    std::vector<int> b{0};
    long long cur = 0;
    for (int c = 0; c < G && cur < T; ++c) {
      long long best = -1;
      for (long long e = std::min<long long>(cur + kt, T); e <= T; ++e) {
        const double v = cta_cost(cur, e, kt);
        if (v <= cap) best = e;
        else if (v > cap + 20000) break;
      }
      if (best < 0) return false;
      cur = best;
      b.push_back((int)cur);
    }
    if (cur < T) return false;
    while ((int)b.size() < G + 1) b.push_back(T);
    if (out) *out = b;
    return true;
  };
  double lo = 0, hi = 1e9;
  for (int i = 0; i < 50; ++i) { const double mid = (lo + hi) / 2; if (fits(mid, nullptr)) hi = mid; else lo = mid; }
  std::vector<int> b;
  fits(hi, &b);
  return b;
}



CUtensorMap Map2d(const void* ptr, int64_t rows, int64_t cols, int box_cols, int box_rows, CUtensorMapSwizzle sw,
                  CUtensorMapL2promotion l2) {
  CUtensorMap m;
  cuuint64_t dims[2] = {(cuuint64_t)cols, (cuuint64_t)rows};
  cuuint64_t strides[1] = {(cuuint64_t)cols * 2};
  cuuint32_t box[2] = {(cuuint32_t)box_cols, (cuuint32_t)box_rows};
  cuuint32_t elem[2] = {1, 1};
  if (xinfer::ffi::TensorMapEncodeTiled()(&m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<void*>(ptr), dims, strides, box, elem,
                CU_TENSOR_MAP_INTERLEAVE_NONE, sw, l2, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) != CUDA_SUCCESS) {
    throw std::runtime_error(std::string(kName) + ": cuTensorMapEncodeTiled failed");
  }
  return m;
}

int SmCount(int device) {
  static int cache[64] = {};
  if (device < 0 || device >= 64) throw std::runtime_error(std::string(kName) + ": device index");
  if (cache[device] == 0) {
    ffi::CheckCudaError(cudaDeviceGetAttribute(&cache[device], cudaDevAttrMultiProcessorCount, device),
                        "cudaDeviceGetAttribute");
  }
  return cache[device];
}

struct Shape {
  int tiles_n, tiles, kt, T;
};
Shape ShapeOf(int64_t m, int64_t n, int64_t k) {
  Shape s;
  s.tiles_n = int(n / BN), s.tiles = int((m + BM - 1) / BM) * s.tiles_n, s.kt = int(k / BK), s.T = s.tiles * s.kt;
  return s;
}
bool ShapeOk(int64_t m, int64_t n, int64_t k, int sms) {
  if (m < 1 || n < BN || k < BK || n % BN || k % BK || m > (int64_t(1) << 24) || n > 65536 || k > 65536) return false;
  const Shape s = ShapeOf(m, n, k);
  return s.T / sms >= s.kt;  // every CTA covers >= K/32 iterations: a tile spans at most two CTAs
}

// Per-CTA iteration bounds of a shape, planned on the host once per (device, M, N, K) and kept on the device.
// The first call of a shape must not be inside a CUDA graph capture (the library rule: run every shape once first).
const int* Bounds(int device, int64_t m, int64_t n, int64_t k, int sms, cudaStream_t stream) {
  static std::mutex mu;
  static std::map<std::tuple<int, int64_t, int64_t, int64_t>, int*> cache;
  std::lock_guard<std::mutex> lock(mu);
  const auto key = std::make_tuple(device, m, n, k);
  auto it = cache.find(key);
  if (it != cache.end()) return it->second;
  cudaStreamCaptureStatus status;
  ffi::CheckCudaError(cudaStreamIsCapturing(stream, &status), "cudaStreamIsCapturing");
  if (status != cudaStreamCaptureStatusNone) {
    throw std::runtime_error(std::string(kName) + ": run each shape once before capturing a CUDA graph");
  }
  const Shape s = ShapeOf(m, n, k);
  const std::vector<int> b = plan_bounds(s.T, sms, s.kt);
  int* d = nullptr;
  ffi::CheckCudaError(cudaMalloc(&d, b.size() * sizeof(int)), "cudaMalloc");
  ffi::CheckCudaError(cudaMemcpy(d, b.data(), b.size() * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy");
  cache.emplace(key, d);
  return d;
}

}  // namespace

int64_t WorkspaceBytes(int64_t m, int64_t n, int64_t k) { return int64_t(2) * ShapeOf(m, n, k).tiles * BM * BN * 4; }
int64_t FlagBytes(int64_t m, int64_t n, int64_t k) { return int64_t(ShapeOf(m, n, k).tiles) * MMA_WARPS * 4; }

bool Supports(int64_t m, int64_t n, int64_t k, int64_t device) { return ShapeOk(m, n, k, SmCount(int(device))); }

void Run(tvm::ffi::TensorView out, tvm::ffi::TensorView x, tvm::ffi::TensorView w, tvm::ffi::TensorView bias,
         tvm::ffi::TensorView workspace, tvm::ffi::TensorView flags) {
  using namespace xinfer::ffi;
  RequireArchitecture(x, 12, 0, kName);
  for (auto* t : {&out, &x, &w, &bias}) {
    CheckCuda(*t, kName);
    CheckBFloat16(*t, kName);
    CheckContiguous(*t, kName);
  }
  CheckUInt8(workspace, "workspace");
  CheckUInt8(flags, "flags");
  if (x.ndim() != 2 || w.ndim() != 2 || out.ndim() != 2) throw std::runtime_error(std::string(kName) + ": rank-2 only");
  const int64_t m = x.size(0), k = x.size(1), n = w.size(0);
  if (w.size(1) != k || out.size(0) != m || out.size(1) != n) {
    throw std::runtime_error(std::string(kName) + ": expected x [M, K], w [N, K], out [M, N]");
  }
  if (NumElements(bias) != n) throw std::runtime_error(std::string(kName) + ": bias must have N elements");
  const int device = x.device().device_id;
  const int sms = SmCount(device);
  if (!ShapeOk(m, n, k, sms)) {
    throw std::runtime_error(std::string(kName) + ": needs N % 128 == 0, K % 32 == 0 and (ceil(M/112) * N/128) * K/32 >= " +
                             std::to_string(sms) + " * K/32 (M=" + std::to_string(m) + " N=" + std::to_string(n) +
                             " K=" + std::to_string(k) + ")");
  }
  if (NumBytes(workspace) < WorkspaceBytes(m, n, k)) throw std::runtime_error(std::string(kName) + ": workspace too small");
  if (NumBytes(flags) < FlagBytes(m, n, k)) throw std::runtime_error(std::string(kName) + ": flags too small");
  for (auto* t : {&x, &w, &out, &workspace}) {
    if (reinterpret_cast<uintptr_t>(t->data_ptr()) % 16) throw std::runtime_error(std::string(kName) + ": 16-byte aligned tensors only");
  }
  cudaStream_t stream = Stream(x);
  const Shape sh = ShapeOf(m, n, k);
  Params p{};
  p.D = static_cast<__nv_bfloat16*>(out.data_ptr());
  p.bias = static_cast<const __nv_bfloat16*>(bias.data_ptr());
  p.partials = static_cast<float*>(workspace.data_ptr());
  p.flags = static_cast<int*>(flags.data_ptr());
  p.bnd = Bounds(device, m, n, k, sms, stream);
  p.M = int(m), p.N = int(n), p.K = int(k), p.tiles_n = sh.tiles_n, p.tiles = sh.tiles, p.kt = sh.kt, p.T = sh.T, p.G = sms;
  const CUtensorMap mA = Map2d(x.data_ptr(), m, k, BK, BM, CU_TENSOR_MAP_SWIZZLE_64B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B);
  const CUtensorMap mB = Map2d(w.data_ptr(), n, k, BK, BN, CU_TENSOR_MAP_SWIZZLE_64B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B);
  const int smem = int(sizeof(Smem)) + 1024;
  static bool attr[64] = {};
  if (!attr[device]) {
    CheckCudaError(cudaFuncSetAttribute(pk6_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem), "cudaFuncSetAttribute");
    attr[device] = true;
  }
  pk6_kernel<<<sms, THREADS, smem, stream>>>(mA, mB, p);
  CheckLastCudaError(kName);
}

}  // namespace xinfer::gemm::interleaved_streamk_gelu_sm120

TVM_FFI_DLL_EXPORT_TYPED_FUNC(interleaved_streamk_bf16_sm120_gelu_tanh, xinfer::gemm::interleaved_streamk_gelu_sm120::Run);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(interleaved_streamk_bf16_sm120_gelu_tanh_supports, xinfer::gemm::interleaved_streamk_gelu_sm120::Supports);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(interleaved_streamk_bf16_sm120_workspace_bytes, xinfer::gemm::interleaved_streamk_gelu_sm120::WorkspaceBytes);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(interleaved_streamk_bf16_sm120_flag_bytes, xinfer::gemm::interleaved_streamk_gelu_sm120::FlagBytes);
