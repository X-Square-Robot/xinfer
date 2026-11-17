// Ported from an earlier standalone Thor fused FFN (prefill_opt/po_ffn.cu). Only the GEMM family calls these entry
// points (mcast_sm110_ffi.cc). Changes: names; tensor maps encoded through cudaGetDriverEntryPoint.
// prefill_ffn: FFN-dimension chunked fusion of P6 (gate_up + GeGLU / SwiGLU) and P7 (down + residual) for Thor (sm_110a).
//
// Derived from prefill_gemm/src/pg_kernels.cu (gemm3: 6-CTA cluster = 3 CTA pairs covering M <= 768, tcgen05
// cta_group::2 UMMA 256x256x16, weight boxes TMA-multicast to the 3 same-parity CTAs, 4-stage smem ring,
// warp0 = TMA producer, warp1 = MMA issuer (pair leader), warps 2..5 = epilogue through TMEM -> smem -> TMA).
//
// What is new here: one persistent kernel per layer executes a static list of "units" per cluster:
//   P6 unit (c, n): gate/up rows [512n, 512n+512) -> h_c[:, 256 j .. 256 j + 256)  (j = n - c*NB/C), K = 2048
//   P7 unit (c, nb, s): x[:, 512nb .. +512) += h_c[:, ks] Wd[512nb.., c*F + ks]^T   (bf16 TMA reduce-add)
// h_c lives in a ring of `nslot` (=2) L2-resident slots [nslot][M][F] (F = I/C, intermediate I = 256 NB).  Cross-cluster dataflow
// through global flags (acquire/release + async-proxy fences):
//   hready[n]   : +1 by each of the 24 epilogue warps after its h stores of P6 unit n completed  (RAW for P7)
//   p7done[c]   : +1 by each pair leader after the last TMA load of a P7 unit of chunk c landed  (WAR for slot reuse)
// Unit lists are topologically ordered (P6(c) < P7(c) < P6(c + nslot)) in every cluster, all 3 clusters are
// co-resident (18 of 20 SMs), so the waits cannot deadlock.  The last CTA to exit re-zeroes the flags.
#include "mcast_ffn_sm110.h"
#include <cuda.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <string>
#include <map>
#include <vector>
#include <algorithm>

namespace mcast_ffn {

typedef __nv_bfloat16 bf16;
constexpr int BK = 64;
constexpr int NEPI = 8;  // epilogue warps: 2 per TMEM lane quadrant, each owns half of the accumulator columns
constexpr int NTHREADS = 64 + 32 * NEPI;
constexpr int BNP = 512;
constexpr int STAGES = 4;
constexpr int NSTG = 1;
constexpr int MAXU = 128;      // dbg timeline slots per CTA
constexpr int MAXQ = 192;      // units per launch
constexpr int SQ = 4;          // in-cluster schedule ring depth
constexpr int NCONS = 5 + 3 + 6 * NEPI;  // schedule consumers per cluster: producers of ranks 1..5, 3 MMA, epi warps
constexpr int FLAG_P7 = 64;     // p7done[c] at flags[64 + c]
constexpr int FLAG_Q = 126;     // global work-queue head
constexpr int FLAG_EXIT = 127;  // exit counter
constexpr int NFLAGS = 128;
constexpr uint32_t UEND = 0xFFFFu;

struct FParams {
  int M, C, F, KBc, S, Slast, nbpc;  // nbpc = P6 blocks per chunk = NB / C (NB = I / 256 gate/up blocks of 512 rows)
  int KBt;   // down-projection k-blocks in total = I / 64
  int act;   // MCAST_ACT_SILU or MCAST_ACT_GELU_TANH
  int nslot;
  int sync;  // 1: in-kernel cross-cluster dataflow flags; 0: ordering by kernel boundaries
  int pol_w, pol_a, pol_hs, pol_hl, pol_x;
  int spin_ns;
  int win;   // half-release overlap window (k-blocks), 0 = off
  int pace;  // MMA pacing: SM clocks per k-block MMA group (both halves = 1024 clk at peak), 0 = off
  int slack; // token bucket: how far (clocks) the pacer may fall behind and then catch up in a burst
  int hdiscard;          // 1: the last CTA to exit discards the (dead) h ring from L2 (no write-back, frees L2)
  char* hbuf;
  long long hbytes;
  int wblk;  // 1: weights in K-blocked layout [nblk][kb][512][64] (each k-block of an N block contiguous, 64 KB)
  int* flags;
  unsigned long long* stamp;
  unsigned long long* dbg;
  const int* pace_dev;   // prefill_opt: device-side pace (graph-invariant pointer, value adapted between replays); NULL -> pace
  int pgran;             // prefill_opt: pacing granularity 0 = per k-block (orig), 1 = per N half, 2 = per MMA instruction
  const char* pf_ptr[2]; // prefill_opt: next-layer weights to L2-prefetch during the tail of this FFN (NULL = off)
  long long pf_bytes[2];
  int pf_q0;             // first queue index whose grab issues a prefetch slice
  int nq;
  uint16_t queue[MAXQ];  // units in topological order; clusters grab them dynamically (flags[FLAG_Q])
};

// ------------------------------------------------------------------------------------------------ PTX helpers
__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ uint32_t cluster_rank() {
  uint32_t r;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
  return r;
}
__device__ __forceinline__ uint32_t mapa(uint32_t addr, uint32_t rank) {
  uint32_t r;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(addr), "r"(rank));
  return r;
}
__device__ __forceinline__ void st_cluster_u32(uint32_t addr, uint32_t v) {
  asm volatile("st.shared::cluster.u32 [%0], %1;" ::"r"(addr), "r"(v) : "memory");
}
__device__ __forceinline__ void cluster_sync() {
  asm volatile("barrier.cluster.arrive.release.aligned;\n\tbarrier.cluster.wait.acquire.aligned;" ::: "memory");
}
__device__ __forceinline__ void mbar_init(uint64_t* bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_u32(bar)), "r"(count));
}
__device__ __forceinline__ void mbar_expect_tx(uint64_t* bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem_u32(bar)), "r"(bytes) : "memory");
}
// remote arrive with default (release.cta) semantics: no MEMBAR.GPU (release.cluster compiles to MEMBAR.ALL.GPU on
// sm_110a).  Used where only the arriving thread's own prior reads must be ordered (TMEM drained, slot read).
__device__ __forceinline__ void mbar_arrive_cluster(uint32_t cluster_addr) {
  asm volatile("mbarrier.arrive.shared::cluster.b64 _, [%0];" ::"r"(cluster_addr) : "memory");
}
// remote arrive that publishes data written with st.shared::cluster (schedule code broadcast)
__device__ __forceinline__ void mbar_arrive_cluster_rel(uint32_t cluster_addr) {
  asm volatile("mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%0];" ::"r"(cluster_addr) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t parity) {
  uint32_t addr = smem_u32(bar);
  asm volatile(
      "{\n\t.reg .pred p;\n\t"
      "WAIT_%=:\n\t"
      "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n\t"
      "@!p bra WAIT_%=;\n\t}" ::"r"(addr),
      "r"(parity)
      : "memory");
}
__device__ __forceinline__ void mbar_wait_cluster(uint64_t* bar, uint32_t parity) {
  uint32_t addr = smem_u32(bar);
  asm volatile(
      "{\n\t.reg .pred p;\n\t"
      "WAIT_%=:\n\t"
      "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%0], %1;\n\t"
      "@!p bra WAIT_%=;\n\t}" ::"r"(addr),
      "r"(parity)
      : "memory");
}
__device__ __forceinline__ void tma_load_2d_2sm(void* dst, const CUtensorMap* map, uint32_t bar_cluster, int c0,
                                                int c1, uint64_t policy) {
  asm volatile(
      "cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], "
      "[%1, {%3, %4}], [%2], %5;" ::"r"(smem_u32(dst)),
      "l"((uint64_t)map), "r"(bar_cluster), "r"(c0), "r"(c1), "l"(policy)
      : "memory");
}
__device__ __forceinline__ void tma_load_3d_2sm(void* dst, const CUtensorMap* map, uint32_t bar_cluster, int c0,
                                                int c1, int c2, uint64_t policy) {
  asm volatile(
      "cp.async.bulk.tensor.3d.cta_group::2.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], "
      "[%1, {%3, %4, %5}], [%2], %6;" ::"r"(smem_u32(dst)),
      "l"((uint64_t)map), "r"(bar_cluster), "r"(c0), "r"(c1), "r"(c2), "l"(policy)
      : "memory");
}
__device__ __forceinline__ uint64_t make_policy(int sel) {
  uint64_t p;
  if (sel == 1)
    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(p));
  else if (sel == 2)
    asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(p));
  else if (sel == 4)
    asm volatile("createpolicy.fractional.L2::evict_unchanged.b64 %0, 1.0;" : "=l"(p));
  else
    asm volatile("createpolicy.fractional.L2::evict_normal.b64 %0, 1.0;" : "=l"(p));
  return p;
}
__device__ __forceinline__ void tc_fence_before() { asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory"); }
__device__ __forceinline__ void tc_fence_after() { asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory"); }

__device__ __forceinline__ uint64_t make_sdesc(uint32_t addr) {
  uint64_t d = 0;
  d |= (uint64_t)((addr >> 4) & 0x3FFF);
  d |= (uint64_t)1 << 16;
  d |= (uint64_t)(1024 >> 4) << 32;
  d |= (uint64_t)1 << 46;
  d |= (uint64_t)2 << 61;
  return d;
}
__host__ __device__ constexpr uint32_t make_idesc(int M, int N) {
  return (1u << 4) | (1u << 7) | (1u << 10) | ((uint32_t)(N >> 3) << 17) | ((uint32_t)(M >> 4) << 24);
}
__device__ __forceinline__ void umma_2sm(uint32_t tmem_d, uint64_t da, uint64_t db, uint32_t idesc, uint32_t acc) {
  asm volatile(
      "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
      "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p;\n\t}\n" ::"r"(tmem_d),
      "l"(da), "l"(db), "r"(idesc), "r"(acc));
}
__device__ __forceinline__ void tmem_ld32(uint32_t taddr, float (&v)[32]) {
  uint32_t* r = reinterpret_cast<uint32_t*>(v);
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x32.b32 "
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
      "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];"
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]), "=r"(r[8]),
        "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15]), "=r"(r[16]),
        "=r"(r[17]), "=r"(r[18]), "=r"(r[19]), "=r"(r[20]), "=r"(r[21]), "=r"(r[22]), "=r"(r[23]), "=r"(r[24]),
        "=r"(r[25]), "=r"(r[26]), "=r"(r[27]), "=r"(r[28]), "=r"(r[29]), "=r"(r[30]), "=r"(r[31])
      : "r"(taddr));
}
__device__ __forceinline__ void tmem_wait_ld() { asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory"); }

__device__ __forceinline__ uint32_t pack_bf16(float a, float b) {
  __nv_bfloat162 h = __floats2bfloat162_rn(a, b);
  return *reinterpret_cast<uint32_t*>(&h);
}
__device__ __forceinline__ float tanh_fast(float x) {
  float y;
  asm("tanh.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
__device__ __forceinline__ float silu(float x) { return x / (1.0f + __expf(-x)); }
__device__ __forceinline__ float gelu_tanh(float x) {
  const float k0 = 0.7978845608028654f, k1 = 0.044715f;
  float u = k0 * fmaf(k1 * x * x, x, x);
  float hx = 0.5f * x;
  return fmaf(hx, tanh_fast(u), hx);
}

__device__ __forceinline__ void tma_store_3d(const CUtensorMap* map, const void* src, int c0, int c1, int c2,
                                             uint64_t pol) {
  asm volatile("cp.async.bulk.tensor.3d.global.shared::cta.bulk_group.L2::cache_hint [%0, {%2, %3, %4}], [%1], %5;" ::"l"(
                   (uint64_t)map),
               "r"(smem_u32(src)), "r"(c0), "r"(c1), "r"(c2), "l"(pol)
               : "memory");
}
__device__ __forceinline__ void tma_reduce_add_2d(const CUtensorMap* map, const void* src, int c0, int c1,
                                                  uint64_t pol) {
  asm volatile(
      "cp.reduce.async.bulk.tensor.2d.global.shared::cta.add.bulk_group.L2::cache_hint [%0, {%2, %3}], [%1], %4;" ::"l"(
          (uint64_t)map),
      "r"(smem_u32(src)), "r"(c0), "r"(c1), "l"(pol)
      : "memory");
}
__device__ __forceinline__ void bulk_commit() { asm volatile("cp.async.bulk.commit_group;" ::: "memory"); }
template <int N>
__device__ __forceinline__ void bulk_wait_read() {
  asm volatile("cp.async.bulk.wait_group.read %0;" ::"n"(N) : "memory");
}
__device__ __forceinline__ void bulk_wait_all() { asm volatile("cp.async.bulk.wait_group 0;" ::: "memory"); }

__device__ __forceinline__ int ld_acquire(const int* p) {
  int v;
  asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void red_release_add(int* p, int v) {
  asm volatile("red.release.gpu.global.add.s32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ void spin_until_ge(const int* p, int target, int ns) {
  while (ld_acquire(p) < target) {
    if (ns > 0) __nanosleep(ns);
  }
}

template <int N>
struct StageRing {
  uint8_t* base;
  int idx;
  int issued;
  __device__ __forceinline__ uint8_t* acquire(int lane) {
    if (issued >= N) {
      if (lane == 0) bulk_wait_read<N - 1>();
      __syncwarp();
    }
    return base + idx * 4096;
  }
  __device__ __forceinline__ void release_issue() {
    idx = (idx + 1 == N) ? 0 : idx + 1;
    if (issued < N) issued++;
  }
};
__device__ __forceinline__ void stage_row32(uint8_t* buf, int lane, int h, const float (&v)[32]) {
  uint8_t* rowp = buf + lane * 128;
#pragma unroll
  for (int q = 0; q < 4; q++) {
    const int j = h * 4 + q;
    uint4 u;
    u.x = pack_bf16(v[8 * q + 0], v[8 * q + 1]);
    u.y = pack_bf16(v[8 * q + 2], v[8 * q + 3]);
    u.z = pack_bf16(v[8 * q + 4], v[8 * q + 5]);
    u.w = pack_bf16(v[8 * q + 6], v[8 * q + 7]);
    *reinterpret_cast<uint4*>(rowp + ((j ^ (lane & 7)) << 4)) = u;
  }
}

struct Unit {
  int type, c, n, s, kb0, kb1;
};
__device__ __forceinline__ Unit decode(const FParams& p, uint32_t e) {
  Unit u;
  u.type = (int)(e >> 15);
  u.c = (int)((e >> 11) & 15);
  if (u.type == 0) {
    u.n = (int)((e >> 4) & 63);
    u.s = 0;
    u.kb0 = 0;
    u.kb1 = 2048 / BK;
  } else {
    u.n = (int)((e >> 4) & 3);
    u.s = (int)((e >> 6) & 31);
    const int S = (u.c == p.C - 1) ? p.Slast : p.S;
    u.kb0 = (u.s * p.KBc) / S;
    u.kb1 = ((u.s + 1) * p.KBc) / S;
  }
  return u;
}

__device__ __forceinline__ unsigned long long gtimer() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
__device__ __forceinline__ void dbg_mark(const FParams& p, int uidx, int slot) {
  if (p.dbg && uidx < MAXU) p.dbg[((size_t)blockIdx.x * MAXU + uidx) * 8 + slot] = gtimer();
}

// ------------------------------------------------------------------------------------------------ fused kernel
__global__ void __launch_bounds__(NTHREADS, 1)
    ffn_kernel(const __grid_constant__ CUtensorMap mapXn, const __grid_constant__ CUtensorMap mapH,
               const __grid_constant__ CUtensorMap mapWgu, const __grid_constant__ CUtensorMap mapWd,
               const __grid_constant__ CUtensorMap mapHst, const __grid_constant__ CUtensorMap mapX,
               const __grid_constant__ FParams p) {
  constexpr int MMA_N = 256;
  constexpr int NMMA = 2;
  constexpr int BH_ROWS = 128;
  constexpr int A_BYTES = 128 * BK * 2;
  constexpr int BH_BYTES = BH_ROWS * BK * 2;
  constexpr int STAGE_BYTES = A_BYTES + NMMA * BH_BYTES;  // 48 KB per CTA
  constexpr uint32_t IDESC = make_idesc(256, MMA_N);
  constexpr int BPH = BH_ROWS / 64;
  constexpr int NBOX = NMMA * BPH;  // 4 boxes of 64 rows

  extern __shared__ __align__(1024) uint8_t smem_raw[];
  uint8_t* smem = (uint8_t*)(((uintptr_t)smem_raw + 1023) & ~(uintptr_t)1023);
  __shared__ __align__(8) uint64_t full_bar[STAGES];
  __shared__ __align__(8) uint64_t empty_bar[STAGES];
  __shared__ __align__(8) uint64_t tfull_bar[1];
  __shared__ __align__(8) uint64_t tempty_bar[2];  // accumulator column halves [0,256) / [256,512) drained
  __shared__ uint32_t tmem_base_sh;
  __shared__ uint32_t sq_code[SQ];
  __shared__ __align__(8) uint64_t sq_full[SQ];
  __shared__ __align__(8) uint64_t sq_empty[SQ];

  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const uint32_t rank = cluster_rank();
  const uint32_t par = rank & 1;
  const int pj = (int)(rank >> 1);
  const int clu = blockIdx.x / 6;
  if (p.stamp && blockIdx.x == 0 && threadIdx.x == 0) {
    p.stamp[0] = clock64();
    p.stamp[2] = gtimer();
  }

  if (threadIdx.x == 0) {
    for (int s = 0; s < STAGES; s++) {
      mbar_init(&full_bar[s], 1);
      mbar_init(&empty_bar[s], 3);
    }
    mbar_init(&tfull_bar[0], 1);
    mbar_init(&tempty_bar[0], 2 * NEPI);
    mbar_init(&tempty_bar[1], 2 * NEPI);
    for (int s = 0; s < SQ; s++) {
      mbar_init(&sq_full[s], 1);
      mbar_init(&sq_empty[s], NCONS);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  if (warp == 0 && lane == 0) {
    asm volatile("prefetch.tensormap [%0];" ::"l"((uint64_t)&mapXn) : "memory");
    asm volatile("prefetch.tensormap [%0];" ::"l"((uint64_t)&mapH) : "memory");
    asm volatile("prefetch.tensormap [%0];" ::"l"((uint64_t)&mapWgu) : "memory");
    asm volatile("prefetch.tensormap [%0];" ::"l"((uint64_t)&mapWd) : "memory");
  }
  if (warp == 2 && lane == 0) {
    asm volatile("prefetch.tensormap [%0];" ::"l"((uint64_t)&mapHst) : "memory");
    asm volatile("prefetch.tensormap [%0];" ::"l"((uint64_t)&mapX) : "memory");
  }
  if (warp == 1) {
    asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(smem_u32(&tmem_base_sh)),
                 "r"(512));
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;");
  }
  tc_fence_before();
  cluster_sync();
  tc_fence_after();
  const uint32_t tmem_base = tmem_base_sh;
  const int m_blk = pj;
  // in-cluster schedule ring: rank 0's producer lane grabs the next unit of the global queue and broadcasts its
  // code to all 6 CTAs; every consumer warp reads it from its own smem and frees the slot on rank 0.
  uint32_t sq_i = 0;
  auto next_unit = [&](bool consumer) -> uint32_t {
    const uint32_t slot = sq_i % SQ, ph = (sq_i / SQ) & 1;
    sq_i++;
    uint32_t code;
    if (!consumer) {  // scheduler (rank 0 producer warp)
      if (lane == 0) {
        mbar_wait_cluster(&sq_empty[slot], ph ^ 1);
        const int q = atomicAdd(&p.flags[FLAG_Q], 1);
        code = q < p.nq ? (uint32_t)p.queue[q] : UEND;
        const uint32_t a_code = smem_u32(&sq_code[slot]), a_full = smem_u32(&sq_full[slot]);
#pragma unroll 1
        for (uint32_t r = 1; r < 6; r++) {
          st_cluster_u32(mapa(a_code, r), code);
          mbar_arrive_cluster_rel(mapa(a_full, r));
        }
        sq_code[slot] = code;
        asm volatile("fence.acq_rel.cluster;" ::: "memory");
      }
      code = __shfl_sync(0xffffffffu, code, 0);
      // rank-0 consumers (its MMA / epilogue warps) wait on the local full barrier too
      if (lane == 0) mbar_arrive_cluster_rel(smem_u32(&sq_full[slot]));
      __syncwarp();
    } else {
      mbar_wait_cluster(&sq_full[slot], ph);
      code = sq_code[slot];
      __syncwarp();
      if (lane == 0) mbar_arrive_cluster(mapa(smem_u32(&sq_empty[slot]), 0));
    }
    return code;
  };

  if (warp == 0) {
    // ---------------- TMA producer (all 6 CTAs)
    const uint64_t pol_w = make_policy(p.pol_w);
    const uint64_t pol_a = make_policy(p.pol_a);
    const uint64_t pol_hl = make_policy(p.pol_hl);
    const uint16_t mc_mask = (uint16_t)((1u << par) | (1u << (par + 2)) | (1u << (par + 4)));
    const int arow = m_blk * 256 + (int)par * 128;
    int stage = 0;
    uint32_t phase = 0;
    for (int i = 0;; i++) {
      const uint32_t code = next_unit(rank != 0);
      if (code == UEND) break;
      const Unit u = decode(p, code);
      const bool p7 = u.type != 0;
      const CUtensorMap* mB = p7 ? &mapWd : &mapWgu;
      int bk_off = p7 ? u.c * p.F : 0;
      int brow = u.n * BNP + (int)par * BH_ROWS;
      int bkstep = BK, browstep = 0;
      if (p.wblk) {  // row = (blk * KB + gkb) * 512 + r, col 0
        brow = ((p7 ? u.n * p.KBt + u.c * p.KBc : u.n * 32) * BNP) + (int)par * BH_ROWS;
        bk_off = 0;
        bkstep = 0;
        browstep = BNP;
      }
      const int slot = p7 ? (u.c % p.nslot) : 0;
      const int blk_base = u.c * p.nbpc;
      int next_blk_kb = u.kb0;  // first kb that needs a new hready check
      for (int kb = u.kb0; kb < u.kb1; kb++) {
        if (p7 && p.sync && kb >= next_blk_kb) {
          if (lane == 0) {
            spin_until_ge(&p.flags[blk_base + (kb >> 2)], 6 * NEPI, p.spin_ns);
            asm volatile("fence.proxy.async.global;" ::: "memory");
          }
          next_blk_kb = (kb & ~3) + 4;
        }
        mbar_wait_cluster(&empty_bar[stage], phase ^ 1);
        if (lane == 0) {
          uint8_t* sA = smem + stage * STAGE_BYTES;
          const uint32_t fb = smem_u32(&full_bar[stage]) & 0xFEFFFFFFu;  // pair leader's barrier
          if (par == 0) mbar_expect_tx(&full_bar[stage], 2 * STAGE_BYTES);
          if (p7)
            tma_load_3d_2sm(sA, &mapH, fb, kb * BK, arow, slot, pol_hl);
          else
            tma_load_2d_2sm(sA, &mapXn, fb, kb * BK, arow, pol_a);
#pragma unroll
          for (int b = 0; b < NBOX; b++) {
            if (b % 3 == pj) {
              const int bi = b / BPH, sub = b % BPH;
              asm volatile(
                  "cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::"
                  "cluster.L2::cache_hint [%0], [%1, {%4, %5}], [%2], %3, %6;" ::"r"(
                      smem_u32(sA + A_BYTES + bi * BH_BYTES + sub * 8192)),
                  "l"((uint64_t)mB), "r"(fb), "h"(mc_mask), "r"(bk_off + kb * bkstep),
                  "r"(brow + kb * browstep + bi * MMA_N + sub * 64), "l"(pol_w)
                  : "memory");
            }
          }
        }
        __syncwarp();
        if (++stage == STAGES) {
          stage = 0;
          phase ^= 1;
        }
      }
    }
  } else if (warp == 1) {
    // ---------------- MMA issuer (pair leaders)
    if (par == 0) {
      int stage = 0;
      uint32_t phase = 0;
      uint32_t acc_phase = 0;
      const uint16_t pair_mask = (uint16_t)(3u << (2 * pj));
      auto issue = [&](int stg, int j, bool acc0) {
        const uint32_t a_addr = smem_u32(smem + stg * STAGE_BYTES);
        const uint64_t da = make_sdesc(a_addr);
        const uint64_t db = make_sdesc(a_addr + A_BYTES + j * BH_BYTES);
#pragma unroll
        for (int k = 0; k < BK / 16; k++) umma_2sm(tmem_base + j * MMA_N, da + 2 * k, db + 2 * k, IDESC, (!acc0 || k > 0));
      };
      auto commit_empty = [&](int stg) {
        asm volatile(
            "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;" ::"r"(
                smem_u32(&empty_bar[stg])),
            "h"((uint16_t)0x3F)
            : "memory");
      };
      const int win = p.win;
      const long long pace = p.pace_dev ? (long long)(*(volatile const int*)p.pace_dev) : (long long)p.pace;
      const long long slack = p.slack;
      const int pgran = p.pgran;
      long long t_tgt = clock64();
      // pgran 4/5: k-block (4) / N-half (5) pacing on a per-pair phase-shifted grid: the 9 pairs' MMA bursts are
      // spread over the pacing period instead of firing in phase (lower aggregate current peaks at equal duty).
      // After a stall the pacer snaps to its next grid slot so the phase offset survives.
      const long long t_ref = t_tgt + (pace * (long long)((3 * (blockIdx.x / 6) + pj) % 9)) / 9;
      const long long slack8 = (pgran == 3) ? (slack > (pace >> 2) ? slack : (pace >> 2)) : slack;
      auto pace_wait8 = [&](int eighths) {  // per-MMA-instruction pacing (lane 0 only)
        if (pace > 0) {
          t_tgt += (pace * eighths) >> 3;
          long long now = clock64();
          if (now - t_tgt > slack8) t_tgt = now - slack8;
          while (now < t_tgt) now = clock64();
        }
      };
      auto issue_paced = [&](int stg, int j, bool acc0) {  // pgran 2: 4 MMAs of one N half, each paced by pace/8
        const uint32_t a_addr = smem_u32(smem + stg * STAGE_BYTES);
        const uint64_t da = make_sdesc(a_addr);
        const uint64_t db = make_sdesc(a_addr + A_BYTES + j * BH_BYTES);
#pragma unroll
        for (int k = 0; k < BK / 16; k++) {
          pace_wait8(1);
          umma_2sm(tmem_base + j * MMA_N, da + 2 * k, db + 2 * k, IDESC, (!acc0 || k > 0));
        }
      };
      auto pace_wait = [&](int halves) {  // keep the tensor pipe below ~100% duty: no over-current (oc3) events
        if (pace > 0 && (pgran < 2 || pgran >= 4)) {
          t_tgt += (pace * halves) >> 1;
          long long now = clock64();
          if (now - t_tgt > slack) {  // fell behind (stall)
            if (pgran >= 4) {
              const long long ps = pgran == 5 ? (pace >> 1) : pace;
              t_tgt = t_ref + ((now - t_ref + ps - 1) / ps) * ps;
            } else {
              t_tgt = now - slack;  // allow a bounded catch-up burst
            }
          }
          while (now < t_tgt) now = clock64();
        }
      };
      for (int i = 0;; i++) {
        const uint32_t code = next_unit(true);
        if (code == UEND) break;
        const Unit u = decode(p, code);
        if (lane == 0) dbg_mark(p, i, 6);
        // half-release overlap: once columns [0,256) are drained, run the N-half-0 MMAs of the first `w` k-blocks
        // (stages already in smem) while the epilogue still drains [256,512); then their N-half-1 MMAs.
        const int nkb = u.kb1 - u.kb0;
        const int w = win < nkb ? win : nkb;
        mbar_wait_cluster(&tempty_bar[0], acc_phase ^ 1);
        tc_fence_after();
        if (lane == 0) dbg_mark(p, i, 0);
        int st0 = stage;
        uint32_t ph0 = phase;
        for (int t = 0; t < w; t++) {
          mbar_wait(&full_bar[stage], phase);
          tc_fence_after();
          pace_wait(1);
          if (lane == 0) { if (pgran == 2 || pgran == 3) issue_paced(stage, 0, t == 0); else issue(stage, 0, t == 0); }
          __syncwarp();
          if (++stage == STAGES) { stage = 0; phase ^= 1; }
        }
        mbar_wait_cluster(&tempty_bar[1], acc_phase ^ 1);
        tc_fence_after();
        if (lane == 0) dbg_mark(p, i, 7);
        for (int t = 0; t < w; t++) {
          pace_wait(1);
          if (lane == 0) {
            if (pgran == 2 || pgran == 3) issue_paced(st0, 1, t == 0); else issue(st0, 1, t == 0);
            commit_empty(st0);
          }
          __syncwarp();
          if (++st0 == STAGES) { st0 = 0; ph0 ^= 1; }
        }
        for (int kb = u.kb0 + w; kb < u.kb1; kb++) {
          mbar_wait(&full_bar[stage], phase);
          tc_fence_after();
          if (pgran == 0 || pgran == 4) {
            pace_wait(2);
            if (lane == 0) {
              issue(stage, 0, kb == u.kb0);
              issue(stage, 1, kb == u.kb0);
              commit_empty(stage);
            }
          } else if (pgran == 1 || pgran == 5) {
            pace_wait(1);
            if (lane == 0) issue(stage, 0, kb == u.kb0);
            __syncwarp();
            pace_wait(1);
            if (lane == 0) {
              issue(stage, 1, kb == u.kb0);
              commit_empty(stage);
            }
          } else {
            if (lane == 0) {
              issue_paced(stage, 0, kb == u.kb0);
              issue_paced(stage, 1, kb == u.kb0);
              commit_empty(stage);
            }
          }
          __syncwarp();
          if (++stage == STAGES) { stage = 0; phase ^= 1; }
        }
        if (lane == 0) {
          asm volatile(
              "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;" ::"r"(
                  smem_u32(&tfull_bar[0])),
              "h"(pair_mask)
              : "memory");
          // every TMA load of this unit has landed (we waited on all its full barriers): h slot reads are done
          if (u.type != 0 && p.sync) red_release_add(&p.flags[FLAG_P7 + u.c], 1);
          dbg_mark(p, i, 1);
        }
        __syncwarp();
        acc_phase ^= 1;
      }
    } else if (lane == 0 && (p.pf_bytes[0] + p.pf_bytes[1]) > 0) {
      // prefill_opt: the MMA warp of the 9 non-leader CTAs is idle -> L2-prefetch the next layer's P2/P4 weights
      // during the tail of this FFN.  72 slices; slice k is issued once the unit queue head passes
      // q0 + k * (nq - q0) / 72 (spreads the extra DRAM reads evenly over the tail, off the producers' path).
      const long long tot = p.pf_bytes[0] + p.pf_bytes[1];
      const int NSL = 72, w = blockIdx.x >> 1;
      for (int k = w; k < NSL; k += 9) {
        const int thr = p.pf_q0 + (int)((long long)k * (p.nq - p.pf_q0) / NSL);
        while (*(volatile const int*)&p.flags[FLAG_Q] < thr) __nanosleep(500);
        const long long a = (tot * k / NSL) & ~1023ll, b = (k == NSL - 1) ? tot : ((tot * (k + 1) / NSL) & ~1023ll);
        for (int r = 0; r < 2; r++) {
          const long long lo = r == 0 ? 0 : p.pf_bytes[0], hi = lo + p.pf_bytes[r];
          const long long x0 = a > lo ? a : lo, x1 = b < hi ? b : hi;
          for (long long o = x0; o < x1; o += 65536) {
            const long long n = (x1 - o) < 65536 ? (x1 - o) : 65536;
            asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(p.pf_ptr[r] + (o - lo)),
                         "r"((uint32_t)((n + 15) & ~15ll))
                         : "memory");
          }
        }
      }
    }
  } else {
    // ---------------- epilogue (all 6 CTAs): warp e = warp-2 reads TMEM lane quadrant (warp & 3) = 32 rows of this
    // CTA's 128, and column half hsel = e / 4 of the 512 accumulator columns.
    const int e = warp - 2;
    const int sub = warp & 3;
    const int hsel = e >> 2;
    uint8_t* sbuf = smem + STAGES * STAGE_BYTES + e * NSTG * 4096;
    int sbuf_busy = 0;
    const uint32_t tempty_leader = smem_u32(&tempty_bar[0]) & 0xFEFFFFFFu;
    const uint64_t pol_hs = make_policy(p.pol_hs);
    const uint64_t pol_x = make_policy(p.pol_x);
    const int row0 = m_blk * 256 + (int)par * 128 + sub * 32;
    uint32_t acc_phase = 0;
    for (int i = 0;; i++) {
      const uint32_t code = next_unit(true);
      if (code == UEND) break;
      const Unit u = decode(p, code);
      if (u.type == 0 && p.sync && u.c >= p.nslot) {
        // WAR: slot (c % nslot) is free once every P7 unit of chunk c - nslot has loaded its h tiles
        if (lane == 0) {
          const int pc = u.c - p.nslot;
          const int tgt = 12 * ((pc == p.C - 1) ? p.Slast : p.S);
          spin_until_ge(&p.flags[FLAG_P7 + pc], tgt, p.spin_ns);
          asm volatile("fence.proxy.async.global;" ::: "memory");
        }
        __syncwarp();
      }
      if (e == 0 && lane == 0) dbg_mark(p, i, 5);
      mbar_wait(&tfull_bar[0], acc_phase);
      tc_fence_after();
      if (e == 0 && lane == 0) dbg_mark(p, i, 2);
      const uint32_t t_row = tmem_base + ((uint32_t)(sub * 32) << 16);
      // column half H = 0 first (all 8 warps), then H = 1; warp pair index w2 = hsel takes 128 of the 256 cols.
      if (u.type == 0) {
        // GeGLU: half H is sub-tile s = H: gate cols H*256 + [0,128), up H*256 + 128 + [0,128) -> h cols
        // j*256 + H*128 + [0,128); this warp: 64 of them (w2*64).
        const int slot = u.c % p.nslot;
        const int hcol = (u.n - u.c * p.nbpc) * 256 + hsel * 64;
#pragma unroll 1
        for (int H = 0; H < 2; H++) {
          const uint32_t tg = t_row + H * 256 + hsel * 64;
          float g0[32], v0[32], g1[32], v1[32];
          tmem_ld32(tg, g0);
          tmem_ld32(tg + 128, v0);
          tmem_ld32(tg + 32, g1);
          tmem_ld32(tg + 128 + 32, v1);
          tmem_wait_ld();
          tc_fence_before();
          __syncwarp();
          if (lane == 0) mbar_arrive_cluster(tempty_leader + 8 * H);
          if (H == 1 && e == 0 && lane == 0) dbg_mark(p, i, 4);
#pragma unroll
          for (int q = 0; q < 32; q++) {
            v0[q] *= p.act == MCAST_ACT_SILU ? silu(g0[q]) : gelu_tanh(g0[q]);
            v1[q] *= p.act == MCAST_ACT_SILU ? silu(g1[q]) : gelu_tanh(g1[q]);
          }
          if (sbuf_busy) {
            if (lane == 0) bulk_wait_read<0>();
            __syncwarp();
          }
          stage_row32(sbuf, lane, 0, v0);
          stage_row32(sbuf, lane, 1, v1);
          asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
          __syncwarp();
          if (lane == 0) {
            tma_store_3d(&mapHst, sbuf, hcol + H * 128, row0, slot, pol_hs);
            bulk_commit();
          }
          sbuf_busy = 1;
        }
        if (p.sync) {
          if (lane == 0) {
            bulk_wait_all();
            asm volatile("fence.proxy.async.global;" ::: "memory");
            red_release_add(&p.flags[u.n], 1);
          }
          __syncwarp();
          sbuf_busy = 0;
        }
      } else {
        // residual: x[:, 512 n + H*256 + w2*128 + [0,128)] += acc (bf16 TMA reduce-add), 2 chunks of 64 per half
#pragma unroll 1
        for (int H = 0; H < 2; H++) {
          const uint32_t tc = t_row + H * 256 + hsel * 128;
          float v0[32], v1[32], v2[32], v3[32];
          tmem_ld32(tc, v0);
          tmem_ld32(tc + 32, v1);
          tmem_ld32(tc + 64, v2);
          tmem_ld32(tc + 96, v3);
          tmem_wait_ld();
          tc_fence_before();
          __syncwarp();
          if (lane == 0) mbar_arrive_cluster(tempty_leader + 8 * H);
          if (H == 1 && e == 0 && lane == 0) dbg_mark(p, i, 4);
#pragma unroll 1
          for (int c = 0; c < 2; c++) {
            if (sbuf_busy) {
              if (lane == 0) bulk_wait_read<0>();
              __syncwarp();
            }
            if (c == 0) {
              stage_row32(sbuf, lane, 0, v0);
              stage_row32(sbuf, lane, 1, v1);
            } else {
              stage_row32(sbuf, lane, 0, v2);
              stage_row32(sbuf, lane, 1, v3);
            }
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            __syncwarp();
            if (lane == 0) {
              tma_reduce_add_2d(&mapX, sbuf, u.n * BNP + H * 256 + hsel * 128 + c * 64, row0, pol_x);
              bulk_commit();
            }
            sbuf_busy = 1;
          }
        }
      }
      if (e == 0 && lane == 0) dbg_mark(p, i, 3);
      acc_phase ^= 1;
    }
    if (lane == 0) bulk_wait_all();
    __syncwarp();
  }

  tc_fence_before();
  cluster_sync();
  tc_fence_after();
  if (warp == 1) asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" ::"r"(tmem_base), "r"(512));
  if (p.hdiscard && p.sync) {
    // the ring now only holds chunks C-nslot..C-1, dead once their P7 loads are done: every CTA waits for that and
    // discards its 1/grid share (invalidate without write-back), so no dirty h lines outlive the kernel.
    if (threadIdx.x == 0) {
      for (int c = p.C - p.nslot; c < p.C; c++)
        if (c >= 0) spin_until_ge(&p.flags[FLAG_P7 + c], 12 * ((c == p.C - 1) ? p.Slast : p.S), p.spin_ns);
      __threadfence();
    }
    __syncthreads();
    const long long nl = p.hbytes >> 7, per = (nl + gridDim.x - 1) / gridDim.x;
    const long long l0 = per * blockIdx.x, l1 = (l0 + per < nl) ? l0 + per : nl;
    for (long long l = l0 + threadIdx.x; l < l1; l += NTHREADS)
      asm volatile("discard.global.L2 [%0], 128;" ::"l"(p.hbuf + (l << 7)) : "memory");
  }
  __shared__ int last_cta;
  if (threadIdx.x == 0) {
    __threadfence();
    const int old = atomicAdd(&p.flags[FLAG_EXIT], 1);
    last_cta = (old == (int)gridDim.x - 1);
  }
  __syncthreads();
  if (last_cta) {
    if (threadIdx.x == 0) {
      for (int f = 0; f < FLAG_EXIT; f++) p.flags[f] = 0;
      __threadfence();
      p.flags[FLAG_EXIT] = 0;
      __threadfence();
    }
  }
  if (p.stamp && threadIdx.x == 0) {
    const unsigned long long t = gtimer();
    if (blockIdx.x == 0) {
      p.stamp[1] = clock64();
      p.stamp[3] = t;
    }
    atomicMax(&p.stamp[4], t);  // kernel-wide end (all CTAs)
  }
}

// ------------------------------------------------------------------------------------------------ host side
// cuTensorMapEncodeTiled through the runtime's driver entry point (no libcuda link dependency).
using EncodeTiledFn = CUresult (*)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*,
                                   const cuuint64_t*, const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave,
                                   CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
static EncodeTiledFn EncodeTiled() {
  static EncodeTiledFn fn = [] {
    void* p = nullptr;
    cudaDriverEntryPointQueryResult q;
    cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &p, cudaEnableDefault, &q);
    return reinterpret_cast<EncodeTiledFn>(p);
  }();
  return fn;
}

static std::map<std::string, int> g_cfg;
static unsigned long long* g_stamp = nullptr;
static unsigned long long* g_dbg = nullptr;
static std::vector<int> g_sched_info;
static int cfg(const std::string& k, int dflt) {
  auto it = g_cfg.find(k);
  return it == g_cfg.end() ? dflt : it->second;
}

static bool encode(CUtensorMap* m, CUtensorMapDataType dt, int rank, const void* ptr, const cuuint64_t* dims,
                   const cuuint64_t* strides, const cuuint32_t* box) {
  cuuint32_t estr[3] = {1, 1, 1};
  CUresult r = EncodeTiled()(m, dt, rank, const_cast<void*>(ptr), dims, strides, box, estr,
                                      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                                      CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (r != CUDA_SUCCESS) {
    fprintf(stderr, "cuTensorMapEncodeTiled failed %d\n", (int)r);
    return false;
  }
  return true;
}
static bool map2d(CUtensorMap* m, const void* ptr, int rows, int cols, int box_rows, long ld = -1) {
  if (ld < 0) ld = cols;
  cuuint64_t dims[2] = {(cuuint64_t)cols, (cuuint64_t)rows};
  cuuint64_t strides[1] = {(cuuint64_t)ld * 2};
  cuuint32_t box[2] = {64, (cuuint32_t)box_rows};
  return encode(m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, ptr, dims, strides, box);
}
static bool map3d(CUtensorMap* m, const void* ptr, int slots, int rows, int cols, int box_rows) {
  cuuint64_t dims[3] = {(cuuint64_t)cols, (cuuint64_t)rows, (cuuint64_t)slots};
  cuuint64_t strides[2] = {(cuuint64_t)cols * 2, (cuuint64_t)rows * cols * 2};
  cuuint32_t box[3] = {64, (cuuint32_t)box_rows, 1};
  return encode(m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, ptr, dims, strides, box);
}

static uint16_t enc_p6(int c, int n) { return (uint16_t)((c << 11) | (n << 4)); }
static uint16_t enc_p7(int c, int nb, int s) { return (uint16_t)(0x8000 | (c << 11) | (s << 6) | (nb << 4)); }

struct Seg {
  std::vector<uint16_t> units;
  std::vector<int> kb;
};

static bool fill_queue(const std::vector<Seg>& segs, FParams& p) {
  p.nq = 0;
  long tot = 0;
  for (const Seg& sg : segs)
    for (size_t i = 0; i < sg.units.size(); i++) {
      if (p.nq >= MAXQ) return false;
      p.queue[p.nq++] = sg.units[i];
      tot += sg.kb[i];
    }
  g_sched_info.assign({p.nq, (int)tot});
  return true;
}

static int g_attr_set = 0;
constexpr int SMEM_BYTES = STAGES * 48 * 1024 + NEPI * NSTG * 4096 + 1024;
static_assert(SMEM_BYTES <= 232448, "smem");

static int launch(const CUtensorMap* maps, const FParams& p, cudaStream_t stream) {
  auto kern = ffn_kernel;
  if (!g_attr_set) {
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES);
    cudaFuncSetAttribute(kern, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    cudaFuncSetAttribute(kern, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared);
    g_attr_set = 1;
  }
  cudaLaunchConfig_t lc = {};
  lc.gridDim = dim3(18);
  lc.blockDim = dim3(NTHREADS);
  lc.dynamicSmemBytes = SMEM_BYTES;
  lc.stream = stream;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeClusterDimension;
  at[0].val.clusterDim.x = 6;
  at[0].val.clusterDim.y = 1;
  at[0].val.clusterDim.z = 1;
  lc.attrs = at;
  lc.numAttrs = 1;
  cudaError_t e = cudaLaunchKernelEx(&lc, kern, maps[0], maps[1], maps[2], maps[3], maps[4], maps[5], p);
  if (e != cudaSuccess) {
    fprintf(stderr, "ffn launch failed: %s\n", cudaGetErrorString(e));
    return -2;
  }
  return 0;
}

// offline (load-time) K-blocked weight layout: dst[(nb*KB + kb)*512 + r][c] = src[nb*512 + r][kb*64 + c]
__global__ void block_weight_kernel(const uint4* src, uint4* dst, int N, int K) {
  const size_t total = (size_t)N * K / 8;  // 16-B vectors
  const int KB = K / 64;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < total; i += (size_t)gridDim.x * blockDim.x) {
    const size_t e = i * 8;
    const int n = (int)(e / K), k = (int)(e % K);
    const int nb = n / 512, r = n % 512, kb = k / 64, c = k % 64;
    dst[((((size_t)nb * KB + kb) * 512 + r) * 64 + c) / 8] = src[i];
  }
}

}  // namespace mcast_ffn

using namespace mcast_ffn;

extern "C" int mcast_ffn_sm110_block_weight(const void* src, void* dst, int N, int K, cudaStream_t stream) {
  if (N % 512 || K % 64) return -3;
  block_weight_kernel<<<512, 256, 0, stream>>>((const uint4*)src, (uint4*)dst, N, K);
  return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

extern "C" int mcast_ffn_sm110_set_config(const char* key, int value) {
  g_cfg[key] = value;
  return 0;
}
static const std::map<std::string, int> kDefaults = {
    {"C", 8}, {"S", 1}, {"mode", 0}, {"nslot", 2}, {"pol_w", 1}, {"pol_a", 2}, {"pol_hs", 2}, {"pol_hl", 2},
    {"pol_x", 3}, {"spin_ns", 0}, {"wblk", 1}, {"win", 4}, {"pace", 1500}, {"slack", 0}, {"hdiscard", 1}, {"pgran", 0}};
// effective value of a knob (set value, else the default; -1 if unknown)
extern "C" int mcast_ffn_sm110_get_config(const char* key) {
  auto d = kDefaults.find(key);
  return cfg(key, d == kDefaults.end() ? -1 : d->second);
}
extern "C" int mcast_ffn_sm110_set_stamp(void* ptr) {
  g_stamp = reinterpret_cast<unsigned long long*>(ptr);
  return 0;
}
extern "C" int mcast_ffn_sm110_set_dbg(void* ptr) {
  g_dbg = reinterpret_cast<unsigned long long*>(ptr);
  return 0;
}
extern "C" size_t mcast_ffn_sm110_flags_bytes(void) { return NFLAGS * sizeof(int); }
// Chunk count for intermediate I (NB = I / 256 gate/up blocks, at most 64): the configured C (default 8) if it is in
// [2, 8] and divides NB, else the largest divisor of NB in [2, 8]; 0 for an I the launch rejects (NB outside [2, 64] or
// without such a divisor).
static int cur_C(int I) {
  const int nb = I / 256, want = cfg("C", 8);
  if (I % 256 || nb < 2 || nb > 64) return 0;
  if (want >= 2 && want <= 8 && nb % want == 0) return want;
  for (int c = 8; c >= 2; c--)
    if (nb % c == 0) return c;
  return 0;
}
static int cur_nslot(int I) {
  const int C = cur_C(I);
  const int ns = cfg("nslot", 2);
  return ns > C ? C : ns;
}
extern "C" size_t mcast_ffn_sm110_hbuf_bytes_ex(int M, int I) {
  const int C = cur_C(I);
  return C ? (size_t)cur_nslot(I) * M * (I / C) * 2 : 0;
}
extern "C" size_t mcast_ffn_sm110_hbuf_bytes(int M) { return mcast_ffn_sm110_hbuf_bytes_ex(M, 16384); }
extern "C" int mcast_ffn_sm110_schedule_info(int* out, int n) {
  int k = 0;
  for (; k < n && k < (int)g_sched_info.size(); k++) out[k] = g_sched_info[k];
  return k;
}

// mode 0 (default): one fused persistent kernel per layer, segments P6(0) | P6(1) P7(0) | ... | P7(C-1)
// mode 1: one kernel per segment (lag-1 pipeline, ordering by kernel boundaries, no flags)
// mode 2: one kernel per phase: P6(0) P7(0) P6(1) P7(1) ... (2C kernels, no flags)
static const int* g_pace_dev = nullptr;
static const void* g_pf_ptr[2] = {nullptr, nullptr};
static long long g_pf_bytes[2] = {0, 0};
extern "C" int mcast_ffn_sm110_set_pace_ptr(const int* pace_dev) { g_pace_dev = pace_dev; return 0; }
// next launch only: L2-prefetch [p0, p0+b0) and [p1, p1+b1) during the last (1 - frac_q0/1000) of the unit queue
static int g_pf_q0_milli = 500;
extern "C" int mcast_ffn_sm110_set_prefetch(const void* p0, long long b0, const void* p1, long long b1, int q0_milli) {
  g_pf_ptr[0] = p0; g_pf_bytes[0] = p0 ? b0 : 0; g_pf_ptr[1] = p1; g_pf_bytes[1] = p1 ? b1 : 0; g_pf_q0_milli = q0_milli;
  return 0;
}
extern "C" int mcast_ffn_sm110_ex(const void* xn, const void* w_gu_il, const void* w_d, void* x, void* hbuf, int* flags,
                                  int M, int I, int act, cudaStream_t stream) {
  if (M <= 512 || M > 768) return -5;
  const int C = cur_C(I);
  if (C == 0 || (act != MCAST_ACT_SILU && act != MCAST_ACT_GELU_TANH)) return -3;
  const int S = cfg("S", 1), Slast = cfg("slast", S);
  const int mode = cfg("mode", 0);
  const int nslot = cur_nslot(I);
  FParams p;
  memset(&p, 0, sizeof(p));
  p.M = M;
  p.C = C;
  p.F = I / C;
  p.KBc = p.F / BK;
  p.KBt = I / BK;
  p.act = act;
  p.S = S;
  p.Slast = Slast;
  p.nbpc = I / 256 / C;
  p.nslot = nslot;
  p.pol_w = cfg("pol_w", 1);
  p.pol_a = cfg("pol_a", 2);
  p.pol_hs = cfg("pol_hs", 2);
  p.pol_hl = cfg("pol_hl", 2);
  p.pol_x = cfg("pol_x", 3);
  p.spin_ns = cfg("spin_ns", 0);
  p.wblk = cfg("wblk", 1);
  p.hdiscard = cfg("hdiscard", 1);
  p.hbuf = (char*)hbuf;
  p.hbytes = (long long)nslot * M * p.F * 2 & ~127ll;
  p.win = cfg("win", STAGES);
  p.pace = cfg("pace", 1500);
  p.slack = cfg("slack", 0);
  if (p.win > STAGES) p.win = STAGES;
  p.flags = flags;
  p.pace_dev = g_pace_dev;
  p.pgran = cfg("pgran", 0);
  for (int r = 0; r < 2; r++) { p.pf_ptr[r] = (const char*)g_pf_ptr[r]; p.pf_bytes[r] = g_pf_bytes[r]; }
  g_pf_ptr[0] = g_pf_ptr[1] = nullptr; g_pf_bytes[0] = g_pf_bytes[1] = 0;
  p.stamp = g_stamp;
  p.dbg = g_dbg;
  if (S < 1 || S > 31 || Slast < 1 || Slast > 31 || p.KBc / (S > Slast ? S : Slast) < 1) return -3;
  if (nslot < 2 && mode == 0 && C > 1) return -3;

  CUtensorMap maps[6];
  if (!map2d(&maps[0], xn, M, 2048, 128) || !map3d(&maps[1], hbuf, nslot, M, p.F, 128) ||
      !(p.wblk ? map2d(&maps[2], w_gu_il, 2 * I * 32, 64, 64) : map2d(&maps[2], w_gu_il, 2 * I, 2048, 64)) ||
      !(p.wblk ? map2d(&maps[3], w_d, 2048 * p.KBt, 64, 64) : map2d(&maps[3], w_d, 2048, I, 64)) ||
      !map3d(&maps[4], hbuf, nslot, M, p.F, 32) || !map2d(&maps[5], x, M, 2048, 32))
    return -1;

  auto p6seg = [&](int c, Seg& s) {
    for (int j = 0; j < p.nbpc; j++) {
      s.units.push_back(enc_p6(c, c * p.nbpc + j));
      s.kb.push_back(2048 / BK);
    }
  };
  auto p7seg = [&](int c, Seg& s) {
    const int SS = (c == C - 1) ? Slast : S;
    for (int sp = 0; sp < SS; sp++)
      for (int nb = 0; nb < 4; nb++) {
        s.units.push_back(enc_p7(c, nb, sp));
        s.kb.push_back(((sp + 1) * p.KBc) / SS - (sp * p.KBc) / SS);
      }
  };
  std::vector<Seg> segs;
  if (mode == 2) {
    for (int c = 0; c < C; c++) {
      Seg a, b;
      p6seg(c, a);
      p7seg(c, b);
      segs.push_back(a);
      segs.push_back(b);
    }
  } else {
    for (int c = 0; c <= C; c++) {
      Seg s;
      if (c < C) p6seg(c, s);
      if (c > 0) p7seg(c - 1, s);
      segs.push_back(s);
    }
  }
  if (mode == 0) {
    p.sync = 1;
    if (!fill_queue(segs, p)) return -6;
    p.pf_q0 = (int)((long long)p.nq * g_pf_q0_milli / 1000);
    return launch(maps, p, stream);
  }
  p.sync = 0;
  for (size_t i = 0; i < segs.size(); i++) {
    std::vector<Seg> one(1, segs[i]);
    if (!fill_queue(one, p)) return -6;
    int r = launch(maps, p, stream);
    if (r) return r;
  }
  return 0;
}
extern "C" int mcast_ffn_sm110(const void* xn, const void* w_gu_il, const void* w_d, void* x, void* hbuf, int* flags, int M,
                               cudaStream_t stream) {
  return mcast_ffn_sm110_ex(xn, w_gu_il, w_d, x, hbuf, flags, M, 16384, MCAST_ACT_GELU_TANH, stream);
}
