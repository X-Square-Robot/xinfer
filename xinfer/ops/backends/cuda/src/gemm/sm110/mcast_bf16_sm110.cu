// Ported from an earlier standalone Thor GEMM package (prefill_gemm/pg_kernels.cu). Only the GEMM family calls these entry
// points (mcast_sm110_ffi.cc). Changes: names, the RoPE table is FP32 [M][128] (row stride 128),
// tensor maps are encoded through cudaGetDriverEntryPoint.
// prefill_gemm kernels: persistent, warp-specialized tcgen05 (UMMA) BF16 GEMM with fused epilogues.
// Target: NVIDIA Thor (sm_110a), 20 SMs.  Build: see build.sh.
//
// Two mainloop variants (same epilogues):
//  * gemm1: 1 CTA/tile, UMMA 128xBN (cta_group::1).
//  * gemm2: CTA pair (cluster 2), UMMA 256x256 (cta_group::2); a pair tile is 256 x BNP (BNP = 256 or 512, i.e.
//    1 or 2 UMMAs per K step).  Each CTA loads half of A (128 rows) and half of every B block (128 rows) and the
//    leader CTA issues the MMAs, so per-pair L2->SM traffic per tile is halved vs. two independent CTAs.  This is
//    the main lever: on Thor the L2 (not DRAM) saturates first with 128x256 tiles at M=712.
// CTA = 192 threads: warp0 = TMA producer, warp1 = TMEM alloc + MMA issuer, warps2..5 = epilogue (128 thr).
// TMEM: 512 fp32 columns; double-buffered accumulators when the tile needs <= 256 columns, so the epilogue of tile
// i overlaps the mainloop of tile i+1 (for BNP=512 the smem ring keeps prefetching during the epilogue).
// Static persistent schedule; unit order (n, split, m) with m fastest so all M tiles of one weight block run at the
// same time (weights come from DRAM once, the other M tiles hit L2).  Optional split-K: partial fp32 tiles go to a
// workspace; the last arriving split (atomic counter) reduces in fixed split order (deterministic) + epilogue.
#include "mcast_bf16_sm110.h"
#include <cuda.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <string>
#include <map>

namespace mcast_bf16_sm110 {

typedef __nv_bfloat16 bf16;

enum { EPI_QKV_ROPE = 0, EPI_RESID = 1, EPI_GEGLU = 2, EPI_SWIGLU = 3, EPI_OUT = 4 };
template <int EPI> constexpr bool kGlu = EPI == EPI_GEGLU || EPI == EPI_SWIGLU;

constexpr int BK = 64;
constexpr int NTHREADS = 192;

struct Params {
  int M, N, K;
  int num_m, num_n, splits, num_units, kblocks;
  int policy_w;
  int order;
  int exact_resid;
  int apf;              // activation rows of a unit's K slab prefetched to L2 as contiguous runs at unit start
  const bf16* a_ptr;    // (apf) the activation, row-major [M][K]
  int wkblk;            // weights K-blocked (prepack layout nk_k64blk): [N/512][K/64][512][64], 64-wide boxes contiguous
  int pdl;              // launched with programmatic dependent launch: prologue overlaps the previous kernel
  int pf_first;
  int pd;               // rolling L2 prefetch distance in k-blocks (0 = off): keeps DRAM requests in flight         // bytes of weights (from the first unit's N block) to L2-prefetch before the PDL wait
  const void* pf_ptr;   // optional: whole-weight L2 prefetch at kernel start (weights that fit in L2: P2/P4)
  long long pf_bytes;  // RESID, no split: x prefetched by TMA at unit start, x = bf16(x + acc) (single rounding)   // L2 cache-hint policy for weights: 0 none, 1 evict_first, 2 evict_last, 3 evict_normal
  bf16* out;      // RESID: x (in/out, ld = N); GEGLU: h (ld = N/2)
  const bf16* bias;       // OUT: [N] or null
  const float* row_scale; // OUT: [M] or null
  int act;                // OUT: MCAST_OUT_ACT_NONE or MCAST_OUT_ACT_GELU_TANH
  const float* cos_t;
  const float* sin_t;
  bf16* q;
  bf16* kc;
  bf16* vc;
  float* ws;      // split-K partials
  int* counters;  // split-K arrival counters (one per 128-row output tile)
  unsigned long long* dbg;    // optional per-unit trace [cta][64][2] globaltimer (epilogue start/end)
  unsigned long long* stamp;  // optional: [clock64 start, end, globaltimer start, end] of CTA 0
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
__device__ __forceinline__ void cluster_sync() {
  asm volatile("barrier.cluster.arrive.release.aligned;\n\tbarrier.cluster.wait.acquire.aligned;" ::: "memory");
}
__device__ __forceinline__ void mbar_init(uint64_t* bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_u32(bar)), "r"(count));
}
__device__ __forceinline__ void mbar_expect_tx(uint64_t* bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem_u32(bar)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint64_t* bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(smem_u32(bar)) : "memory");
}
__device__ __forceinline__ void mbar_arrive_cluster(uint32_t cluster_addr) {
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
  // acquire at cluster scope (arrivals come from the peer CTA)
  uint32_t addr = smem_u32(bar);
  asm volatile(
      "{\n\t.reg .pred p;\n\t"
      "WAIT_%=:\n\t"
      "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%0], %1;\n\t"
      "@!p bra WAIT_%=;\n\t}" ::"r"(addr),
      "r"(parity)
      : "memory");
}
// 1-CTA TMA load, mbarrier in own CTA
__device__ __forceinline__ void tma_load_2d(void* dst, const CUtensorMap* map, uint64_t* bar, int c0, int c1,
                                            uint64_t policy) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {%3, "
      "%4}], [%2], %5;" ::"r"(smem_u32(dst)),
      "l"((uint64_t)map), "r"(smem_u32(bar)), "r"(c0), "r"(c1), "l"(policy)
      : "memory");
}
// 2-CTA TMA load: data to own smem, completion signalled on the leader CTA's mbarrier (cluster address)
__device__ __forceinline__ void tma_load_2d_2sm(void* dst, const CUtensorMap* map, uint32_t bar_cluster, int c0,
                                                int c1, uint64_t policy) {
  asm volatile(
      "cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], "
      "[%1, {%3, %4}], [%2], %5;" ::"r"(smem_u32(dst)),
      "l"((uint64_t)map), "r"(bar_cluster), "r"(c0), "r"(c1), "l"(policy)
      : "memory");
}
__device__ __forceinline__ uint64_t make_policy(int sel) {
  uint64_t p;
  if (sel == 1)
    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(p));
  else if (sel == 2)
    asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(p));
  else
    asm volatile("createpolicy.fractional.L2::evict_normal.b64 %0, 1.0;" : "=l"(p));
  return p;
}

__device__ __forceinline__ void tc_fence_before() { asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory"); }
__device__ __forceinline__ void tc_fence_after() { asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory"); }

// K-major operand tile written by TMA with SWIZZLE_128B: rows of 128 B, 8-row core groups of 1024 B.
__device__ __forceinline__ uint64_t make_sdesc(uint32_t addr) {
  uint64_t d = 0;
  d |= (uint64_t)((addr >> 4) & 0x3FFF);
  d |= (uint64_t)1 << 16;             // LBO (ignored for swizzled K-major)
  d |= (uint64_t)(1024 >> 4) << 32;   // SBO = 1024 B
  d |= (uint64_t)1 << 46;             // descriptor version (sm100)
  d |= (uint64_t)2 << 61;             // SWIZZLE_128B
  return d;
}

__host__ __device__ constexpr uint32_t make_idesc(int M, int N) {
  // D=F32, A=B=BF16, K-major both, N>>3 @17, M>>4 @24
  return (1u << 4) | (1u << 7) | (1u << 10) | ((uint32_t)(N >> 3) << 17) | ((uint32_t)(M >> 4) << 24);
}

__device__ __forceinline__ void umma_1sm(uint32_t tmem_d, uint64_t da, uint64_t db, uint32_t idesc, uint32_t acc) {
  asm volatile(
      "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t}\n" ::"r"(tmem_d),
      "l"(da), "l"(db), "r"(idesc), "r"(acc));
}
__device__ __forceinline__ void umma_2sm(uint32_t tmem_d, uint64_t da, uint64_t db, uint32_t idesc, uint32_t acc) {
  asm volatile(
      "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
      "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p;\n\t}\n" ::"r"(tmem_d),
      "l"(da), "l"(db), "r"(idesc), "r"(acc));
}
__device__ __forceinline__ void umma_commit_1sm(uint64_t* bar) {
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" ::"r"(smem_u32(bar))
               : "memory");
}
__device__ __forceinline__ void umma_commit_2sm_mc(uint64_t* bar) {
  // arrive on the barrier at the same smem offset in both CTAs of the pair
  asm volatile(
      "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;" ::"r"(
          smem_u32(bar)),
      "h"((uint16_t)3)
      : "memory");
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

#ifndef PG_FENCE_MODE
#define PG_FENCE_MODE 0
#endif
__device__ __forceinline__ void PG_FENCE() {
#if PG_FENCE_MODE == 0
  __threadfence();
#elif PG_FENCE_MODE == 1
  asm volatile("fence.acq_rel.gpu;" ::: "memory");
#else
#endif
}
__device__ __forceinline__ void epi_bar() { asm volatile("bar.sync 1, 128;" ::: "memory"); }

__device__ __forceinline__ uint32_t pack_bf16(float a, float b) {
  __nv_bfloat162 h = __floats2bfloat162_rn(a, b);
  return *reinterpret_cast<uint32_t*>(&h);
}
// store 32 floats as bf16 (64 B, 32B-aligned) with two 256-bit stores (full sectors per instruction)
__device__ __forceinline__ void store32(bf16* dst, const float (&v)[32]) {
  uint32_t u[16];
#pragma unroll
  for (int i = 0; i < 16; i++) u[i] = pack_bf16(v[2 * i], v[2 * i + 1]);
#pragma unroll
  for (int h = 0; h < 2; h++) {
    asm volatile("st.global.v8.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8};" ::"l"(dst + 16 * h), "r"(u[8 * h + 0]),
                 "r"(u[8 * h + 1]), "r"(u[8 * h + 2]), "r"(u[8 * h + 3]), "r"(u[8 * h + 4]), "r"(u[8 * h + 5]),
                 "r"(u[8 * h + 6]), "r"(u[8 * h + 7])
                 : "memory");
  }
}
__device__ __forceinline__ void add_bf16x32(float (&v)[32], const bf16* src) {
#pragma unroll
  for (int h = 0; h < 2; h++) {
    uint32_t u[8];
    asm volatile("ld.global.v8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                 : "=r"(u[0]), "=r"(u[1]), "=r"(u[2]), "=r"(u[3]), "=r"(u[4]), "=r"(u[5]), "=r"(u[6]), "=r"(u[7])
                 : "l"(src + 16 * h));
#pragma unroll
    for (int j = 0; j < 8; j++) {
      float2 f = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&u[j]));
      v[16 * h + 2 * j] += f.x;
      v[16 * h + 2 * j + 1] += f.y;
    }
  }
}
__device__ __forceinline__ float tanh_fast(float x) {
  float y;
  asm("tanh.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
__device__ __forceinline__ float gelu_tanh(float x) {
  const float k0 = 0.7978845608028654f, k1 = 0.044715f;
  float u = k0 * fmaf(k1 * x * x, x, x);
  float hx = 0.5f * x;
  return fmaf(hx, tanh_fast(u), hx);
}
__device__ __forceinline__ float silu(float x) { return x / (1.0f + __expf(-x)); }

// ------------------------------------------------------------------------------------------------ epilogue
// One thread = one output row (TMEM lane), BN accumulator columns starting at TMEM address t_row.
// tile_id identifies the 128-row output tile (split-K bookkeeping).
// ---- TMA store helpers (epilogue).  Per-warp staging ring of NSTG buffers, each [32 rows][64 bf16] = 4 KB in the
// TMA SWIZZLE_128B layout (16-B chunk j of row r at position j ^ (r & 7)) -> conflict-free row-wise smem writes.
__device__ __forceinline__ void tma_store_2d(const CUtensorMap* map, const void* src, int c0, int c1) {
  asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%2, %3}], [%1];" ::"l"((uint64_t)map),
               "r"(smem_u32(src)), "r"(c0), "r"(c1)
               : "memory");
}
__device__ __forceinline__ void tma_reduce_add_2d(const CUtensorMap* map, const void* src, int c0, int c1) {
  asm volatile("cp.reduce.async.bulk.tensor.2d.global.shared::cta.add.bulk_group [%0, {%2, %3}], [%1];" ::"l"(
                   (uint64_t)map),
               "r"(smem_u32(src)), "r"(c0), "r"(c1)
               : "memory");
}
__device__ __forceinline__ void bulk_commit() { asm volatile("cp.async.bulk.commit_group;" ::: "memory"); }
template <int N>
__device__ __forceinline__ void bulk_wait_read() {
  asm volatile("cp.async.bulk.wait_group.read %0;" ::"n"(N) : "memory");
}
__device__ __forceinline__ void bulk_wait_all() { asm volatile("cp.async.bulk.wait_group 0;" ::: "memory"); }

struct OutMaps {
  CUtensorMap m[3];  // P2: q [M][2048], k [M][256], v [M][256];  P6: h [M][16384];  P4/P7: x [M][2048]
};

template <int NSTG>
struct StageRing {
  uint8_t* base;  // NSTG * 4096 bytes, 1024-aligned
  int idx;        // next buffer
  int issued;     // groups committed so far (saturates at NSTG)
  // acquire next buffer: make sure the TMA that last read it has finished reading
  __device__ __forceinline__ uint8_t* acquire(int lane) {
    if (issued >= NSTG) {
      if (lane == 0) bulk_wait_read<NSTG - 1>();
      __syncwarp();
    }
    return base + idx * 4096;
  }
  __device__ __forceinline__ void release_issue() {
    idx = (idx + 1 == NSTG) ? 0 : idx + 1;
    if (issued < NSTG) issued++;
  }
};

// write 32 fp32 (as bf16) of this lane's row into half `h` (cols h*32..h*32+31) of a 64-col staging buffer
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
// all lanes: make smem writes visible to the async proxy; lane 0 issues the store (or reduce-add) + commit
template <bool REDUCE>
__device__ __forceinline__ void stage_flush(const CUtensorMap* map, const uint8_t* buf, int col, int row0, int lane) {
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncwarp();
  if (lane == 0) {
    if (REDUCE)
      tma_reduce_add_2d(map, buf, col, row0);
    else
      tma_store_2d(map, buf, col, row0);
    bulk_commit();
  }
}

// One thread = one output row (TMEM lane); BN accumulator columns at TMEM address t_row.  row0 = first row of
// this warp's 32-row slice.  Outputs leave through TMA (rows >= M are clipped by the tensor map).
// RESID: x += acc via TMA reduce-add (bf16), which also implements split-K (each split adds its partial).
struct EpiCtx {
  int cid;          // split-K counter id (one per 32-row warp slice of an output tile)
  uint64_t* ebar;   // this warp's mbarrier for epilogue TMA loads
  uint32_t ephase;  // its phase
  uint32_t tempty;  // accumulator-free barrier: cluster address (cluster kernels) or 0 -> local pointer below
  uint64_t* tempty_local;
  bool released;
  int ephase_q;          // QKV BN=128: staging buffer-pair toggle
  float cs[32], sn[32];  // QKV: prefetched cos/sin (first half; cos[:, i] == cos[:, i+128]) for the next chunk
  // tell the MMA warp this warp is done reading the TMEM accumulator (call after the last tcgen05.ld/wait)
  __device__ __forceinline__ void release_tmem() {
    if (released) return;
    tc_fence_before();
    __syncwarp();
    if ((threadIdx.x & 31) == 0) {
      if (tempty)
        mbar_arrive_cluster(tempty);
      else
        mbar_arrive(tempty_local);
    }
    released = true;
  }
};

__device__ __forceinline__ void stage_row32_f32(uint8_t* buf, int lane, const float (&v)[32]) {
  uint8_t* rowp = buf + lane * 128;
#pragma unroll
  for (int j = 0; j < 8; j++)
    *reinterpret_cast<float4*>(rowp + ((j ^ (lane & 7)) << 4)) =
        make_float4(v[4 * j], v[4 * j + 1], v[4 * j + 2], v[4 * j + 3]);
}

__device__ __forceinline__ void rope_load(const Params& p, int row, int col, EpiCtx& ec) {
  row = row < p.M ? row : p.M - 1;
  const float4* cp = reinterpret_cast<const float4*>(p.cos_t + (size_t)row * 128 + col);
  const float4* sp4 = reinterpret_cast<const float4*>(p.sin_t + (size_t)row * 128 + col);
#pragma unroll
  for (int i = 0; i < 8; i++) {
    const float4 a = __ldg(cp + i), b = __ldg(sp4 + i);
    ec.cs[4 * i] = a.x; ec.cs[4 * i + 1] = a.y; ec.cs[4 * i + 2] = a.z; ec.cs[4 * i + 3] = a.w;
    ec.sn[4 * i] = b.x; ec.sn[4 * i + 1] = b.y; ec.sn[4 * i + 2] = b.z; ec.sn[4 * i + 3] = b.w;
  }
}
template <int EPI, int BN>
__device__ __forceinline__ void rope_prefetch(const Params& p, int row0, int n_blk, EpiCtx& ec) {
  if constexpr (EPI == EPI_QKV_ROPE) {
    if (BN == 256 && n_blk < 9) rope_load(p, row0 + (threadIdx.x & 31), 0, ec);
    if (BN == 128 && n_blk < 18) rope_load(p, row0 + (threadIdx.x & 31), 64 * (n_blk & 1), ec);
  }
}

// RESID exact (no split, BN == 256, NSTG == 4): prefetch this warp's x slice [32 rows][256] into the 4 ring
// buffers before waiting for the accumulator, so the load latency hides behind the mainloop.
template <int BN, int EPI, int NSTG>
__device__ __forceinline__ void resid_prefetch(const Params& p, const OutMaps& om, StageRing<NSTG>& ring, int row0,
                                               int n_blk, EpiCtx& ec) {
  if constexpr (EPI == EPI_RESID && BN <= 256 && NSTG == 4) {
    const int lane = threadIdx.x & 31;
    if (p.exact_resid && p.splits == 1) {
      if (lane == 0) {
        bulk_wait_read<0>();
        mbar_expect_tx(ec.ebar, (BN / 64) * 4096);
        const uint64_t pol = make_policy(3);
#pragma unroll
        for (int c = 0; c < BN / 64; c++) tma_load_2d(ring.base + c * 4096, &om.m[0], ec.ebar, n_blk * BN + c * 64, row0, pol);
      }
      __syncwarp();
    }
  }
}

template <int BN, int EPI, int NSTG>
__device__ __forceinline__ void run_epilogue(const Params& p, const OutMaps& om, StageRing<NSTG>& ring,
                                             uint32_t t_row, int row0, int n_blk, EpiCtx& ec) {
  const int lane = threadIdx.x & 31;
  if constexpr (EPI == EPI_RESID && NSTG >= 4) {
    if (p.splits > 1 && p.ws != nullptr) {
      // ---- exact split-K: fp32 partial -> TMA reduce-add into fp32 workspace (om.m[1]); the last split of this
      // 32-row slice (atomic counter) computes x = bf16(x + ws) once and re-zeroes the workspace slice.
#pragma unroll 1
      for (int c = 0; c < BN / 32; c++) {
        float v[32];
        tmem_ld32(t_row + c * 32, v);
        tmem_wait_ld();
        uint8_t* buf = ring.acquire(lane);
        stage_row32_f32(buf, lane, v);
        stage_flush<true>(&om.m[1], buf, n_blk * BN + c * 32, row0, lane);
        ring.release_issue();
      }
      ec.release_tmem();
      int last = 0;
      if (lane == 0) {
        bulk_wait_all();
        asm volatile("fence.proxy.async.global;" ::: "memory");
        __threadfence();
        const int old = atomicAdd(&p.counters[ec.cid], 1);
        last = (old == p.splits - 1);
        if (last) {
          p.counters[ec.cid] = 0;
          __threadfence();
          asm volatile("fence.proxy.async.global;" ::: "memory");
        }
      }
      last = __shfl_sync(0xffffffffu, last, 0);
      if (!last) return;
      // all previous groups are complete (bulk_wait_all above) -> the ring buffers are free
      uint8_t* bw0 = ring.base;
      uint8_t* bw1 = ring.base + 4096;
      uint8_t* bx = ring.base + 8192;
      uint8_t* bz = ring.base + 12288;
#pragma unroll
      for (int j = 0; j < 8; j++) *reinterpret_cast<float4*>(bz + lane * 128 + j * 16) = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll 1
      for (int c = 0; c < BN / 64; c++) {
        const int col = n_blk * BN + c * 64;
        if (c > 0) {  // previous chunk's stores must have finished reading bx before we overwrite it
          if (lane == 0) bulk_wait_read<0>();
          __syncwarp();
        }
        if (lane == 0) {
          mbar_expect_tx(ec.ebar, 3 * 4096);
          tma_load_2d(bw0, &om.m[1], ec.ebar, col, row0, make_policy(1));
          tma_load_2d(bw1, &om.m[1], ec.ebar, col + 32, row0, make_policy(1));
          tma_load_2d(bx, &om.m[0], ec.ebar, col, row0, make_policy(3));
        }
        mbar_wait(ec.ebar, ec.ephase);
        ec.ephase ^= 1;
        // lane = row: 64 values; x chunk j (8 bf16) at (j ^ (lane&7)), ws chunk j (4 fp32) at (j ^ (lane&7))
#pragma unroll
        for (int j = 0; j < 8; j++) {
          uint4* xp = reinterpret_cast<uint4*>(bx + lane * 128 + ((j ^ (lane & 7)) << 4));
          uint4 xu = *xp;
          const int wj0 = 2 * j, wj1 = 2 * j + 1;  // fp32 16-B chunks covering these 8 columns
          const uint8_t* wb = (wj0 < 8) ? bw0 : bw1;
          const float4 w0 = *reinterpret_cast<const float4*>(wb + lane * 128 + (((wj0 & 7) ^ (lane & 7)) << 4));
          const float4 w1 = *reinterpret_cast<const float4*>(wb + lane * 128 + (((wj1 & 7) ^ (lane & 7)) << 4));
          __nv_bfloat162* xh = reinterpret_cast<__nv_bfloat162*>(&xu);
          float2 f0 = __bfloat1622float2(xh[0]), f1 = __bfloat1622float2(xh[1]);
          float2 f2 = __bfloat1622float2(xh[2]), f3 = __bfloat1622float2(xh[3]);
          xu.x = pack_bf16(f0.x + w0.x, f0.y + w0.y);
          xu.y = pack_bf16(f1.x + w0.z, f1.y + w0.w);
          xu.z = pack_bf16(f2.x + w1.x, f2.y + w1.y);
          xu.w = pack_bf16(f3.x + w1.z, f3.y + w1.w);
          *xp = xu;
        }
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        __syncwarp();
        if (lane == 0) {
          tma_store_2d(&om.m[0], bx, col, row0);
          tma_store_2d(&om.m[1], bz, col, row0);
          tma_store_2d(&om.m[1], bz, col + 32, row0);
          bulk_commit();
        }
      }
      if (lane == 0) bulk_wait_read<0>();
      __syncwarp();
      ring.issued = NSTG;  // ring state: everything may still be in flight -> force waits
      return;
    }
  }
  if constexpr (EPI == EPI_RESID && BN <= 256 && NSTG == 4) {
    if (p.exact_resid && p.splits == 1) {
      mbar_wait(ec.ebar, ec.ephase);
      ec.ephase ^= 1;
#pragma unroll 1
      for (int c = 0; c < BN / 64; c++) {
        uint8_t* bx = ring.base + c * 4096;
#pragma unroll 1
        for (int h = 0; h < 2; h++) {
          float v[32];
          tmem_ld32(t_row + c * 64 + h * 32, v);
          tmem_wait_ld();
#pragma unroll
          for (int q = 0; q < 4; q++) {
            uint4* xp = reinterpret_cast<uint4*>(bx + lane * 128 + (((h * 4 + q) ^ (lane & 7)) << 4));
            uint4 xu = *xp;
            __nv_bfloat162* xh = reinterpret_cast<__nv_bfloat162*>(&xu);
            float2 f0 = __bfloat1622float2(xh[0]), f1 = __bfloat1622float2(xh[1]);
            float2 f2 = __bfloat1622float2(xh[2]), f3 = __bfloat1622float2(xh[3]);
            xu.x = pack_bf16(f0.x + v[8 * q + 0], f0.y + v[8 * q + 1]);
            xu.y = pack_bf16(f1.x + v[8 * q + 2], f1.y + v[8 * q + 3]);
            xu.z = pack_bf16(f2.x + v[8 * q + 4], f2.y + v[8 * q + 5]);
            xu.w = pack_bf16(f3.x + v[8 * q + 6], f3.y + v[8 * q + 7]);
            *xp = xu;
          }
        }
        stage_flush<false>(&om.m[0], bx, n_blk * BN + c * 64, row0, lane);
      }
      ring.issued = NSTG;
      return;
    }
  }
  if constexpr (EPI == EPI_RESID) {
#pragma unroll 1
    for (int c = 0; c < BN / 64; c++) {
      float v0[32], v1[32];
      tmem_ld32(t_row + c * 64, v0);
      tmem_ld32(t_row + c * 64 + 32, v1);
      tmem_wait_ld();
      uint8_t* buf = ring.acquire(lane);
      stage_row32(buf, lane, 0, v0);
      stage_row32(buf, lane, 1, v1);
      stage_flush<true>(&om.m[0], buf, n_blk * BN + c * 64, row0, lane);
      ring.release_issue();
    }
  } else if constexpr (EPI == EPI_OUT) {
    // out = act(row_scale[row] * acc + bias) stored as BF16 (rows >= M and columns >= N are clipped by the map)
    const int row = row0 + lane;
    const float rs = (p.row_scale != nullptr && row < p.M) ? __ldg(p.row_scale + row) : 1.0f;
#pragma unroll 1
    for (int c = 0; c < BN / 64; c++) {
      const int col = n_blk * BN + c * 64;
      float v[2][32];
      tmem_ld32(t_row + c * 64, v[0]);
      tmem_ld32(t_row + c * 64 + 32, v[1]);
      tmem_wait_ld();
#pragma unroll
      for (int h = 0; h < 2; h++) {
#pragma unroll
        for (int i = 0; i < 32; i++) {
          const int n = col + h * 32 + i;
          float y = v[h][i] * rs;
          if (p.bias != nullptr && n < p.N) y += __bfloat162float(p.bias[n]);
          v[h][i] = p.act == MCAST_OUT_ACT_GELU_TANH ? gelu_tanh(y) : y;
        }
      }
      uint8_t* buf = ring.acquire(lane);
      stage_row32(buf, lane, 0, v[0]);
      stage_row32(buf, lane, 1, v[1]);
      stage_flush<false>(&om.m[0], buf, col, row0, lane);
      ring.release_issue();
    }
  } else if constexpr (kGlu<EPI>) {
    // Each 256-col sub-tile s: cols [0,128) gate, [128,256) up, for h cols n_blk*BN/2 + s*128 + j;
    // h = act(gate) * up with GELU-tanh (EPI_GEGLU) or SiLU (EPI_SWIGLU) on the FP32 accumulators
    static_assert(BN % 256 == 0, "GLU tile must be a multiple of 256");
#pragma unroll 1
    for (int c = 0; c < BN / 128; c++) {  // 64 output cols per step
      const int s = c >> 1, cc = c & 1;
      uint8_t* buf = ring.acquire(lane);
#pragma unroll 1
      for (int h = 0; h < 2; h++) {
        float g[32], v[32];
        tmem_ld32(t_row + s * 256 + cc * 64 + h * 32, g);
        tmem_ld32(t_row + s * 256 + 128 + cc * 64 + h * 32, v);
        tmem_wait_ld();
#pragma unroll
        for (int i = 0; i < 32; i++) v[i] *= EPI == EPI_SWIGLU ? silu(g[i]) : gelu_tanh(g[i]);
        stage_row32(buf, lane, h, v);
      }
      stage_flush<false>(&om.m[0], buf, n_blk * (BN / 2) + s * 128 + cc * 64, row0, lane);
      ring.release_issue();
    }
  } else if constexpr (BN == 128) {  // EPI_QKV_ROPE, half-head tiles (weights permuted, see pg_permute_qk_rows)
    static_assert(NSTG == 4, "QKV epilogue needs 4 staging buffers");
    // tile t of head hd = n_blk/2 holds dims [64t, 64t+64) in cols [0,64) and [128+64t, ...) in cols [64,128)
    const int hd = n_blk >> 1, t = n_blk & 1;
    if (hd < 9) {
      const CUtensorMap* map = hd < 8 ? &om.m[0] : &om.m[1];
      const int colbase = (hd < 8 ? hd * 256 : 0) + 64 * t;
      if (lane == 0) bulk_wait_read<2>();
      __syncwarp();
      const int pb = (ec.ephase_q ^= 1);  // alternate buffer pairs (0,1) / (2,3)
      uint8_t* blo = ring.base + (2 * pb) * 4096;
      uint8_t* bhi = ring.base + (2 * pb + 1) * 4096;
#pragma unroll 1
      for (int h = 0; h < 2; h++) {
        float cs[32], sn[32];
#pragma unroll
        for (int i = 0; i < 32; i++) { cs[i] = ec.cs[i]; sn[i] = ec.sn[i]; }
        if (h == 0) rope_load(p, row0 + lane, 64 * t + 32, ec);
        float lo[32], hi[32];
        tmem_ld32(t_row + h * 32, lo);
        tmem_ld32(t_row + 64 + h * 32, hi);
        tmem_wait_ld();
        if (h == 1) ec.release_tmem();
#pragma unroll
        for (int j = 0; j < 32; j++) {
          const float a = lo[j], b = hi[j];
          lo[j] = a * cs[j] - b * sn[j];
          hi[j] = b * cs[j] + a * sn[j];
        }
        stage_row32(blo, lane, h, lo);
        stage_row32(bhi, lane, h, hi);
      }
      stage_flush<false>(map, blo, colbase, row0, lane);
      stage_flush<false>(map, bhi, colbase + 128, row0, lane);
    } else {
      if (lane == 0) bulk_wait_read<2>();
      __syncwarp();
      const int pb = (ec.ephase_q ^= 1);
#pragma unroll 1
      for (int c = 0; c < 2; c++) {
        float v0[32], v1[32];
        tmem_ld32(t_row + c * 64, v0);
        tmem_ld32(t_row + c * 64 + 32, v1);
        tmem_wait_ld();
        if (c == 1) ec.release_tmem();
        uint8_t* buf = ring.base + (2 * pb + c) * 4096;
        stage_row32(buf, lane, 0, v0);
        stage_row32(buf, lane, 1, v1);
        stage_flush<false>(&om.m[2], buf, (n_blk - 18) * 128 + c * 64, row0, lane);
      }
    }
  } else {  // EPI_QKV_ROPE: one head per 256-col tile
    static_assert(BN == 256 && NSTG == 4, "QKV epilogue needs BN == 256 (one head per tile) and 4 staging buffers");
    if (n_blk < 9) {
      const CUtensorMap* map = n_blk < 8 ? &om.m[0] : &om.m[1];
      const int colbase = n_blk < 8 ? n_blk * 256 : 0;
      // cos/sin for chunk 0 were prefetched into ec.cs/ec.sn before the accumulator wait (rope_prefetch)
#pragma unroll 1
      for (int c = 0; c < 2; c++) {  // output cols [c*64, c*64+64) and [128+c*64, ...): buffers 2c, 2c+1
        if (lane == 0) bulk_wait_read<2>();
        __syncwarp();
        uint8_t* blo = ring.base + (2 * c) * 4096;
        uint8_t* bhi = ring.base + (2 * c + 1) * 4096;
#pragma unroll 1
        for (int h = 0; h < 2; h++) {
          const int ch = c * 2 + h;  // 32-col chunk index 0..3
          float cs[32], sn[32];
#pragma unroll
          for (int i = 0; i < 32; i++) { cs[i] = ec.cs[i]; sn[i] = ec.sn[i]; }
          if (ch < 3) rope_load(p, row0 + lane, (ch + 1) * 32, ec);  // software-pipelined next chunk
          float lo[32], hi[32];
          tmem_ld32(t_row + ch * 32, lo);
          tmem_ld32(t_row + 128 + ch * 32, hi);
          tmem_wait_ld();
#pragma unroll
          for (int j = 0; j < 32; j++) {
            const float a = lo[j], b = hi[j];
            lo[j] = a * cs[j] - b * sn[j];
            hi[j] = b * cs[j] + a * sn[j];
          }
          stage_row32(blo, lane, h, lo);
          stage_row32(bhi, lane, h, hi);
        }
        if (c == 1) ec.release_tmem();
        stage_flush<false>(map, blo, colbase + c * 64, row0, lane);
        stage_flush<false>(map, bhi, colbase + 128 + c * 64, row0, lane);
      }
    } else {
#pragma unroll 1
      for (int c = 0; c < 4; c++) {
        float v0[32], v1[32];
        tmem_ld32(t_row + c * 64, v0);
        tmem_ld32(t_row + c * 64 + 32, v1);
        tmem_wait_ld();
        if (c == 3) ec.release_tmem();
        if (lane == 0) bulk_wait_read<1>();
        __syncwarp();
        uint8_t* buf = ring.base + (2 + (c & 1)) * 4096;
        stage_row32(buf, lane, 0, v0);
        stage_row32(buf, lane, 1, v1);
        stage_flush<false>(&om.m[2], buf, c * 64, row0, lane);
      }
    }
  }
}

// Unit order.  order 0: (n, split, m) with m fastest -> M tiles sharing a weight block run together.
// order 1: (split, m, n) with n fastest -> one K-slab of A and W at a time (fits L2 for split-K on P7).
__device__ __forceinline__ void decode_unit_o(int u, int num_m, int num_n, int splits, int KB, int order, int& m_blk,
                                              int& sp, int& n_blk, int& kb0, int& kb1) {
  if (order == 0) {
    m_blk = u % num_m;
    const int ns = u / num_m;
    sp = ns % splits;
    n_blk = ns / splits;
  } else {
    n_blk = u % num_n;
    const int ms = u / num_n;
    m_blk = ms % num_m;
    sp = ms / num_m;
  }
  kb0 = (sp * KB) / splits;
  kb1 = ((sp + 1) * KB) / splits;
}

__device__ __forceinline__ void dbg_mark(const Params& p, int uidx, int slot) {
  if (p.dbg && uidx < 64) {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    p.dbg[(blockIdx.x * 64 + uidx) * 4 + slot] = t;
  }
}
__device__ __forceinline__ void stamp_begin(const Params& p) {
  if (p.stamp && blockIdx.x == 0 && threadIdx.x == 0) {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    p.stamp[0] = clock64();
    p.stamp[2] = t;
  }
}
__device__ __forceinline__ void stamp_end(const Params& p) {
  if (p.stamp && blockIdx.x == 0 && threadIdx.x == 0) {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    p.stamp[1] = clock64();
    p.stamp[3] = t;
  }
}

__device__ __forceinline__ void tma_prefetch_2d(const CUtensorMap* map, int c0, int c1) {
  asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global [%0, {%1, %2}];" ::"l"((uint64_t)map), "r"(c0), "r"(c1)
               : "memory");
}
__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
// TMA coordinates of the weight box (k block kb, first row `row`): row-major [N][K], or K-blocked
// [N/512][K/64][512][64] read through a [N * K/64][64] map (a box never crosses a 512-row group).
__device__ __forceinline__ int wc0(const Params& p, int kb) { return p.wkblk ? 0 : kb * BK; }
__device__ __forceinline__ int wc1(const Params& p, int kb, int row) {
  return p.wkblk ? ((row >> 9) * p.kblocks + kb) * 512 + (row & 511) : row;
}
// L2-prefetch [ptr, ptr+bytes) split over `parts` cooperating CTAs (this one is `part`)
__device__ __forceinline__ void l2_prefetch_range(const void* ptr, long long bytes, int part, int parts) {
  const long long share = ((bytes / parts) + 4095) & ~4095ll;
  long long beg = share * part, end = beg + share;
  if (end > bytes) end = bytes;
  for (long long o = beg; o < end; o += 65536) {
    const long long n = (end - o) < 65536 ? (end - o) : 65536;
    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"((const char*)ptr + o), "r"((uint32_t)n) : "memory");
  }
}
__device__ __forceinline__ void l2_prefetch_share(const Params& p) {
  // each CTA prefetches its contiguous 1/gridDim share of the weight bytes into L2 (bulk, async, no smem)
  if (p.pf_ptr == nullptr || p.pf_bytes <= 0) return;
  const long long share = ((p.pf_bytes / gridDim.x) + 4095) & ~4095ll;
  long long beg = share * blockIdx.x, end = beg + share;
  if (end > p.pf_bytes) end = p.pf_bytes;
  for (long long o = beg; o < end; o += 65536) {
    const long long n = (end - o) < 65536 ? (end - o) : 65536;
    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"((const char*)p.pf_ptr + o), "r"((uint32_t)n)
                 : "memory");
  }
}

// ------------------------------------------------------------------------------------------------ 1-CTA kernel
template <int BN, int STAGES, int EPI, int NSTG>
__global__ void __launch_bounds__(NTHREADS, 1)
    gemm1_kernel(const __grid_constant__ CUtensorMap mapA, const __grid_constant__ CUtensorMap mapB,
                 const __grid_constant__ OutMaps om, Params p) {
  constexpr int BM = 128;
  constexpr int A_BYTES = BM * BK * 2;
  constexpr int B_BYTES = BN * BK * 2;
  constexpr int STAGE_BYTES = A_BYTES + B_BYTES;
  constexpr int TMEM_COLS = 2 * BN;
  constexpr uint32_t IDESC = make_idesc(128, BN);

  extern __shared__ __align__(1024) uint8_t smem_raw[];
  uint8_t* smem = (uint8_t*)(((uintptr_t)smem_raw + 1023) & ~(uintptr_t)1023);
  __shared__ __align__(8) uint64_t full_bar[STAGES];
  __shared__ __align__(8) uint64_t empty_bar[STAGES];
  __shared__ __align__(8) uint64_t tfull_bar[2];
  __shared__ __align__(8) uint64_t tempty_bar[2];
  __shared__ uint32_t tmem_base_sh;
  __shared__ __align__(8) uint64_t epi_tma_bar[4];

  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  stamp_begin(p);

  if (threadIdx.x == 0) {
    for (int s = 0; s < STAGES; s++) {
      mbar_init(&full_bar[s], 1);
      mbar_init(&empty_bar[s], 1);
    }
    for (int s = 0; s < 2; s++) {
      mbar_init(&tfull_bar[s], 1);
      mbar_init(&tempty_bar[s], 4);
    }
    for (int s = 0; s < 4; s++) mbar_init(&epi_tma_bar[s], 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  if (warp == 0 && lane == 0) {
    asm volatile("prefetch.tensormap [%0];" ::"l"((uint64_t)&mapA) : "memory");
    asm volatile("prefetch.tensormap [%0];" ::"l"((uint64_t)&mapB) : "memory");
  }
  if (warp == 1) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(smem_u32(&tmem_base_sh)),
                 "r"(TMEM_COLS));
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
  }
  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  const uint32_t tmem_base = tmem_base_sh;
  const int num_m = p.num_m, splits = p.splits, KB = p.kblocks;

  if (warp == 0) {
    if (lane == 0) l2_prefetch_share(p);
    if (p.pdl) pdl_wait();
    const uint64_t pol_w = make_policy(p.policy_w == 0 ? 3 : p.policy_w);
    const uint64_t pol_a = make_policy(2);
    int stage = 0;
    uint32_t phase = 0;
    for (int u = blockIdx.x; u < p.num_units; u += gridDim.x) {
      int m_blk, sp, n_blk, kb0, kb1;
      decode_unit_o(u, num_m, p.num_n, splits, KB, p.order, m_blk, sp, n_blk, kb0, kb1);
      for (int kb = kb0; kb < kb1; kb++) {
        mbar_wait(&empty_bar[stage], phase ^ 1);
        if (lane == 0) {
          uint8_t* sA = smem + stage * STAGE_BYTES;
          mbar_expect_tx(&full_bar[stage], STAGE_BYTES);
          tma_load_2d(sA, &mapA, &full_bar[stage], kb * BK, m_blk * BM, pol_a);
          tma_load_2d(sA + A_BYTES, &mapB, &full_bar[stage], kb * BK, n_blk * BN, pol_w);
        }
        __syncwarp();
        if (++stage == STAGES) { stage = 0; phase ^= 1; }
      }
    }
  } else if (warp == 1) {
    int stage = 0;
    uint32_t phase = 0;
    int acc = 0;
    uint32_t acc_phase = 0;
    for (int u = blockIdx.x; u < p.num_units; u += gridDim.x) {
      int m_blk, sp, n_blk, kb0, kb1;
      decode_unit_o(u, num_m, p.num_n, splits, KB, p.order, m_blk, sp, n_blk, kb0, kb1);
      mbar_wait(&tempty_bar[acc], acc_phase ^ 1);
      tc_fence_after();
      const uint32_t d_tmem = tmem_base + acc * BN;
      for (int kb = kb0; kb < kb1; kb++) {
        mbar_wait(&full_bar[stage], phase);
        tc_fence_after();
        if (lane == 0) {
          const uint32_t a_addr = smem_u32(smem + stage * STAGE_BYTES);
          const uint64_t da = make_sdesc(a_addr), db = make_sdesc(a_addr + A_BYTES);
#pragma unroll
          for (int k = 0; k < BK / 16; k++) umma_1sm(d_tmem, da + 2 * k, db + 2 * k, IDESC, (kb > kb0 || k > 0));
          umma_commit_1sm(&empty_bar[stage]);
        }
        __syncwarp();
        if (++stage == STAGES) { stage = 0; phase ^= 1; }
      }
      if (lane == 0) umma_commit_1sm(&tfull_bar[acc]);
      __syncwarp();
      acc ^= 1;
      if (acc == 0) acc_phase ^= 1;
    }
  } else {
    const int sub = warp & 3;
    if (p.pdl) pdl_wait();
    StageRing<NSTG> ring{smem + STAGES * STAGE_BYTES + (warp - 2) * NSTG * 4096, 0, 0};
    EpiCtx ec{};
    ec.ebar = &epi_tma_bar[warp - 2];
    int acc = 0;
    uint32_t acc_phase = 0;
    for (int u = blockIdx.x; u < p.num_units; u += gridDim.x) {
      int m_blk, sp, n_blk, kb0, kb1;
      decode_unit_o(u, num_m, p.num_n, splits, KB, p.order, m_blk, sp, n_blk, kb0, kb1);
      mbar_wait(&tfull_bar[acc], acc_phase);
      tc_fence_after();
      const uint32_t t_row = tmem_base + ((uint32_t)(sub * 32) << 16) + acc * BN;
      ec.cid = (n_blk * num_m + m_blk) * 4 + sub;
      ec.tempty = 0;
      ec.tempty_local = &tempty_bar[acc];
      ec.released = false;
      run_epilogue<BN, EPI, NSTG>(p, om, ring, t_row, m_blk * BM + sub * 32, n_blk, ec);
      ec.release_tmem();
      acc ^= 1;
      if (acc == 0) acc_phase ^= 1;
    }
    if (lane == 0) bulk_wait_all();
    __syncwarp();
  }

  tc_fence_before();
  __syncthreads();
  tc_fence_after();
  if (warp == 1) asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(tmem_base), "r"(TMEM_COLS));
  stamp_end(p);
}

// ------------------------------------------------------------------------------------------------ 2-CTA kernel
// Pair tile = 256 (M) x BNP (N); NMMA = BNP/256 UMMAs (256x256x16) per 16-wide K step.
// smem stage per CTA: A half [128 x 64] + NMMA B halves [128 x 64].
template <int BNP, int STAGES, int EPI, int NSTG>
__global__ void __launch_bounds__(NTHREADS, 1)
    gemm2_kernel(const __grid_constant__ CUtensorMap mapA, const __grid_constant__ CUtensorMap mapB,
                 const __grid_constant__ OutMaps om, Params p) {
  constexpr int MMA_N = BNP < 256 ? BNP : 256;
  constexpr int NMMA = BNP / MMA_N;
  constexpr int BH_ROWS = MMA_N / 2;  // B rows per CTA per UMMA
  constexpr int A_BYTES = 128 * BK * 2;
  constexpr int BH_BYTES = BH_ROWS * BK * 2;
  constexpr int STAGE_BYTES = A_BYTES + NMMA * BH_BYTES;  // per CTA
  constexpr int NBUF = (512 / BNP) > 4 ? 4 : (512 / BNP);  // TMEM accumulator buffers
  constexpr uint32_t IDESC = make_idesc(256, MMA_N);

  extern __shared__ __align__(1024) uint8_t smem_raw[];
  uint8_t* smem = (uint8_t*)(((uintptr_t)smem_raw + 1023) & ~(uintptr_t)1023);
  __shared__ __align__(8) uint64_t full_bar[STAGES];
  __shared__ __align__(8) uint64_t empty_bar[STAGES];
  __shared__ __align__(8) uint64_t tfull_bar[NBUF];
  __shared__ __align__(8) uint64_t tempty_bar[NBUF];
  __shared__ uint32_t tmem_base_sh;
  __shared__ __align__(8) uint64_t epi_tma_bar[4];

  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const uint32_t rank = cluster_rank();
  const int pair = blockIdx.x >> 1, npairs = gridDim.x >> 1;
  stamp_begin(p);

  if (threadIdx.x == 0) {
    for (int s = 0; s < STAGES; s++) {
      mbar_init(&full_bar[s], 1);
      mbar_init(&empty_bar[s], 1);
    }
    for (int s = 0; s < NBUF; s++) {
      mbar_init(&tfull_bar[s], 1);
      mbar_init(&tempty_bar[s], 8);  // 4 epilogue warps x 2 CTAs (only the leader's is used)
    }
    for (int s = 0; s < 4; s++) mbar_init(&epi_tma_bar[s], 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  if (warp == 0 && lane == 0) {
    asm volatile("prefetch.tensormap [%0];" ::"l"((uint64_t)&mapA) : "memory");
    asm volatile("prefetch.tensormap [%0];" ::"l"((uint64_t)&mapB) : "memory");
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
  const int num_m = p.num_m, splits = p.splits, KB = p.kblocks;

  if (warp == 0) {
    // ---------------- TMA producer (both CTAs); completion bytes counted on the leader's full barrier
    if (lane == 0) l2_prefetch_share(p);
    if (p.pdl) {
      if (lane == 0 && p.pf_first > 0 && pair < p.num_units) {
        int m_blk, sp, n_blk, kb0, kb1;
        decode_unit_o(pair, num_m, p.num_n, splits, KB, p.order, m_blk, sp, n_blk, kb0, kb1);
        const char* wb = (const char*)p.pf_ptr + ((size_t)n_blk * BNP * p.K + (size_t)kb0 * BK) * 2;
        // rows n_blk*BNP .. +BNP, K range of this split: prefetch row by row is too fine; prefetch whole rows
        long long bytes = (long long)BNP * p.K * 2;
        if (bytes > p.pf_first) bytes = p.pf_first;
        l2_prefetch_range((const char*)p.pf_ptr + (size_t)n_blk * BNP * p.K * 2, bytes, (int)rank, 2);
        (void)wb;
      }
      pdl_wait();
    }
    const uint64_t pol_w = make_policy(p.policy_w == 0 ? 3 : p.policy_w);
    const uint64_t pol_a = make_policy(2);
    int stage = 0;
    uint32_t phase = 0;
    for (int u = pair; u < p.num_units; u += npairs) {
      int m_blk, sp, n_blk, kb0, kb1;
      decode_unit_o(u, num_m, p.num_n, splits, KB, p.order, m_blk, sp, n_blk, kb0, kb1);
      for (int kb = kb0; kb < kb1; kb++) {
        mbar_wait(&empty_bar[stage], phase ^ 1);
        if (lane == 0) {
          uint8_t* sA = smem + stage * STAGE_BYTES;
          const uint32_t fb = mapa(smem_u32(&full_bar[stage]), 0);
          if (rank == 0) mbar_expect_tx(&full_bar[stage], 2 * STAGE_BYTES);
          tma_load_2d_2sm(sA, &mapA, fb, kb * BK, m_blk * 256 + rank * 128, pol_a);
#pragma unroll
          for (int j = 0; j < NMMA; j++)
            tma_load_2d_2sm(sA + A_BYTES + j * BH_BYTES, &mapB, fb, wc0(p, kb),
                            wc1(p, kb, n_blk * BNP + j * MMA_N + (int)rank * BH_ROWS), pol_w);
        }
        __syncwarp();
        if (++stage == STAGES) { stage = 0; phase ^= 1; }
      }
    }
  } else if (warp == 1) {
    // ---------------- MMA issuer (leader CTA only)
    if (rank == 0) {
      int stage = 0;
      uint32_t phase = 0;
      int acc = 0;
      uint32_t acc_phase = 0;
      for (int u = pair; u < p.num_units; u += npairs) {
        int m_blk, sp, n_blk, kb0, kb1;
        decode_unit_o(u, num_m, p.num_n, splits, KB, p.order, m_blk, sp, n_blk, kb0, kb1);
        mbar_wait_cluster(&tempty_bar[acc], acc_phase ^ 1);
        tc_fence_after();
        const uint32_t d_tmem = tmem_base + acc * BNP;
        for (int kb = kb0; kb < kb1; kb++) {
          mbar_wait(&full_bar[stage], phase);
          tc_fence_after();
          if (lane == 0) {
            const uint32_t a_addr = smem_u32(smem + stage * STAGE_BYTES);
            const uint64_t da = make_sdesc(a_addr);
#pragma unroll
            for (int j = 0; j < NMMA; j++) {
              const uint64_t db = make_sdesc(a_addr + A_BYTES + j * BH_BYTES);
#pragma unroll
              for (int k = 0; k < BK / 16; k++)
                umma_2sm(d_tmem + j * MMA_N, da + 2 * k, db + 2 * k, IDESC, (kb > kb0 || k > 0));
            }
            umma_commit_2sm_mc(&empty_bar[stage]);
          }
          __syncwarp();
          if (++stage == STAGES) { stage = 0; phase ^= 1; }
        }
        if (lane == 0) umma_commit_2sm_mc(&tfull_bar[acc]);
        __syncwarp();
        if (++acc == NBUF) { acc = 0; acc_phase ^= 1; }
      }
    }
  } else {
    // ---------------- epilogue (both CTAs): this CTA's 128 rows of the 256-row pair tile
    const int sub = warp & 3;
    if (p.pdl) pdl_wait();
    StageRing<NSTG> ring{smem + STAGES * STAGE_BYTES + (warp - 2) * NSTG * 4096, 0, 0};
    EpiCtx ec{};
    ec.ebar = &epi_tma_bar[warp - 2];
    const uint32_t tempty_leader0 = mapa(smem_u32(&tempty_bar[0]), 0);
    int acc = 0;
    uint32_t acc_phase = 0;
    for (int u = pair; u < p.num_units; u += npairs) {
      int m_blk, sp, n_blk, kb0, kb1;
      decode_unit_o(u, num_m, p.num_n, splits, KB, p.order, m_blk, sp, n_blk, kb0, kb1);
      const int uidx = (u - pair) / npairs;
      resid_prefetch<BNP, EPI, NSTG>(p, om, ring, m_blk * 256 + rank * 128 + sub * 32, n_blk, ec);
      rope_prefetch<EPI, BNP>(p, m_blk * 256 + rank * 128 + sub * 32, n_blk, ec);
      mbar_wait(&tfull_bar[acc], acc_phase);
      tc_fence_after();
      if (sub == 0 && lane == 0) dbg_mark(p, uidx, 0);
      const uint32_t t_row = tmem_base + ((uint32_t)(sub * 32) << 16) + acc * BNP;
      ec.cid = ((n_blk * num_m + m_blk) * 2 + (int)rank) * 4 + sub;
      ec.tempty = tempty_leader0 + acc * 8;
      ec.released = false;
      run_epilogue<BNP, EPI, NSTG>(p, om, ring, t_row, m_blk * 256 + rank * 128 + sub * 32, n_blk, ec);
      if (sub == 0 && lane == 0) dbg_mark(p, uidx, 1);
      ec.release_tmem();
      if (++acc == NBUF) { acc = 0; acc_phase ^= 1; }
    }
    if (lane == 0) bulk_wait_all();
    __syncwarp();
  }

  tc_fence_before();
  cluster_sync();
  tc_fence_after();
  if (warp == 1) asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" ::"r"(tmem_base), "r"(512));
  stamp_end(p);
}

// ------------------------------------------------------------------------------------------------ 6-CTA kernel
// Cluster of 6 CTAs = 3 CTA pairs; pair j owns rows [256j, 256j+256) (so the cluster covers M <= 768), all three
// pairs work on the same N block.  Every weight (B) tile is fetched from L2 ONCE per cluster and TMA-multicast to
// the 3 CTAs of the same pair-parity; A (activations) is loaded per pair.  Per-cluster L2->SM traffic per K step:
// 3 x A(256 rows) + 1 x B(BNP rows), vs 3 x (A + B) for three independent pairs.  Uses 18 of 20 SMs.
template <int BNP, int STAGES, int EPI, int NSTG>
__global__ void __launch_bounds__(NTHREADS, 1)
    gemm3_kernel(const __grid_constant__ CUtensorMap mapA, const __grid_constant__ CUtensorMap mapB,
                 const __grid_constant__ OutMaps om, Params p) {
  constexpr int MMA_N = BNP < 256 ? BNP : 256;
  constexpr int NMMA = BNP / MMA_N;
  constexpr int BH_ROWS = MMA_N / 2;
  constexpr int A_BYTES = 128 * BK * 2;
  constexpr int BH_BYTES = BH_ROWS * BK * 2;
  constexpr int STAGE_BYTES = A_BYTES + NMMA * BH_BYTES;  // per CTA
  constexpr int NBUF = (512 / BNP) > 4 ? 4 : (512 / BNP);
  constexpr uint32_t IDESC = make_idesc(256, MMA_N);
  constexpr int BPH = BH_ROWS / 64;  // 64-row boxes per B half
  constexpr int NBOX = NMMA * BPH;   // 64-row B boxes per CTA per stage

  extern __shared__ __align__(1024) uint8_t smem_raw[];
  uint8_t* smem = (uint8_t*)(((uintptr_t)smem_raw + 1023) & ~(uintptr_t)1023);
  __shared__ __align__(8) uint64_t full_bar[STAGES];
  __shared__ __align__(8) uint64_t empty_bar[STAGES];
  __shared__ __align__(8) uint64_t tfull_bar[NBUF];
  __shared__ __align__(8) uint64_t tempty_bar[NBUF];
  __shared__ uint32_t tmem_base_sh;
  __shared__ __align__(8) uint64_t epi_tma_bar[4];

  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const uint32_t rank = cluster_rank();  // 0..5
  const uint32_t par = rank & 1;         // position inside the pair
  const int pj = (int)(rank >> 1);       // pair index inside the cluster = m tile
  const int clu = blockIdx.x / 6, nclu = gridDim.x / 6;
  stamp_begin(p);
  if (threadIdx.x == 0) dbg_mark(p, 63, 0);

  if (threadIdx.x == 0) {
    for (int s = 0; s < STAGES; s++) {
      mbar_init(&full_bar[s], 1);
      mbar_init(&empty_bar[s], 3);  // one MMA commit from each of the 3 pair leaders
    }
    for (int s = 0; s < NBUF; s++) {
      mbar_init(&tfull_bar[s], 1);
      mbar_init(&tempty_bar[s], 8);
    }
    for (int s = 0; s < 4; s++) mbar_init(&epi_tma_bar[s], 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  if (warp == 0 && lane == 0) {
    asm volatile("prefetch.tensormap [%0];" ::"l"((uint64_t)&mapA) : "memory");
    asm volatile("prefetch.tensormap [%0];" ::"l"((uint64_t)&mapB) : "memory");
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
  const int splits = p.splits, KB = p.kblocks, num_n = p.num_n;
  const int m_blk = pj;

  auto decode = [&](int u, int& sp, int& n_blk, int& kb0, int& kb1) {
    if (p.order == 0) {
      sp = u % splits;
      n_blk = u / splits;
    } else {
      n_blk = u % num_n;
      sp = u / num_n;
    }
    kb0 = (sp * KB) / splits;
    kb1 = ((sp + 1) * KB) / splits;
  };

  if (warp == 0) {
    if (lane == 0) l2_prefetch_share(p);
    const uint64_t pol_w = make_policy(p.policy_w == 0 ? 3 : p.policy_w);
    const uint64_t pol_a = make_policy(2);
    const uint16_t mc_mask = (uint16_t)((1u << par) | (1u << (par + 2)) | (1u << (par + 4)));
    if (p.pdl) {
      if (lane == 0 && p.pf_first > 0 && clu < p.num_units) {
        int sp, n_blk, kb0, kb1;
        decode(clu, sp, n_blk, kb0, kb1);
        long long bytes = (long long)BNP * p.K * 2;
        if (bytes > p.pf_first) bytes = p.pf_first;
        l2_prefetch_range((const char*)p.pf_ptr + (size_t)n_blk * BNP * p.K * 2, bytes, (int)rank, 6);
      }
      pdl_wait();
    }
    int stage = 0;
    uint32_t phase = 0;
    for (int u = clu; u < p.num_units; u += nclu) {
      int sp, n_blk, kb0, kb1;
      decode(u, sp, n_blk, kb0, kb1);
      if (p.apf) {
        // this CTA's 128 activation rows over the unit's K slab: one contiguous run per row (the TMA boxes read them
        // 128 B at a time, a row stride apart, which the DRAM serves slowly)
        const int r0 = m_blk * 256 + (int)par * 128;
        const uint32_t bytes = (uint32_t)(kb1 - kb0) * BK * 2;
        for (int r = lane; r < 128 && r0 + r < p.M; r += 32)
          asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(p.a_ptr + (size_t)(r0 + r) * p.K + kb0 * BK),
                       "r"(bytes)
                       : "memory");
      }
      for (int kb = kb0; kb < kb1; kb++) {
        if (p.pd > 0 && lane == 0) {
          // rolling L2 prefetch of the boxes this CTA will load pd k-blocks from now (possibly in the next unit)
          int tkb = kb + p.pd, tn = n_blk, tu = u;
          bool ok = true;
          if (tkb >= kb1) {
            tu = u + nclu;
            ok = tu < p.num_units;
            if (ok) {
              int tsp, tkb0, tkb1;
              decode(tu, tsp, tn, tkb0, tkb1);
              tkb = tkb0 + (tkb - kb1);
              ok = tkb < tkb1;
            }
          }
          if (ok) {
            tma_prefetch_2d(&mapA, tkb * BK, m_blk * 256 + par * 128);
#pragma unroll
            for (int b = 0; b < NBOX; b++)
              if (b % 3 == pj)
                tma_prefetch_2d(&mapB, wc0(p, tkb),
                                wc1(p, tkb, tn * BNP + (b / BPH) * MMA_N + (int)par * BH_ROWS + (b % BPH) * 64));
          }
        }
        mbar_wait_cluster(&empty_bar[stage], phase ^ 1);
        if (lane == 0) {
          uint8_t* sA = smem + stage * STAGE_BYTES;
          const uint32_t fb = smem_u32(&full_bar[stage]) & 0xFEFFFFFFu;  // pair leader's barrier
          if (par == 0) mbar_expect_tx(&full_bar[stage], 2 * STAGE_BYTES);
          tma_load_2d_2sm(sA, &mapA, fb, kb * BK, m_blk * 256 + par * 128, pol_a);
#pragma unroll
          for (int b = 0; b < NBOX; b++) {
            if (b % 3 == pj) {
              const int i = b / BPH, sub = b % BPH;
              asm volatile(
                  "cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::"
                  "cluster.L2::cache_hint [%0], [%1, {%4, %5}], [%2], %3, %6;" ::"r"(
                      smem_u32(sA + A_BYTES + i * BH_BYTES + sub * 8192)),
                  "l"((uint64_t)&mapB), "r"(fb), "h"(mc_mask), "r"(wc0(p, kb)),
                  "r"(wc1(p, kb, n_blk * BNP + i * MMA_N + (int)par * BH_ROWS + sub * 64)), "l"(pol_w)
                  : "memory");
            }
          }
        }
        __syncwarp();
        if (++stage == STAGES) { stage = 0; phase ^= 1; }
      }
    }
  } else if (warp == 1) {
    if (par == 0) {
      int stage = 0;
      uint32_t phase = 0;
      int acc = 0;
      uint32_t acc_phase = 0;
      const uint16_t pair_mask = (uint16_t)(3u << (2 * pj));
      for (int u = clu; u < p.num_units; u += nclu) {
        int sp, n_blk, kb0, kb1;
        decode(u, sp, n_blk, kb0, kb1);
        mbar_wait_cluster(&tempty_bar[acc], acc_phase ^ 1);
        tc_fence_after();
        const int uidx = (u - clu) / nclu;
        if (lane == 0) dbg_mark(p, uidx, 2);
        const uint32_t d_tmem = tmem_base + acc * BNP;
        for (int kb = kb0; kb < kb1; kb++) {
          mbar_wait(&full_bar[stage], phase);
          tc_fence_after();
          if (lane == 0 && kb == kb0) dbg_mark(p, uidx, 3);
          if (lane == 0) {
            const uint32_t a_addr = smem_u32(smem + stage * STAGE_BYTES);
            const uint64_t da = make_sdesc(a_addr);
#pragma unroll
            for (int j = 0; j < NMMA; j++) {
              const uint64_t db = make_sdesc(a_addr + A_BYTES + j * BH_BYTES);
#pragma unroll
              for (int k = 0; k < BK / 16; k++)
                umma_2sm(d_tmem + j * MMA_N, da + 2 * k, db + 2 * k, IDESC, (kb > kb0 || k > 0));
            }
            // stage free in all 6 CTAs once all three pairs consumed it
            asm volatile(
                "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;" ::"r"(
                    smem_u32(&empty_bar[stage])),
                "h"((uint16_t)0x3F)
                : "memory");
          }
          __syncwarp();
          if (++stage == STAGES) { stage = 0; phase ^= 1; }
        }
        if (lane == 0)
          asm volatile(
              "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;" ::"r"(
                  smem_u32(&tfull_bar[acc])),
              "h"(pair_mask)
              : "memory");
        __syncwarp();
        if (++acc == NBUF) { acc = 0; acc_phase ^= 1; }
      }
    }
  } else {
    const int sub = warp & 3;
    if (p.pdl) pdl_wait();
    StageRing<NSTG> ring{smem + STAGES * STAGE_BYTES + (warp - 2) * NSTG * 4096, 0, 0};
    EpiCtx ec{};
    ec.ebar = &epi_tma_bar[warp - 2];
    const uint32_t tempty_leader0 = smem_u32(&tempty_bar[0]) & 0xFEFFFFFFu;
    int acc = 0;
    uint32_t acc_phase = 0;
    for (int u = clu; u < p.num_units; u += nclu) {
      int sp, n_blk, kb0, kb1;
      decode(u, sp, n_blk, kb0, kb1);
      const int uidx = (u - clu) / nclu;
      resid_prefetch<BNP, EPI, NSTG>(p, om, ring, m_blk * 256 + par * 128 + sub * 32, n_blk, ec);
      rope_prefetch<EPI, BNP>(p, m_blk * 256 + par * 128 + sub * 32, n_blk, ec);
      mbar_wait(&tfull_bar[acc], acc_phase);
      tc_fence_after();
      if (sub == 0 && lane == 0) dbg_mark(p, uidx, 0);
      const uint32_t t_row = tmem_base + ((uint32_t)(sub * 32) << 16) + acc * BNP;
      ec.cid = ((n_blk * 3 + m_blk) * 2 + (int)par) * 4 + sub;
      ec.tempty = tempty_leader0 + acc * 8;
      ec.released = false;
      run_epilogue<BNP, EPI, NSTG>(p, om, ring, t_row, m_blk * 256 + par * 128 + sub * 32, n_blk, ec);
      ec.release_tmem();
      if (sub == 0 && lane == 0) dbg_mark(p, uidx, 1);
      if (++acc == NBUF) { acc = 0; acc_phase ^= 1; }
    }
    if (lane == 0) bulk_wait_all();
    __syncwarp();
  }

  tc_fence_before();
  cluster_sync();
  tc_fence_after();
  if (warp == 1) asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" ::"r"(tmem_base), "r"(512));
  stamp_end(p);
}

__global__ void interleave_kernel(const bf16* wg, const bf16* wu, bf16* out, int I, int K, int blk) {
  size_t total = (size_t)2 * I * K / 8;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < total; i += (size_t)gridDim.x * blockDim.x) {
    size_t e = i * 8;
    int r = (int)(e / K), k = (int)(e % K);
    int b = r / (2 * blk), w = r % (2 * blk);
    const bf16* src = w < blk ? wg + ((size_t)(b * blk + w) * K + k) : wu + ((size_t)(b * blk + w - blk) * K + k);
    *reinterpret_cast<uint4*>(out + e) = *reinterpret_cast<const uint4*>(src);
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
static int cfg(const std::string& k, int dflt) {
  auto it = g_cfg.find(k);
  return it == g_cfg.end() ? dflt : it->second;
}

static int g_num_sms = 0;
static int num_sms() {
  if (!g_num_sms) {
    int dev;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&g_num_sms, cudaDevAttrMultiProcessorCount, dev);
  }
  return g_num_sms;
}

// 2D bf16 tensor [rows][cols] with row stride ld (elements); box = 64 cols x box_rows, 128B swizzle
static bool make_map(CUtensorMap* m, const void* ptr, int rows, int cols, int box_rows, long ld = -1) {
  if (ld < 0) ld = cols;
  cuuint64_t dims[2] = {(cuuint64_t)cols, (cuuint64_t)rows};
  cuuint64_t strides[1] = {(cuuint64_t)ld * 2};
  cuuint32_t box[2] = {64, (cuuint32_t)box_rows};
  cuuint32_t estr[2] = {1, 1};
  CUresult r = EncodeTiled()(m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<void*>(ptr), dims, strides,
                                      box, estr, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                                      CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (r != CUDA_SUCCESS) {
    fprintf(stderr, "cuTensorMapEncodeTiled failed %d\n", (int)r);
    return false;
  }
  return true;
}

// 2D fp32 tensor [rows][cols]; box = 32 cols (128 B) x box_rows, 128B swizzle
// weight map: row-major [N][K], or K-blocked [N/512][K/64][512][64] viewed as [N * K/64][64]
static bool make_wmap(CUtensorMap* m, const void* ptr, int n, int k, int box_rows, int kblk) {
  if (!kblk) return make_map(m, ptr, n, k, box_rows);
  if (n % 512 != 0 || k % BK != 0) return false;
  return make_map(m, ptr, n * (k / BK), BK, box_rows);
}
static bool make_map_f32(CUtensorMap* m, const void* ptr, int rows, int cols, int box_rows) {
  cuuint64_t dims[2] = {(cuuint64_t)cols, (cuuint64_t)rows};
  cuuint64_t strides[1] = {(cuuint64_t)cols * 4};
  cuuint32_t box[2] = {32, (cuuint32_t)box_rows};
  cuuint32_t estr[2] = {1, 1};
  CUresult r = EncodeTiled()(m, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 2, const_cast<void*>(ptr), dims, strides,
                                      box, estr, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                                      CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  return r == CUDA_SUCCESS;
}

static void fill_common(Params& p, int M, int N, int K, int splits, int bm, int bn) {
  p.M = M;
  p.N = N;
  p.K = K;
  p.num_m = (M + bm - 1) / bm;
  p.num_n = (N + bn - 1) / bn;
  p.splits = splits;
  p.num_units = p.num_m * p.num_n * splits;
  p.kblocks = K / BK;
  p.stamp = g_stamp;
  p.dbg = g_dbg;
}

template <int BN, int STAGES, int EPI, int NSTG>
static int launch1(const bf16* A, const bf16* W, int M, int N, int K, int splits, Params p, const OutMaps& om,
                   int grid_cap, cudaStream_t stream) {
  constexpr int SMEM = STAGES * (128 * BK * 2 + BN * BK * 2) + 4 * NSTG * 4096 + 1024;
  static_assert(SMEM <= 227 * 1024, "smem");
  static bool attr_set = false;
  auto kern = gemm1_kernel<BN, STAGES, EPI, NSTG>;
  if (!attr_set) {
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
    attr_set = true;
  }
  CUtensorMap ma, mb;
  if (!make_map(&ma, A, M, K, 128) || !make_map(&mb, W, N, K, BN)) return -1;
  fill_common(p, M, N, K, splits, 128, BN);
  int grid = num_sms();
  if (grid_cap > 0 && grid_cap < grid) grid = grid_cap;
  if (grid > p.num_units) grid = p.num_units;
  kern<<<grid, NTHREADS, SMEM, stream>>>(ma, mb, om, p);
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) {
    fprintf(stderr, "launch failed: %s\n", cudaGetErrorString(e));
    return -2;
  }
  return 0;
}

template <int BNP, int STAGES, int EPI, int NSTG>
static int launch2(const bf16* A, const bf16* W, int M, int N, int K, int splits, Params p, const OutMaps& om,
                   int grid_cap, cudaStream_t stream) {
  constexpr int SMEM = STAGES * (128 * BK * 2 + (BNP / 2) * BK * 2) + 4 * NSTG * 4096 + 1024;
  static_assert(SMEM <= 227 * 1024, "smem");
  static bool attr_set = false;
  auto kern = gemm2_kernel<BNP, STAGES, EPI, NSTG>;
  if (!attr_set) {
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
    attr_set = true;
  }
  CUtensorMap ma, mb;
  if (!make_map(&ma, A, M, K, 128) || !make_wmap(&mb, W, N, K, (BNP < 256 ? BNP : 256) / 2, p.wkblk)) return -1;
  fill_common(p, M, N, K, splits, 256, BNP);
  int npairs = num_sms() / 2;
  if (grid_cap > 0 && grid_cap / 2 < npairs) npairs = grid_cap / 2;
  if (npairs > p.num_units) npairs = p.num_units;
  cudaLaunchConfig_t lc = {};
  lc.gridDim = dim3(2 * npairs);
  lc.blockDim = dim3(NTHREADS);
  lc.dynamicSmemBytes = SMEM;
  lc.stream = stream;
  cudaLaunchAttribute at[2];
  at[0].id = cudaLaunchAttributeClusterDimension;
  at[0].val.clusterDim.x = 2;
  at[0].val.clusterDim.y = 1;
  at[0].val.clusterDim.z = 1;
  at[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[1].val.programmaticStreamSerializationAllowed = 1;
  lc.attrs = at;
  lc.numAttrs = p.pdl ? 2 : 1;
  cudaError_t e = cudaLaunchKernelEx(&lc, kern, ma, mb, om, p);
  if (e != cudaSuccess) {
    fprintf(stderr, "launch2 failed: %s\n", cudaGetErrorString(e));
    return -2;
  }
  return 0;
}

template <int BNP, int STAGES, int EPI, int NSTG>
static int launch3(const bf16* A, const bf16* W, int M, int N, int K, int splits, Params p, const OutMaps& om,
                   cudaStream_t stream) {
  constexpr int SMEM = STAGES * (128 * BK * 2 + (BNP / 2) * BK * 2) + 4 * NSTG * 4096 + 1024;
  static_assert(SMEM <= 227 * 1024, "smem");
  static bool attr_set = false;
  auto kern = gemm3_kernel<BNP, STAGES, EPI, NSTG>;
  if (!attr_set) {
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
    cudaFuncSetAttribute(kern, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    attr_set = true;
  }
  if (M > 768 || M <= 512) return -5;
  CUtensorMap ma, mb;
  if (!make_map(&ma, A, M, K, 128) || !make_wmap(&mb, W, N, K, 64, p.wkblk)) return -1;
  fill_common(p, M, N, K, splits, 256, BNP);
  p.num_units = p.num_n * splits;
  int ncl = num_sms() / 6;
  if (ncl > p.num_units) ncl = p.num_units;
  cudaLaunchConfig_t lc = {};
  lc.gridDim = dim3(6 * ncl);
  lc.blockDim = dim3(NTHREADS);
  lc.dynamicSmemBytes = SMEM;
  lc.stream = stream;
  cudaLaunchAttribute at[2];
  at[0].id = cudaLaunchAttributeClusterDimension;
  at[0].val.clusterDim.x = 6;
  at[0].val.clusterDim.y = 1;
  at[0].val.clusterDim.z = 1;
  at[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[1].val.programmaticStreamSerializationAllowed = 1;
  lc.attrs = at;
  lc.numAttrs = p.pdl ? 2 : 1;
  cudaError_t e = cudaLaunchKernelEx(&lc, kern, ma, mb, om, p);
  if (e != cudaSuccess) {
    fprintf(stderr, "launch3 failed: %s\n", cudaGetErrorString(e));
    return -2;
  }
  return 0;
}

// Dispatch on (impl, bn): impl 1 = gemm1 (BN 128/256), impl 2 = gemm2 (BNP 256/512)
template <int EPI>
static int dispatch(const std::string& op, const bf16* A, const bf16* W, int M, int N, int K, Params p,
                    const OutMaps& om, cudaStream_t s, int d_impl, int d_bn, int d_splits, int d_stages,
                    int d_order = 0, int d_policy = 0) {
  const int impl = cfg(op + "_impl", d_impl), bn = cfg(op + "_bn", d_bn), splits = cfg(op + "_splits", d_splits);
  const int grid = cfg(op + "_grid", 0), st = cfg(op + "_stages", d_stages);
  p.policy_w = cfg(op + "_policy", d_policy);
  p.order = cfg(op + "_order", d_order);
  p.pdl = cfg("pdl", 0);
  p.pd = cfg(op + "_pd", 0);
  p.pf_first = cfg(op + "_pff", 4 << 20);
  if (p.pdl && !p.pf_ptr) p.pf_ptr = W;
  if (cfg(op + "_pf", 0)) {
    p.pf_ptr = W;
    p.pf_bytes = (long long)N * K * 2;
  }
  if (splits > 1 && EPI != EPI_RESID) return -3;
  if (impl == 3 && (M > 768 || M <= 512)) {
    // the 6-CTA cluster kernel covers 512 < M <= 768 in one cluster; otherwise use CTA pairs
    return dispatch<EPI>(op + "_fb", A, W, M, N, K, p, om, s, 2, bn, splits, st, p.order);
  }
  if (impl == 3) {
    if constexpr (!kGlu<EPI>) {
      if (bn == 128) return launch3<128, 6, EPI, 4>(A, W, M, N, K, splits, p, om, s);
    }
    if (bn == 256) {
      if constexpr (EPI == EPI_RESID) {
        if (st == 6 && !p.exact_resid && splits == 1) return launch3<256, 6, EPI, 2>(A, W, M, N, K, splits, p, om, s);
      }
      if (st == 5) return launch3<256, 5, EPI, 4>(A, W, M, N, K, splits, p, om, s);
      return launch3<256, 4, EPI, 4>(A, W, M, N, K, splits, p, om, s);
    }
    if constexpr (EPI != EPI_QKV_ROPE) {
      if (bn == 512) {
        if (st == 3 || (EPI == EPI_RESID && splits > 1 && p.ws)) return launch3<512, 3, EPI, 4>(A, W, M, N, K, splits, p, om, s);
        return launch3<512, 4, EPI, 2>(A, W, M, N, K, splits, p, om, s);
      }
    }
    return -4;
  }
  if (impl == 1) {
    if (splits > 1 && p.ws) return -6;  // exact split-K not wired for gemm1
    if (p.wkblk) return -7;             // K-blocked weights only in the 2-SM kernels
    if constexpr (EPI != EPI_QKV_ROPE) {
      if (bn == 256) return launch1<256, 4, EPI, 2>(A, W, M, N, K, splits, p, om, grid, s);
    }
    if constexpr (EPI == EPI_RESID) {
      if (bn == 128) return launch1<128, 6, EPI, 2>(A, W, M, N, K, splits, p, om, grid, s);
    }
  } else {
    if constexpr (!kGlu<EPI>) {
      if (bn == 128) return launch2<128, 6, EPI, 4>(A, W, M, N, K, splits, p, om, grid, s);
    }
    if (bn == 256) {
      if constexpr (EPI != EPI_QKV_ROPE) {
        if (st == 6 && !(EPI == EPI_RESID && splits > 1 && p.ws))
          return launch2<256, 6, EPI, 2>(A, W, M, N, K, splits, p, om, grid, s);
      }
      if (st == 5) return launch2<256, 5, EPI, 4>(A, W, M, N, K, splits, p, om, grid, s);
      return launch2<256, 4, EPI, 4>(A, W, M, N, K, splits, p, om, grid, s);
    }
    if constexpr (EPI != EPI_QKV_ROPE) {
      if (bn == 512) {
        if (st == 3 || (EPI == EPI_RESID && splits > 1 && p.ws)) return launch2<512, 3, EPI, 4>(A, W, M, N, K, splits, p, om, grid, s);
        return launch2<512, 4, EPI, 2>(A, W, M, N, K, splits, p, om, grid, s);
      }
    }
  }
  return -4;
}

}  // namespace mcast_bf16_sm110

using namespace mcast_bf16_sm110;

extern "C" int mcast_bf16_sm110_set_config(const char* key, int value) {
  g_cfg[key] = value;
  return 0;
}
extern "C" int mcast_bf16_sm110_set_dbg(void* ptr) {
  g_dbg = reinterpret_cast<unsigned long long*>(ptr);
  return 0;
}
extern "C" int mcast_bf16_sm110_set_stamp(void* ptr) {
  g_stamp = reinterpret_cast<unsigned long long*>(ptr);
  return 0;
}

extern "C" int mcast_bf16_sm110_qkv_rope(const bf16* xn, const bf16* w, const float* cos_t, const float* sin_t, bf16* q_out,
                           bf16* kc, bf16* vc, int M, cudaStream_t stream) {
  Params p{};
  p.cos_t = cos_t;
  p.sin_t = sin_t;
  OutMaps om;
  if (!make_map(&om.m[0], q_out, M, 2048, 32) || !make_map(&om.m[1], kc, M, 256, 32) ||
      !make_map(&om.m[2], vc, M, 256, 32))
    return -1;
  return dispatch<EPI_QKV_ROPE>("p2", xn, w, M, 2560, 2048, p, om, stream, 3, 256, 1, 4);
}

// split-K workspace: 4096 int counters + fp32 accumulator [ceil(M/256)*256][2048]; zero once at allocation
extern "C" size_t mcast_bf16_sm110_workspace_bytes(int M) {
  const int mp = (M + 255) / 256 * 256;
  return 4096 * sizeof(int) + (size_t)mp * 2048 * sizeof(float);
}
static bool resid_maps(OutMaps& om, Params& p, bf16* x, int M, void* ws, bool exact) {
  if (!make_map(&om.m[0], x, M, 2048, 32)) return false;
  om.m[1] = om.m[0];
  om.m[2] = om.m[0];
  p.exact_resid = exact ? 1 : 0;
  if (exact && ws) {
    const int mp = (M + 255) / 256 * 256;
    p.counters = reinterpret_cast<int*>(ws);
    p.ws = reinterpret_cast<float*>(reinterpret_cast<char*>(ws) + 4096 * sizeof(int));
    if (!make_map_f32(&om.m[1], p.ws, mp, 2048, 32)) return false;
  }
  return true;
}
extern "C" size_t mcast_bf16_sm110_down_workspace_bytes(int M) { return mcast_bf16_sm110_workspace_bytes(M); }

extern "C" int mcast_bf16_sm110_o_proj(const bf16* o, const bf16* w, bf16* x, int M, void* ws, cudaStream_t stream) {
  Params p{};
  OutMaps om;
  if (!resid_maps(om, p, x, M, ws, cfg("p4_exact", 1) != 0)) return -1;
  return dispatch<EPI_RESID>("p4", o, w, M, 2048, 2048, p, om, stream, 3, 256, 1, 4);
}

extern "C" int mcast_bf16_sm110_gemm_out(const bf16* x, const bf16* w, bf16* out, const bf16* bias,
                                          const float* row_scale, int act, int M, int N, int K, cudaStream_t stream) {
  if (N <= 0 || N % 8 != 0 || K <= 0 || K % BK != 0 || (act != MCAST_OUT_ACT_NONE && act != MCAST_OUT_ACT_GELU_TANH)) {
    return -2;
  }
  Params p{};
  p.bias = bias;
  p.row_scale = row_scale;
  p.act = act;
  OutMaps om;
  if (!make_map(&om.m[0], out, M, N, 32)) return -1;
  om.m[1] = om.m[0];
  om.m[2] = om.m[0];
  return dispatch<EPI_OUT>("p8", x, w, M, N, K, p, om, stream, 3, 256, 1, 4);
}

extern "C" int mcast_bf16_sm110_gate_up(const bf16* xn, const bf16* w, bf16* h, int M, int I, int act,
                                         cudaStream_t stream) {
  if (I <= 0 || I % 256 != 0 || I > MCAST_MAX_INTER || (act != MCAST_ACT_SILU && act != MCAST_ACT_GELU_TANH)) return -2;
  Params p{};
  OutMaps om;
  if (!make_map(&om.m[0], h, M, I, 32)) return -1;
  om.m[1] = om.m[0];
  om.m[2] = om.m[0];
  // SiLU: weights evict_first in the cluster kernel (each weight block is read once there), so h stays in L2 for the
  // down projection that reads it next: WALL-OSS-0.5 prefill (M 768, I 11008) gate/up + down -9% per layer on Thor. The
  // GELU-tanh path keeps its default.
  if (act == MCAST_ACT_SILU) return dispatch<EPI_SWIGLU>("p6", xn, w, M, 2 * I, 2048, p, om, stream, 3, 512, 1, 4, 0, 1);
  return dispatch<EPI_GEGLU>("p6", xn, w, M, 2 * I, 2048, p, om, stream, 3, 512, 1, 4);
}
extern "C" int mcast_bf16_sm110_gate_up_geglu(const bf16* xn, const bf16* w, bf16* h, int M, cudaStream_t stream) {
  return mcast_bf16_sm110_gate_up(xn, w, h, M, 16384, MCAST_ACT_GELU_TANH, stream);
}

extern "C" int mcast_bf16_sm110_down_ex(const bf16* h, const bf16* w, bf16* x, int M, int K, void* ws, int w_kblk,
                                        cudaStream_t stream) {
  if (K <= 0 || K % 64 != 0 || K > MCAST_MAX_INTER) return -2;
  Params p{};
  p.wkblk = w_kblk;
  p.a_ptr = h;
  p.apf = cfg("p7_apf", 1);  // cluster kernel only; S2 -8..-13% at M 640-768, S0 within noise
  OutMaps om;
  if (!resid_maps(om, p, x, M, ws, cfg("p7_exact", 0) != 0)) return -1;
  // order 1: K slab outermost; weights evict_first in the cluster kernel (each weight block is read exactly once, so
  // the activation keeps its L2 lines: -10% at M 640-768 with the activation in L2, no change when it is not). The
  // CTA-pair fallback keeps the default policy: there several M tiles re-read each weight block from L2.
  return dispatch<EPI_RESID>("p7", h, w, M, 2048, K, p, om, stream, 3, 512, 6, 4, 1, 1);
}
extern "C" int mcast_bf16_sm110_down(const bf16* h, const bf16* w, bf16* x, int M, void* ws, int w_kblk,
                                     cudaStream_t stream) {
  return mcast_bf16_sm110_down_ex(h, w, x, M, 16384, ws, w_kblk, stream);
}

// P2 half-head tiles (bn=128) need the q/k weight rows of every head reordered to
// [0..63, 128..191, 64..127, 192..255] (so rotate_half partners share a 128-col tile). v rows unchanged.
// Outputs (q_out, KV cache) keep the ORIGINAL dim order: the epilogue writes each dim to its true column.
__global__ void permute_qk_kernel(const bf16* w, bf16* out, int K) {
  const int r = blockIdx.x;  // destination row 0..2559
  int src = r;
  if (r < 2304) {
    const int hd = r / 256, d = r % 256;
    const int blk = d / 64;  // 0:[0,64) 1:[128,192) 2:[64,128) 3:[192,256)
    const int srcd = (blk == 0 ? 0 : blk == 1 ? 128 : blk == 2 ? 64 : 192) + d % 64;
    src = hd * 256 + srcd;
  }
  for (int k = threadIdx.x * 8; k < K; k += blockDim.x * 8)
    *reinterpret_cast<uint4*>(out + (size_t)r * K + k) = *reinterpret_cast<const uint4*>(w + (size_t)src * K + k);
}
extern "C" int mcast_bf16_sm110_permute_qk_rows(const bf16* wqkv, bf16* out, int K, cudaStream_t stream) {
  permute_qk_kernel<<<2560, 256, 0, stream>>>(wqkv, out, K);
  return cudaGetLastError() == cudaSuccess ? 0 : -1;
}
extern "C" int mcast_bf16_sm110_qkv_tile_n(void) { return cfg("p2_bn", 256); }

extern "C" int mcast_bf16_sm110_interleave_gate_up(const bf16* wg, const bf16* wu, bf16* out, int I, int K, cudaStream_t stream) {
  interleave_kernel<<<512, 256, 0, stream>>>(wg, wu, out, I, K, 128);
  return cudaGetLastError() == cudaSuccess ? 0 : -1;
}
