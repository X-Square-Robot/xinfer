// attention_kvcache, split-KV backend for small M (generic CUDA: mma.sync, cp.async; SM80+).
//
// Grouped-query rows: the G = Hq / Hkv query heads that share a KV head are packed with the M query tokens into
// one tile of G * M rows (row r = token * G + head-in-group), so every KV row is read once per KV head and every
// MMA row is a real query row.
//
// Phase 1 (splitkv_partial_kernel): grid (chunks, Hkv x row groups, B). A row group is up to ROWS of the G * M
// rows (one warp per 16 rows); CTA (c, row group) owns the visible keys u in [c * CHUNK,
// (c + 1) * CHUNK) of the sequence u = 0..n-1, n = prefix_len - leftpad + M, key u at cache row leftpad + u
// (u < prefix_len - leftpad) or new_pos + (u - prefix_len + leftpad), in cache row cache_batch_idx[b] (or b). A chunk may straddle the two segments: rows are copied one by one (cp.async), so
// the chunk boundaries depend only on CHUNK, never on M or on where the second segment sits. Warp w owns rows
// [16 w, 16 w + 16): S = Q K^T and O = P V with mma.sync m16n8k16, a softmax local to the chunk in registers,
// the unnormalised FP32 O and (max, sum) written to the workspace. With k_new the CTA owning a new key also
// stores it into the cache (each new key belongs to exactly one CTA).
//
// Phase 2 (splitkv_finalize_kernel, PDL-chained): one warp per output row merges the chunk partials in chunk
// order, divides, and writes BF16 or MXFP8 (1x32 / 1x128 E8M0 scales, SWIZZLE_32_4_4).
//
// The summation order of a row is fixed by CHUNK and the key layout: a row's result does not depend on M.
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "common/cuda_pdl.cuh"
#include "quant/mxfp8_format_core.cuh"

namespace xinfer::attention_kvcache {

enum OutQuant : int { kOutBf16 = 0, kOutMxfp8Blk32 = 1, kOutMxfp8Blk128 = 2 };

struct SplitkvParams {
  const __nv_bfloat16* q;
  int64_t q_sb, q_sm, q_sh;  // element strides: batch, token, head (last dim contiguous)
  __nv_bfloat16* k_cache;
  __nv_bfloat16* v_cache;
  int64_t kc_sb, kc_sr, kc_sh;
  int64_t vc_sb, vc_sr, vc_sh;
  const __nv_bfloat16* k_new;  // nullptr: the new keys are already in the cache
  const __nv_bfloat16* v_new;
  int64_t kn_sb, kn_sm, kn_sh;
  int64_t vn_sb, vn_sm, vn_sh;
  const int* prefix_len;       // [B]
  const int* cache_batch_idx;  // nullptr: request b uses cache row b; else int32 [B], cache row of request b
  const int* leftpad;          // nullptr: 0; else int32 [B]: the prefix keys of request b are rows [leftpad, prefix_len)
  int pool;                    // cache rows (dim 0 of k_cache / v_cache)
  const int* flash_mask;       // nullptr or int32 [B, 1 | Hkv, L, 2] (LTS, UTE) per key column (cache row)
  int64_t fm_sb, fm_sh;        // element strides (fm_sh = 0 for one shared head)
  float* ws_o;                 // [B, Hkv, nchunk, G * M, D] FP32
  float2* ws_ml;               // [B, Hkv, nchunk, G * M] (max in log2 units, sum)
  void* out;                   // BF16 or E4M3 [B, M, Hq * D]
  uint8_t* out_scale;          // E8M0 SWIZZLE_32_4_4 for [B * M, Hq * D] (MXFP8 only)
  int B, M, L, Hkv, new_pos, causal, nchunk, out_quant;
  int row_groups;  // ceil(G * M / ROWS): CTAs per (chunk, KV head)
  float scale_log2;  // softmax_scale * log2(e)
};

namespace splitkv_detail {

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
// byte offset of 16-byte unit `unit` of row `row` in a [rows][D] BF16 tile, XOR-swizzled within 128 B
template <int D>
__device__ __forceinline__ uint32_t swz(int row, int unit) {
  return static_cast<uint32_t>(row * (D * 2) + ((unit ^ (row & 7)) << 4));
}
__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst), "l"(src) : "memory");
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
template <int N>
__device__ __forceinline__ void cp_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory");
}
__device__ __forceinline__ void st_shared16(uint32_t dst, uint4 v) {
  asm volatile("st.shared.v4.u32 [%0], {%1, %2, %3, %4};\n" ::"r"(dst), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w)
               : "memory");
}
__device__ __forceinline__ void ldsm4(uint32_t (&r)[4], uint32_t a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(a));
}
__device__ __forceinline__ void ldsm4t(uint32_t (&r)[4], uint32_t a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(a));
}
__device__ __forceinline__ void mma16816(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) {
  __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&v);
}
__device__ __forceinline__ float ex2(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

}  // namespace splitkv_detail

// Shared memory bytes of one partial CTA (WG = 2: plus the S exchange tile, CHUNK FP32 per row).
template <int D, int CHUNK, int WG = 1>
constexpr size_t SplitkvSmemBytes(int rows_pad) {
  return static_cast<size_t>(rows_pad + 2 * CHUNK) * D * 2 +
         (WG == 2 ? static_cast<size_t>(rows_pad) * CHUNK * 4 + 2048 : 0);
}

// ROWS: query rows per CTA (64: more CTAs on a wide GPU; 256: every row of a KV head in one CTA).
// WG: warp groups per row tile. WG = 2: the two warps of a 16-row tile split the work of the chunk without
// changing any arithmetic: each computes the S = Q K^T columns of half the keys
// shared memory, each masks and exponentiates its own columns, the row maxima and the P columns are exchanged
// (group 0 sums the row in the WG = 1 order), and each computes half of the O = P V columns. The result is
// bit-identical to WG = 1; the per-warp MMA and softmax chains are halved.
template <int D, int G, int CHUNK, int ROWS, int WG = 1>
__global__ void __launch_bounds__(ROWS * 2 * WG) splitkv_partial_kernel(const SplitkvParams p) {
  using namespace splitkv_detail;
  constexpr int U = D / 8;  // 16-byte units per row
  static_assert(CHUNK % 16 == 0 && D % 64 == 0, "tile shape");
  extern __shared__ __align__(128) unsigned char smem[];
  const int rows = G * p.M;  // all rows of the KV head; this CTA owns [row0, row0 + rows_here)
  const int chunk = blockIdx.x, h = blockIdx.y / p.row_groups, b = blockIdx.z;
  const int row0 = (blockIdx.y % p.row_groups) * ROWS;
  const int rows_here = min(ROWS, rows - row0);
  const int rows_pad = (rows_here + 15) & ~15;
  const uint32_t sQ = smem_u32(smem);
  const uint32_t sK = sQ + rows_pad * D * 2;
  const uint32_t sV = sK + CHUNK * D * 2;
  // WG = 2 only: P exchange [rows_pad / 16 tiles][CHUNK / 8][32 lanes] float4, then row maxima [tiles][2][32] float2
  const uint32_t sS = sV + CHUNK * D * 2;
  static_assert(WG == 1 || (WG == 2 && CHUNK % 32 == 0 && D % 32 == 0), "warp groups split S and O in halves");
  constexpr int kThreads = ROWS * 2 * WG;
  constexpr int RS = kThreads / U;  // rows per pass of the copy loops (thread = one 16-byte unit of a row)
  static_assert(RS == G || G % RS == 0 || RS % G == 0, "a pass covers whole tokens or a whole fraction of one");
  const int tid = threadIdx.x, nthr = kThreads;
  const int cu = tid % U, rr = tid / U;  // this thread's unit and first row of every copy loop

  // prefix_len and the prefix rows are not produced by the preceding kernel (contract of pdl=True), so they
  // are read before the dependency wait.
  // With leftpad lp the visible keys are the rows [lp, prefix_len) and [p, p + M) of cache row cb: the key
  // sequence is counted from row lp (prefix and new position below are relative to lp), so without leftpad /
  // cache_batch_idx (lp = 0, cb = b) the kernel is the one before these parameters.
  const int cb = p.cache_batch_idx != nullptr ? p.cache_batch_idx[b] : b;
  const int lp = p.leftpad != nullptr ? p.leftpad[b] : 0;
  const int prefix_abs = p.prefix_len[b];
  const int pos_abs = p.new_pos >= 0 ? p.new_pos : prefix_abs;
  if (cb < 0 || cb >= p.pool || lp < 0 || lp > prefix_abs || prefix_abs > pos_abs || pos_abs + p.M > p.L) {
    if (chunk == 0 && h == 0 && tid == 0) asm volatile("trap;");
    return;
  }
  const int prefix = prefix_abs - lp;
  const int pos_new = pos_abs - lp;
  const int n = prefix + p.M;
  const int u0 = chunk * CHUNK;
  if (u0 >= n) {
    // no visible key in this chunk: an empty partial (max -inf, sum 0) for the merge, which then reads every
    // chunk of L without depending on prefix_len (after the wait: the previous call may still read ws_ml)
    xinfer::cuda::pdl_wait();
    xinfer::cuda::pdl_trigger();
    const int64_t part = (static_cast<int64_t>(b) * p.Hkv + h) * p.nchunk + chunk;
    for (int r = tid; r < rows_here; r += nthr) p.ws_ml[part * rows + row0 + r] = make_float2(-INFINITY, 0.f);
    return;
  }
  const int nk = min(CHUNK, n - u0);
  const int npre = min(max(prefix - u0, 0), nk);  // keys [0, npre) of the chunk are prefix rows
  __nv_bfloat16* const kcb = p.k_cache + cb * p.kc_sb + h * p.kc_sh + static_cast<int64_t>(lp) * p.kc_sr;
  __nv_bfloat16* const vcb = p.v_cache + cb * p.vc_sb + h * p.vc_sh + static_cast<int64_t>(lp) * p.vc_sr;
  const __nv_bfloat16* kc = kcb;
  const __nv_bfloat16* vc = vcb;

  // Copy loops: thread (rr, cu) moves unit cu of rows rr, rr + RS, ...; global pointers advance by RS rows.
  const int64_t kstep = static_cast<int64_t>(RS) * p.kc_sr, vstep = static_cast<int64_t>(RS) * p.vc_sr;
  // ---- prefix K (independent of the preceding kernel) ----
  {
    const __nv_bfloat16* src = kc + static_cast<int64_t>(u0 + rr) * p.kc_sr + cu * 8;
    for (int r = rr; r < npre; r += RS, src += kstep) cp_async16(sK + swz<D>(r, cu), src);
  }
  cp_commit();
  xinfer::cuda::pdl_wait();
  xinfer::cuda::pdl_trigger();

  // ---- Q (rows past G * M zero), new keys, zero rows past n ----
  {
    // row R = row0 + r is token R / G, head R % G
    const __nv_bfloat16* qbase = p.q + b * p.q_sb + static_cast<int64_t>(h) * G * p.q_sh + cu * 8;
    int r = rr;
    if constexpr (RS == G) {  // a pass of RS = G rows is one token: the pointer advances one token per pass
      const __nv_bfloat16* src =
          qbase + static_cast<int64_t>((row0 + rr) / G) * p.q_sm + ((row0 + rr) % G) * p.q_sh;
      for (; r < rows_here; r += RS, src += p.q_sm) cp_async16(sQ + swz<D>(r, cu), src);
    } else {
      for (; r < rows_here; r += RS) {
        const int R = row0 + r;
        cp_async16(sQ + swz<D>(r, cu), qbase + static_cast<int64_t>(R / G) * p.q_sm + (R % G) * p.q_sh);
      }
    }
    for (; r < rows_pad; r += RS) st_shared16(sQ + swz<D>(r, cu), make_uint4(0, 0, 0, 0));
  }
  const int nsuf = nk - npre;
  const int t0 = u0 + npre - prefix;  // first new-token index of the chunk (if nsuf > 0)
  if (p.k_new == nullptr) {
    const __nv_bfloat16* src = kc + static_cast<int64_t>(pos_new + t0 + rr) * p.kc_sr + cu * 8;
    for (int r = rr; r < nsuf; r += RS, src += kstep) cp_async16(sK + swz<D>(npre + r, cu), src);
  }
  cp_commit();
  {  // V last: its arrival overlaps S = Q K^T and the softmax
    const __nv_bfloat16* src = vc + static_cast<int64_t>(u0 + rr) * p.vc_sr + cu * 8;
    for (int r = rr; r < npre; r += RS, src += vstep) cp_async16(sV + swz<D>(r, cu), src);
  }
  if (p.k_new == nullptr) {
    const __nv_bfloat16* src = vc + static_cast<int64_t>(pos_new + t0 + rr) * p.vc_sr + cu * 8;
    for (int r = rr; r < nsuf; r += RS, src += vstep) cp_async16(sV + swz<D>(npre + r, cu), src);
  } else {
    // append: k_new / v_new -> cache rows pos_new + t (side effect) and into the tile
    const __nv_bfloat16* kn = p.k_new + b * p.kn_sb + h * p.kn_sh + static_cast<int64_t>(t0 + rr) * p.kn_sm + cu * 8;
    const __nv_bfloat16* vn = p.v_new + b * p.vn_sb + h * p.vn_sh + static_cast<int64_t>(t0 + rr) * p.vn_sm + cu * 8;
    __nv_bfloat16* kcw = kcb + static_cast<int64_t>(pos_new + t0 + rr) * p.kc_sr + cu * 8;
    __nv_bfloat16* vcw = vcb + static_cast<int64_t>(pos_new + t0 + rr) * p.vc_sr + cu * 8;
    const int64_t knstep = static_cast<int64_t>(RS) * p.kn_sm, vnstep = static_cast<int64_t>(RS) * p.vn_sm;
    for (int r = rr; r < nsuf; r += RS, kn += knstep, vn += vnstep, kcw += kstep, vcw += vstep) {
      const uint4 kv = *reinterpret_cast<const uint4*>(kn);
      const uint4 vv = *reinterpret_cast<const uint4*>(vn);
      if (row0 == 0) {  // one row group per chunk writes the cache
        *reinterpret_cast<uint4*>(kcw) = kv;
        *reinterpret_cast<uint4*>(vcw) = vv;
      }
      st_shared16(sK + swz<D>(npre + r, cu), kv);
      st_shared16(sV + swz<D>(npre + r, cu), vv);
    }
  }
  cp_commit();
  // keys past n: zero (stale cache rows may hold NaN / Inf, and 0 * NaN in P V would poison the row)
  for (int r = nk + rr; r < CHUNK; r += RS) {
    st_shared16(sK + swz<D>(r, cu), make_uint4(0, 0, 0, 0));
    st_shared16(sV + swz<D>(r, cu), make_uint4(0, 0, 0, 0));
  }
  cp_wait<1>();  // Q and K (V may still be in flight)
  __syncthreads();

  const int warp = tid >> 5, lane = tid & 31;
  constexpr int kTiles = ROWS / 16;  // 16-row tiles per CTA
  const int wt = WG == 1 ? warp : warp % kTiles;  // this warp's row tile
  const int wg = WG == 1 ? 0 : warp / kTiles;     // and its warp group
  const int wrow = wt * 16;
  const int g = lane >> 2, q4 = lane & 3;
  // ---- S = Q K^T (warps past the tile rows only help with the loads) ----
  const bool active = wrow < rows_pad;
  float s[CHUNK / 8][4];
#pragma unroll
  for (int j = 0; j < CHUNK / 8; ++j) s[j][0] = s[j][1] = s[j][2] = s[j][3] = 0.f;
  const int r_lo = row0 + wrow + g, r_hi = r_lo + 8;
  const int64_t part = (static_cast<int64_t>(b) * p.Hkv + h) * p.nchunk + chunk;
  if (active) {
    if constexpr (WG == 1) {
#pragma unroll 4
      for (int kk = 0; kk < D / 16; ++kk) {
        uint32_t a[4];
        ldsm4(a, sQ + swz<D>(wrow + (lane & 15), kk * 2 + (lane >> 4)));
#pragma unroll
        for (int j = 0; j < CHUNK / 8; j += 2) {
          uint32_t bb[4];
          ldsm4(bb, sK + swz<D>((j + (lane >> 4)) * 8 + (lane & 7), kk * 2 + ((lane >> 3) & 1)));
          mma16816(s[j], a, bb[0], bb[1]);
          mma16816(s[j + 1], a, bb[2], bb[3]);
        }
      }
    } else {
      // this group's key columns j0 .. j0 + CHUNK / 16 into s[0 .. CHUNK / 16), each column in the same kk order
      constexpr int JH = CHUNK / 16;
      const int j0 = wg * JH;
#pragma unroll 4
      for (int kk = 0; kk < D / 16; ++kk) {
        uint32_t a[4];
        ldsm4(a, sQ + swz<D>(wrow + (lane & 15), kk * 2 + (lane >> 4)));
#pragma unroll
        for (int j = 0; j < JH; j += 2) {
          uint32_t bb[4];
          ldsm4(bb, sK + swz<D>((j0 + j + (lane >> 4)) * 8 + (lane & 7), kk * 2 + ((lane >> 3) & 1)));
          mma16816(s[j], a, bb[0], bb[1]);
          mma16816(s[j + 1], a, bb[2], bb[3]);
        }
      }
    }
    // ---- mask + softmax over the chunk (rows r_lo and r_hi) ----
    const int t_lo = r_lo / G, t_hi = r_hi / G;
    // mask columns are cache rows; the key sequence starts at row lp
    const int* fm = p.flash_mask ? p.flash_mask + b * p.fm_sb + h * p.fm_sh + 2 * static_cast<int64_t>(lp) : nullptr;
    // key kidx of the chunk is visible to a row iff kidx < limit: past nk there is no key; causal hides the new
    // keys after the row's token (new key u - prefix <= t  <=>  kidx < prefix - u0 + t + 1). Compared against
    // j * 8 + e with the lane's column offset 2 q4 folded into the limit.
    int lim_lo = nk, lim_hi = nk;
    if (p.causal) {
      lim_lo = min(nk, prefix - u0 + t_lo + 1);
      lim_hi = min(nk, prefix - u0 + t_hi + 1);
    }
    lim_lo -= 2 * q4;
    lim_hi -= 2 * q4;
    float m_lo = -INFINITY, m_hi = -INFINITY;
    float l_lo = 0.f, l_hi = 0.f;
    if constexpr (WG == 1) {
#pragma unroll
      for (int j = 0; j < CHUNK / 8; ++j) {
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          bool ok_lo = j * 8 + e < lim_lo, ok_hi = j * 8 + e < lim_hi;
          if (fm != nullptr) {
            const int kidx = j * 8 + q4 * 2 + e;  // key in chunk
            const int u = u0 + kidx;
            if (kidx < nk) {
              const int col = u < prefix ? u : pos_new + (u - prefix);
              const int2 se = *reinterpret_cast<const int2*>(fm + 2 * static_cast<int64_t>(col));
              ok_lo = ok_lo && !(t_lo >= se.x || t_lo < se.y);
              ok_hi = ok_hi && !(t_hi >= se.x || t_hi < se.y);
            }
          }
          s[j][e] = ok_lo ? s[j][e] * p.scale_log2 : -INFINITY;
          s[j][2 + e] = ok_hi ? s[j][2 + e] * p.scale_log2 : -INFINITY;
          m_lo = fmaxf(m_lo, s[j][e]);
          m_hi = fmaxf(m_hi, s[j][2 + e]);
        }
      }
      m_lo = fmaxf(m_lo, __shfl_xor_sync(0xffffffffu, m_lo, 1));
      m_lo = fmaxf(m_lo, __shfl_xor_sync(0xffffffffu, m_lo, 2));
      m_hi = fmaxf(m_hi, __shfl_xor_sync(0xffffffffu, m_hi, 1));
      m_hi = fmaxf(m_hi, __shfl_xor_sync(0xffffffffu, m_hi, 2));
      const float mz_lo = m_lo == -INFINITY ? 0.f : m_lo, mz_hi = m_hi == -INFINITY ? 0.f : m_hi;
#pragma unroll
      for (int j = 0; j < CHUNK / 8; ++j) {
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          s[j][e] = ex2(s[j][e] - mz_lo);
          s[j][2 + e] = ex2(s[j][2 + e] - mz_hi);
          l_lo += s[j][e];
          l_hi += s[j][2 + e];
        }
      }
      l_lo += __shfl_xor_sync(0xffffffffu, l_lo, 1);
      l_lo += __shfl_xor_sync(0xffffffffu, l_lo, 2);
      l_hi += __shfl_xor_sync(0xffffffffu, l_hi, 1);
      l_hi += __shfl_xor_sync(0xffffffffu, l_hi, 2);
    } else {
      // Each group masks, scales and exponentiates its own key columns (element-wise: exact), the row maxima are
      // combined through shared memory (a max is exact in any order), the P columns are exchanged, and group 0
      // sums the full row in the WG = 1 order. Every value equals the WG = 1 one bit for bit.
      constexpr int JH = CHUNK / 16;
      const int j0 = wg * JH;
#pragma unroll
      for (int jj = 0; jj < JH; ++jj) {
        const int j = j0 + jj;
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          bool ok_lo = j * 8 + e < lim_lo, ok_hi = j * 8 + e < lim_hi;
          if (fm != nullptr) {
            const int kidx = j * 8 + q4 * 2 + e;  // key in chunk
            const int u = u0 + kidx;
            if (kidx < nk) {
              const int col = u < prefix ? u : pos_new + (u - prefix);
              const int2 se = *reinterpret_cast<const int2*>(fm + 2 * static_cast<int64_t>(col));
              ok_lo = ok_lo && !(t_lo >= se.x || t_lo < se.y);
              ok_hi = ok_hi && !(t_hi >= se.x || t_hi < se.y);
            }
          }
          s[jj][e] = ok_lo ? s[jj][e] * p.scale_log2 : -INFINITY;
          s[jj][2 + e] = ok_hi ? s[jj][2 + e] * p.scale_log2 : -INFINITY;
          m_lo = fmaxf(m_lo, s[jj][e]);
          m_hi = fmaxf(m_hi, s[jj][2 + e]);
        }
      }
      m_lo = fmaxf(m_lo, __shfl_xor_sync(0xffffffffu, m_lo, 1));
      m_lo = fmaxf(m_lo, __shfl_xor_sync(0xffffffffu, m_lo, 2));
      m_hi = fmaxf(m_hi, __shfl_xor_sync(0xffffffffu, m_hi, 1));
      m_hi = fmaxf(m_hi, __shfl_xor_sync(0xffffffffu, m_hi, 2));
      const uint32_t xm = sS + static_cast<uint32_t>(rows_pad * CHUNK * 4) + static_cast<uint32_t>((wt * 2 * 32) * 8);
      asm volatile("st.shared.v2.f32 [%0], {%1, %2};\n" ::"r"(xm + (wg * 32 + lane) * 8), "f"(m_lo), "f"(m_hi)
                   : "memory");
      asm volatile("bar.sync %0, 64;\n" ::"r"(1 + wt) : "memory");  // the two warps of tile wt
      float o_lo, o_hi;
      asm volatile("ld.shared.v2.f32 {%0, %1}, [%2];\n"
                   : "=f"(o_lo), "=f"(o_hi)
                   : "r"(xm + ((1 - wg) * 32 + lane) * 8)
                   : "memory");
      m_lo = fmaxf(m_lo, o_lo);
      m_hi = fmaxf(m_hi, o_hi);
      const float mz_lo = m_lo == -INFINITY ? 0.f : m_lo, mz_hi = m_hi == -INFINITY ? 0.f : m_hi;
      const uint32_t xs = sS + static_cast<uint32_t>(wt * (CHUNK / 8) * 32 + lane) * 16;
#pragma unroll
      for (int jj = 0; jj < JH; ++jj) {
        asm volatile("st.shared.v4.f32 [%0], {%1, %2, %3, %4};\n" ::"r"(xs + (j0 + jj) * 512),
                     "f"(ex2(s[jj][0] - mz_lo)), "f"(ex2(s[jj][1] - mz_lo)), "f"(ex2(s[jj][2] - mz_hi)),
                     "f"(ex2(s[jj][3] - mz_hi))
                     : "memory");
      }
      asm volatile("bar.sync %0, 64;\n" ::"r"(1 + wt) : "memory");
#pragma unroll
      for (int j = 0; j < CHUNK / 8; ++j) {
        asm volatile("ld.shared.v4.f32 {%0, %1, %2, %3}, [%4];\n"
                     : "=f"(s[j][0]), "=f"(s[j][1]), "=f"(s[j][2]), "=f"(s[j][3])
                     : "r"(xs + j * 512)
                     : "memory");
      }
      if (wg == 0) {
#pragma unroll
        for (int j = 0; j < CHUNK / 8; ++j) {
#pragma unroll
          for (int e = 0; e < 2; ++e) {
            l_lo += s[j][e];
            l_hi += s[j][2 + e];
          }
        }
        l_lo += __shfl_xor_sync(0xffffffffu, l_lo, 1);
        l_lo += __shfl_xor_sync(0xffffffffu, l_lo, 2);
        l_hi += __shfl_xor_sync(0xffffffffu, l_hi, 1);
        l_hi += __shfl_xor_sync(0xffffffffu, l_hi, 2);
      }
    }
    if (q4 == 0 && wg == 0) {
      if (r_lo < rows) p.ws_ml[part * rows + r_lo] = make_float2(m_lo, l_lo);
      if (r_hi < rows) p.ws_ml[part * rows + r_hi] = make_float2(m_hi, l_hi);
    }
  }
  // ---- O = P V (WG = 2: this group's half of the D / 8 column blocks) ----
  cp_wait<0>();
  __syncthreads();
  if (!active) return;
  constexpr int JO = D / 8 / WG;
  const int jo0 = wg * JO;
  float o[JO][4];
#pragma unroll
  for (int j = 0; j < JO; ++j) o[j][0] = o[j][1] = o[j][2] = o[j][3] = 0.f;
#pragma unroll
  for (int k2 = 0; k2 < CHUNK / 16; ++k2) {
    const uint32_t a[4] = {pack_bf16(s[2 * k2][0], s[2 * k2][1]), pack_bf16(s[2 * k2][2], s[2 * k2][3]),
                           pack_bf16(s[2 * k2 + 1][0], s[2 * k2 + 1][1]),
                           pack_bf16(s[2 * k2 + 1][2], s[2 * k2 + 1][3])};
#pragma unroll
    for (int j = 0; j < JO; j += 2) {
      uint32_t bb[4];
      ldsm4t(bb, sV + swz<D>(k2 * 16 + (lane & 7) + ((lane >> 3) & 1) * 8, jo0 + j + (lane >> 4)));
      mma16816(o[j], a, bb[0], bb[1]);
      mma16816(o[j + 1], a, bb[2], bb[3]);
    }
  }
  float* op = p.ws_o + part * rows * D;
#pragma unroll
  for (int j = 0; j < JO; ++j) {
    const int col = (jo0 + j) * 8 + q4 * 2;
    if (r_lo < rows) *reinterpret_cast<float2*>(op + static_cast<int64_t>(r_lo) * D + col) = make_float2(o[j][0], o[j][1]);
    if (r_hi < rows) *reinterpret_cast<float2*>(op + static_cast<int64_t>(r_hi) * D + col) = make_float2(o[j][2], o[j][3]);
  }
}

// One warp per output row (b, token t, query head qh); lanes own D / 32 consecutive columns. Every chunk of L is
// merged in chunk order (chunks without visible keys hold (-inf, 0) and are skipped: their weight is 0), so the
// kernel does not read prefix_len. OQ (output format) is a template parameter: the block width and the lanes per
// scale block are compile-time.
// BATCH: chunk partials in flight per warp (SM120: 4, SM110: 2: fastest measured). It changes no result.
template <int D, int G, int CHUNK, int OQ, int BATCH>
__global__ void __launch_bounds__(256) splitkv_finalize_kernel(const SplitkvParams p) {
  using namespace splitkv_detail;
  constexpr int E = D / 32;  // columns per lane
  const int Hq = p.Hkv * G;
  const int gw = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31;
  xinfer::cuda::pdl_wait();
  xinfer::cuda::pdl_trigger();
  if (gw >= p.B * p.M * Hq) return;
  const int qh = gw % Hq, t = (gw / Hq) % p.M, b = gw / (Hq * p.M);
  const int h = qh / G, r = t * G + (qh % G);
  const int rows = G * p.M;
  const int nchunk = p.nchunk;
  const float2* ml_base = p.ws_ml + (static_cast<int64_t>(b) * p.Hkv + h) * nchunk * rows + r;
  const float* o_base = p.ws_o + ((static_cast<int64_t>(b) * p.Hkv + h) * nchunk * rows + r) * D + lane * E;
  const int64_t o_stride = static_cast<int64_t>(rows) * D;  // between chunks
  float mx = -INFINITY;
  float2 ml_lane = make_float2(-INFINITY, 0.f);
  if (nchunk <= 32) {
    if (lane < nchunk) ml_lane = __ldcg(ml_base + static_cast<int64_t>(lane) * rows);
    mx = ml_lane.x;
  } else {
    for (int c = lane; c < nchunk; c += 32) mx = fmaxf(mx, __ldcg(ml_base + static_cast<int64_t>(c) * rows).x);
  }
#pragma unroll
  for (int sh = 16; sh; sh >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, sh));
  float acc[E];
#pragma unroll
  for (int i = 0; i < E; ++i) acc[i] = 0.f;
  float den = 0.f;
  for (int c0 = 0; c0 < nchunk; c0 += 32) {
    float w = 0.f;
    if (c0 + lane < nchunk) {
      const float2 ml = nchunk <= 32 ? ml_lane : __ldcg(ml_base + static_cast<int64_t>(c0 + lane) * rows);
      // ml.x is the chunk max relative to which ml.y and the partial O were accumulated (sum 0: empty chunk)
      w = ml.y > 0.f ? ex2(ml.x - mx) : 0.f;
      den = fmaf(w, ml.y, den);
    }
    const int cnt = min(32, nchunk - c0);
    if constexpr (BATCH == 1) {
      for (int i = 0; i < cnt; ++i) {
        const float wc = __shfl_sync(0xffffffffu, w, i);
        if (wc != 0.f) {  // warp-uniform; a zero weight adds exactly nothing
          const float4* src = reinterpret_cast<const float4*>(o_base + (c0 + i) * o_stride);
#pragma unroll
          for (int v = 0; v < E / 4; ++v) {
            const float4 x = __ldcg(src + v);
            acc[4 * v] = fmaf(wc, x.x, acc[4 * v]);
            acc[4 * v + 1] = fmaf(wc, x.y, acc[4 * v + 1]);
            acc[4 * v + 2] = fmaf(wc, x.z, acc[4 * v + 2]);
            acc[4 * v + 3] = fmaf(wc, x.w, acc[4 * v + 3]);
          }
        }
      }
      continue;
    }
    constexpr int kBatch = BATCH;  // partials in flight, accumulated in chunk order
    for (int i0 = 0; i0 < cnt; i0 += kBatch) {
      float wc[kBatch];
      float4 x[kBatch][E / 4];
#pragma unroll
      for (int i = 0; i < kBatch; ++i) {
        wc[i] = __shfl_sync(0xffffffffu, w, (i0 + i) & 31);
        if (i0 + i >= cnt) wc[i] = 0.f;
        if (wc[i] != 0.f) {  // warp-uniform; a zero weight adds exactly nothing
          const float4* src = reinterpret_cast<const float4*>(o_base + (c0 + i0 + i) * o_stride);
#pragma unroll
          for (int v = 0; v < E / 4; ++v) x[i][v] = __ldcg(src + v);
        }
      }
#pragma unroll
      for (int i = 0; i < kBatch; ++i) {
        if (wc[i] != 0.f) {
#pragma unroll
          for (int v = 0; v < E / 4; ++v) {
            acc[4 * v] = fmaf(wc[i], x[i][v].x, acc[4 * v]);
            acc[4 * v + 1] = fmaf(wc[i], x[i][v].y, acc[4 * v + 1]);
            acc[4 * v + 2] = fmaf(wc[i], x[i][v].z, acc[4 * v + 2]);
            acc[4 * v + 3] = fmaf(wc[i], x[i][v].w, acc[4 * v + 3]);
          }
        }
      }
    }
  }
#pragma unroll
  for (int sh = 16; sh; sh >>= 1) den += __shfl_xor_sync(0xffffffffu, den, sh);
  const float inv = den > 0.f ? 1.f / den : 0.f;
#pragma unroll
  for (int i = 0; i < E; ++i) acc[i] *= inv;
  const int orow = b * p.M + t;
  const int col0 = qh * D + lane * E;
  if constexpr (OQ == kOutBf16) {
    __nv_bfloat16* dst = static_cast<__nv_bfloat16*>(p.out) + static_cast<int64_t>(orow) * Hq * D + col0;
    if constexpr (E >= 8) {
#pragma unroll
      for (int v = 0; v < E; v += 8) {
        *reinterpret_cast<uint4*>(dst + v) =
            make_uint4(pack_bf16(acc[v], acc[v + 1]), pack_bf16(acc[v + 2], acc[v + 3]),
                       pack_bf16(acc[v + 4], acc[v + 5]), pack_bf16(acc[v + 6], acc[v + 7]));
      }
    } else {
      *reinterpret_cast<uint2*>(dst) = make_uint2(pack_bf16(acc[0], acc[1]), pack_bf16(acc[2], acc[3]));
    }
  } else {
    constexpr int kBlock = OQ == kOutMxfp8Blk32 ? 32 : 128;
    constexpr int kLanes = kBlock / E;  // lanes per scale block (power of two, <= 32)
    float am = 0.f;
#pragma unroll
    for (int i = 0; i < E; ++i) am = fmaxf(am, fabsf(acc[i]));
#pragma unroll
    for (int sh = 1; sh < kLanes; sh <<= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, sh));
    const uint8_t e = xinfer::mxfp8::amax_to_e8m0_ceil(am);
    const float qs = xinfer::mxfp8::e8m0_inv(e);
    uint8_t* qdst = static_cast<uint8_t*>(p.out) + static_cast<int64_t>(orow) * Hq * D + col0;
#pragma unroll
    for (int v = 0; v < E; v += 4) {
      uint32_t pk = 0;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const __nv_fp8_storage_t f = __nv_cvt_float_to_fp8(acc[v + i] * qs, __NV_SATFINITE, __NV_E4M3);
        pk |= static_cast<uint32_t>(f) << (8 * i);
      }
      *reinterpret_cast<uint32_t*>(qdst + v) = pk;
    }
    if ((lane & (kLanes - 1)) == 0) {
      // SWIZZLE_32_4_4: 512-byte groups of (32 rows x 4 blocks); Hq * D is a multiple of 128 for every kernel
      const int kb = col0 / 32;
      const int off = ((orow >> 7) * (Hq * D / 128) + (kb >> 2)) * 512 + (orow & 31) * 16 + ((orow & 127) >> 5) * 4;
      if constexpr (kBlock == 32) {
        p.out_scale[off + (kb & 3)] = e;
      } else {  // 1x128: replicated into the four 1x32 slots of the group
        *reinterpret_cast<uint32_t*>(p.out_scale + off) = 0x01010101u * e;
      }
    }
  }
}

}  // namespace xinfer::attention_kvcache
