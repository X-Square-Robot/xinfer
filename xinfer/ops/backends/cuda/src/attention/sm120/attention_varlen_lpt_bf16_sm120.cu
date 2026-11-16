// attention_varlen, SM120 backend varlen_lpt_bf16_sm120: packed variable-length segments, non-causal, head_dim 64,
// q / k / v heads equal, BF16 output.
//
// Persistent, one CTA per SM with two independent teams (4 MMA warps + 1 TMA producer warp each, own Q / K / V
// pipelines). Work units (segment, head, 64 query rows) sorted by key block count (descending) are cut into G equal
// runs of key blocks, one per SM (a unit crossing an SM boundary is split by key blocks into two pieces); each SM's
// pieces are assigned to its two teams by LPT. The host-side plan is built by the Python wrapper (it needs the
// segment lengths on the host) and passed as `pieces` / `tstart`.
// Split units: both pieces write FP32 partial results (o, m, l) to the workspace and count on a per-pair counter;
// whichever piece finishes last merges both in a fixed order (front, back) and writes the rows, then resets the
// counter. No CTA waits for another CTA; results are bitwise deterministic.
//
// attention_varlen_lpt_bf16_sm120(out, q, k, v, pieces, tstart, ws, cnt, softmax_scale)
//   q / k / v  [T, H, 64] BF16 views (last dim contiguous, row stride a 16-byte multiple, 16-byte aligned base)
//   out        [T, H * 64] BF16 contiguous
//   pieces     int32 [n, 8] (seg, len, head, row0, kb0, kb1, part, pid); tstart int32 [2G + 1] (G = grid size)
//   ws         FP32 >= pairs * 2 * 128 * 36 floats; cnt int32 >= pairs, zero (left zero by the kernel)
// Caller stream, asynchronous, no allocation.
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/function.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "common/tvm_ffi_cuda_tensor.h"
#include "common/tensor_map.h"

#define PA_STAGES 2

namespace xinfer::attention_varlen::lpt_sm120 {
namespace {

constexpr int D = 64, BQ = 64, BKV = 64, STAGES = PA_STAGES, MMA_WARPS = 4;
constexpr int THREADS = (MMA_WARPS + 1) * 32;
constexpr int TILE_BYTES = 64 * 128;  // 64 rows x 64 BF16 columns, 128 B per row

__device__ __forceinline__ uint32_t smem_u32(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }
__device__ __forceinline__ void mbar_init(uint64_t* b, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_u32(b)), "r"(count));
}
__device__ __forceinline__ void mbar_expect_tx(uint64_t* b, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem_u32(b)), "r"(bytes));
}
__device__ __forceinline__ void mbar_arrive(uint64_t* b) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(smem_u32(b)) : "memory"); }
__device__ __forceinline__ void mbar_wait(uint64_t* b, uint32_t parity) {
  asm volatile(
      "{\n .reg .pred p;\n"
      "WAIT_%=:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
      " @!p bra WAIT_%=;\n}\n" ::"r"(smem_u32(b)),
      "r"(parity) : "memory");
}
__device__ __forceinline__ void tma_load_2d(void* dst, const CUtensorMap* map, uint64_t* bar, int x, int y) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];" ::"r"(
          smem_u32(dst)),
      "l"(map), "r"(smem_u32(bar)), "r"(x), "r"(y)
      : "memory");
}
__device__ __forceinline__ void ldsm_x4(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
__device__ __forceinline__ void ldsm_x4_t(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
__device__ __forceinline__ void mma16816(float* c, const uint32_t* a, const uint32_t* b) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ float ex2(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) {
  const __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<const uint32_t*>(&v);
}
// 128B swizzle: 64 rows x 128 B tile, the 16 B chunk index XOR (row & 7)
__device__ __forceinline__ uint32_t swz(uint32_t base, int row, int chunk) { return base + row * 128 + ((chunk ^ (row & 7)) << 4); }

struct Piece {
  int seg, len, head, row0;  // segment start row, segment length, head, first query row of the unit (in the segment)
  int kb0, kb1;              // key block range
  int part;                  // -1 whole unit; 0 / 1 front / back piece of a unit split across two SMs
  int pid;                   // index of the split pair
};
struct TeamSmem {  // a multiple of 1024 bytes (barriers live outside so two padded teams stay within 99 KB)
  alignas(1024) uint8_t q[2][TILE_BYTES];
  alignas(1024) uint8_t k[STAGES][TILE_BYTES];
  alignas(1024) uint8_t v[STAGES][TILE_BYTES];
};
struct TeamBars {
  uint64_t qfull[2], qempty[2];
  uint64_t full[STAGES], empty[STAGES];
};
struct Smem {
  TeamSmem t[2];
  TeamBars b[2];
  int last[2];
};
struct Params {
  const Piece* pieces;
  const int* tstart;  // [2G+1]: piece range of team (cta * 2 + team)
  __nv_bfloat16* O;
  int H;
  float scale_log2;
  float* ws;  // [pair][piece][128][36] FP32 partial results of split units
  int* cnt;
};
constexpr int TEAM_WARPS = MMA_WARPS + 1;
constexpr int CTA_THREADS = 2 * TEAM_WARPS * 32;

__global__ void __launch_bounds__(CTA_THREADS, 1) VarlenLptKernel(const __grid_constant__ CUtensorMap mapQ, const __grid_constant__ CUtensorMap mapK,
                                                                 const __grid_constant__ CUtensorMap mapV, const Params p) {
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  Smem& s = *reinterpret_cast<Smem*>((reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t(1023));
  const int gw = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int team = gw / TEAM_WARPS, warp = gw % TEAM_WARPS;
  TeamSmem& ts = s.t[team];
  TeamBars& tb = s.b[team];
  const int tix = blockIdx.x * 2 + team;
  const int pc0 = p.tstart[tix], pc1 = p.tstart[tix + 1];
  if (warp == 0 && lane == 0) {
    for (int i = 0; i < 2; ++i) mbar_init(&tb.qfull[i], 1), mbar_init(&tb.qempty[i], MMA_WARPS);
    for (int i = 0; i < STAGES; ++i) mbar_init(&tb.full[i], 1), mbar_init(&tb.empty[i], MMA_WARPS);
    asm volatile("fence.mbarrier_init.release.cluster;");
  }
  __syncthreads();
  if (warp == MMA_WARPS) {  // this team's TMA producer: prefetches across pieces
    if (lane == 0) {
      int g = 0;
      for (int pi = pc0; pi < pc1; ++pi) {
        const Piece u = p.pieces[pi];
        const int qb = (pi - pc0) & 1, k = (pi - pc0) >> 1;
        if (k >= 1) mbar_wait(&tb.qempty[qb], (k - 1) & 1);
        mbar_expect_tx(&tb.qfull[qb], TILE_BYTES);
        tma_load_2d(ts.q[qb], &mapQ, &tb.qfull[qb], u.head * D, u.seg + u.row0);
        const int kcol = u.head * D;
        for (int j = u.kb0; j < u.kb1; ++j, ++g) {
          const int st = g % STAGES;
          if (g >= STAGES) mbar_wait(&tb.empty[st], ((g / STAGES) - 1) & 1);
          mbar_expect_tx(&tb.full[st], 2 * TILE_BYTES);
          tma_load_2d(ts.k[st], &mapK, &tb.full[st], kcol, u.seg + j * BKV);
          tma_load_2d(ts.v[st], &mapV, &tb.full[st], kcol, u.seg + j * BKV);
        }
      }
    }
    return;
  }
  const int r0 = warp * 16;
  const int t4 = warp * 32 + lane;  // the team's 128 compute threads
  int g = 0;
  for (int pi = pc0; pi < pc1; ++pi) {
    const Piece u = p.pieces[pi];
    const int qb = (pi - pc0) & 1;
    const bool active = u.row0 + r0 < u.len;
    uint32_t qa[4][4];
    mbar_wait(&tb.qfull[qb], ((pi - pc0) >> 1) & 1);
    const uint32_t sq = smem_u32(ts.q[qb]);
#pragma unroll
    for (int kk = 0; kk < 4; ++kk) ldsm_x4(swz(sq, r0 + (lane & 15), kk * 2 + (lane >> 4)), qa[kk][0], qa[kk][1], qa[kk][2], qa[kk][3]);
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");  // as for K / V below
    __syncwarp();
    if (lane == 0) mbar_arrive(&tb.qempty[qb]);
    float o[8][4];
#pragma unroll
    for (int i = 0; i < 8; ++i) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.f;
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;
    for (int j = u.kb0; j < u.kb1; ++j, ++g) {
      const int st = g % STAGES;
      mbar_wait(&tb.full[st], (g / STAGES) & 1);
      const uint32_t sk = smem_u32(ts.k[st]), sv = smem_u32(ts.v[st]);
      float sc[8][4];
#pragma unroll
      for (int i = 0; i < 8; ++i) sc[i][0] = sc[i][1] = sc[i][2] = sc[i][3] = 0.f;
#pragma unroll
      for (int kk = 0; kk < 4; ++kk)
#pragma unroll
        for (int nj = 0; nj < 4; ++nj) {
          uint32_t b0, b1, b2, b3;
          ldsm_x4(swz(sk, nj * 16 + (lane & 7) + ((lane >> 4) << 3), kk * 2 + ((lane >> 3) & 1)), b0, b1, b2, b3);
          const uint32_t bb0[2] = {b0, b1}, bb1[2] = {b2, b3};
          mma16816(sc[2 * nj], qa[kk], bb0);
          mma16816(sc[2 * nj + 1], qa[kk], bb1);
        }
      const int kbase = j * BKV;
      if (kbase + BKV > u.len) {
#pragma unroll
        for (int i = 0; i < 8; ++i)
#pragma unroll
          for (int e = 0; e < 4; ++e)
            if (kbase + i * 8 + (lane & 3) * 2 + (e & 1) >= u.len) sc[i][e] = -INFINITY;
      }
      // row max on the raw scores, scale applied once per row; lazy rescale: O and l are rescaled only when the new max exceeds the reference by 2^8 (or on the first block)
      float r0 = -INFINITY, r1 = -INFINITY;
#pragma unroll
      for (int i = 0; i < 8; ++i) r0 = fmaxf(r0, fmaxf(sc[i][0], sc[i][1])), r1 = fmaxf(r1, fmaxf(sc[i][2], sc[i][3]));
      r0 = fmaxf(r0, __shfl_xor_sync(0xffffffffu, r0, 1));
      r0 = fmaxf(r0, __shfl_xor_sync(0xffffffffu, r0, 2));
      r1 = fmaxf(r1, __shfl_xor_sync(0xffffffffu, r1, 1));
      r1 = fmaxf(r1, __shfl_xor_sync(0xffffffffu, r1, 2));
      r0 *= p.scale_log2, r1 *= p.scale_log2;
      if (__any_sync(0xffffffffu, r0 > m0 + 8.f || r1 > m1 + 8.f)) {
        const float n0 = fmaxf(m0, r0), n1 = fmaxf(m1, r1);
        const float c0 = ex2(m0 - n0), c1 = ex2(m1 - n1);
        m0 = n0, m1 = n1;
        l0 *= c0, l1 *= c1;
#pragma unroll
        for (int i = 0; i < 8; ++i) o[i][0] *= c0, o[i][1] *= c0, o[i][2] *= c1, o[i][3] *= c1;
      }
      const float mx0 = m0, mx1 = m1;
      uint32_t pa[4][4];
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const float p0 = ex2(fmaf(sc[i][0], p.scale_log2, -mx0)), p1 = ex2(fmaf(sc[i][1], p.scale_log2, -mx0));
        const float p2 = ex2(fmaf(sc[i][2], p.scale_log2, -mx1)), p3 = ex2(fmaf(sc[i][3], p.scale_log2, -mx1));
        l0 += p0 + p1, l1 += p2 + p3;
        pa[i >> 1][(i & 1) * 2 + 0] = pack_bf16(p0, p1);
        pa[i >> 1][(i & 1) * 2 + 1] = pack_bf16(p2, p3);
      }
#pragma unroll
      for (int t = 0; t < 4; ++t)
#pragma unroll
        for (int nj = 0; nj < 4; ++nj) {
          uint32_t b0, b1, b2, b3;
          ldsm_x4_t(swz(sv, t * 16 + (lane & 7) + (((lane >> 3) & 1) << 3), nj * 2 + (lane >> 4)), b0, b1, b2, b3);
          const uint32_t bb0[2] = {b0, b1}, bb1[2] = {b2, b3};
          mma16816(o[2 * nj], pa[t], bb0);
          mma16816(o[2 * nj + 1], pa[t], bb1);
        }
      // Every lane orders its generic-proxy reads of the stage before the producer's next TMA (async-proxy) write into
      // it; without this fence a TMA refill could overwrite V while a slow warp was still reading it (seen as rare
      // run-to-run differences in one warp's output columns when two teams share an SM).
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
      __syncwarp();
      if (lane == 0) mbar_arrive(&tb.empty[st]);
    }
    l0 += __shfl_xor_sync(0xffffffffu, l0, 1);
    l0 += __shfl_xor_sync(0xffffffffu, l0, 2);
    l1 += __shfl_xor_sync(0xffffffffu, l1, 1);
    l1 += __shfl_xor_sync(0xffffffffu, l1, 2);
    if (u.part >= 0) {  // split across two SMs: write the partial result; whichever piece finishes last merges both in a fixed order (front, back)
      // layout [pair][piece][float4 j][128 threads]: one instruction reads / writes 512 contiguous bytes
      float4* mine = reinterpret_cast<float4*>(p.ws) + ((size_t)u.pid * 2 + u.part) * 9 * 128 + t4;
#pragma unroll
      for (int i = 0; i < 8; ++i) __stcg(mine + i * 128, make_float4(o[i][0], o[i][1], o[i][2], o[i][3]));
      __stcg(mine + 8 * 128, make_float4(m0, m1, l0, l1));
      __threadfence();
      asm volatile("bar.sync %0, 128;" ::"r"(1 + team) : "memory");
      if (t4 == 0) {
        const int old = atomicAdd(&p.cnt[u.pid], 1);
        s.last[team] = old == 1;
        if (old == 1) p.cnt[u.pid] = 0;
      }
      asm volatile("bar.sync %0, 128;" ::"r"(1 + team) : "memory");
      if (!s.last[team]) continue;
      __threadfence();
      const float4* a = reinterpret_cast<const float4*>(p.ws) + ((size_t)u.pid * 2 + 0) * 9 * 128 + t4;
      const float4* b = a + 9 * 128;
      const float4 sa = __ldcg(a + 8 * 128);
      const float4 sb = __ldcg(b + 8 * 128);
      const float M0 = fmaxf(sa.x, sb.x), M1 = fmaxf(sa.y, sb.y);
      const float wa0 = ex2(sa.x - M0), wb0 = ex2(sb.x - M0), wa1 = ex2(sa.y - M1), wb1 = ex2(sb.y - M1);
      l0 = sa.z * wa0 + sb.z * wb0, l1 = sa.w * wa1 + sb.w * wb1;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const float4 xa = __ldcg(a + i * 128), xb = __ldcg(b + i * 128);
        o[i][0] = xa.x * wa0 + xb.x * wb0, o[i][1] = xa.y * wa0 + xb.y * wb0;
        o[i][2] = xa.z * wa1 + xb.z * wb1, o[i][3] = xa.w * wa1 + xb.w * wb1;
      }
    }
    if (!active) continue;
    const float i0 = 1.f / l0, i1 = 1.f / l1;
    const int ra = u.row0 + r0 + (lane >> 2), rb = ra + 8;
    const int ldo = p.H * D;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int col = u.head * D + i * 8 + (lane & 3) * 2;
      if (ra < u.len) *reinterpret_cast<uint32_t*>(p.O + (size_t)(u.seg + ra) * ldo + col) = pack_bf16(o[i][0] * i0, o[i][1] * i0);
      if (rb < u.len) *reinterpret_cast<uint32_t*>(p.O + (size_t)(u.seg + rb) * ldo + col) = pack_bf16(o[i][2] * i1, o[i][3] * i1);
    }
  }
}



// [T, H * 64] view of a [T, H, 64] tensor with row stride `ld` elements: 64 x 64 boxes, 128B swizzle.
CUtensorMap HeadMap(const tvm::ffi::TensorView& t) {
  CUtensorMap m;
  cuuint64_t dims[2] = {(cuuint64_t)(t.size(1) * D), (cuuint64_t)t.size(0)};
  cuuint64_t strides[1] = {(cuuint64_t)t.stride(0) * 2};
  cuuint32_t box[2] = {64, 64};
  cuuint32_t elem[2] = {1, 1};
  if (xinfer::ffi::TensorMapEncodeTiled()(&m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, t.data_ptr(), dims, strides, box, elem, CU_TENSOR_MAP_INTERLEAVE_NONE,
                CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) != CUDA_SUCCESS) {
    throw std::runtime_error("attention_varlen_lpt_bf16_sm120: cuTensorMapEncodeTiled failed (16-byte aligned base and row stride needed)");
  }
  return m;
}

void CheckHeads(const tvm::ffi::TensorView& t, const tvm::ffi::TensorView& ref, const char* name) {
  namespace ffi = xinfer::ffi;
  ffi::CheckCuda(t, name);
  ffi::CheckBFloat16(t, name);
  if (t.ndim() != 3 || t.size(0) != ref.size(0) || t.size(1) != ref.size(1) || t.size(2) != D) {
    throw std::runtime_error(std::string("attention_varlen_lpt_bf16_sm120: ") + name + " must be [T, H, 64] like q");
  }
  if (t.stride(2) != 1 || t.stride(1) != D || (t.stride(0) * 2) % 16 || reinterpret_cast<uintptr_t>(t.data_ptr()) % 16) {
    throw std::runtime_error(std::string("attention_varlen_lpt_bf16_sm120: ") + name +
                             " needs dense [H, 64] rows, a 16-byte multiple row stride and a 16-byte aligned base");
  }
}

}  // namespace

constexpr int kPieceInts = 8;
static_assert(sizeof(Piece) == kPieceInts * 4, "Piece layout");

int64_t WorkspaceFloatsPerPair() { return 2 * 128 * 36; }

void Run(tvm::ffi::TensorView out, tvm::ffi::TensorView q, tvm::ffi::TensorView k, tvm::ffi::TensorView v,
         tvm::ffi::TensorView pieces, tvm::ffi::TensorView tstart, tvm::ffi::TensorView ws, tvm::ffi::TensorView cnt,
         double softmax_scale) {
  namespace ffi = xinfer::ffi;
  const char* op = "attention_varlen_lpt_bf16_sm120";
  ffi::RequireArchitecture(q, 12, 0, op);
  CheckHeads(q, q, "q");
  CheckHeads(k, q, "k");
  CheckHeads(v, q, "v");
  ffi::CheckCuda(out, "out");
  ffi::CheckBFloat16(out, "out");
  ffi::CheckContiguous(out, "out");
  const int64_t t = q.size(0), h = q.size(1);
  if (out.ndim() != 2 || out.size(0) != t || out.size(1) != h * D) {
    throw std::runtime_error("attention_varlen_lpt_bf16_sm120: out must be [T, H * 64]");
  }
  for (auto* x : {&pieces, &tstart, &cnt}) {
    ffi::CheckCuda(*x, "plan");
    ffi::CheckInt32(*x, "plan");
    ffi::CheckContiguous(*x, "plan");
  }
  ffi::CheckCuda(ws, "ws");
  ffi::CheckFloat32(ws, "ws");
  if (pieces.ndim() != 2 || pieces.size(1) != kPieceInts) throw std::runtime_error("attention_varlen_lpt_bf16_sm120: pieces must be int32 [n, 8]");
  const int64_t g2 = ffi::NumElements(tstart) - 1;
  if (g2 < 2 || g2 % 2) throw std::runtime_error("attention_varlen_lpt_bf16_sm120: tstart must be int32 [2G + 1]");
  if (!(softmax_scale > 0)) throw std::runtime_error("attention_varlen_lpt_bf16_sm120: softmax_scale must be positive");
  // pairs <= G (at most one split per SM boundary)
  const int64_t grid = g2 / 2;
  if (ffi::NumElements(cnt) < grid || ffi::NumElements(ws) < grid * WorkspaceFloatsPerPair()) {
    throw std::runtime_error("attention_varlen_lpt_bf16_sm120: workspace too small (ws >= G * 9216 floats, cnt >= G)");
  }
  const CUtensorMap mq = HeadMap(q), mk = HeadMap(k), mv = HeadMap(v);
  Params p{static_cast<const Piece*>(pieces.data_ptr()), static_cast<const int*>(tstart.data_ptr()),
           static_cast<__nv_bfloat16*>(out.data_ptr()), int(h), float(softmax_scale * 1.4426950408889634),
           static_cast<float*>(ws.data_ptr()), static_cast<int*>(cnt.data_ptr())};
  const int smem = int(sizeof(Smem)) + 1024;
  // Stream() also makes q's device current: the function attributes below are per device.
  const cudaStream_t stream = ffi::Stream(q);
  static bool attr[64] = {};
  const int device = q.device().device_id;
  if (device < 0 || device >= 64) throw std::runtime_error("attention_varlen_lpt_bf16_sm120: device index");
  if (!attr[device]) {
    ffi::CheckCudaError(cudaFuncSetAttribute(VarlenLptKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem),
                        "cudaFuncSetAttribute");
    attr[device] = true;
  }
  VarlenLptKernel<<<int(grid), CTA_THREADS, smem, stream>>>(mq, mk, mv, p);
  ffi::CheckLastCudaError(op);
}

}  // namespace xinfer::attention_varlen::lpt_sm120

TVM_FFI_DLL_EXPORT_TYPED_FUNC(attention_varlen_lpt_bf16_sm120, xinfer::attention_varlen::lpt_sm120::Run);
