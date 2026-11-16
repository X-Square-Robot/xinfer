// tile_mxfp8_sm120: whole-tile MXFP8 GEMMs, one output tile per CTA over the whole K and no cross-CTA
// synchronization. Two engines on the same packed weight (layout nk_tile256):
//   (16 R) x 256 tiles, K <= 1024 (tile_mxfp8_kernel):
//     gelu_quant128  bias + tanh GELU + MXFP8 output with one scale per 128 columns
//     qkv_rope       bias + rotate-half RoPE (head_dim 64, pairs4 table) on the first rope_cols columns, BF16 output
//   (16 R) x 128 tiles with a loader warp, any K (residual_kernel):
//     residual       bias + BF16 residual, BF16 output
//
// 256-column engine. 8 MMA warps: wm = warp >> 2 owns the upper ceil(R / 2) or the lower floor(R / 2) m16
// fragments, wn = warp & 3 owns 32 columns of each of the tile's two 128-column groups. Warps w and w + 4 share an
// SMSP, so with odd R every SMSP holds one warp of each half.
// Shared memory: a ring of 3 stages [A (16 R) x 64 B | W 256 x 64 B | W scales 1 KB], and the A scales of the whole
// tile (its two SWIZZLE_32_4_4 row blocks, all K), copied once. The weight is pre-tiled per (column tile, 64-K step)
// in the stage layout with its rows permuted inside 128-row groups so that a thread owns contiguous outputs
// (gelu_quant128: 8 per row and group) or both halves of its rotary pairs (qkv_rope).
// Numerics: FP32 accumulators. gelu_quant128: gelu_tanh(x) = 0.5 x (1 + tanh(x (0.79788456 + 0.035677408 x^2))) in
// FP32 with tanh.approx, E8M0 ceil scale, E4M3 round to nearest (satfinite). qkv_rope: y = BF16(acc + bias), then
// BF16(y1 c - y2 s), BF16(y2 c + y1 s), fused as the generic path. Deterministic.
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/function.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "common/tvm_ffi_cuda_tensor.h"
#include "quant/mxfp8_format_utils.cuh"
#include "tile_mxfp8_sm120_common.cuh"

namespace xinfer::gemm::tile_mxfp8_sm120 {

using namespace xinfer::gemm::sm120::mxfp8_tile;

bool Supports(int64_t m, int64_t n, int64_t k, int64_t device, bool residual);
int64_t PackedWeightBytes(int64_t n, int64_t k);

namespace {

constexpr int kStages = 3;
constexpr int kMaxRows16 = 10;   // tile rows / 16: 16 accumulators per fragment + ~90 registers must stay under 255
constexpr int kMaxSteps = 16;    // K <= 1024: the tile's A scales (2 x K / 128 x 512 B) must fit beside the ring

// MI fragments in the wm = 0 half, MI - kOdd in the wm = 1 half.
template <int MI, int kOdd>
struct Layout {
  static constexpr int kBM = 32 * MI - 16 * kOdd, kHalf = 16 * MI;  // tile rows, rows of the wm = 0 half
  static constexpr int kABytes = kBM * kBK;
  static constexpr int kStageBytes = (kABytes + kWBlock + 1023) / 1024 * 1024;
  static constexpr int kOffW = kABytes, kOffWScale = kABytes + kWBytes;
  static constexpr int kOffBars = kStages * kStageBytes;
  static constexpr int kOffAScale = kOffBars + 1024;
  static constexpr int kOffAmax = kOffAScale + 2 * (kMaxSteps / 2) * 512;  // row amax exchange [2 groups][kBM][4 wn]
  static constexpr int kSmemBytes = kOffAmax + 2 * kBM * 4 * 4;
};

enum Epilogue { kGeluQuant128 = 0, kQkvRope = 1 };

struct Params {
  const uint8_t* a_scale;  // SWIZZLE_32_4_4 storage of x
  const uint8_t* w;        // [N / 256][steps][kWBlock]
  const __nv_bfloat16* bias;
  const __nv_bfloat16* residual;  // residual: BF16 [M, N] (may be the output)
  uint8_t* out;            // [M, N]: E4M3 (gelu_quant128) or BF16 (qkv_rope, residual)
  uint32_t* q_scale;       // gelu_quant128: SWIZZLE_32_4_4 storage of [M, N], as words
  const float4* rope;      // qkv_rope: pairs4 table [16][M]{c, s, c, s}
  int m, n, tiles_m, row_blocks, steps, rope_cols;
};

// Refill of one stage: the A box and the weight block. elect: the whole converged warp calls and elect.sync picks the
// issuing lane; otherwise one thread calls.
__device__ __forceinline__ void Refill(bool elect, uint32_t stage, uint32_t bar, const CUtensorMap* map_a, int ax,
                                       int ay, uint32_t w_dst, const void* w_src, uint32_t w_bytes,
                                       uint32_t total_bytes) {
  if (elect) {
    asm volatile(
        "{\n .reg .pred p;\n elect.sync _|p, 0xffffffff;\n"
        " @p mbarrier.arrive.expect_tx.shared.b64 _, [%1], %8;\n"
        " @p cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes [%0], [%2, {%3, %4}], [%1];\n"
        " @p cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%5], [%6], %7, [%1];\n}\n"
        ::"r"(stage), "r"(bar), "l"(map_a), "r"(ax), "r"(ay), "r"(w_dst), "l"(w_src), "r"(w_bytes), "r"(total_bytes)
        : "memory");
  } else {
    asm volatile(
        "mbarrier.arrive.expect_tx.shared.b64 _, [%1], %8;\n"
        "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes [%0], [%2, {%3, %4}], [%1];\n"
        "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%5], [%6], %7, [%1];\n"
        ::"r"(stage), "r"(bar), "l"(map_a), "r"(ax), "r"(ay), "r"(w_dst), "l"(w_src), "r"(w_bytes), "r"(total_bytes)
        : "memory");
  }
}
// Two bulk copies of `bytes` each on one barrier.
__device__ __forceinline__ void BulkPair(uint32_t bar, uint32_t d0, const void* s0, uint32_t d1, const void* s1,
                                         uint32_t bytes) {
  asm volatile(
      "mbarrier.arrive.expect_tx.shared.b64 _, [%0], %6;\n"
      "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%1], [%2], %5, [%0];\n"
      "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%3], [%4], %5, [%0];\n"
      ::"r"(bar), "r"(d0), "l"(s0), "r"(d1), "l"(s1), "r"(bytes), "r"(2 * bytes) : "memory");
}
template <int MI, int kOdd, int kEpilogue>
__global__ void __launch_bounds__(kThreads, 1) tile_mxfp8_kernel(const __grid_constant__ CUtensorMap map_a,
                                                                const Params p) {
  using L = Layout<MI, kOdd>;
  extern __shared__ __align__(1024) uint8_t smem[];
  const uint32_t ring = Shared(smem), full = ring + L::kOffBars, empty = full + 8 * kStages;
  const uint32_t a_scale_bar = empty + 8 * kStages, a_scale = ring + L::kOffAScale;
  float* amax = reinterpret_cast<float*>(smem + L::kOffAmax);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int tile = blockIdx.x, nt = tile / p.tiles_m, mt = tile - nt * p.tiles_m;
  const int row0 = mt * L::kBM;
  const uint32_t a_scale_block = (p.steps / 2) * 512;  // one 128-row block of A scales, all K
  if (threadIdx.x == 0) {
    for (int s = 0; s < kStages; ++s) {
      BarrierInit(full + 8 * s, 1);
      BarrierInit(empty + 8 * s, kWarps);
    }
    BarrierInit(a_scale_bar, 1);
    BarrierInit(a_scale_bar + 8, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  // Load n is step n and lands in stage n % 3; the weight scales travel with the even steps.
  auto issue = [&](int n, bool elect) {
    const uint32_t stage = ring + static_cast<uint32_t>(n % kStages) * L::kStageBytes;
    const uint32_t w_bytes = (n & 1) ? kWBytes : kWBlock;
    Refill(elect, stage, full + 8 * (n % kStages), &map_a, n * kBK, row0, stage + L::kOffW,
           p.w + (static_cast<size_t>(nt) * p.steps + n) * kWBlock, w_bytes, L::kABytes + w_bytes);
  };
  if (threadIdx.x == 0) {
    // Loads are served first come first served: stage 0 goes first with the A scales of the first 128 K; stages 1
    // and 2 and the other A scales follow once it has landed.
    const int rb0 = row0 >> 7, rb1 = min(rb0 + 1, p.row_blocks - 1);  // past the last block: the first again (unused)
    const uint8_t* s0 = p.a_scale + static_cast<size_t>(rb0) * a_scale_block;
    const uint8_t* s1 = p.a_scale + static_cast<size_t>(rb1) * a_scale_block;
    issue(0, false);
    BulkPair(a_scale_bar, a_scale, s0, a_scale + a_scale_block, s1, 512);
    while (!BarrierTry(full, 0)) {
    }
    for (int n = 1; n < kStages && n < p.steps; ++n) issue(n, false);
    if (p.steps > 2) {
      BulkPair(a_scale_bar + 8, a_scale + 512, s0 + 512, a_scale + a_scale_block + 512, s1 + 512, a_scale_block - 512);
    }
  }

  const int wm = warp >> 2, wn = warp & 3;
  const int g = lane >> 2, t4 = lane & 3;
  const int j8 = lane >> 3, r8 = lane & 7;
  // ldmatrix addresses of the two k32 halves (64-byte swizzle)
  const int a_row = (j8 & 1) * 8 + r8, a_c = j8 >> 1, w_row = (j8 >> 1) * 8 + r8, w_c = j8 & 1;
  uint32_t a_off[2], w_off[2];
  for (int h = 0; h < 2; ++h) {
    a_off[h] = (L::kHalf * wm + a_row) * 64 + (((2 * h + a_c) ^ ((a_row >> 1) & 3)) << 4);
    w_off[h] = L::kOffW + (32 * wn + w_row) * 64 + (((2 * h + w_c) ^ ((w_row >> 1) & 3)) << 4);
  }
  // A scale of fragment mi: row (row0 & 127) + 16 MI wm + 16 mi + g + 8 (lane & 1) of the two copied blocks
  const int a_scale_row = (row0 & 127) + L::kHalf * wm + g + 8 * (lane & 1);
  const uint32_t w_scale_off = L::kOffWScale + (wn * 8 + g) * 32;

  float acc[MI][8][4];
#pragma unroll
  for (int i = 0; i < MI * 32; ++i) (&acc[0][0][0])[i] = 0.f;
  uint32_t sa[MI], sw[8];
  // Warp w refills loads w, w + 8, ... (the first three were issued above).
  int pend = warp >= kStages ? warp : warp + 8;
  int pend_q = pend / 3, pend_r = pend - 3 * (pend / 3);  // pend = 3 pend_q + pend_r

  // parity_tag: the 64-K step's half of its 128 K; frag_tag: the m16 fragments of this warp. Both are types, so each
  // combination is its own straight-line body (a runtime test per fragment compiles to predicated QMMAs).
  auto step = [&](int it, int stage_index, uint32_t phase, auto parity_tag, auto frag_tag) {
    constexpr int kParity = std::decay_t<decltype(parity_tag)>::value;
    constexpr int kFrags = std::decay_t<decltype(frag_tag)>::value;
    const uint32_t base = ring + stage_index * L::kStageBytes;
    auto refill_due = [&]() {
      return pend <= it + 2 && pend < p.steps && BarrierTest(empty + 8 * pend_r, (pend_q - 1) & 1);
    };
    auto advance = [&]() {
      pend += 8, pend_r += 2, pend_q += 2;
      if (pend_r >= 3) pend_r -= 3, ++pend_q;
    };
    // The usual refill is issued once with elect.sync before the wait; the wait loop keeps a lane-0 form for a stage
    // that frees up while this warp waits (elect.sync inside the loop adds re-convergence code to every iteration).
    if (refill_due()) {
      issue(pend, true);
      advance();
    }
    while (true) {
      if (refill_due()) {
        if (lane == 0) issue(pend, false);
        advance();
      }
      if (BarrierTry(full + 8 * stage_index, phase)) break;
    }
    if (!kParity) {
      if (it == 2) {
        while (!BarrierTry(a_scale_bar + 8, 0)) {
        }
      }
#pragma unroll
      for (int mi = 0; mi < kFrags; ++mi) {
        const int r = a_scale_row + 16 * mi;
        sa[mi] = LoadShared32(a_scale + (r >> 7) * a_scale_block + (it >> 1) * 512 + (r & 31) * 16 + ((r & 127) >> 5) * 4);
      }
      LoadShared128(base + w_scale_off, sw);
      LoadShared128(base + w_scale_off + 16, sw + 4);
    }
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      uint32_t a[MI][4], b[4][4];
#pragma unroll
      for (int mi = 0; mi < kFrags; ++mi) LoadMatrix4(base + a_off[h] + 1024 * mi, a[mi]);
#pragma unroll
      for (int q = 0; q < 4; ++q) LoadMatrix4(base + w_off[h] + ((q >> 1) * 128 + (q & 1) * 16) * 64, b[q]);
      if (h == 1) {  // the stage is in registers: release it
        __syncwarp();
        ArriveElect(empty + 8 * stage_index);
      }
#pragma unroll
      for (int q = 0; q < 4; ++q)
#pragma unroll
        for (int mi = 0; mi < kFrags; ++mi)
#pragma unroll
          for (int j = 0; j < 2; ++j) {
            float* d = acc[mi][2 * q + j];
            const uint32_t b0 = b[q][2 * j], b1 = b[q][2 * j + 1];
            if (kParity == 0 && h == 0) Mma<0>(d, a[mi], b0, b1, sa[mi], sw[2 * q + j]);
            if (kParity == 0 && h == 1) Mma<1>(d, a[mi], b0, b1, sa[mi], sw[2 * q + j]);
            if (kParity == 1 && h == 0) Mma<2>(d, a[mi], b0, b1, sa[mi], sw[2 * q + j]);
            if (kParity == 1 && h == 1) Mma<3>(d, a[mi], b0, b1, sa[mi], sw[2 * q + j]);
          }
    }
  };

  // gelu_quant128. Per thread: 8 contiguous outputs per row in each of the tile's two 128-column groups. One m16
  // fragment at a time: activation, row amax (2 shuffles, then the 4 wn warps through `amax`), then the stores.
  auto gelu_quant128 = [&](auto frag_tag) {
    constexpr int kFrags = std::decay_t<decltype(frag_tag)>::value;
    float bias[2][8];
#pragma unroll
    for (int grp = 0; grp < 2; ++grp) {
      const uint4 raw = *reinterpret_cast<const uint4*>(p.bias + nt * kBN + grp * 128 + 32 * wn + 8 * t4);
      const __nv_bfloat162* pairs = reinterpret_cast<const __nv_bfloat162*>(&raw);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const float2 f = __bfloat1622float2(pairs[i]);
        bias[grp][2 * i] = f.x, bias[grp][2 * i + 1] = f.y;
      }
    }
#pragma unroll
    for (int mi = 0; mi < kFrags; ++mi) {
      float v[2][2][8];  // [group][row half][output]: 2 gelu (the 0.5 goes into the row's reciprocal scale)
#pragma unroll
      for (int grp = 0; grp < 2; ++grp)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          float m = 0.f;
#pragma unroll
          for (int n = 0; n < 4; ++n)
#pragma unroll
            for (int j = 0; j < 2; ++j) {
              const float x = acc[mi][4 * grp + n][2 * h + j] + bias[grp][2 * n + j];
              float th;
              asm("tanh.approx.f32 %0, %1;" : "=f"(th) : "f"(x * fmaf(0.035677408f, x * x, 0.79788456f)));
              const float y2 = fmaf(x, th, x);
              v[grp][h][2 * n + j] = y2;
              m = fmaxf(m, fabsf(y2));
            }
          m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 1));
          m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 2));
          if (t4 == 0) amax[(grp * L::kBM + L::kHalf * wm + 16 * mi + g + 8 * h) * 4 + wn] = m;
        }
      // the four wn warps of this wm half (a literal barrier id per half: a register id reserves all 16 barriers)
      if (wm == 0) {
        asm volatile("bar.sync 1, 128;" ::: "memory");
      } else {
        asm volatile("bar.sync 2, 128;" ::: "memory");
      }
#pragma unroll
      for (int grp = 0; grp < 2; ++grp)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int r = L::kHalf * wm + 16 * mi + g + 8 * h, row = row0 + r;
          const float4 mm = *reinterpret_cast<const float4*>(amax + 4 * (grp * L::kBM + r));
          // amax of 2 gelu: halving is exact, so e is the scale of gelu itself
          const uint8_t e = xinfer::mxfp8::amax_to_e8m0_ceil(0.5f * fmaxf(fmaxf(mm.x, mm.y), fmaxf(mm.z, mm.w)));
          // 0.5 x 2^(127 - e) from its bits (exp2f is a MUFU); e > 252 only for Inf / NaN rows
          const float inv = __uint_as_float(static_cast<uint32_t>(max(253 - static_cast<int>(e), 1)) << 23);
          const float* x = v[grp][h];
          const uint2 packed = make_uint2(ToE4M3x4(x[0] * inv, x[1] * inv, x[2] * inv, x[3] * inv),
                                          ToE4M3x4(x[4] * inv, x[5] * inv, x[6] * inv, x[7] * inv));
          if (row < p.m) {
            *reinterpret_cast<uint2*>(p.out + static_cast<size_t>(row) * p.n + nt * kBN + grp * 128 + 32 * wn + 8 * t4) =
                packed;
            // the four 1x32 slots of (row, group) are four consecutive bytes
            if (wn == 0 && t4 == 0) {
              p.q_scale[xinfer::mxfp8::scale_offset(row, 4 * (2 * nt + grp), p.n / 128) / 4] = 0x01010101u * e;
            }
          }
        }
    }
  };

  // qkv_rope. In a rotary 128-column group (2 heads of 64) the packed row order gives a thread the 4 pairs (j, j + 32),
  // j = 16 (wn & 1) + 4 t4 + 0..3, of head wn >> 1; elsewhere 8 contiguous outputs per row. The table values of a
  // thread depend on its pair index and row only (the same for both groups), so they are all loaded once, up front.
  auto qkv_rope = [&](auto frag_tag) {
    constexpr int kFrags = std::decay_t<decltype(frag_tag)>::value;
    __nv_bfloat16* out = reinterpret_cast<__nv_bfloat16*>(p.out);
    const int pair0 = 16 * (wn & 1) + 4 * t4;
    float4 table[MI][2][2];
    if (nt * kBN < p.rope_cols) {
#pragma unroll
      for (int mi = 0; mi < kFrags; ++mi)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int row = min(row0 + L::kHalf * wm + 16 * mi + g + 8 * h, p.m - 1);
          table[mi][h][0] = __ldg(p.rope + static_cast<size_t>(pair0 >> 1) * p.m + row);
          table[mi][h][1] = __ldg(p.rope + static_cast<size_t>((pair0 >> 1) + 1) * p.m + row);
        }
    }
#pragma unroll
    for (int grp = 0; grp < 2; ++grp) {
      const int col0 = nt * kBN + grp * 128;
      if (col0 < p.rope_cols) {
        const int col = col0 + 64 * (wn >> 1) + pair0;  // first of the thread's 4 pairs (col, col + 32)
        float b1[4], b2[4];
        {
          const uint2 r1 = *reinterpret_cast<const uint2*>(p.bias + col);
          const uint2 r2 = *reinterpret_cast<const uint2*>(p.bias + col + 32);
          const __nv_bfloat162* p1 = reinterpret_cast<const __nv_bfloat162*>(&r1);
          const __nv_bfloat162* p2 = reinterpret_cast<const __nv_bfloat162*>(&r2);
#pragma unroll
          for (int i = 0; i < 2; ++i) {
            const float2 f1 = __bfloat1622float2(p1[i]), f2 = __bfloat1622float2(p2[i]);
            b1[2 * i] = f1.x, b1[2 * i + 1] = f1.y, b2[2 * i] = f2.x, b2[2 * i + 1] = f2.y;
          }
        }
#pragma unroll
        for (int mi = 0; mi < kFrags; ++mi)
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            const int row = row0 + L::kHalf * wm + 16 * mi + g + 8 * h;
            if (row >= p.m) continue;
            const float4 t0 = table[mi][h][0], t1 = table[mi][h][1];
            const float cs[4][2] = {{t0.x, t0.y}, {t0.z, t0.w}, {t1.x, t1.y}, {t1.z, t1.w}};
            uint32_t w1[2], w2[2];
#pragma unroll
            for (int i = 0; i < 2; ++i) {
              // y = BF16(acc + bias), back to FP32 for the rotation (as the generic path)
              const float2 y1 = __bfloat1622float2(__floats2bfloat162_rn(acc[mi][4 * grp + i][2 * h] + b1[2 * i],
                                                                         acc[mi][4 * grp + i][2 * h + 1] + b1[2 * i + 1]));
              const float2 y2 = __bfloat1622float2(__floats2bfloat162_rn(acc[mi][4 * grp + 2 + i][2 * h] + b2[2 * i],
                                                                         acc[mi][4 * grp + 2 + i][2 * h + 1] + b2[2 * i + 1]));
              const float* c0 = cs[2 * i];
              const float* c1 = cs[2 * i + 1];
              // The products are fused as gemm_epilogue.cu's RopeKernel compiles, fma(y1, c, -(y2 s)) and
              // fma(y1, s, y2 c): other fusions differ from the generic path at BF16 rounding ties.
              const __nv_bfloat162 q1 = __floats2bfloat162_rn(fmaf(y1.x, c0[0], -__fmul_rn(y2.x, c0[1])),
                                                              fmaf(y1.y, c1[0], -__fmul_rn(y2.y, c1[1])));
              const __nv_bfloat162 q2 = __floats2bfloat162_rn(fmaf(y1.x, c0[1], __fmul_rn(y2.x, c0[0])),
                                                              fmaf(y1.y, c1[1], __fmul_rn(y2.y, c1[0])));
              w1[i] = *reinterpret_cast<const uint32_t*>(&q1);
              w2[i] = *reinterpret_cast<const uint32_t*>(&q2);
            }
            *reinterpret_cast<uint2*>(out + static_cast<size_t>(row) * p.n + col) = make_uint2(w1[0], w1[1]);
            *reinterpret_cast<uint2*>(out + static_cast<size_t>(row) * p.n + col + 32) = make_uint2(w2[0], w2[1]);
          }
      } else {
        const int col = col0 + 32 * wn + 8 * t4;
        const uint4 raw = *reinterpret_cast<const uint4*>(p.bias + col);
        const __nv_bfloat162* pairs = reinterpret_cast<const __nv_bfloat162*>(&raw);
#pragma unroll
        for (int mi = 0; mi < kFrags; ++mi)
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            const int row = row0 + L::kHalf * wm + 16 * mi + g + 8 * h;
            uint4 packed;
            uint32_t* words = reinterpret_cast<uint32_t*>(&packed);
#pragma unroll
            for (int n = 0; n < 4; ++n) {
              const float2 f = __bfloat1622float2(pairs[n]);
              const __nv_bfloat162 y =
                  __floats2bfloat162_rn(acc[mi][4 * grp + n][2 * h] + f.x, acc[mi][4 * grp + n][2 * h + 1] + f.y);
              words[n] = *reinterpret_cast<const uint32_t*>(&y);
            }
            if (row < p.m) *reinterpret_cast<uint4*>(out + static_cast<size_t>(row) * p.n + col) = packed;
          }
      }
    }
  };

  while (!BarrierTry(a_scale_bar, 0)) {
  }
  using Long = std::integral_constant<int, MI>;
  using Short = std::integral_constant<int, MI - kOdd>;
  using Even = std::integral_constant<int, 0>;
  using Odd = std::integral_constant<int, 1>;
  const bool short_half = kOdd && wm == 1;
  int stage_index = 0;
  uint32_t phase = 0;
  auto next_stage = [&]() {
    if (++stage_index == kStages) stage_index = 0, phase ^= 1u;
  };
#pragma unroll 1
  for (int it = 0; it < p.steps; it += 2) {
    if (short_half) step(it, stage_index, phase, Even{}, Short{}); else step(it, stage_index, phase, Even{}, Long{});
    next_stage();
    if (short_half) step(it + 1, stage_index, phase, Odd{}, Short{}); else step(it + 1, stage_index, phase, Odd{}, Long{});
    next_stage();
  }
  if constexpr (kEpilogue == kGeluQuant128) {
    if (short_half) gelu_quant128(Short{}); else gelu_quant128(Long{});
  } else {
    if (short_half) qkv_rope(Short{}); else qkv_rope(Long{});
  }
}

// ---- (16 R) x 128 tiles with a loader warp: bias + BF16 residual ----
// For long K and narrow N. A 128-column tile is one half of a packed 256-column tile. A ninth warp issues every load,
// so an MMA warp's step is: one wait, fragments to registers, one release, MMAs (with the refill bookkeeping in the
// MMA warps all 8 leave the QMMA stream together at every step, and the step is half as long as with 256 columns).
// Ring of 4 stages [A (16 R) x 64 B | A scales 2 x 512 B | W 128 x 64 B | W scales 1 KB (both groups of the 256
// column tile)]; the scales travel with the even steps, so K is not bounded by shared memory.
constexpr int kResStages = 4, kResThreads = kThreads + 32;
template <int MI, int kOdd>
struct ResidualLayout {
  static constexpr int kBM = 32 * MI - 16 * kOdd, kHalf = 16 * MI;
  static constexpr int kABytes = kBM * kBK;
  static constexpr int kOffAScale = kABytes, kOffW = kABytes + 1024, kOffWScale = kOffW + kWBytes / 2;
  static constexpr int kStageBytes = (kOffWScale + 1024 + 1023) / 1024 * 1024;
  static constexpr int kOffBars = kResStages * kStageBytes;
  static constexpr int kSmemBytes = kOffBars + 1024;
};

template <int MI, int kOdd>
__global__ void __launch_bounds__(kResThreads, 1) residual_kernel(const __grid_constant__ CUtensorMap map_a,
                                                                 const Params p) {
  using L = ResidualLayout<MI, kOdd>;
  extern __shared__ __align__(1024) uint8_t smem[];
  const uint32_t ring = Shared(smem), full = ring + L::kOffBars, empty = full + 8 * kResStages;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  // column tile nt of 128 = group nt & 1 of the packed 256-column tile nt >> 1
  const int tile = blockIdx.x, nt = tile / p.tiles_m, mt = tile - nt * p.tiles_m, grp = nt & 1;
  const int row0 = mt * L::kBM;
  if (threadIdx.x == 0) {
    for (int s = 0; s < kResStages; ++s) {
      BarrierInit(full + 8 * s, 1);
      BarrierInit(empty + 8 * s, kWarps);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  if (warp == kWarps) {
    // Loader. Load n is step n and lands in stage n % 4. Stage 0 goes alone first (loads are served first come first
    // served); afterwards a stage is refilled as soon as the 8 MMA warps have released it.
    if (lane != 0) return;
    const int rb0 = row0 >> 7, rb1 = min(rb0 + 1, p.row_blocks - 1);  // past the last block: the first again (unused)
    const size_t a_scale_block = static_cast<size_t>(p.steps / 2) * 512;
    const uint8_t* w = p.w + static_cast<size_t>(nt >> 1) * p.steps * kWBlock;
#pragma unroll 1
    for (int n = 0; n < p.steps; ++n) {
      const int s = n % kResStages;
      const uint32_t stage = ring + static_cast<uint32_t>(s) * L::kStageBytes, bar = full + 8 * s;
      if (n >= kResStages) {
        while (!BarrierTry(empty + 8 * s, ((n / kResStages) - 1) & 1)) {
        }
      } else if (n == 1) {
        while (!BarrierTry(full, 0)) {
        }
      }
      const uint8_t* block = w + static_cast<size_t>(n) * kWBlock;
      const bool scales = !(n & 1);
      asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;"
                   ::"r"(bar), "r"(L::kABytes + kWBytes / 2 + (scales ? 2048 : 0)) : "memory");
      asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
                   ::"r"(stage), "l"(&map_a), "r"(n * kBK), "r"(row0), "r"(bar) : "memory");
      // group 1's rows are followed by the scales in the packed block: one copy
      const uint32_t w_bytes = kWBytes / 2 + (scales && grp ? 1024 : 0);
      asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                   ::"r"(stage + L::kOffW), "l"(block + grp * (kWBytes / 2)), "r"(w_bytes), "r"(bar) : "memory");
      if (scales) {
        if (!grp) {
          asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], 1024, [%2];"
                       ::"r"(stage + L::kOffWScale), "l"(block + kWBytes), "r"(bar) : "memory");
        }
        const uint8_t* s0 = p.a_scale + rb0 * a_scale_block + (n >> 1) * 512;
        const uint8_t* s1 = p.a_scale + rb1 * a_scale_block + (n >> 1) * 512;
        asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], 512, [%2];\n"
                     "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%3], [%4], 512, [%2];"
                     ::"r"(stage + L::kOffAScale), "l"(s0), "r"(bar), "r"(stage + L::kOffAScale + 512), "l"(s1) : "memory");
      }
    }
    return;
  }
  const int wm = warp >> 2, wn = warp & 3;
  const int g = lane >> 2, t4 = lane & 3;
  const int j8 = lane >> 3, r8 = lane & 7;
  const int a_row = (j8 & 1) * 8 + r8, a_c = j8 >> 1, w_row = (j8 >> 1) * 8 + r8, w_c = j8 & 1;
  uint32_t a_off[2], w_off[2];
  for (int h = 0; h < 2; ++h) {
    a_off[h] = (L::kHalf * wm + a_row) * 64 + (((2 * h + a_c) ^ ((a_row >> 1) & 3)) << 4);
    w_off[h] = L::kOffW + (32 * wn + w_row) * 64 + (((2 * h + w_c) ^ ((w_row >> 1) & 3)) << 4);
  }
  const int a_scale_row = (row0 & 127) + L::kHalf * wm + g + 8 * (lane & 1);
  const uint32_t w_scale_off = L::kOffWScale + (wn * 8 + g) * 32 + 16 * grp;

  float acc[MI][4][4];
#pragma unroll
  for (int i = 0; i < MI * 16; ++i) (&acc[0][0][0])[i] = 0.f;
  uint32_t sa[MI], sw[4];

  auto step = [&](int stage_index, uint32_t phase, auto parity_tag, auto frag_tag) {
    constexpr int kParity = std::decay_t<decltype(parity_tag)>::value;
    constexpr int kFrags = std::decay_t<decltype(frag_tag)>::value;
    const uint32_t base = ring + stage_index * L::kStageBytes;
    while (!BarrierTry(full + 8 * stage_index, phase)) {
    }
    if (!kParity) {
#pragma unroll
      for (int mi = 0; mi < kFrags; ++mi) {
        const int r = a_scale_row + 16 * mi;
        sa[mi] = LoadShared32(base + L::kOffAScale + (r >> 7) * 512 + (r & 31) * 16 + ((r & 127) >> 5) * 4);
      }
      LoadShared128(base + w_scale_off, sw);
    }
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      uint32_t a[MI][4], b[2][4];
#pragma unroll
      for (int mi = 0; mi < kFrags; ++mi) LoadMatrix4(base + a_off[h] + 1024 * mi, a[mi]);
#pragma unroll
      for (int q = 0; q < 2; ++q) LoadMatrix4(base + w_off[h] + q * 16 * 64, b[q]);
      if (h == 1) {  // the stage is in registers: release it
        __syncwarp();
        ArriveElect(empty + 8 * stage_index);
      }
#pragma unroll
      for (int q = 0; q < 2; ++q)
#pragma unroll
        for (int mi = 0; mi < kFrags; ++mi)
#pragma unroll
          for (int j = 0; j < 2; ++j) {
            float* d = acc[mi][2 * q + j];
            const uint32_t b0 = b[q][2 * j], b1 = b[q][2 * j + 1];
            if (kParity == 0 && h == 0) Mma<0>(d, a[mi], b0, b1, sa[mi], sw[2 * q + j]);
            if (kParity == 0 && h == 1) Mma<1>(d, a[mi], b0, b1, sa[mi], sw[2 * q + j]);
            if (kParity == 1 && h == 0) Mma<2>(d, a[mi], b0, b1, sa[mi], sw[2 * q + j]);
            if (kParity == 1 && h == 1) Mma<3>(d, a[mi], b0, b1, sa[mi], sw[2 * q + j]);
          }
    }
  };

  // out = BF16(BF16(acc + bias) + residual), the two roundings of the generic path. A thread owns 8 contiguous
  // outputs per row; it reads all its residuals before it stores (the output may be the residual).
  auto epilogue = [&](auto frag_tag) {
    constexpr int kFrags = std::decay_t<decltype(frag_tag)>::value;
    __nv_bfloat16* out = reinterpret_cast<__nv_bfloat16*>(p.out);
    const int col = nt * 128 + 32 * wn + 8 * t4;
    const uint4 bias_raw = *reinterpret_cast<const uint4*>(p.bias + col);
    const __nv_bfloat162* bias = reinterpret_cast<const __nv_bfloat162*>(&bias_raw);
    uint4 res[MI][2];
#pragma unroll
    for (int mi = 0; mi < kFrags; ++mi)
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int row = min(row0 + L::kHalf * wm + 16 * mi + g + 8 * h, p.m - 1);
        res[mi][h] = *reinterpret_cast<const uint4*>(p.residual + static_cast<size_t>(row) * p.n + col);
      }
#pragma unroll
    for (int mi = 0; mi < kFrags; ++mi)
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int row = row0 + L::kHalf * wm + 16 * mi + g + 8 * h;
        const __nv_bfloat162* r = reinterpret_cast<const __nv_bfloat162*>(&res[mi][h]);
        uint4 packed;
        uint32_t* words = reinterpret_cast<uint32_t*>(&packed);
#pragma unroll
        for (int n = 0; n < 4; ++n) {
          const float2 fb = __bfloat1622float2(bias[n]), fr = __bfloat1622float2(r[n]);
          const float2 y =
              __bfloat1622float2(__floats2bfloat162_rn(acc[mi][n][2 * h] + fb.x, acc[mi][n][2 * h + 1] + fb.y));
          const __nv_bfloat162 o = __floats2bfloat162_rn(y.x + fr.x, y.y + fr.y);
          words[n] = *reinterpret_cast<const uint32_t*>(&o);
        }
        if (row < p.m) *reinterpret_cast<uint4*>(out + static_cast<size_t>(row) * p.n + col) = packed;
      }
  };

  using Long = std::integral_constant<int, MI>;
  using Short = std::integral_constant<int, MI - kOdd>;
  using Even = std::integral_constant<int, 0>;
  using Odd = std::integral_constant<int, 1>;
  const bool short_half = kOdd && wm == 1;
  int stage_index = 0;
  uint32_t phase = 0;
  auto next_stage = [&]() {
    if (++stage_index == kResStages) stage_index = 0, phase ^= 1u;
  };
#pragma unroll 1
  for (int it = 0; it < p.steps; it += 2) {
    if (short_half) step(stage_index, phase, Even{}, Short{}); else step(stage_index, phase, Even{}, Long{});
    next_stage();
    if (short_half) step(stage_index, phase, Odd{}, Short{}); else step(stage_index, phase, Odd{}, Long{});
    next_stage();
  }
  if (short_half) epilogue(Short{}); else epilogue(Long{});
}


// Tile rows / 16: the smallest height with at most one tile per SM; 0 if even the tallest needs a second wave.
int TileRows16(int64_t m, int64_t n, int sms) {
  for (int r = 1; r <= kMaxRows16; ++r) {
    if ((m + 16 * r - 1) / (16 * r) * (n / kBN) <= sms) return r;
  }
  return 0;
}

// Residual engine: the smallest tile height whose tile count is at most a share of the SMs that grows with the
// height (48% up to 48 rows, 62% at 64, 76% at 80 and 96, all SMs from 112 rows); if none qualifies, the smallest
// height with at most one tile per SM. 0: none fits. The shares were fitted to measured times on an RTX 5090.
int ResidualRows16(int64_t m, int64_t n, int sms) {
  int fits = 0;
  for (int r = 1; r <= kMaxRows16; ++r) {
    const int64_t tiles = (m + 16 * r - 1) / (16 * r) * (n / 128);
    if (tiles > sms) continue;
    if (fits == 0) fits = r;
    const int share = r <= 3 ? 48 : r == 4 ? 62 : r <= 6 ? 76 : 100;
    if (tiles * 100 <= static_cast<int64_t>(sms) * share) return r;
  }
  return fits;
}

template <int kRows16 = 1>
void LaunchResidual(int rows16, const CUtensorMap& map_a, const Params& p, int device, cudaStream_t stream) {
  if (rows16 != kRows16) {
    if constexpr (kRows16 < kMaxRows16) LaunchResidual<kRows16 + 1>(rows16, map_a, p, device, stream);
    return;
  }
  constexpr int MI = (kRows16 + 1) / 2, kOdd = kRows16 % 2;
  using L = ResidualLayout<MI, kOdd>;
  static bool attr[64] = {};
  if (!attr[device]) {
    xinfer::ffi::CheckCudaError(cudaFuncSetAttribute(residual_kernel<MI, kOdd>,
                                                       cudaFuncAttributeMaxDynamicSharedMemorySize, L::kSmemBytes),
                                  "cudaFuncSetAttribute");
    attr[device] = true;
  }
  residual_kernel<MI, kOdd><<<p.tiles_m * (p.n / 128), kResThreads, L::kSmemBytes, stream>>>(map_a, p);
}

// kRows16 = 2 MI - kOdd.
template <int kEpilogue, int kRows16 = 1>
void Launch(int rows16, const CUtensorMap& map_a, const Params& p, int device, cudaStream_t stream) {
  if (rows16 != kRows16) {
    if constexpr (kRows16 < kMaxRows16) Launch<kEpilogue, kRows16 + 1>(rows16, map_a, p, device, stream);
    return;
  }
  constexpr int MI = (kRows16 + 1) / 2, kOdd = kRows16 % 2;
  using L = Layout<MI, kOdd>;
  static bool attr[64] = {};
  if (!attr[device]) {  // per device (Stream() made the tensors' device current)
    xinfer::ffi::CheckCudaError(cudaFuncSetAttribute(tile_mxfp8_kernel<MI, kOdd, kEpilogue>,
                                                       cudaFuncAttributeMaxDynamicSharedMemorySize, L::kSmemBytes),
                                  "cudaFuncSetAttribute");
    attr[device] = true;
  }
  tile_mxfp8_kernel<MI, kOdd, kEpilogue><<<p.tiles_m * (p.n / kBN), kThreads, L::kSmemBytes, stream>>>(map_a, p);
}

// Checks shared by the two entries; returns the launch parameters without the epilogue's own fields.
Params Common(const tvm::ffi::TensorView& out, const tvm::ffi::TensorView& x, const tvm::ffi::TensorView& x_scale,
              const tvm::ffi::TensorView& w, const tvm::ffi::TensorView& bias, uintptr_t out_align, bool residual,
              int* rows16, int* device, cudaStream_t* stream) {
  using namespace xinfer::ffi;
  const char* op = "tile_mxfp8_sm120";
  RequireArchitecture(x, 12, 0, op);
  for (auto* t : {&out, &x, &x_scale, &w, &bias}) {
    CheckCuda(*t, op);
    CheckContiguous(*t, op);
    CheckSameDevice(*t, x, op);
  }
  CheckFloat8E4M3Storage(x, "x");
  CheckBFloat16(bias, "bias");
  if (x.ndim() != 2 || out.ndim() != 2) throw std::runtime_error("tile_mxfp8_sm120: x and the output must be rank-2");
  const int64_t m = x.size(0), k = x.size(1), n = out.size(1);
  *device = x.device().device_id;
  *stream = Stream(x);
  if (out.size(0) != m || n < kBN || n % kBN || !Supports(m, n, k, *device, residual)) {
    throw std::runtime_error(std::string("tile_mxfp8_sm120: needs x [M, K], output [M, N] with N % 256 == 0, "
                                         "K % 128 == 0, ") +
                             (residual ? "at most one 160 x 128 tile per SM" : "K <= 1024 and at most one 160 x 256 tile per SM") +
                             " (M=" + std::to_string(m) + " N=" + std::to_string(n) + " K=" + std::to_string(k) + ")");
  }
  xinfer::mxfp8::CheckScaleStorage(x_scale, m, k, "x_scale");
  if (!IsUInt8(w.dtype()) || NumElements(w) != PackedWeightBytes(n, k)) {
    throw std::runtime_error("tile_mxfp8_sm120: w must be the uint8 nk_tile256 packing of an [N, K] weight");
  }
  if (NumElements(bias) != n) throw std::runtime_error("tile_mxfp8_sm120: bias must have N elements");
  auto aligned = [](const tvm::ffi::TensorView& t, uintptr_t bytes) {
    return reinterpret_cast<uintptr_t>(t.data_ptr()) % bytes == 0;
  };
  if (!aligned(x, 16) || !aligned(w, 16) || !aligned(x_scale, 16) || !aligned(bias, 16) || !aligned(out, out_align)) {
    throw std::runtime_error("tile_mxfp8_sm120: x, x_scale, w, bias must be 16-byte aligned, the output " +
                             std::to_string(out_align) + "-byte");
  }
  *rows16 = residual ? ResidualRows16(m, n, SmCount(*device)) : TileRows16(m, n, SmCount(*device));
  Params p{};
  p.a_scale = ConstData<uint8_t>(x_scale), p.w = ConstData<uint8_t>(w), p.bias = ConstData<__nv_bfloat16>(bias);
  p.out = Data<uint8_t>(out);
  p.m = static_cast<int>(m), p.n = static_cast<int>(n), p.steps = static_cast<int>(k / kBK);
  p.tiles_m = static_cast<int>((m + 16 * *rows16 - 1) / (16 * *rows16)), p.row_blocks = static_cast<int>((m + 127) / 128);
  return p;
}

}  // namespace

// Shapes the kernels run on this device: N a multiple of 256, K of 128, at most one tile per SM; K <= 1024 for the
// 256-column engine (residual = false).
bool Supports(int64_t m, int64_t n, int64_t k, int64_t device, bool residual) {
  if (m < 1 || m > (int64_t(1) << 24) || n < kBN || n % kBN || n > (int64_t(1) << 20) || k < 128 || k % 128) return false;
  const int sms = SmCount(static_cast<int>(device));
  if (residual) return k <= (int64_t(1) << 20) && ResidualRows16(m, n, sms) != 0;
  return k <= kMaxSteps * kBK && TileRows16(m, n, sms) != 0;
}

// Bytes of the packed weight of an [N, K] matrix (prepack_weight layout "nk_tile256").
int64_t PackedWeightBytes(int64_t n, int64_t k) { return (n / kBN) * (k / kBK) * kWBlock; }

// q, q_scale = MXFP8 (1x128 scales) of gelu_tanh(x @ W^T + bias). x [M, K] E4M3 with SWIZZLE_32_4_4 scales, w the
// packed weight, bias BF16 [N]; q [M, N] E4M3, q_scale the SWIZZLE_32_4_4 storage of [M, N].
void GeluQuant128(tvm::ffi::TensorView q, tvm::ffi::TensorView q_scale, tvm::ffi::TensorView x,
                  tvm::ffi::TensorView x_scale, tvm::ffi::TensorView w, tvm::ffi::TensorView bias) {
  using namespace xinfer::ffi;
  int rows16, device;
  cudaStream_t stream;
  Params p = Common(q, x, x_scale, w, bias, 8, false, &rows16, &device, &stream);
  CheckFloat8E4M3Storage(q, "q");
  CheckCuda(q_scale, "q_scale");
  CheckContiguous(q_scale, "q_scale");
  CheckSameDevice(q_scale, x, "q_scale");
  xinfer::mxfp8::CheckScaleStorage(q_scale, p.m, p.n, "q_scale");
  if (reinterpret_cast<uintptr_t>(q_scale.data_ptr()) % 4) {
    throw std::runtime_error("tile_mxfp8_sm120: q_scale must be 4-byte aligned");
  }
  p.q_scale = Data<uint32_t>(q_scale);
  Launch<kGeluQuant128>(rows16, MapA(x.data_ptr(), p.m, x.size(1), 16 * rows16), p, device, stream);
  CheckLastCudaError("tile_mxfp8_sm120");
}

// out (BF16 [M, N]) = x @ W^T + bias with rotate-half RoPE on the first rope_cols columns (heads of 64, both halves
// of a pair inside one head; rope_cols % 128 == 0). rope_table: FP32 pairs4 [16, M, 4]. The weight is packed with
// the rotary row order of its first rope_cols rows (prepack_weight op "qkv_proj_rope").
void QkvRope(tvm::ffi::TensorView out, tvm::ffi::TensorView x, tvm::ffi::TensorView x_scale, tvm::ffi::TensorView w,
             tvm::ffi::TensorView bias, tvm::ffi::TensorView rope_table, int64_t rope_cols) {
  using namespace xinfer::ffi;
  int rows16, device;
  cudaStream_t stream;
  Params p = Common(out, x, x_scale, w, bias, 16, false, &rows16, &device, &stream);
  CheckBFloat16(out, "out");
  CheckCuda(rope_table, "rope_table");
  CheckContiguous(rope_table, "rope_table");
  CheckSameDevice(rope_table, x, "rope_table");
  if (rope_cols < 0 || rope_cols > p.n || rope_cols % 128) {
    throw std::runtime_error("tile_mxfp8_sm120: rope_cols must be a multiple of 128 within [0, N]");
  }
  const DLDataType dt = rope_table.dtype();
  if (dt.code != kDLFloat || dt.bits != 32 || NumElements(rope_table) != int64_t(16) * p.m * 4 ||
      reinterpret_cast<uintptr_t>(rope_table.data_ptr()) % 16) {
    throw std::runtime_error("tile_mxfp8_sm120: rope_table must be the 16-byte aligned FP32 pairs4 table [16, M, 4]");
  }
  p.rope = ConstData<float4>(rope_table), p.rope_cols = static_cast<int>(rope_cols);
  Launch<kQkvRope>(rows16, MapA(x.data_ptr(), p.m, x.size(1), 16 * rows16), p, device, stream);
  CheckLastCudaError("tile_mxfp8_sm120");
}

// out (BF16 [M, N]) = BF16(BF16(x @ W^T + bias) + residual). residual BF16 [M, N], may be `out` itself (no other
// overlap). rows16: tile rows / 16 (1..10) for tests and tuning, 0 picks ResidualRows16.
void Residual(tvm::ffi::TensorView out, tvm::ffi::TensorView x, tvm::ffi::TensorView x_scale, tvm::ffi::TensorView w,
              tvm::ffi::TensorView bias, tvm::ffi::TensorView residual, int64_t rows16_override) {
  using namespace xinfer::ffi;
  int rows16, device;
  cudaStream_t stream;
  Params p = Common(out, x, x_scale, w, bias, 16, true, &rows16, &device, &stream);
  CheckBFloat16(out, "out");
  CheckBFloat16(residual, "residual");
  CheckCuda(residual, "residual");
  CheckContiguous(residual, "residual");
  CheckSameDevice(residual, x, "residual");
  if (residual.ndim() != 2 || residual.size(0) != p.m || residual.size(1) != p.n ||
      reinterpret_cast<uintptr_t>(residual.data_ptr()) % 16) {
    throw std::runtime_error("tile_mxfp8_sm120: residual must be a 16-byte aligned BF16 [M, N] tensor");
  }
  const auto* r0 = static_cast<const char*>(residual.data_ptr());
  const auto* o0 = static_cast<const char*>(out.data_ptr());
  const int64_t bytes = int64_t(2) * p.m * p.n;
  if (r0 != o0 && r0 < o0 + bytes && o0 < r0 + bytes) {
    throw std::runtime_error("tile_mxfp8_sm120: residual may be the output itself but must not overlap it otherwise");
  }
  if (rows16_override != 0) {
    if (rows16_override < 1 || rows16_override > kMaxRows16 ||
        (p.m + 16 * rows16_override - 1) / (16 * rows16_override) * (p.n / 128) > SmCount(device)) {
      throw std::runtime_error("tile_mxfp8_sm120: rows16 must be 0 or a tile height in 1..10 with at most one tile per SM");
    }
    rows16 = static_cast<int>(rows16_override);
  }
  p.tiles_m = (p.m + 16 * rows16 - 1) / (16 * rows16);
  p.residual = ConstData<__nv_bfloat16>(residual);
  LaunchResidual<>(rows16, MapA(x.data_ptr(), p.m, x.size(1), 16 * rows16), p, device, stream);
  CheckLastCudaError("tile_mxfp8_sm120");
}

}  // namespace xinfer::gemm::tile_mxfp8_sm120

TVM_FFI_DLL_EXPORT_TYPED_FUNC(tile_mxfp8_gelu_quant128_sm120, xinfer::gemm::tile_mxfp8_sm120::GeluQuant128);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(tile_mxfp8_qkv_rope_sm120, xinfer::gemm::tile_mxfp8_sm120::QkvRope);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(tile_mxfp8_residual_sm120, xinfer::gemm::tile_mxfp8_sm120::Residual);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(tile_mxfp8_sm120_supports, xinfer::gemm::tile_mxfp8_sm120::Supports);
