// chunk32 backend kernels (chunk 32, H_v = RAT * H_k, RAT in {1, 2}):
//   prep  (k head, chunk): RAT v heads; conv + SiLU + L2 norm, beta / g / chunk-local cumsum, Gkk, (I + A)^-1, w / u
//   state (v head, dv slice of 16 NG): recurrence with the state transposed in registers (accumulator fragments reused as A
//         fragments), NS-stage cp.async + mbarrier ring, writes the state before each chunk and v_new
//   out   (v head, chunk): Gqk before the PDL wait, q S + (Gqk * decay * causal) v_new, gated RMSNorm in the epilogue
#pragma once
#include "gdr_device.cuh"

namespace xinfer::gated_delta_rule_prefill::chunk32 {
using namespace dev;
constexpr int BT = 32;
// 命名屏障必须用立即数编号：寄存器编号会让编译器按 16 个屏障全占计，每 SM 只能驻留 1 个 CTA（实测 occupancy 1）
template <int ID, int N>
__device__ __forceinline__ void bar_c() { asm volatile("bar.sync %0, %1;\n" ::"n"(ID), "n"(N) : "memory"); }
__device__ __forceinline__ void bar_sel128(int sel) {
  if (sel == 0) {
    bar_c<1, 128>();
  } else if (sel == 1) {
    bar_c<2, 128>();
  } else {
    bar_c<3, 128>();
  }
}

// conv + SiLU：NCH 通道 × NR 行，行 ≥ T 或 < 0 读 0
template <int NCH, int NR, int TANH, int PRE, typename F>
__device__ __forceinline__ void conv_rows(const bf16* __restrict__ qkv, int ldq, const bf16* __restrict__ convk, const bf16* __restrict__ cbias, int ch, int t0,
                                          int r0, int T, F emit) {
  using V = typename std::conditional<NCH == 8, uint4, uint2>::type;
  if constexpr (PRE) {  // 输入已是 conv + SiLU 之后的 BF16：直接读本行，无前 3 行
    V xr[NR];
#pragma unroll
    for (int i = 0; i < NR; ++i) {
      const int tr = t0 + r0 + i;
      xr[i] = tr < T ? __ldg(reinterpret_cast<const V*>(qkv + (size_t)tr * ldq + ch)) : V{};
    }
#pragma unroll
    for (int i = 0; i < NR; ++i) {
      float y[NCH];
      const bf16* b = reinterpret_cast<const bf16*>(&xr[i]);
#pragma unroll
      for (int e = 0; e < NCH; ++e) y[e] = __bfloat162float(b[e]);
      emit(r0 + i, y);
    }
    return;
  }
  float wt[NCH][4];
#pragma unroll
  for (int i = 0; i < NCH / 2; ++i) {
    const uint4 v = *reinterpret_cast<const uint4*>(convk + (size_t)ch * 4 + i * 8);
    const bf16* b = reinterpret_cast<const bf16*>(&v);
#pragma unroll
    for (int e = 0; e < 8; ++e) wt[2 * i + e / 4][e % 4] = __bfloat162float(b[e]);
  }
  V xr[NR + 3];
#pragma unroll
  for (int i = 0; i < NR + 3; ++i) {
    const int tr = t0 + r0 + i - 3;
    xr[i] = (tr >= 0 && tr < T) ? __ldg(reinterpret_cast<const V*>(qkv + (size_t)tr * ldq + ch)) : V{};
  }
  float xw[3][NCH];
#pragma unroll
  for (int i = -3; i < NR; ++i) {
    float x[NCH];
    const bf16* b = reinterpret_cast<const bf16*>(&xr[i + 3]);
#pragma unroll
    for (int e = 0; e < NCH; ++e) x[e] = __bfloat162float(b[e]);
    if (i >= 0) {
      float y[NCH];
#pragma unroll
      for (int e = 0; e < NCH; ++e) {
        float acc = cbias ? __bfloat162float(cbias[ch + e]) : 0.f;
#pragma unroll
        for (int w = 0; w < 3; ++w) acc = fmaf(wt[e][w], xw[w][e], acc);
        acc = fmaf(wt[e][3], x[e], acc);
        if (TANH) {
          float t;
          asm("tanh.approx.f32 %0, %1;" : "=f"(t) : "f"(0.5f * acc));
          y[e] = rbf(fmaf(0.5f * acc, t, 0.5f * acc));
        } else {
          y[e] = rbf(__fdividef(acc, 1.0f + fexp(-acc)));
        }
      }
      emit(r0 + i, y);
    }
#pragma unroll
    for (int e = 0; e < NCH; ++e) xw[0][e] = xw[1][e], xw[1][e] = xw[2][e], xw[2][e] = x[e];
  }
}

// ======================================================================= K1
template <int RAT>
struct K1L {
  static constexpr int K = 0, V = 8192, A = V + RAT * 8192, Tt = A + RAT * 3072, U = Tt + RAT * 3072, P = U + RAT * 4096, SMEM = P + RAT * 256;
};
template <int RAT, int TANH, int PRE>
__global__ void __launch_bounds__(256, 2) prep_kernel(const bf16* __restrict__ qkv, int ldq, const bf16* __restrict__ zba, int ldz, int HK,
                                                     const bf16* __restrict__ convk, const float* __restrict__ A_log, const float* __restrict__ dt_bias,
                                                     bf16* __restrict__ qh, bf16* __restrict__ kh, bf16* __restrict__ wo, bf16* __restrict__ uo,
                                                     float* __restrict__ gco, int T, int Tp, int pdl, const bf16* __restrict__ cbias) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 900
  __trap();
#else
  using L = K1L<RAT>;
  extern __shared__ __align__(128) uint8_t sm[];
  const int HV = HK * RAT, g = blockIdx.x % HK, n = blockIdx.x / HK;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, t0 = n * BT, gq = lane >> 2, q = lane & 3;
  float* sBeta = reinterpret_cast<float*>(sm + L::P);  // [RAT][32]
  float* sGc = sBeta + RAT * 32;                        // [RAT][32]
  float* Af = reinterpret_cast<float*>(sm + L::A);     // [RAT][3][16][16]
  float* Tf = reinterpret_cast<float*>(sm + L::Tt);
  if (pdl) pdl_wait();
  pdl_launch();
  // P0：β、g、段内前缀和（warp hh 管 v head g·RAT+hh，lane = 行）
  if (warp < RAT) {
    const int h = g * RAT + warp, tr = t0 + lane;
    float beta = 0.f, gv = 0.f;
    if (tr < T) {
      const float b = __bfloat162float(zba[(size_t)tr * ldz + HV * D + h]);
      const float a = __bfloat162float(zba[(size_t)tr * ldz + HV * D + HV + h]);
      if (b >= 0.f) {
        beta = __fdividef(1.0f, 1.0f + __expf(-b));
      } else {
        const float z = __expf(b);
        beta = __fdividef(z, 1.0f + z);
      }
      const float u = a + dt_bias[h];
      const float sp = u > 20.0f ? u : log1pf(__expf(u));
      gv = -__expf(A_log[h]) * sp;
    }
    float incl = gv;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const float y = __shfl_up_sync(0xffffffff, incl, o);
      if (lane >= o) incl += y;
    }
    sBeta[warp * 32 + lane] = rbf(beta);
    sGc[warp * 32 + lane] = incl;
    gco[(size_t)h * Tp + t0 + lane] = incl;
  }
  // P1：q、k conv（线程 = 4 通道 × 8 行；warp 半组：cg<32 为 q，否则 k；一个 warp = 一行 128 通道）
  {
    const int cg = tid & 63, seg = tid >> 6, isk = cg >> 5, c4 = (cg & 31) * 4;
    conv_rows<4, 8, TANH, PRE>(qkv, ldq, convk, cbias, isk * HK * D + g * D + c4, t0, seg * 8, T, [&](int row, float* y) {
      float ss = 0.f;
#pragma unroll
      for (int e = 0; e < 4; ++e) ss = fmaf(y[e], y[e], ss);
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) ss += __shfl_xor_sync(0xffffffff, ss, o);
      const float rstd = rsqrt_approx(ss + 1.0e-6f);
      uint2 v = make_uint2(pk(y[0] * rstd, y[1] * rstd), pk(y[2] * rstd, y[3] * rstd));
      if (t0 + row >= T) v = make_uint2(0, 0);
      *reinterpret_cast<uint2*>((isk ? kh : qh) + ((size_t)g * Tp + t0 + row) * D + c4) = v;
      if (isk) *reinterpret_cast<uint2*>(sm + L::K + sw256(row, c4 >> 3) + (c4 & 4) * 2) = v;
    });
    // v conv：RAT·128 通道 × 32 行（线程 = 4 通道 × 32/(8/RAT) 行）
    constexpr int NCG = RAT * 32, NSEG = 256 / NCG, NR = 32 / NSEG;
    const int vcg = tid % NCG, vseg = tid / NCG, vc = vcg * 4, hv = vc >> 7, col = vc & 127;
    conv_rows<4, NR, TANH, PRE>(qkv, ldq, convk, cbias, 2 * HK * D + g * RAT * D + vc, t0, vseg * NR, T, [&](int row, float* y) {
      *reinterpret_cast<uint2*>(sm + L::V + hv * 8192 + sw256(row, col >> 3) + (col & 4) * 2) = make_uint2(pk(y[0], y[1]), pk(y[2], y[3]));
    });
  }
  __syncthreads();
  // P2：Gkk 下三角 3 块（warp 0：(0,0)，1：(1,0)，2：(1,1)）→ A_h = β_i·G·e^{gc_i−gc_j}（i > j）
  if (warp < 3) {
    const int r = warp ? 1 : 0, p = warp == 2 ? 1 : 0;
    float acc[2][4] = {};
#pragma unroll
    for (int kk = 0; kk < 8; ++kk) {
      uint32_t a[4], b[4];
      ldsm4(su32(sm + L::K + sw256(16 * r + (lane & 15), 2 * kk + (lane >> 4))), a[0], a[1], a[2], a[3]);
      ldsm4(su32(sm + L::K + sw256(16 * p + (lane & 7) + ((lane >> 4) << 3), 2 * kk + ((lane >> 3) & 1))), b[0], b[1], b[2], b[3]);
      mma(acc[0], a, b[0], b[1]);
      mma(acc[1], a, b[2], b[3]);
    }
#pragma unroll
    for (int nt = 0; nt < 2; ++nt)
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int i = 16 * r + gq + (e >> 1) * 8, j = 16 * p + 8 * nt + 2 * q + (e & 1);
#pragma unroll
        for (int hh = 0; hh < RAT; ++hh)
          Af[hh * 768 + warp * 256 + (i & 15) * 16 + (j & 15)] = i > j ? acc[nt][e] * fexp(sGc[hh * 32 + i] - sGc[hh * 32 + j]) * sBeta[hh * 32 + i] : 0.f;
      }
  }
  __syncthreads();
  // P3：对角 16×16 块前代（每线程一列）
  if (tid < 32 * RAT) {
    const int hh = tid >> 5, b = (tid >> 4) & 1, j = tid & 15, blk = hh * 3 + (b ? 2 : 0);
    const float* A = Af + blk * 256;
    float t[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      float acc = 0.f;
#pragma unroll
      for (int k = 0; k < i; ++k) acc = fmaf(A[i * 16 + k], t[k], acc);
      t[i] = i > j ? -acc : (i == j ? 1.f : 0.f);
    }
#pragma unroll
    for (int i = 0; i < 16; ++i) Tf[blk * 256 + i * 16 + j] = t[i];
  }
  __syncthreads();
  // P4：非对角块 T[1][0]（bf16×3 tensor core，X 交换区借 A 的对角块 0）
  if (warp < RAT) level_mma<1>(Af + warp * 768, Tf + warp * 768, Af + warp * 768, 0, 0, lane);
  __syncthreads();
  // P5：Tw / Tu tile [RAT][2][32][32] bf16（sw64），下三角
  for (int idx = tid; idx < RAT * 2 * 32 * 16; idx += 256) {
    const int hh = idx >> 10, isu = (idx >> 9) & 1, i = (idx >> 4) & 31, jc = 2 * (idx & 15);
    float t0v = 0.f, t1v = 0.f;
    if (jc <= i) {
      const int bi = i >> 4, bj = jc >> 4, blk = bi * (bi + 1) / 2 + bj;
      const float* t = Tf + hh * 768 + blk * 256 + (i & 15) * 16 + (jc & 15);
      t0v = t[0], t1v = jc + 1 <= i ? t[1] : 0.f;
    }
    float s0 = sBeta[hh * 32 + jc], s1 = sBeta[hh * 32 + jc + 1];
    if (!isu) s0 *= fexp(sGc[hh * 32 + jc]), s1 *= fexp(sGc[hh * 32 + jc + 1]);
    *reinterpret_cast<uint32_t*>(sm + L::U + (hh * 2 + isu) * 2048 + sw64(i, jc >> 3) + (jc & 7) * 2) = pk(t0v * s0, t1v * s1);
  }
  __syncthreads();
  // P6：w / u（作业 = (head, w|u, 行块)，RAT=1 时每作业两 warp 各管 64 列）
  {
    constexpr int CPJ = 2 / RAT, NCOL = 128 / CPJ, NJ = NCOL / 8;
    const int job = warp / CPJ, cp = warp % CPJ, hh = job >> 2, isu = (job >> 1) & 1, rb = job & 1;
    const uint8_t* tile = sm + L::U + (hh * 2 + isu) * 2048;
    const uint8_t* Bb = isu ? sm + L::V + hh * 8192 : sm + L::K;
    float acc[NJ][4] = {};
#pragma unroll
    for (int kk = 0; kk < 2; ++kk) {
      if (kk > rb) break;
      uint32_t a[4];
      ldsm4(su32(tile + sw64(16 * rb + (lane & 15), 2 * kk + (lane >> 4))), a[0], a[1], a[2], a[3]);
#pragma unroll
      for (int pp = 0; pp < NJ / 2; ++pp) {
        uint32_t b[4];
        ldsm4t(su32(Bb + sw256(16 * kk + (lane & 7) + (((lane >> 3) & 1) << 3), cp * (NCOL / 8) + 2 * pp + (lane >> 4))), b[0], b[1], b[2], b[3]);
        mma(acc[2 * pp], a, b[0], b[1]);
        mma(acc[2 * pp + 1], a, b[2], b[3]);
      }
    }
    bf16* dst = (isu ? uo : wo) + ((size_t)(g * RAT + hh) * Tp + t0 + 16 * rb) * D + cp * NCOL;
#pragma unroll
    for (int j = 0; j < NJ; j += 2) {
      st_pair(dst + (size_t)gq * D + 8 * j, pk(acc[j][0], acc[j][1]), pk(acc[j + 1][0], acc[j + 1][1]), q);
      st_pair(dst + (size_t)(gq + 8) * D + 8 * j, pk(acc[j][2], acc[j][3]), pk(acc[j + 1][2], acc[j + 1][3]), q);
    }
  }
#endif
}

// ======================================================================= K2
// CTA = (v head h, dv 片 s，宽 16·NG)，NG 组 × 4 warp；组 p 的 warp m：P 的 t 列 8m..8m+7（完整 dk），更新的 dk 列 32m..32m+31。
// 装载：p = NG−1 组为生产者（cp.async + mbarrier，NS 级环）；其余组读完一个 stage 在 empty 上登记。
template <int NG, int NS>
struct K2L {
  static constexpr int W = 0, K = 8192, U = 16384, G = U + 2048, SS = G + 128;  // stage
  static constexpr int XH = NS * SS, XV = XH + NG * 4096, XN = XV + NG * 1024, MB = XN + NG * 1024, SMEM = MB + 8 * 2 * NS;
};
template <int NG, int NS>
__global__ void __launch_bounds__(128 * NG, 1) state_kernel(const bf16* __restrict__ kh, const bf16* __restrict__ wo, const bf16* __restrict__ uo,
                                                           const float* __restrict__ gco, bf16* __restrict__ hst, bf16* __restrict__ vnT, int HK, int HV,
                                                           int T, int Tp, int pdl) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 900
  __trap();
#else
  using L = K2L<NG, NS>;
  extern __shared__ __align__(128) uint8_t sm[];
  constexpr int SPH = 128 / (16 * NG);
  const int h = blockIdx.x / SPH, s = blockIdx.x % SPH, g = h / (HV / HK);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, p = warp >> 2, m = warp & 3, gq = lane >> 2, q = lane & 3;
  const int NT = Tp / BT, dvb = 16 * NG * s;
  uint64_t* full = reinterpret_cast<uint64_t*>(sm + L::MB);
  uint64_t* empty = full + NS;
  auto load = [&](int n, int st) {
    uint8_t* base = sm + st * L::SS;
    const int t1 = tid - 128 * (NG - 1);
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const int c = t1 + 128 * k, row = c >> 4, ch = c & 15;
      cp16(su32(base + L::W + sw256(row, ch)), wo + ((size_t)h * Tp + n * BT + row) * D + ch * 8);
      cp16(su32(base + L::K + sw256(row, ch)), kh + ((size_t)g * Tp + n * BT + row) * D + ch * 8);
    }
    if (NG == 2) {
      const int row = t1 >> 2, cc = t1 & 3;  // 32 行 × 4 块 16 B
      cp16(su32(base + L::U + sw64(row, cc)), uo + ((size_t)h * Tp + n * BT + row) * D + dvb + 8 * cc);
    } else if (t1 < 64) {
      const int row = t1 >> 1, cc = t1 & 1;  // 32 行 × 2 块
      cp16(su32(base + L::U + sw64(row, cc)), uo + ((size_t)h * Tp + n * BT + row) * D + dvb + 8 * cc);
    }
    if (t1 < 8) cp16(su32(base + L::G + t1 * 16), gco + (size_t)h * Tp + n * BT + t1 * 4);
    mb_cp_arrive(&full[st]);
  };
  if (tid == 0) {
    for (int i = 0; i < NS; ++i) {
      mb_init(&full[i], 128);
      if (NG > 1) mb_init(&empty[i], 128 * (NG - 1));
    }
  }
  __syncthreads();
  if (pdl) pdl_wait();
  pdl_launch();
  if (p == NG - 1) {
#pragma unroll
    for (int i = 0; i < NS; ++i)
      if (i < NT) load(i, i);
  }
  float hacc[4][4] = {};  // hᵀ[dv16][dk 32m + 8j]
  const int dv0 = dvb + 16 * p + gq;
  uint4* XH = reinterpret_cast<uint4*>(sm + L::XH);   // [p][kk 8][lane]
  uint2* XV = reinterpret_cast<uint2*>(sm + L::XV);   // [p][kk2 2][lane][half 2]
  uint32_t* XN = reinterpret_cast<uint32_t*>(sm + L::XN);  // [p][dv 16][t 32] bf16 → u32 [p][16][16]
  for (int n = 0; n < NT; ++n) {
    const int st = n % NS;
    mb_wait(&full[st], (n / NS) & 1);
    const uint8_t* base = sm + st * L::SS;
    const float* G = reinterpret_cast<const float*>(base + L::G);
    // a) hᵀ 自己的 dk 段 → bf16 A 片段：交换区 + 写出段前状态
    {
      uint32_t a0[4], a1[4];
      a0[0] = pk(hacc[0][0], hacc[0][1]), a0[1] = pk(hacc[0][2], hacc[0][3]), a0[2] = pk(hacc[1][0], hacc[1][1]), a0[3] = pk(hacc[1][2], hacc[1][3]);
      a1[0] = pk(hacc[2][0], hacc[2][1]), a1[1] = pk(hacc[2][2], hacc[2][3]), a1[2] = pk(hacc[3][0], hacc[3][1]), a1[3] = pk(hacc[3][2], hacc[3][3]);
      XH[(p * 8 + 2 * m) * 32 + lane] = make_uint4(a0[0], a0[1], a0[2], a0[3]);
      XH[(p * 8 + 2 * m + 1) * 32 + lane] = make_uint4(a1[0], a1[1], a1[2], a1[3]);
      bf16* hp = hst + (((size_t)h * NT + n) * D + dv0) * D + 32 * m;
      {
        st_pair(hp, a0[0], a0[2], q);
        st_pair(hp + 8 * D, a0[1], a0[3], q);
        st_pair(hp + 16, a1[0], a1[2], q);
        st_pair(hp + 8 * D + 16, a1[1], a1[3], q);
      }
    }
    bar_sel128(p);
    // b) P = hᵀ · wᵀ[:, t 8m..8m+7]（两条累加链）
    float P[4] = {}, P2[4] = {};
#pragma unroll
    for (int k2 = 0; k2 < 4; ++k2) {
      uint32_t b[4];
      ldsm4(su32(base + L::W + sw256(8 * m + (lane & 7), 4 * k2 + (lane >> 3))), b[0], b[1], b[2], b[3]);
      const uint4 av0 = XH[(p * 8 + 2 * k2) * 32 + lane], av1 = XH[(p * 8 + 2 * k2 + 1) * 32 + lane];
      const uint32_t a0[4] = {av0.x, av0.y, av0.z, av0.w}, a1[4] = {av1.x, av1.y, av1.z, av1.w};
      mma(P, a0, b[0], b[1]);
      mma(P2, a1, b[2], b[3]);
    }
#pragma unroll
    for (int e = 0; e < 4; ++e) P[e] += P2[e];
    // c) v_new = u − P；vd = bf16(v_new·e^{gl−gc}) → A 片段的一半；v_new 写 XN
    const float gl = G[BT - 1];
    float v[4];
    const int tA = 8 * m + 2 * q;
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const int t = tA + (e & 1), dvl = 16 * p + gq + 8 * (e >> 1);
      const float u = __bfloat162float(*reinterpret_cast<const bf16*>(base + L::U + sw64(t, dvl >> 3) + (dvl & 7) * 2));
      v[e] = u - P[e];
    }
    {
      const float e0 = fexp(gl - G[tA]), e1 = fexp(gl - G[tA + 1]);
      XV[((p * 2 + (m >> 1)) * 32 + lane) * 2 + (m & 1)] = make_uint2(pk(v[0] * e0, v[1] * e1), pk(v[2] * e0, v[3] * e1));
      XN[(p * 16 + gq) * 16 + 4 * m + q] = pk(v[0], v[1]);
      XN[(p * 16 + gq + 8) * 16 + 4 * m + q] = pk(v[2], v[3]);
    }
    bar_sel128(p);
    // d) v_new 按 dv 行写出（warp m 写 dv 行 4m..4m+3，每行 64 B）；hᵀ[:, dk 段] = e^{gl}·hᵀ + vdᵀ k̂
    {
      const int r = 4 * m + (lane >> 3), c = lane & 7;
      const uint2 x = *reinterpret_cast<const uint2*>(&XN[(p * 16 + r) * 16 + 2 * c]);
      *reinterpret_cast<uint2*>(vnT + ((size_t)h * D + dvb + 16 * p + r) * Tp + n * BT + 4 * c) = x;
    }
    const float dl = fexp(gl);
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
      for (int e = 0; e < 4; ++e) hacc[j][e] *= dl;
#pragma unroll
    for (int kk = 0; kk < 2; ++kk) {
      const uint2 lo = XV[((p * 2 + kk) * 32 + lane) * 2], hi = XV[((p * 2 + kk) * 32 + lane) * 2 + 1];
      const uint32_t a[4] = {lo.x, lo.y, hi.x, hi.y};
#pragma unroll
      for (int pp = 0; pp < 2; ++pp) {
        uint32_t b[4];
        ldsm4t(su32(base + L::K + sw256(16 * kk + (lane & 7) + (((lane >> 3) & 1) << 3), 4 * m + 2 * pp + (lane >> 4))), b[0], b[1], b[2], b[3]);
        mma(hacc[2 * pp], a, b[0], b[1]);
        mma(hacc[2 * pp + 1], a, b[2], b[3]);
      }
    }
    // e) 释放 stage；生产者补装 n+NS
    if (NG == 1) {
      bar_c<1, 128>();
      if (n + NS < NT) load(n + NS, st);
    } else if (p == 0) {
      mb_arrive(&empty[st]);
    } else {
      bar_c<3, 128>();
      if (n + NS < NT) {
        mb_wait(&empty[st], (n / NS) & 1);
        load(n + NS, st);
      }
    }
  }
#endif
}

// ======================================================================= K3
// CTA = (v head h, 段 n)，8 warp：rb = warp>>2（行块 16 行），qd = warp&3（dv 32 列）。
// smem：Q 8K（后作 SS）| K 8K | S 片环 2×8K（hᵀ_n[:, dk 32i..+31]）| VN 8K（vnT[dv][t]）| GK 4K（Gqk fp32 [32][32]）| G 128 —— 44.6 KB，每 SM 2 个
struct K3L {
  static constexpr int Q = 0, K = 8192, SR = 16384, VN = SR + 16384, GK = VN + 8192, SSQ = Q /* Q 区在 q̂S 第 0 片后已无用 */, G = GK + 4096, SMEM = G + 128;
};
__device__ __forceinline__ void k3_piece(uint8_t* dst, const bf16* hst, size_t off, int i, int tid) {
  const bf16* src = hst + off + 32 * i;
#pragma unroll
  for (int k = 0; k < 2; ++k) {
    const int cc = tid + 256 * k, row = cc >> 2, ch = cc & 3;
    cp16(su32(dst + sw64(row, ch)), src + (size_t)row * D + ch * 8);
  }
}
__global__ void __launch_bounds__(256, 2) out_kernel(const bf16* __restrict__ qh, const bf16* __restrict__ kh, const bf16* __restrict__ hst,
                                                    const bf16* __restrict__ vnT, const float* __restrict__ gco, const bf16* __restrict__ zba, int ldz,
                                                    const bf16* __restrict__ nw, bf16* __restrict__ out, int ldo, int HK, int HV, int T, int Tp,
                                                    float scale, float eps, int pdl) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 900
  __trap();
#else
  using L = K3L;
  extern __shared__ __align__(128) uint8_t sm[];
  const int h = blockIdx.x % HV, n = blockIdx.x / HV, g = h / (HV / HK), NT = Tp / BT, t0 = n * BT;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, rb = warp >> 2, qd = warp & 3, gq = lane >> 2, q = lane & 3;
  // C0：q̂、k̂、gc（K1 的输出：本 kernel 由 K2 以 PDL 触发，K2 已等待 K1 完成）
  {
    const bf16* qs = qh + ((size_t)g * Tp + t0) * D;
    const bf16* ks = kh + ((size_t)g * Tp + t0) * D;
#pragma unroll
    for (int k = 0; k < 2; ++k) {
      const int cc = tid + 256 * k, row = cc >> 4, ch = cc & 15;
      cp16(su32(sm + L::Q + sw256(row, ch)), qs + (size_t)row * D + ch * 8);
      cp16(su32(sm + L::K + sw256(row, ch)), ks + (size_t)row * D + ch * 8);
    }
    if (tid < 8) cp16(su32(sm + L::G + tid * 16), gco + (size_t)h * Tp + t0 + tid * 4);
    cp_commit();
  }
  // 门控因子 w·silu(z) 的原始值（z 是 in_proj 输出）
  const int row0 = 16 * rb + gq, row1 = row0 + 8;
  uint32_t zv[4][2], wv[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int col = 32 * qd + 8 * j + 2 * q;
    wv[j] = *reinterpret_cast<const uint32_t*>(nw + col);
#pragma unroll
    for (int e = 0; e < 2; ++e) {
      const int tr = t0 + (e ? row1 : row0);
      zv[j][e] = tr < T ? *reinterpret_cast<const uint32_t*>(zba + (size_t)tr * ldz + h * D + col) : 0u;
    }
  }
  cp_wait<0>();
  __syncthreads();
  // q̂ A 片段（本行块 16 行 × dk 128）
  uint32_t qA[8][4];
#pragma unroll
  for (int kk = 0; kk < 8; ++kk) ldsm4(su32(sm + L::Q + sw256(16 * rb + (lane & 15), 2 * kk + (lane >> 4))), qA[kk][0], qA[kk][1], qA[kk][2], qA[kk][3]);
  // Gqk：行块 rb 的 n8 列块 jt ≤ 2rb+1，分给本行块的 4 个 warp（jt = qd 与 qd+4）
  float* GK = reinterpret_cast<float*>(sm + L::GK);
#pragma unroll
  for (int jj = 0; jj < 2; ++jj) {
    const int jt = qd + 4 * jj;
    if (jt <= 2 * rb + 1) {
      float acc[4] = {};
#pragma unroll
      for (int k2 = 0; k2 < 4; ++k2) {
        uint32_t b[4];
        ldsm4(su32(sm + L::K + sw256(8 * jt + (lane & 7), 4 * k2 + (lane >> 3))), b[0], b[1], b[2], b[3]);
        mma(acc, qA[2 * k2], b[0], b[1]);
        mma(acc, qA[2 * k2 + 1], b[2], b[3]);
      }
      *reinterpret_cast<float2*>(GK + row0 * 32 + 8 * jt + 2 * q) = make_float2(acc[0], acc[1]);
      *reinterpret_cast<float2*>(GK + row1 * 32 + 8 * jt + 2 * q) = make_float2(acc[2], acc[3]);
    }
  }
  if (pdl) pdl_wait();  // hst / vnT 来自 K2
  const size_t soff = (((size_t)h * NT + n) * D) * D;
  uint8_t* ring = sm + L::SR;
  k3_piece(ring, hst, soff, 0, tid);
  cp_commit();
  k3_piece(ring + 8192, hst, soff, 1, tid);
  cp_commit();
  {
    const bf16* src = vnT + (size_t)h * D * Tp + t0;
#pragma unroll
    for (int k = 0; k < 2; ++k) {
      const int cc = tid + 256 * k, row = cc >> 2, ch = cc & 3;
      cp16(su32(sm + L::VN + sw64(row, ch)), src + (size_t)row * Tp + ch * 8);
    }
    cp_commit();
  }
  // q̂·S：4 片（dk 32i..32i+31），2 槽
  float o[4][4] = {};
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    if (i < 2) {
      cp_wait<2>();
    } else if (i == 2) {
      cp_wait<1>();
    } else {
      cp_wait<0>();
    }
    __syncthreads();
    const uint8_t* piece = ring + (i & 1) * 8192;
#pragma unroll
    for (int jp = 0; jp < 2; ++jp) {
      uint32_t b[4], c[4];
      // n8 列块 2jp、2jp+1（dv 32qd + 16jp + ...），k = 本片 dk 0..15 与 16..31
      ldsm4(su32(piece + sw64(32 * qd + 16 * jp + (lane & 7) + ((lane >> 4) << 3), (lane >> 3) & 1)), b[0], b[1], b[2], b[3]);
      ldsm4(su32(piece + sw64(32 * qd + 16 * jp + (lane & 7) + ((lane >> 4) << 3), 2 + ((lane >> 3) & 1))), c[0], c[1], c[2], c[3]);
      mma(o[2 * jp], qA[2 * i], b[0], b[1]);
      mma(o[2 * jp + 1], qA[2 * i], b[2], b[3]);
      mma(o[2 * jp], qA[2 * i + 1], c[0], c[1]);
      mma(o[2 * jp + 1], qA[2 * i + 1], c[2], c[3]);
    }
    if (i < 2) {
      __syncthreads();
      k3_piece(ring + i * 8192, hst, soff, i + 2, tid);
      cp_commit();
    }
  }
  // o·e^{gc} + Aqk·v_new
  const float* Gs = reinterpret_cast<const float*>(sm + L::G);
  const float e0 = fexp(Gs[row0]), e1 = fexp(Gs[row1]);
#pragma unroll
  for (int j = 0; j < 4; ++j) o[j][0] *= e0, o[j][1] *= e0, o[j][2] *= e1, o[j][3] *= e1;
#pragma unroll
  for (int kk = 0; kk < 2; ++kk) {
    if (kk > rb) break;
    uint32_t pA[4];
#pragma unroll
    for (int hf = 0; hf < 2; ++hf) {
      const int j0 = 16 * kk + 8 * hf + 2 * q;
      float x[4];
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int i = e < 2 ? row0 : row1, j = j0 + (e & 1);
        x[e] = i >= j ? GK[i * 32 + j] * fexp(Gs[i] - Gs[j]) : 0.f;
      }
      pA[2 * hf] = pk(x[0], x[1]);
      pA[2 * hf + 1] = pk(x[2], x[3]);
    }
#pragma unroll
    for (int jp = 0; jp < 2; ++jp) {
      uint32_t b[4];
      ldsm4(su32(sm + L::VN + sw64(32 * qd + 16 * jp + (lane & 7) + ((lane >> 4) << 3), 2 * kk + ((lane >> 3) & 1))), b[0], b[1], b[2], b[3]);
      mma(o[2 * jp], pA, b[0], b[1]);
      mma(o[2 * jp + 1], pA, b[2], b[3]);
    }
  }
  // bf16(o·scale)，gated RMSNorm（一行 128 = 本行块 4 个 warp 的 32 列）
  float ss0 = 0.f, ss1 = 0.f;
#pragma unroll
  for (int j = 0; j < 4; ++j)
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      o[j][e] = rbf(o[j][e] * scale);
      if (e < 2) {
        ss0 = fmaf(o[j][e], o[j][e], ss0);
      } else {
        ss1 = fmaf(o[j][e], o[j][e], ss1);
      }
    }
#pragma unroll
  for (int x = 1; x < 4; x <<= 1) ss0 += __shfl_xor_sync(0xffffffff, ss0, x), ss1 += __shfl_xor_sync(0xffffffff, ss1, x);
  float* SSQ = reinterpret_cast<float*>(sm + L::SSQ);  // [32 行][4 qd]
  if (q == 0) SSQ[row0 * 4 + qd] = ss0, SSQ[row1 * 4 + qd] = ss1;
  bar_sel128(rb);
  const float4 a0 = *reinterpret_cast<const float4*>(SSQ + row0 * 4), a1 = *reinterpret_cast<const float4*>(SSQ + row1 * 4);
  const float i0 = rsqrt_approx(__fdividef((a0.x + a0.y) + (a0.z + a0.w), 128.f) + eps), i1 = rsqrt_approx(__fdividef((a1.x + a1.y) + (a1.z + a1.w), 128.f) + eps);
  uint32_t yo[4][2];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const __nv_bfloat162 w2 = *reinterpret_cast<const __nv_bfloat162*>(&wv[j]);
    const float w0 = __bfloat162float(w2.x), w1 = __bfloat162float(w2.y);
#pragma unroll
    for (int e = 0; e < 2; ++e) {
      const __nv_bfloat162 z2 = *reinterpret_cast<const __nv_bfloat162*>(&zv[j][e]);
      const float za = __bfloat162float(z2.x), zb = __bfloat162float(z2.y), inv = e ? i1 : i0;
      yo[j][e] = pk(o[j][2 * e] * inv * w0 * silu_g(za), o[j][2 * e + 1] * inv * w1 * silu_g(zb));
    }
  }
#pragma unroll
  for (int j = 0; j < 4; j += 2)
#pragma unroll
    for (int e = 0; e < 2; ++e) {
      const int tr = t0 + (e ? row1 : row0);
      st_pair_if(out + (size_t)(tr < T ? tr : 0) * ldo + h * D + 32 * qd + 8 * j, yo[j][e], yo[j + 1][e], q, tr < T);
    }
#endif
}
}  // namespace xinfer::gated_delta_rule_prefill::chunk32
