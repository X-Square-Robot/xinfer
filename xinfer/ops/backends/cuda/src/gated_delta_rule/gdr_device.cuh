// Device helpers shared by the gated_delta_rule_prefill backends (mma.sync / ldmatrix / cp.async / mbarrier / PDL).
// Needs SM90+ (griddepcontrol, mbarrier.try_wait); every kernel body is compiled only for __CUDA_ARCH__ >= 900.
#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <type_traits>

namespace xinfer::gated_delta_rule_prefill::dev {

typedef __nv_bfloat16 bf16;
constexpr int D = 128, CH = 64;

__device__ __forceinline__ uint32_t su32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void ldsm4(uint32_t a, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a));
}
__device__ __forceinline__ void ldsm4t(uint32_t a, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a));
}
__device__ __forceinline__ void mma(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t pk(float lo, float hi) {
  __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&v);
}
__device__ __forceinline__ float rbf(float x) { return __bfloat162float(__float2bfloat16(x)); }
__device__ __forceinline__ void cp16(uint32_t dst, const void* src) { asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst), "l"(src)); }
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}
// e^x：ex2.approx.ftz（一条 MUFU，无非规格数修正）；结果与 __expf 只在非规格数区不同
__device__ __forceinline__ float fexp(float x) {
  float r;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x * 1.4426950408889634f));
  return r;
}
// 门控 silu：ex2 + 近似除法（2 次 MUFU）
__device__ __forceinline__ float silu_g(float x) { return __fdividef(x, 1.f + fexp(-x)); }
__device__ __forceinline__ float rsqrt_approx(float x) {
  float r;
  asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
  return r;
}
// mbarrier
__device__ __forceinline__ void mb_init(void* b, int cnt) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(su32(b)), "r"(cnt)); }
__device__ __forceinline__ void mb_arrive(void* b) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" ::"r"(su32(b)) : "memory"); }
__device__ __forceinline__ void mb_cp_arrive(void* b) { asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n" ::"r"(su32(b)) : "memory"); }
// 等待带看门狗：自旋 2^28 次仍未完成则 trap（不让卡死的 CTA 挂住 GPU）
__device__ __forceinline__ void mb_wait(void* b, int parity) {
  asm volatile(
      "{\n .reg .pred P;\n .reg .u32 n;\n mov.u32 n, 0;\n WAIT_%=:\n mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n @P bra DONE_%=;\n"
      " add.u32 n, n, 1;\n setp.gt.u32 P, n, 268435456;\n @P trap;\n bra WAIT_%=;\n DONE_%=:\n}\n" ::"r"(su32(b)),
      "r"(parity)
      : "memory");
}
// PDL：等待前一个 grid 完成（其写出可见）/ 允许下一个 grid 提前上 SM
__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;\n" ::: "memory"); }
__device__ __forceinline__ void pdl_launch() { asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory"); }
__device__ __forceinline__ void bar_pair128(int id) { asm volatile("bar.sync %0, 128;\n" ::"r"(id)); }
// 两个相邻 n8 片段（同一行 cols 8j+2q.. 与 8j+8+2q..）经一次 shfl 拼成每线程连续 4 个 bf16，四个线程正好写满一个 32B sector
__device__ __forceinline__ void st_pair(bf16* row, uint32_t x, uint32_t y, int q) {
  const uint32_t r = __shfl_xor_sync(0xffffffff, (q & 1) ? x : y, 1);
  const uint2 v = (q & 1) ? make_uint2(r, y) : make_uint2(x, r);
  *reinterpret_cast<uint2*>(row + (q & 1) * 8 + (q >> 1) * 4) = v;
}
// 同上，但只在 ok 为真时写出。shfl 必须由整个 warp 执行，所以不能把 st_pair 放进按行分叉的 if 里（行有效性在 warp 内不一致时
// 全掩码 shfl 会死锁：S 的尾段把一个 warp 的行分成有效 / 无效两部分时）。
__device__ __forceinline__ void st_pair_if(bf16* row, uint32_t x, uint32_t y, int q, bool ok) {
  const uint32_t r = __shfl_xor_sync(0xffffffff, (q & 1) ? x : y, 1);
  const uint2 v = (q & 1) ? make_uint2(r, y) : make_uint2(x, r);
  if (ok) *reinterpret_cast<uint2*>(row + (q & 1) * 8 + (q >> 1) * 4) = v;
}
// 16B 块 XOR 交错：每行 128 / 64 / 32 个 bf16
__device__ __forceinline__ int sw256(int row, int ch) { return row * 256 + ((ch ^ (row & 7)) << 4); }
__device__ __forceinline__ int sw128(int row, int ch) { return row * 128 + ((ch ^ (row & 7)) << 4); }
__device__ __forceinline__ int sw64(int row, int ch) { return row * 64 + ((ch ^ ((row >> 1) & 3)) << 4); }

__device__ __forceinline__ void bar_grp(int id, int n) { asm volatile("bar.sync %0, %1;\n" ::"r"(id), "r"(n)); }
// 16×16 fp32 块（行主序，smem）→ bf16 hi/lo 片段。A 片段（m16k16）：
__device__ __forceinline__ void frag_a_split(const float* P, int lane, uint32_t* hi, uint32_t* lo) {
  const int gq = lane >> 2, q = lane & 3;
  const float2 v[4] = {*reinterpret_cast<const float2*>(P + gq * 16 + 2 * q), *reinterpret_cast<const float2*>(P + (gq + 8) * 16 + 2 * q),
                       *reinterpret_cast<const float2*>(P + gq * 16 + 2 * q + 8), *reinterpret_cast<const float2*>(P + (gq + 8) * 16 + 2 * q + 8)};
#pragma unroll
  for (int r = 0; r < 4; ++r) {
    const __nv_bfloat162 h = __floats2bfloat162_rn(v[r].x, v[r].y);
    hi[r] = *reinterpret_cast<const uint32_t*>(&h);
    lo[r] = pk(v[r].x - __bfloat162float(h.x), v[r].y - __bfloat162float(h.y));
  }
}
// B 片段（k16 × n8，列块 nt）：b0 = (P[2q][n], P[2q+1][n])，b1 = (P[2q+8][n], P[2q+9][n])，n = 8nt + gq
__device__ __forceinline__ void frag_b_split(const float* P, int nt, int lane, uint32_t* hi, uint32_t* lo) {
  const int gq = lane >> 2, q = lane & 3, n = 8 * nt + gq;
  const float v[4] = {P[(2 * q) * 16 + n], P[(2 * q + 1) * 16 + n], P[(2 * q + 8) * 16 + n], P[(2 * q + 9) * 16 + n]};
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    const __nv_bfloat162 h = __floats2bfloat162_rn(v[2 * r], v[2 * r + 1]);
    hi[r] = *reinterpret_cast<const uint32_t*>(&h);
    lo[r] = pk(v[2 * r] - __bfloat162float(h.x), v[2 * r + 1] - __bfloat162float(h.y));
  }
}
// acc[nt] += A·B（16×16×16），bf16×3：hi·hi + hi·lo + lo·hi（相对误差约 2^-17）
__device__ __forceinline__ void mma3_16(float (*acc)[4], const float* A, const float* B, int lane) {
  uint32_t ah[4], al[4];
  frag_a_split(A, lane, ah, al);
#pragma unroll
  for (int nt = 0; nt < 2; ++nt) {
    uint32_t bh[2], bl[2];
    frag_b_split(B, nt, lane, bh, bl);
    mma(acc[nt], al, bh[0], bh[1]);
    mma(acc[nt], ah, bl[0], bl[1]);
    mma(acc[nt], ah, bh[0], bh[1]);
  }
}
// 第 L 级非对角块（一个 warp 一块，tensor core）：X = Σ_m A[bi][m]·T[m][bj]，T[bi][bj] = −T[bi][bi]·X；X 经 Xs（1 KB）转成 B 片段
template <int L>
__device__ __forceinline__ void level_mma(const float* Af, float* Tf, float* Xs, int hh, int bj, int lane) {
  const int gq = lane >> 2, q = lane & 3, bi = bj + L;
  float acc[2][4] = {};
#pragma unroll
  for (int mm = 0; mm < L; ++mm) {
    const int m = bj + mm;
    mma3_16(acc, Af + (hh * 10 + bi * (bi + 1) / 2 + m) * 256, Tf + (hh * 10 + m * (m + 1) / 2 + bj) * 256, lane);
  }
#pragma unroll
  for (int nt = 0; nt < 2; ++nt) {
    *reinterpret_cast<float2*>(Xs + gq * 16 + 8 * nt + 2 * q) = make_float2(acc[nt][0], acc[nt][1]);
    *reinterpret_cast<float2*>(Xs + (gq + 8) * 16 + 8 * nt + 2 * q) = make_float2(acc[nt][2], acc[nt][3]);
  }
  __syncwarp();
  float y[2][4] = {};
  mma3_16(y, Tf + (hh * 10 + bi * (bi + 3) / 2) * 256, Xs, lane);
  float* dst = Tf + (hh * 10 + bi * (bi + 1) / 2 + bj) * 256;
#pragma unroll
  for (int nt = 0; nt < 2; ++nt) {
    *reinterpret_cast<float2*>(dst + gq * 16 + 8 * nt + 2 * q) = make_float2(-y[nt][0], -y[nt][1]);
    *reinterpret_cast<float2*>(dst + (gq + 8) * 16 + 8 * nt + 2 * q) = make_float2(-y[nt][2], -y[nt][3]);
  }
}

}  // namespace xinfer::gated_delta_rule_prefill::dev
