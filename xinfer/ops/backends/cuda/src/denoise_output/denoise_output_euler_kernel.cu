// denoise_output_euler kernel (all architectures with programmatic dependent launch).
//
// grid = (ceil(B*M / RPC), ceil(A / OPC)), 256 threads; thread t owns k = [4t, 4t + 4) of K = 1024.
// p = rho * sum_k h_k f_k W_ak + c_a  with f = 1 + scale (adarms) / norm weight (rmsnorm) / 1 (none) and
// c = sum_k shift_k W_ak + bias_a. W, f and c depend only on constants, so they are read and reduced BEFORE
// griddepcontrol.wait (overlapping the predecessor; dependents are released at the start); after the wait one block reduction gives sum h^2 and the data
// terms together. Every per-row sum is a fixed tree (warp butterfly over xor distances 16..1, then the 8 warps in
// order), independent of RPC / OPC, so a row's result does not depend on M, B or its position in the batch.
// No CTA waits for another one.
#include <cuda_bf16.h>

#include "denoise_output_euler.cuh"

namespace xinfer::denoise_output {
namespace {

__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void pdl_launch_dependents() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }

template <typename T>
__device__ __forceinline__ void load4(const void* base, int64_t off, float* f);
template <>
__device__ __forceinline__ void load4<float>(const void* base, int64_t off, float* f) {
  const float4 v = __ldg(reinterpret_cast<const float4*>(static_cast<const float*>(base) + off));
  f[0] = v.x; f[1] = v.y; f[2] = v.z; f[3] = v.w;
}
template <>
__device__ __forceinline__ void load4<__nv_bfloat16>(const void* base, int64_t off, float* f) {
  const uint2 u = __ldg(reinterpret_cast<const uint2*>(static_cast<const __nv_bfloat16*>(base) + off));
  const float2 a = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u.x));
  const float2 b = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u.y));
  f[0] = a.x; f[1] = a.y; f[2] = b.x; f[3] = b.y;
}
// dependent data: L2 (not the read-only path, the producer kernel may have just written it)
template <typename T>
__device__ __forceinline__ float4 load4_cg(const void* base, int64_t off);
template <>
__device__ __forceinline__ float4 load4_cg<float>(const void* base, int64_t off) {
  return __ldcg(reinterpret_cast<const float4*>(static_cast<const float*>(base) + off));
}
template <>
__device__ __forceinline__ float4 load4_cg<__nv_bfloat16>(const void* base, int64_t off) {
  const uint2 u = __ldcg(reinterpret_cast<const uint2*>(static_cast<const __nv_bfloat16*>(base) + off));
  const float2 a = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u.x));
  const float2 b = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u.y));
  return make_float4(a.x, a.y, b.x, b.y);
}
__device__ __forceinline__ float bias_at(const void* b, bool bf16, int a) {
  return bf16 ? __bfloat162float(static_cast<const __nv_bfloat16*>(b)[a]) : static_cast<const float*>(b)[a];
}

// warp reduction of CNT accumulators by recursive halving (xor distances S, S/2, ..., 1); afterwards lane groups hold
// disjoint slots starting at `base`
template <int CNT, int S>
__device__ __forceinline__ void wred(float* acc, int lane, int& base) {
  if constexpr (S >= 1) {
    if constexpr (CNT > 1) {
      constexpr int h = CNT / 2;
      const bool up = lane & S;
#pragma unroll
      for (int i = 0; i < h; i++) {
        const float send = up ? acc[i] : acc[i + h];
        const float keep = up ? acc[i + h] : acc[i];
        acc[i] = keep + __shfl_xor_sync(0xffffffff, send, S);
      }
      if (up) base += h;
      wred<h, S / 2>(acc, lane, base);
    } else {
      acc[0] += __shfl_xor_sync(0xffffffff, acc[0], S);
      wred<1, S / 2>(acc, lane, base);
    }
  }
}
template <int N>
__device__ __forceinline__ void block_reduce(float (&acc)[N], float (*red)[N], int lane, int warp) {
  int base = 0;
  wred<N, 16>(acc, lane, base);
  constexpr int NV = N >= 32 ? N / 32 : 1;
  constexpr int LM = N >= 32 ? 0 : (32 / N - 1);
  if ((lane & LM) == 0) {
#pragma unroll
    for (int i = 0; i < NV; i++) red[warp][base + i] = acc[i];
  }
}

// PN: prenorm (compile time); FEAT: any of row mask / dof blend / pinned rows / per-request dt (compile time, so the
// plain step carries no epilogue branches)
template <int RPC, int OPC, typename TH, typename TW, int PN, bool FEAT>
__global__ void __launch_bounds__(256, 1) tail_kernel(const Params p) {
  constexpr int NA0 = RPC + RPC * OPC;  // sum h^2 per row + data term per (row, out)
  constexpr int NA = NA0 <= 16 ? 16 : (NA0 <= 32 ? 32 : (NA0 <= 64 ? 64 : 128));
  const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
  const int rows = p.B * p.M;
  const int r0 = blockIdx.x * RPC, o0 = blockIdx.y * OPC;
  constexpr bool kWBf16 = sizeof(TW) == 2;
  __shared__ float redC[8][16];
  __shared__ float redA[8][NA];
  __shared__ float sC[OPC];

  // the dependent kernel may start its own constant prologue now; it waits for this grid's completion before reading
  // anything this kernel writes
  pdl_launch_dependents();
  // ---- constants (before the PDL wait): every load is issued up front, then the shift reduction ----
  float wa[OPC][4];
#pragma unroll
  for (int o = 0; o < OPC; o++) {
    if (o0 + o < p.A) {
      load4<TW>(p.w, static_cast<int64_t>(o0 + o) * 1024 + 4 * t, wa[o]);
    } else {
#pragma unroll
      for (int i = 0; i < 4; i++) wa[o][i] = 0.f;
    }
  }
  float f[4] = {1.f, 1.f, 1.f, 1.f}, sh[4] = {0.f, 0.f, 0.f, 0.f};
  if (PN != kNone) {
    if (p.mod_bf16) load4<__nv_bfloat16>(p.norm_w, 4 * t, f); else load4<float>(p.norm_w, 4 * t, f);
  }
  if (PN == kAdaRms) {
    if (p.mod_bf16) load4<__nv_bfloat16>(p.shift, 4 * t, sh); else load4<float>(p.shift, 4 * t, sh);
  }
  const float bias_v = (t < OPC && p.bias != nullptr && o0 + t < p.A) ? bias_at(p.bias, kWBf16, o0 + t) : 0.f;
  if (PN == kAdaRms) {
    float c[16];
#pragma unroll
    for (int o = 0; o < 16; o++) c[o] = 0.f;
#pragma unroll
    for (int o = 0; o < OPC; o++)
      c[o] = fmaf(sh[0], wa[o][0], fmaf(sh[1], wa[o][1], fmaf(sh[2], wa[o][2], sh[3] * wa[o][3])));
    block_reduce<16>(c, redC, lane, warp);
    __syncthreads();
  }
  if (t < OPC) {
    float v = bias_v;
    if (PN == kAdaRms) {
#pragma unroll
      for (int k = 0; k < 8; k++) v += redC[k][t];
    }
    sC[t] = v;
  }
  if (PN != kNone) {
#pragma unroll
    for (int o = 0; o < OPC; o++)
#pragma unroll
      for (int i = 0; i < 4; i++) wa[o][i] *= (PN == kAdaRms ? 1.f + f[i] : f[i]);
  }
  // row addresses depend only on the parameters: computed before the wait, off the dependent chain
  int hoff[RPC];  // element offsets (the FFI bounds h to 2^31 elements)
#pragma unroll
  for (int r = 0; r < RPC; r++) {
    const int i = r0 + r;
    const int b = static_cast<int>((static_cast<float>(i) + 0.5f) * p.inv_m), tt = i - b * p.M;
    hoff[r] = i < rows ? static_cast<int>(b * p.h_sb + tt * p.h_st) + 4 * t : -1;
  }
  const int jr = t / OPC, jo = t % OPC, ji = r0 + jr, ja = o0 + jo;
  const bool owner = t < RPC * OPC && ji < rows && ja < p.A;
  const int jb = static_cast<int>((static_cast<float>(ji) + 0.5f) * p.inv_m), jt = ji - jb * p.M;
  const int64_t e = static_cast<int64_t>(ji) * p.A + ja;
  pdl_wait();

  // ---- dependent data ----
  float4 hv[RPC];
#pragma unroll
  for (int r = 0; r < RPC; r++)
    hv[r] = hoff[r] >= 0 ? load4_cg<TH>(p.h, hoff[r]) : make_float4(0.f, 0.f, 0.f, 0.f);
  float xv = 0.f;
  if (owner) xv = __ldcg(p.x + e);

  float acc[NA];
#pragma unroll
  for (int i = 0; i < NA; i++) acc[i] = 0.f;
#pragma unroll
  for (int r = 0; r < RPC; r++) {
    const float4 hh = hv[r];
    acc[r] = hh.x * hh.x + hh.y * hh.y + hh.z * hh.z + hh.w * hh.w;
#pragma unroll
    for (int o = 0; o < OPC; o++)
      acc[RPC + r * OPC + o] = fmaf(hh.x, wa[o][0], fmaf(hh.y, wa[o][1], fmaf(hh.z, wa[o][2], hh.w * wa[o][3])));
  }
  block_reduce<NA>(acc, redA, lane, warp);
  __syncthreads();

  if (owner) {
    float ss = 0.f, dot = 0.f;
#pragma unroll
    for (int k = 0; k < 8; k++) { ss += redA[k][jr]; dot += redA[k][RPC + t]; }
    float pred;
    if (PN == kNone) {
      pred = dot + sC[jo];
    } else {
      const float rho = rsqrtf(ss * (1.f / 1024.f) + p.eps);
      pred = fmaf(rho, dot, sC[jo]);
    }
    const int b = jb, tt = jt;
    if (FEAT && p.row_mask != nullptr && p.row_mask[b * p.rm_sb + tt * p.rm_st] == 0) pred = 0.f;
    float v = pred;
    if (FEAT && p.dof != nullptr) {
      const int64_t de = b * p.dof_sb + tt * p.dof_st + ja;
      const float m = p.dof_f32 ? static_cast<const float*>(p.dof)[de]
                                : static_cast<float>(static_cast<const uint8_t*>(p.dof)[de] != 0);
      v = __fadd_rn(__fmul_rn(p.vpad[e], __fsub_rn(1.f, m)), __fmul_rn(pred, m));
    }
    const float dt = (FEAT && p.dt_t != nullptr) ? p.dt_t[b] : p.dt;
    float xn = __fadd_rn(xv, __fmul_rn(dt, v));
    if (FEAT && p.pinned != nullptr && tt < p.pinned[b]) xn = p.pinned_values != nullptr ? p.pinned_values[e] : xv;
    p.out[e] = xn;
  }
}

template <typename TH, typename TW, int PN, bool FEAT>
cudaError_t LaunchTyped(const Params& p, bool pdl, cudaStream_t stream) {
  const int rows = p.B * p.M;
  cudaLaunchConfig_t cfg = {};
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.blockDim = dim3(256);
  cfg.stream = stream;
  cfg.attrs = attr;
  cfg.numAttrs = pdl ? 1 : 0;
  constexpr int OPC = 8;
  const unsigned gy = (p.A + OPC - 1) / OPC;
  // rows per CTA: more CTAs for small row counts (latency), fewer redundant weight reads for large ones
  if (rows <= 12) {
    cfg.gridDim = dim3((rows + 1) / 2, gy);
    return cudaLaunchKernelEx(&cfg, tail_kernel<2, OPC, TH, TW, PN, FEAT>, p);
  }
  if (rows <= 20) {
    cfg.gridDim = dim3((rows + 3) / 4, gy);
    return cudaLaunchKernelEx(&cfg, tail_kernel<4, OPC, TH, TW, PN, FEAT>, p);
  }
  cfg.gridDim = dim3((rows + 7) / 8, gy);
  return cudaLaunchKernelEx(&cfg, tail_kernel<8, OPC, TH, TW, PN, FEAT>, p);
}

template <typename TH, typename TW>
cudaError_t LaunchPn(const Params& p, bool pdl, cudaStream_t stream) {
  const bool feat = p.row_mask != nullptr || p.dof != nullptr || p.pinned != nullptr || p.dt_t != nullptr;
  switch (p.prenorm) {
    case kNone:
      return feat ? LaunchTyped<TH, TW, kNone, true>(p, pdl, stream) : LaunchTyped<TH, TW, kNone, false>(p, pdl, stream);
    case kRmsNorm:
      return feat ? LaunchTyped<TH, TW, kRmsNorm, true>(p, pdl, stream)
                  : LaunchTyped<TH, TW, kRmsNorm, false>(p, pdl, stream);
    default:
      return feat ? LaunchTyped<TH, TW, kAdaRms, true>(p, pdl, stream)
                  : LaunchTyped<TH, TW, kAdaRms, false>(p, pdl, stream);
  }
}

}  // namespace

cudaError_t Launch(const Params& p, bool h_bf16, bool w_bf16, bool pdl, cudaStream_t stream) {
  if (p.K != 1024 || p.A < 1 || p.A > 32 || p.B < 1 || p.M < 1) return cudaErrorInvalidValue;
  if (p.B * p.M > (1 << 20)) return cudaErrorInvalidValue;  // the row -> request split uses an FP32 reciprocal
  Params q = p;
  q.inv_m = 1.f / static_cast<float>(p.M);
  if (h_bf16) {
    return w_bf16 ? LaunchPn<__nv_bfloat16, __nv_bfloat16>(q, pdl, stream)
                  : LaunchPn<__nv_bfloat16, float>(q, pdl, stream);
  }
  return w_bf16 ? LaunchPn<float, __nv_bfloat16>(q, pdl, stream) : LaunchPn<float, float>(q, pdl, stream);
}

}  // namespace xinfer::denoise_output
