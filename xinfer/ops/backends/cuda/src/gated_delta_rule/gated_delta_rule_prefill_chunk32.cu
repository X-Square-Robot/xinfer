// chunk32 backend of gated_delta_rule_prefill: launches prep / state / out.
#include "chunk32_kernels.cuh"
#include "gated_delta_rule_prefill.cuh"

namespace xinfer::gated_delta_rule_prefill {
namespace {
template <typename K>
cudaError_t Prepare(K kernel, int smem) {
  cudaError_t e = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  if (e == cudaSuccess) e = cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
  return e;
}
}  // namespace

cudaError_t LaunchChunk32(const Params& p, cudaStream_t stream) {
  using namespace chunk32;
  const Workspace ws = CarveWorkspace(p.workspace, kChunk32, p.T, p.HK, p.HV);
  const int rat = p.HV / p.HK;
  const int ldq = static_cast<int>(p.ldq), ldz = static_cast<int>(p.ldz), ldo = static_cast<int>(p.ldo);
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = 1;
  cudaLaunchConfig_t cfg = {};
  cfg.stream = stream;
  cfg.attrs = at;
  cfg.numAttrs = p.pdl ? 1 : 0;
  const int pdl1 = p.pdl ? 1 : 0;
  // prep：silu 用 tanh.approx（conv 段 SIMT 发射受限，−11%，精度仍好于参照链）
  cfg.gridDim = dim3(p.HK * ws.NT);
  cfg.blockDim = dim3(256);
  cudaError_t e;
#define GDR_PREP(R, PRE)                                                                                                      \
  {                                                                                                                           \
    cfg.dynamicSmemBytes = K1L<R>::SMEM;                                                                                      \
    if ((e = Prepare(prep_kernel<R, 1, PRE>, K1L<R>::SMEM)) != cudaSuccess) return e;                                         \
    e = cudaLaunchKernelEx(&cfg, prep_kernel<R, 1, PRE>, p.qkv, ldq, p.zba, ldz, p.HK, p.conv_w, p.A_log, p.dt_bias, ws.qh, ws.kh, \
                           ws.w, ws.u, ws.gc, p.T, ws.Tp, pdl1, p.conv_b);                                                     \
  }
  if (rat == 1 && !p.preconv) GDR_PREP(1, 0) else if (rat == 1) GDR_PREP(1, 1) else if (!p.preconv) GDR_PREP(2, 0) else GDR_PREP(2, 1)
#undef GDR_PREP
  if (e != cudaSuccess) return e;
  // state / out 总是以 PDL 接在前一个 kernel 之后
  cfg.numAttrs = 1;
  if (rat == 1) {
    using L = K2L<1, 3>;
    cfg.gridDim = dim3(p.HV * 8);
    cfg.blockDim = dim3(128);
    cfg.dynamicSmemBytes = L::SMEM;
    if ((e = Prepare(state_kernel<1, 3>, L::SMEM)) != cudaSuccess) return e;
    e = cudaLaunchKernelEx(&cfg, state_kernel<1, 3>, (const __nv_bfloat16*)ws.kh, (const __nv_bfloat16*)ws.w, (const __nv_bfloat16*)ws.u,
                           (const float*)ws.gc, ws.hst, ws.vnT, p.HK, p.HV, p.T, ws.Tp, 1);
  } else {
    using L = K2L<2, 3>;
    cfg.gridDim = dim3(p.HV * 4);
    cfg.blockDim = dim3(256);
    cfg.dynamicSmemBytes = L::SMEM;
    if ((e = Prepare(state_kernel<2, 3>, L::SMEM)) != cudaSuccess) return e;
    e = cudaLaunchKernelEx(&cfg, state_kernel<2, 3>, (const __nv_bfloat16*)ws.kh, (const __nv_bfloat16*)ws.w, (const __nv_bfloat16*)ws.u,
                           (const float*)ws.gc, ws.hst, ws.vnT, p.HK, p.HV, p.T, ws.Tp, 1);
  }
  if (e != cudaSuccess) return e;
  cfg.gridDim = dim3(p.HV * ws.NT);
  cfg.blockDim = dim3(256);
  cfg.dynamicSmemBytes = K3L::SMEM;
  if ((e = Prepare(out_kernel, K3L::SMEM)) != cudaSuccess) return e;
  return cudaLaunchKernelEx(&cfg, out_kernel, (const __nv_bfloat16*)ws.qh, (const __nv_bfloat16*)ws.kh, (const __nv_bfloat16*)ws.hst,
                            (const __nv_bfloat16*)ws.vnT, (const float*)ws.gc, p.zba, ldz, p.norm_w, p.out, ldo, p.HK, p.HV, p.T, ws.Tp,
                            p.scale, p.eps, 1);
}

}  // namespace xinfer::gated_delta_rule_prefill
