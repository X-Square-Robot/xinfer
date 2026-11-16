// gated_delta_rule_prefill: gated delta rule (linear attention) prefill from the in_proj outputs to the gated RMSNorm output
// in three PDL-chained kernels:
//   prep  -- causal conv (k = 4) + SiLU, q/k L2 normalization, beta / g / chunk-local cumsum, (I + A)^-1, w and u
//   state -- inter-chunk recurrence (state kept in registers), writes the state before each chunk and v_new
//   out   -- o = scale * (e^gc * q S + (q k^T * decay * causal) v_new), then gated RMSNorm
// chunk32: chunk 32, H_v in {H_k, 2 H_k}; chunk64: chunk 64, H_k = 16 and H_v = 32 only.
#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace xinfer::gated_delta_rule_prefill {

enum Backend : int { kChunk32 = 0, kChunk64 = 1 };
constexpr int kHeadDim = 128;
constexpr int kMaxSeq = 1024;

struct Params {
  const __nv_bfloat16* qkv;     // [T, (2 H_k + H_v) * 128], row stride ldq (elements); post-conv when preconv
  const __nv_bfloat16* zba;     // [T, H_v * 130] = [z | beta logits | decay logits], row stride ldz
  const __nv_bfloat16* conv_w;  // [(2 H_k + H_v) * 128, 4] contiguous
  const __nv_bfloat16* conv_b;  // [(2 H_k + H_v) * 128] or nullptr
  const float* A_log;           // [H_v]
  const float* dt_bias;         // [H_v]
  const __nv_bfloat16* norm_w;  // [128]
  __nv_bfloat16* out;           // [T, H_v * 128], row stride ldo
  void* workspace;              // WorkspaceBytes(backend, T, H_k, H_v) bytes, 256-byte aligned
  int64_t ldq, ldz, ldo;
  int T, HK, HV;
  float eps;
  float scale;   // o = scale * (...) before the norm; 128^-0.5 for the public interface
  bool pdl;
  bool preconv;  // qkv already holds conv + SiLU output (chunk32 only; conv_w / conv_b unused)
};

// Chunk length of a backend and the workspace it needs.
int ChunkLength(int backend);
size_t WorkspaceBytes(int backend, int T, int HK, int HV);

// Launch the three kernels on `stream` (the first one as a programmatic dependent launch when p.pdl).
cudaError_t LaunchChunk32(const Params& p, cudaStream_t stream);

// Workspace layout shared by both backends (Tp = T rounded up to the chunk, NT = Tp / chunk):
// qh, kh [H_k][Tp][128] BF16 | w, u [H_v][Tp][128] BF16 | gc [H_v][Tp] FP32 | hst [H_v][NT][128][128] BF16 | vnT [H_v][128][Tp] BF16
struct Workspace {
  __nv_bfloat16 *qh, *kh, *w, *u;
  float* gc;
  __nv_bfloat16 *hst, *vnT;
  int Tp, NT;
};
Workspace CarveWorkspace(void* base, int backend, int T, int HK, int HV);

}  // namespace xinfer::gated_delta_rule_prefill
