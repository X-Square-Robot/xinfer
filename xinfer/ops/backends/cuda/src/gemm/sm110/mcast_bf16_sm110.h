// Weight-multicast BF16 tcgen05 GEMMs with fused epilogues (GEMM family backend mcast_bf16_sm110, SM110a):
// hidden 2048, 8 query / 1 KV head of 256, intermediate 16384, M up to a few hundred tokens.
// All launchers: pure device pointers + stream, no allocation, no synchronization -> CUDA-Graph capturable.
// Layouts (row-major, bf16 unless noted):
//   activations A [M][K]; weights W [N][K] (nn.Linear layout, after the offline transforms in README).
#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include "mcast_sm110_codes.h"

#ifdef __cplusplus
extern "C" {
#endif

// P2: y = xn[M][2048] . Wqkv[2560][2048]^T ; q (cols 0..2047, 8 heads x 256) and k (2048..2303) get RoPE
// (HF rotate_half: pairs (i, i+128), cos/sin fp32 [M][128]); q -> q_out[M][8][256];
// k -> kc[token][256], v -> vc[token][256] (kc/vc = layer-l slice of the KV cache, row stride 256).
int mcast_bf16_sm110_qkv_rope(const __nv_bfloat16* xn, const __nv_bfloat16* w, const float* cos_t, const float* sin_t,
                __nv_bfloat16* q_out, __nv_bfloat16* kc, __nv_bfloat16* vc, int M, cudaStream_t stream);

// P4: x[M][2048] += o[M][2048] . Wo[2048][2048]^T   (in place). ws: split-K workspace (may be NULL if the
// configured split count is 1), size mcast_bf16_sm110_workspace_bytes(M), zero-initialised once (kernel keeps counters zero).
int mcast_bf16_sm110_o_proj(const __nv_bfloat16* o, const __nv_bfloat16* w, __nv_bfloat16* x, int M, void* ws,
              cudaStream_t stream);

// P8: out[M][N] = act(row_scale[m] * (x.W^T) + bias) in BF16, act MCAST_OUT_ACT_NONE or MCAST_OUT_ACT_GELU_TANH; bias [N] and row_scale
// [M] (FP32) optional; W [N][K] row-major; N a multiple of 8 (a partial last tile is clipped), K a multiple of 64.
// Returns -2 for an unsupported N / K / act.
int mcast_bf16_sm110_gemm_out(const __nv_bfloat16* x, const __nv_bfloat16* w, __nv_bfloat16* out,
                              const __nv_bfloat16* bias, const float* row_scale, int act, int M, int N, int K,
                              cudaStream_t stream);

// P6: h[M][I] = act(xn.Wg^T) * (xn.Wu^T), act MCAST_ACT_SILU or MCAST_ACT_GELU_TANH, on the FP32 accumulators; I a
// multiple of 256, at most MCAST_MAX_INTER; w = interleaved Wgu [2I][2048] (see mcast_bf16_sm110_interleave_gate_up).
// Returns -2 for an unsupported I / act.
int mcast_bf16_sm110_gate_up(const __nv_bfloat16* xn, const __nv_bfloat16* w_gu_il, __nv_bfloat16* h, int M, int I,
                             int act, cudaStream_t stream);
// The same with I = 16384 and GELU-tanh.
int mcast_bf16_sm110_gate_up_geglu(const __nv_bfloat16* xn, const __nv_bfloat16* w_gu_il, __nv_bfloat16* h, int M,
                     cudaStream_t stream);

// P7: x[M][2048] += h[M][K] . Wd[2048][K]^T  (in place), K the intermediate size: a multiple of 64, at most
// MCAST_MAX_INTER (16384, 11008, ...). ws/counters: split-K workspace (size mcast_bf16_sm110_workspace_bytes(M),
// zero-initialised once; the kernel leaves the counters zero). w_kblk: Wd stored K-blocked [4][K/64][512][64] (prepack
// layout nk_k64blk) instead of row-major. Returns -2 for an unsupported K.
int mcast_bf16_sm110_down_ex(const __nv_bfloat16* h, const __nv_bfloat16* w, __nv_bfloat16* x, int M, int K, void* ws,
                             int w_kblk, cudaStream_t stream);
// The same with K = 16384.
int mcast_bf16_sm110_down(const __nv_bfloat16* h, const __nv_bfloat16* w, __nv_bfloat16* x, int M, void* ws,
            int w_kblk, cudaStream_t stream);
size_t mcast_bf16_sm110_down_workspace_bytes(int M);
size_t mcast_bf16_sm110_workspace_bytes(int M);
// Debug: if non-NULL, CTA 0 writes {clock64 start, end, globaltimer start, end} (ns) -> real SM clock.
int mcast_bf16_sm110_set_stamp(void* dev_ptr4xu64);

// Offline (load-time) weight transform for P6: Wg[16384][2048], Wu[16384][2048] -> Wgu_il[32768][2048],
// blocks of 128 rows: [g(0:128), u(0:128), g(128:256), u(128:256), ...]. Runs on device.
int mcast_bf16_sm110_interleave_gate_up(const __nv_bfloat16* wg, const __nv_bfloat16* wu, __nv_bfloat16* w_gu_il, int I, int K,
                          cudaStream_t stream);

// Generic knobs for experiments (tile config ids); returns 0 on success.
int mcast_bf16_sm110_set_config(const char* key, int value);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif
// Offline transform for P2 when configured with 128-wide N tiles (p2_bn=128): reorder q/k rows of each head to
// [0..63,128..191,64..127,192..255]; v rows unchanged. Output q/K/V layouts are NOT affected.
int mcast_bf16_sm110_permute_qk_rows(const __nv_bfloat16* wqkv, __nv_bfloat16* out, int K, cudaStream_t stream);
int mcast_bf16_sm110_qkv_tile_n(void);  // currently configured P2 N tile (256: no permutation needed, 128: permuted weights)
#ifdef __cplusplus
}
#endif
