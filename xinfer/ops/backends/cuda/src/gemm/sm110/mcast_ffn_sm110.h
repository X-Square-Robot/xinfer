// prefill_ffn: fused (FFN-chunked) P6+P7 for a hidden-2048 MLP (intermediate 16384, or I with _ex; M <= 768) on Thor SM110a.
// x[M][2048] += (act(xn Wg^T) * (xn Wu^T)) Wd^T, act GELU-tanh or SiLU, with the intermediate h kept in L2 (never in DRAM).
// All entry points: plain device pointers + stream, no allocation / sync -> CUDA-graph capturable.
#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stddef.h>

#include "mcast_sm110_codes.h"

#ifdef __cplusplus
extern "C" {
#endif

// One FFN layer.  xn: RMSNorm output [M][2048] bf16; w_gu_il: [32768][2048] gate/up rows interleaved per 128
// (same layout as prefill_gemm, pg_interleave_gate_up); w_d: [2048][16384]; x: residual [M][2048] (in/out);
// hbuf: mcast_ffn_sm110_hbuf_bytes(M) scratch (L2-resident chunk ring); flags: mcast_ffn_sm110_flags_bytes() ints, zeroed ONCE at allocation
// (the kernel leaves them zero again).  Requires 512 < M <= 768 and 18 free SMs (3 co-resident 6-CTA clusters).
int mcast_ffn_sm110(const void* xn, const void* w_gu_il, const void* w_d, void* x, void* hbuf, int* flags, int M,
           cudaStream_t stream);
size_t mcast_ffn_sm110_hbuf_bytes(int M);
// The same for intermediate I (I / 256 gate/up blocks, at most 64, divisible by a chunk count in [2, 8]: w_gu_il
// [2I][2048], w_d [2048][I]) and act MCAST_ACT_SILU or MCAST_ACT_GELU_TANH. hbuf_bytes_ex is 0 for an I the launch
// rejects.
int mcast_ffn_sm110_ex(const void* xn, const void* w_gu_il, const void* w_d, void* x, void* hbuf, int* flags, int M, int I,
                       int act, cudaStream_t stream);
size_t mcast_ffn_sm110_hbuf_bytes_ex(int M, int I);
// Load-time weight transform for the default wblk=1 layout (apply to the interleaved W_gu [2I][2048] and to
// W_d [2048][I], N rows by K columns): dst[(nb*K/64 + kb)*512 + r][c] = src[nb*512 + r][kb*64 + c].  src != dst.
int mcast_ffn_sm110_block_weight(const void* src, void* dst, int N, int K, cudaStream_t stream);
size_t mcast_ffn_sm110_flags_bytes(void);
// Experiment knobs (see README): C, S, slast, mode, nslot, pol_w, pol_hs, pol_hl, pol_x, pol_a, ...
int mcast_ffn_sm110_set_config(const char* key, int value);
int mcast_ffn_sm110_get_config(const char* key);
int mcast_ffn_sm110_set_stamp(void* ptr);  // [8] int64: [0..3] CTA0 clock64 s/e + globaltimer s/e, [4] kernel-wide end (atomicMax)
int mcast_ffn_sm110_set_dbg(void* ptr);    // per-CTA per-unit timeline [grid][128][8] globaltimer (NULL = off)
int mcast_ffn_sm110_schedule_info(int* out, int n);
// prefill_opt additions
int mcast_ffn_sm110_set_pace_ptr(const int* pace_dev);   // device int read by every later launch at kernel start (NULL: cfg pace)
int mcast_ffn_sm110_set_prefetch(const void* p0, long long b0, const void* p1, long long b1, int q0_milli);  // next launch only  // last launch: {queue length, total k-blocks}

#ifdef __cplusplus
}
#endif
