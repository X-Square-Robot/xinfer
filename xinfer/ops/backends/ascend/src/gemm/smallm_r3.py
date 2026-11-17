"""smallm-r3: action-loop M16 GEMM chain kernels (Round-3 SMALLM-K).

Four kernels for the wall_oss action loop (180 calls each per request), all
BF16-in, N-partitioned to exactly 32 blocks (one wave on the 32-AIC 950DT),
FP32 L0C accumulation:

  smallm_r3_o_proj   A_bnsd[16,16,128] x W[1024,2048] -> OUT[16,1024] BF16.
      Replaces the shipped 16-block smallm_bnsd_o_proj (e2e 6.885us, mte2
      5.50us serialized on half the cores) + the follow-up FP32->BF16 cast
      slot (1.85us). Single-shot full-K W stream; the packed A is preloaded
      once (16 pipelined head copies) so the Cube sees the same 16x128-step
      k order as the shipped kernel (FP32-accum order preserved).
  smallm_r3_gateup_glu  x[16,1024] x W_int[4096,1024] -> silu(g)*u -> OUT
      [16,2048] BF16. W_int is the one-time row-interleaved weight (even rows
      gate, odd rows up). Fuses the aclnnMm (6.22us) and the
      gemm_glu_fp32_finish_split_2048 slot (3.50us) into one launch; the
      L0C->UB dual_copy N-split keeps each (g,u) pair inside one AIV half
      (smallm_kn bias-kernel structure). Scalar witness silu
      g/(1.0+exp(-g))*u and the integer-RNE store.
  smallm_r3_down      x[16,2048] x W[1024,2048] -> OUT[16,1024] BF16.
      Single-shot bn=32 (32 blocks x one 128KB W DMA each); the BF16 store
      folds the cast round (RNE, the bnsd_m768 argument).
  smallm_r3_qkv       x[16,1024] x W[2560,1024] -> OUT[16,2560] FP32.
      Plain swap (bn=80 -> 32 blocks); keeps the FP32 accumulator contract
      the m16 rope finish consumes.

Dissection evidence: smallm-r3/results/bench-dissect.json,
bench-fusion*.json, prof-dissect.json (wave-1/2, card 0, rot-16 graph p50).
"""

import tilelang.ascend.language as T

from .silu_and_mul import _store_rne

HEADS, HEAD_DIM = 16, 128


def o_proj(bn=32):
    columns, reduction = 1024, HEADS * HEAD_DIM

    @T.prim_func
    def main(a_handle: T.handle, weight_handle: T.handle, output_handle: T.handle):
        A = T.match_buffer(a_handle, (HEADS, 16, HEAD_DIM), "bfloat16")
        W = T.match_buffer(weight_handle, (columns, reduction), "bfloat16")
        OUT = T.match_buffer(output_handle, (16, columns), "bfloat16")
        with T.Kernel(columns // bn) as block:
            accum = T.alloc_l0c((16, bn), "float32")
            a_full = T.alloc_l1((16, HEADS * HEAD_DIM), "bfloat16")
            w_l1 = T.alloc_l1((bn, HEADS * HEAD_DIM), "bfloat16")
            n0 = T.cast(block * bn, "int32")
            for h in range(HEADS):
                T.copy(A[h, 0:16, 0:HEAD_DIM],
                       a_full[0:16, h * HEAD_DIM : (h + 1) * HEAD_DIM])
            T.copy(W[n0 : n0 + bn, 0 : HEADS * HEAD_DIM], w_l1)
            T.gemm(a_full, w_l1, accum, transpose_B=True, clear_accum=True,
                   unit_flag_ctrl=3)
            T.copy(accum, OUT[0:16, n0 : n0 + bn], unit_flag_ctrl=3)

    return main


def gateup_glu(bn_pairs=64, bk=128, stages=3):
    columns, reduction, half = 4096, 1024, 2048
    rows_per_block = 2 * bn_pairs

    @T.prim_func
    def main(x_handle: T.handle, weight_handle: T.handle, output_handle: T.handle):
        X = T.match_buffer(x_handle, (16, reduction), "bfloat16")
        W = T.match_buffer(weight_handle, (columns, reduction), "bfloat16")
        OUT = T.match_buffer(output_handle, (16, half), "bfloat16")
        with T.Kernel(half // bn_pairs) as block:
            accum = T.alloc_l0c((16, rows_per_block), "float32")
            x_l1 = T.alloc_l1((16, bk), "bfloat16")
            w_l1 = T.alloc_l1((rows_per_block, bk), "bfloat16")
            r0 = T.cast(block * rows_per_block, "int32")
            k_tiles = reduction // bk
            for kt in T.Pipelined(k_tiles, num_stages=stages):
                k0 = T.cast(kt * bk, "int32")
                T.copy(X[0:16, k0 : k0 + bk], x_l1)
                T.copy(W[r0 : r0 + rows_per_block, k0 : k0 + bk], w_l1)
                T.gemm(x_l1, w_l1, accum, transpose_B=True, clear_accum=kt == 0,
                       unit_flag_ctrl=T.Select(kt == k_tiles - 1, 3, 2))
            accum_ub = T.alloc_shared((16, bn_pairs), "float32")
            out_ub = T.alloc_shared((16, bn_pairs // 2), "bfloat16")
            T.dual_copy(accum, accum_ub, unit_flag_ctrl=3)
            with T.SimtVF(threads=128):
                for i, j in T.Parallel(16, bn_pairs // 2):
                    g = accum_ub[i, 2 * j]
                    u = accum_ub[i, 2 * j + 1]
                    value = g / (1.0 + T.exp(-g)) * u
                    out_ub[i, j] = _store_rne(value, "bfloat16")
            T.dual_copy(out_ub, OUT[0:16, block * bn_pairs : (block + 1) * bn_pairs])

    return main


def down():
    K, N, bn = 2048, 1024, 32

    @T.prim_func
    def main(x_handle: T.handle, weight_handle: T.handle, output_handle: T.handle):
        X = T.match_buffer(x_handle, (16, K), "bfloat16")
        W = T.match_buffer(weight_handle, (N, K), "bfloat16")
        OUT = T.match_buffer(output_handle, (16, N), "bfloat16")
        with T.Kernel(N // bn) as block:
            accum = T.alloc_l0c((16, bn), "float32")
            x_l1 = T.alloc_l1((16, K), "bfloat16")
            w_l1 = T.alloc_l1((bn, K), "bfloat16")
            n0 = T.cast(block * bn, "int32")
            T.copy(X[0:16, 0:K], x_l1)
            T.copy(W[n0 : n0 + bn, 0:K], w_l1)
            T.gemm(x_l1, w_l1, accum, transpose_B=True, clear_accum=True,
                   unit_flag_ctrl=3)
            T.copy(accum, OUT[0:16, n0 : n0 + bn], unit_flag_ctrl=3)

    return main


def qkv():
    K, N, bn, bk = 1024, 2560, 80, 128

    @T.prim_func
    def main(x_handle: T.handle, weight_handle: T.handle, output_handle: T.handle):
        X = T.match_buffer(x_handle, (16, K), "bfloat16")
        W = T.match_buffer(weight_handle, (N, K), "bfloat16")
        OUT = T.match_buffer(output_handle, (16, N), "float32")
        with T.Kernel(N // bn) as block:
            accum = T.alloc_l0c((16, bn), "float32")
            x_l1 = T.alloc_l1((16, bk), "bfloat16")
            w_l1 = T.alloc_l1((bn, bk), "bfloat16")
            n0 = T.cast(block * bn, "int32")
            k_tiles = K // bk
            for kt in T.Pipelined(k_tiles, num_stages=3):
                k0 = T.cast(kt * bk, "int32")
                T.copy(X[0:16, k0 : k0 + bk], x_l1)
                T.copy(W[n0 : n0 + bn, k0 : k0 + bk], w_l1)
                T.gemm(x_l1, w_l1, accum, transpose_B=True, clear_accum=kt == 0,
                       unit_flag_ctrl=T.Select(kt == k_tiles - 1, 3, 2))
            T.copy(accum, OUT[0:16, n0 : n0 + bn], unit_flag_ctrl=3)

    return main


def build_programs():
    yield "smallm_r3_o_proj", o_proj(32)
    yield "smallm_r3_gateup_glu", gateup_glu(64, 128, 3)
    yield "smallm_r3_down", down()
    yield "smallm_r3_qkv", qkv()
