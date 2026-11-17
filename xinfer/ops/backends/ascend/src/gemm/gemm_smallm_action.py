# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""Small-M action-stage Cube GEMMs reading the persistent KN weights directly.

The r4 o-projection route: the ACLNN FIA V5 attention writes its output BNSD
[B, H, M, D] (strided out writes are rejected), and the wall_oss action loop
used to pay a 4.0 us out-finish transpose per call so the o-projection Mm
could read a packed [M, H*D] A. This kernel reads the BNSD-laid A directly:
the k-th 128-wide column group of the logical [M, K] A is the contiguous
[M, D] block A_bnsd[k // D, :, :], so one-head k-steps need no permutation
copies at all. The K accumulation order (16 sequential D=128 Cube steps in
one L0C accumulator) matches the ACLNN Mm bit-for-bit on the o-projection
shape (probe 2026-11-11 r4: aclnn_delta 0.0, oracle_rel 1e-6).

Tile geometry bn=64 / stages=3 measured fastest (5.46 us in-graph vs the
7.61 us transpose+ACLNN composite; probe-smallm-sweep.json).

qkv-batch r1: the same kernel at ROWS=32/64 (the B2/B4 request-batch
o/down projections, A = [16, B*16, 128]). The per-row K order is unchanged
(the same 16 sequential head steps), so the batch forms keep the ACLNN
bitwise match, and the 64-row tile carries no M padding at all.
"""

import tilelang.ascend.language as T

HEADS, HEAD_DIM = 16, 128
BN = 64


def bnsd_o_proj(rows):
    """OUT[rows, N] FP32 = A_bnsd[H, rows, D] @ W[N, H*D]^T for the action shape."""
    columns = 1024
    reduction = HEADS * HEAD_DIM

    @T.prim_func
    def main(a_handle: T.handle, weight_handle: T.handle, output_handle: T.handle):
        A = T.match_buffer(a_handle, (HEADS, rows, HEAD_DIM), "bfloat16")
        W = T.match_buffer(weight_handle, (columns, reduction), "bfloat16")
        OUT = T.match_buffer(output_handle, (rows, columns), "float32")
        with T.Kernel(columns // BN) as block:
            accum = T.alloc_l0c((rows, BN), "float32")
            a_l1 = T.alloc_l1((rows, HEAD_DIM), "bfloat16")
            w_l1 = T.alloc_l1((BN, HEAD_DIM), "bfloat16")
            n0 = T.cast(block * BN, "int32")
            for h in T.Pipelined(HEADS, num_stages=3):
                T.copy(A[h, 0:rows, 0:HEAD_DIM], a_l1)
                k0 = T.cast(h * HEAD_DIM, "int32")
                T.copy(W[n0 : n0 + BN, k0 : k0 + HEAD_DIM], w_l1)
                T.gemm(a_l1, w_l1, accum, transpose_B=True, clear_accum=h == 0,
                       unit_flag_ctrl=T.Select(h == HEADS - 1, 3, 2))
            T.copy(accum, OUT[0:rows, n0 : n0 + BN], unit_flag_ctrl=3)

    return main


M768_ROWS, M768_BM, M768_BN = 768, 256, 256


def bnsd_o_proj_m768_bf16():
    """OUT[768, 2048] BF16 = A_bnsd[16, 768, 128] @ W[2048, 2048]^T for the prefill.

    r8: the prefill causal out is born BNSD [1, H, P, D] and the packed
    o.reshape(768, -1) transpose copy plus the FP32->BF16 finish cast cost
    ~17 us/call serially over 35 layers. This blocked kernel reads the BNSD A
    directly (one [BM, D] L1 slab per head, the k-th 128-wide column group of
    the logical [M, K] A is the contiguous [M, D] block A_bnsd[k // D]) and
    rounds in the final store: bitwise vs the reshape+ACLNN+cast chain
    (probe r8: aclnn_cast flips 0, fp32-route aclnn_max_abs 0.0; the K order
    stays 16 sequential head-steps in one L0C accumulator per output element).
    bm=bn=256/stages=2 measured fastest in the r8 sweep.
    """
    columns, reduction = 2048, HEADS * HEAD_DIM
    bm, bn = M768_BM, M768_BN

    @T.prim_func
    def main(a_handle: T.handle, weight_handle: T.handle, output_handle: T.handle):
        A = T.match_buffer(a_handle, (HEADS, M768_ROWS, HEAD_DIM), "bfloat16")
        Wt = T.match_buffer(weight_handle, (columns, reduction), "bfloat16")
        OUT = T.match_buffer(output_handle, (M768_ROWS, columns), "bfloat16")
        nblocks = columns // bn
        with T.Kernel((M768_ROWS // bm) * nblocks) as block:
            accum = T.alloc_l0c((bm, bn), "float32")
            a_l1 = T.alloc_l1((bm, HEAD_DIM), "bfloat16")
            w_l1 = T.alloc_l1((bn, HEAD_DIM), "bfloat16")
            m0 = T.cast((block // nblocks) * bm, "int32")
            n0 = T.cast((block % nblocks) * bn, "int32")
            for h in T.Pipelined(HEADS, num_stages=2):
                T.copy(A[h, m0 : m0 + bm, 0:HEAD_DIM], a_l1)
                k0 = T.cast(h * HEAD_DIM, "int32")
                T.copy(Wt[n0 : n0 + bn, k0 : k0 + HEAD_DIM], w_l1)
                T.gemm(a_l1, w_l1, accum, transpose_B=True, clear_accum=h == 0,
                       unit_flag_ctrl=T.Select(h == HEADS - 1, 3, 2))
            T.copy(accum, OUT[m0 : m0 + bm, n0 : n0 + bn], unit_flag_ctrl=3)

    return main


def build_programs():
    yield "smallm_bnsd_o_proj", bnsd_o_proj(16)
    yield "smallm_bnsd_o_proj_m32", bnsd_o_proj(32)
    yield "smallm_bnsd_o_proj_m64", bnsd_o_proj(64)
    yield "bnsd_o_proj_m768_bf16", bnsd_o_proj_m768_bf16()
