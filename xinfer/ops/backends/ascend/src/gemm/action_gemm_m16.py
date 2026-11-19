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

"""M16 GEMMs for the Wall-OSS action output, gated MLP and QKV projections.

The kernels partition output channels across Cube blocks and accumulate
BF16 inputs in FP32. The O projection reads head-major input, SwiGLU reads
interleaved gate/up weights, and QKV exposes an FP32 accumulator to the
subsequent bias/RoPE/cache finish."""

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
    yield "action_gemm_m16_o_proj", o_proj(32)
    yield "action_gemm_m16_gateup_glu", gateup_glu(64, 128, 3)
    yield "action_gemm_m16_down", down()
    yield "action_gemm_m16_qkv", qkv()
