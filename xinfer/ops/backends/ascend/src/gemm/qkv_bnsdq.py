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

"""Finish the Wall-X 2B QKV projection for cache attention.
Apply per-head RMS normalization and RoPE to Q/K, append K/V to the cache
and write head-major Q and a separate packed gate for the gated attention store."""

import tilelang.ascend.language as T


def qkv_2b_norm_gate_cache_bnsdq():
    @T.prim_func
    def main(
        qh: T.handle,
        gh: T.handle,
        kh: T.handle,
        vh: T.handle,
        ah: T.handle,
        qfh: T.handle,
        kfh: T.handle,
        ch: T.handle,
        sh: T.handle,
        rows: T.int64,
        capacity: T.int64,
        start: T.int64,
    ):
        query = T.match_buffer(qh, (8 * rows * 256,), "bfloat16")
        gate = T.match_buffer(gh, (rows * 2048,), "bfloat16")
        key = T.match_buffer(kh, (2 * capacity * 256,), "bfloat16")
        value = T.match_buffer(vh, (2 * capacity * 256,), "bfloat16")
        accum = T.match_buffer(ah, (rows * 5120,), "float32")
        q_factor = T.match_buffer(qfh, (256,), "float32")
        k_factor = T.match_buffer(kfh, (256,), "float32")
        cosine = T.match_buffer(ch, (rows * 32,), "float32")
        sine = T.match_buffer(sh, (rows * 32,), "float32")
        jobs = rows * 90
        blocks = T.ceildiv(jobs, 8)
        with T.Kernel(blocks) as block:
            with T.SimtVF(threads=256):
                for lane in T.Parallel(256):
                    warp = T.cast(lane // 32, "int64")
                    pos = T.cast(lane % 32, "int64")
                    job = T.cast(block, "int64") * 8 + warp
                    if job < rows * 10:
                        row = job // 10
                        head = job % 10
                        base = head * 256
                        total = T.alloc_var("float32", init=0.0)
                        for stripe in T.serial(8):
                            j = stripe * 32 + pos
                            a = accum[row * 5120 + base + j]
                            total = total + a * a
                        sumsq = T.warp_reduce_sum(total)
                        inverse = T.rsqrt(
                            sumsq / T.float32(256.0) + T.float32(1e-6)
                        )
                        for stripe in T.serial(8):
                            j = stripe * 32 + pos
                            a = accum[row * 5120 + base + j]
                            w = T.Select(
                                head < 8, q_factor[j], k_factor[j]
                            )
                            t1 = T.alloc_var("float32", init=a * inverse)
                            z = T.alloc_var("float32", init=t1 * w)
                            result = T.alloc_var("float32", init=T.cast(z, "float32"))
                            if j < 64:
                                pair = T.Select(j < 32, j, j - 32)
                                partner_j = T.Select(j < 32, j + 32, j - 32)
                                ap = accum[row * 5120 + base + partner_j]
                                wp = T.Select(
                                    head < 8, q_factor[partner_j], k_factor[partner_j]
                                )
                                t1p = T.alloc_var("float32", init=ap * inverse)
                                zp = T.alloc_var("float32", init=t1p * wp)
                                c = cosine[row * 32 + pair]
                                s = sine[row * 32 + pair]
                                zc = T.alloc_var("float32", init=z * c)
                                zps = T.alloc_var("float32", init=zp * s)
                                result = T.Select(
                                    j < 32,
                                    zc - zps,
                                    zc + zps,
                                )
                            if head < 8:
                                query[head * (rows * 256) + row * 256 + j] = T.cast(
                                    result, "bfloat16"
                                )
                            else:
                                key[
                                    (head - 8) * capacity * 256
                                    + (start + row) * 256
                                    + j
                                ] = T.cast(result, "bfloat16")
                    elif job < rows * 26:
                        flat = (job - rows * 10) * 32 + pos
                        row = flat // 512
                        c = flat % 512
                        value[
                            (c // 256) * capacity * 256
                            + (start + row) * 256
                            + c % 256
                        ] = T.cast(accum[row * 5120 + 2560 + c], "bfloat16")
                    elif job < jobs:
                        flat = (job - rows * 26) * 32 + pos
                        row = flat // 2048
                        c = flat % 2048
                        gate[row * 2048 + c] = T.cast(
                            accum[row * 5120 + 3072 + c], "bfloat16"
                        )

    return main


def build_programs():
    yield "gemm_qkv_2b_norm_gate_cache_bnsdq", qkv_2b_norm_gate_cache_bnsdq()
