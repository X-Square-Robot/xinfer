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

"""Build-only bit-preserving movement for canonical QKV cache outputs."""

# Kernel and SIMT are distinct hardware scopes.
# ruff: noqa: SIM117
import tilelang.ascend.language as T

from .silu_and_mul import _store_rne

THREADS, MAX_BLOCKS = 256, 128



def rotate_fp32_append():
    """Raw NK multicast epilogue for eight Q and one K/V head, each width 256."""

    @T.prim_func
    def main(qh: T.handle, kh: T.handle, vh: T.handle, ah: T.handle,
             ch: T.handle, sh: T.handle, rows: T.int64, start: T.int64):
        query = T.match_buffer(qh, (rows * 2048,), "bfloat16")
        key = T.match_buffer(kh, ((start + rows) * 256,), "bfloat16")
        value = T.match_buffer(vh, ((start + rows) * 256,), "bfloat16")
        accum = T.match_buffer(ah, (rows * 2560,), "float32")
        cos = T.match_buffer(ch, (rows * 128,), "float32")
        sin = T.match_buffer(sh, (rows * 128,), "float32")
        # A lane handles one rotary pair or one V element, so both halves are read only once.
        count = rows * 1408
        blocks = T.min(T.ceildiv(count, THREADS), MAX_BLOCKS)
        with T.Kernel(blocks) as block:
            with T.SimtVF(threads=THREADS):
                for step in T.serial(T.ceildiv(count, blocks * THREADS)):
                    for lane in T.Parallel(THREADS):
                        index = (step * blocks + T.cast(block, "int64")) * THREADS + lane
                        if index < count:
                            row, local = index // 1408, index % 1408
                            cache = (start + row) * 256
                            if local < 1152:
                                head, pair = local // 128, local % 128
                                column = head * 256 + pair
                                a = accum[row * 2560 + column]
                                b = accum[row * 2560 + column + 128]
                                c, s = cos[row * 128 + pair], sin[row * 128 + pair]
                                low = _store_rne(a * c - b * s, "bfloat16")
                                high = _store_rne(b * c + a * s, "bfloat16")
                                if head < 8:
                                    query[row * 2048 + column] = low
                                    query[row * 2048 + column + 128] = high
                                else:
                                    key[cache + pair] = low
                                    key[cache + pair + 128] = high
                            else:
                                column = local - 1152
                                value[cache + column] = _store_rne(accum[row * 2560 + 2304 + column], "bfloat16")

    return main


def bias_rotate_fp32_append_m16():
    """Q16/KV2 D128: add BF16 bias in FP32, rotate Q/K, append head-major K/V.

    Q is written directly in BNSD [H, M, D] order (r3): the values are the
    same elements the [M, H*D] packing carried, at head-major addresses, so
    the ACLNN FIA query needs no per-call transpose copy (the strided-query
    descriptor route is rejected by ACLNN, error 561103, probe r2)."""

    @T.prim_func
    def main(qh: T.handle, kh: T.handle, vh: T.handle, ah: T.handle, bh: T.handle,
             ch: T.handle, sh: T.handle, capacity: T.int64, start: T.int64):
        query = T.match_buffer(qh, (16, 16, 128), "bfloat16")
        key = T.match_buffer(kh, (2, capacity, 128), "bfloat16")
        value = T.match_buffer(vh, (2, capacity, 128), "bfloat16")
        accum = T.match_buffer(ah, (16, 2560), "float32")
        bias = T.match_buffer(bh, (2560,), "bfloat16")
        cos = T.match_buffer(ch, (16, 64), "float32")
        sin = T.match_buffer(sh, (16, 64), "float32")
        # Per row, 18 groups handle Q/K pairs and four groups handle V.
        with T.Kernel(16 * 22 * 64 // THREADS) as block:
            with T.SimtVF(threads=THREADS):
                for lane in T.Parallel(THREADS):
                    index = T.cast(block * THREADS + lane, "int32")
                    row, group, col = index // 1408, index % 1408 // 64, index % 64
                    if group < 18:
                        first = group * 128 + col
                        a = T.alloc_var("float32", init=accum[row, first] + T.cast(bias[first], "float32"))
                        b = T.alloc_var("float32", init=accum[row, first + 64] + T.cast(bias[first + 64], "float32"))
                        c, s = cos[row, col], sin[row, col]
                        ac, bs = T.alloc_var("float32", init=a * c), T.alloc_var("float32", init=b * s)
                        bc, sa = T.alloc_var("float32", init=b * c), T.alloc_var("float32", init=a * s)
                        lo, hi = _store_rne(ac - bs, "bfloat16"), _store_rne(bc + sa, "bfloat16")
                        if group < 16:
                            query[group, row, col] = lo
                            query[group, row, col + 64] = hi
                        else:
                            key[group - 16, start + row, col] = lo
                            key[group - 16, start + row, col + 64] = hi
                    else:
                        component = (group - 18) * 64 + col
                        column = 2304 + component
                        z = accum[row, column] + T.cast(bias[column], "float32")
                        value[component // 128, start + row, component % 128] = _store_rne(z, "bfloat16")

    return main


def rotate_fp32_append_ub8():
    """M512 multicast: stage eight full rows per AIV and append the BF16 results."""
    rows, tile_rows = 512, 8

    @T.prim_func
    def main(qh: T.handle, kh: T.handle, vh: T.handle, ah: T.handle,
             ch: T.handle, sh: T.handle, start: T.int64):
        query = T.match_buffer(qh, (rows, 2048), "bfloat16")
        key = T.match_buffer(kh, (start + rows, 256), "bfloat16")
        value = T.match_buffer(vh, (start + rows, 256), "bfloat16")
        accum = T.match_buffer(ah, (rows, 2560), "float32")
        cos = T.match_buffer(ch, (rows, 128), "float32")
        sin = T.match_buffer(sh, (rows, 128), "float32")
        with T.Kernel(rows // tile_rows) as block:
            acc_ub = T.alloc_shared((tile_rows, 2560), "float32")
            cos_ub = T.alloc_shared((tile_rows, 128), "float32")
            sin_ub = T.alloc_shared((tile_rows, 128), "float32")
            q_ub = T.alloc_shared((tile_rows, 2048), "bfloat16")
            k_ub = T.alloc_shared((tile_rows, 256), "bfloat16")
            v_ub = T.alloc_shared((tile_rows, 256), "bfloat16")
            first_row = T.cast(block * tile_rows, "int32")
            cache_row = T.cast(start + first_row, "int32")
            T.copy(accum[first_row : first_row + tile_rows, :], acc_ub)
            T.copy(cos[first_row : first_row + tile_rows, :], cos_ub)
            T.copy(sin[first_row : first_row + tile_rows, :], sin_ub)
            with T.SimtVF(threads=THREADS):
                for index in T.Parallel(tile_rows * 1408):
                    row, local = index // 1408, index % 1408
                    if local < 1152:
                        head, pair = local // 128, local % 128
                        column = head * 256 + pair
                        a, b = acc_ub[row, column], acc_ub[row, column + 128]
                        c, s = cos_ub[row, pair], sin_ub[row, pair]
                        low = _store_rne(a * c - b * s, "bfloat16")
                        high = _store_rne(b * c + a * s, "bfloat16")
                        if head < 8:
                            q_ub[row, column] = low
                            q_ub[row, column + 128] = high
                        else:
                            k_ub[row, pair] = low
                            k_ub[row, pair + 128] = high
                    else:
                        column = local - 1152
                        v_ub[row, column] = _store_rne(acc_ub[row, 2304 + column], "bfloat16")
            T.copy(q_ub, query[first_row : first_row + tile_rows, :])
            T.copy(k_ub, key[cache_row : cache_row + tile_rows, :])
            T.copy(v_ub, value[cache_row : cache_row + tile_rows, :])

    return main



def build_programs():
    yield "gemm_qkv_cache_rotate_fp32", rotate_fp32_append()
    yield "gemm_qkv_cache_rotate_fp32_ub8", rotate_fp32_append_ub8()
    yield "gemm_qkv_cache_bias_rope_m16", bias_rotate_fp32_append_m16()
    yield "gemm_qkv_cache_bias_rope_prefill", bias_rotate_fp32_append_prefill()


def bias_rotate_fp32_append_prefill():
    """Q16/KV2 D128 prefill: add BF16 bias in FP32, rotate Q/K, append K/V, write Q BNSD.

    The r6 fused finish of the prefill qkv_rope_kvcache call (m 768, n 2560):
    the m16 kernel's group structure (18 rotary pair groups + 4 V groups of 64
    per row) over ub-staged 4-row tiles, one DMA burst per tile per tensor.
    Q lands BNSD [16, rows, 128] (the r7 layout: the same elements the packed
    [rows, 2048] write carried, at the head-major addresses the prefill FIA
    query consumes, so the per-call transpose copy disappears -- the m16
    kernel's r3 pattern; the head-major ub staging keeps one bulk store per
    tile); K/V append head-major [2, capacity, 128] exactly as the torch
    chain.
    """

    R = 4
    cores = 64

    @T.prim_func
    def main(qh: T.handle, kh: T.handle, vh: T.handle, ah: T.handle, bh: T.handle,
             ch: T.handle, sh: T.handle, capacity: T.int64, start: T.int64, rows: T.int64):
        query = T.match_buffer(qh, (16, rows, 128), "bfloat16")
        key = T.match_buffer(kh, (2, capacity, 128), "bfloat16")
        value = T.match_buffer(vh, (2, capacity, 128), "bfloat16")
        accum = T.match_buffer(ah, (rows, 2560), "float32")
        bias = T.match_buffer(bh, (2560,), "bfloat16")
        cos = T.match_buffer(ch, (rows, 64), "float32")
        sin = T.match_buffer(sh, (rows, 64), "float32")
        with T.Kernel(cores) as core:
            acc_ub = T.alloc_shared((R, 2560), "float32")
            cs_ub = T.alloc_shared((R, 64), "float32")
            sn_ub = T.alloc_shared((R, 64), "float32")
            bias_ub = T.alloc_shared((2560,), "bfloat16")
            q_ub = T.alloc_shared((16, R, 128), "bfloat16")
            k_ub = T.alloc_shared((2, R, 128), "bfloat16")
            v_ub = T.alloc_shared((2, R, 128), "bfloat16")
            T.annotate_buffer_versions({acc_ub: 2, cs_ub: 2, sn_ub: 2})
            T.copy(bias, bias_ub)
            for step in T.Pipelined(T.ceildiv(rows, cores * R), num_stages=2):
                r0 = T.cast((step * cores + core) * R, "int32")
                if r0 < rows:
                    T.copy(accum[r0 : r0 + R, :], acc_ub)
                    T.copy(cos[r0 : r0 + R, :], cs_ub)
                    T.copy(sin[r0 : r0 + R, :], sn_ub)
                    with T.SimtVF(threads=THREADS):
                        for r, idx in T.Parallel(R, 1408):
                            group, col = idx // 64, idx % 64
                            if group < 18:
                                first = group * 128 + col
                                a = acc_ub[r, first] + T.cast(bias_ub[first], "float32")
                                b = acc_ub[r, first + 64] + T.cast(bias_ub[first + 64], "float32")
                                c, s = cs_ub[r, col], sn_ub[r, col]
                                lo = T.cast(a * c - b * s, "bfloat16")
                                hi = T.cast(b * c + a * s, "bfloat16")
                                if group < 16:
                                    q_ub[group, r, col] = lo
                                    q_ub[group, r, col + 64] = hi
                                else:
                                    kv = group - 16
                                    k_ub[kv, r, col] = lo
                                    k_ub[kv, r, col + 64] = hi
                            else:
                                component = (group - 18) * 64 + col
                                column = 2304 + component
                                z = acc_ub[r, column] + T.cast(bias_ub[column], "float32")
                                v_ub[component // 128, r, component % 128] = T.cast(z, "bfloat16")
                    T.copy(q_ub, query[:, r0 : r0 + R, :])
                    for h in range(2):
                        T.copy(k_ub[h, 0:R, :], key[h, start + r0 : start + r0 + R, :])
                        T.copy(v_ub[h, 0:R, :], value[h, start + r0 : start + r0 + R, :])

    return main
