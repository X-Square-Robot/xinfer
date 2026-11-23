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

"""SwiGLU over the FP32 accumulator of a paired or split gate/up projection.

Use the non-flushing vector exponential and hardware division for the SiLU,
then round the product once to BF16. Paired input interleaves gate and up
values at stride two; split input stores contiguous halves. Split spans pad
the unified buffer to a 64-lane boundary without writing padded lanes to output."""

import tilelang.ascend.language as T

from .silu_and_mul import _store_rne

CORES = 64
THREADS = 256
LANES = 64  # FP32 lanes per CCE vector register


def _segment(columns):
    # Largest divisor of columns that stays within one 2048-wide tile so the
    # staged spans never cross a row boundary and keep full-copy bursts. The
    # naive ceildiv shortcut mis-sizes 11008 (1834 does not divide the row),
    # so take the real largest divisor <= 2048 with 32B-aligned FP32 spans
    # (11008 -> 1376, an exact 8-slab split of the row); when no 8-aligned
    # divisor exists, use the largest exact divisor.
    seg = min(2048, columns)
    while seg > 0 and (columns % seg or seg % 8):
        seg -= 8
    if seg <= 0 or columns % seg:
        seg = min(columns, 2048)
        while columns % seg:
            seg -= 1
    return seg


def _simd_silu(g, u, m, one):
    """g / (1 + exp(-g)) * u on vector registers, scalar-bit-identical."""
    e = T.simd.vexp(T.simd.vneg(g, m), m, precision="ftz_false")
    d = T.simd.vadd(one, e, m)
    s = T.simd.vdiv(g, d, m)
    return T.simd.vmul(s, u, m)


def paired_finish(columns):
    """Interleaved gate/up rows: one contiguous ``2*seg`` staged flat span.

    The 2-periodic interleave keeps the accumulator span of any flat output
    tile ``[base, base+seg)`` contiguous at ``[2*base, 2*base+2*seg)``, so the
    tiles never need row alignment and the vector loop needs no guard.
    """
    seg = 2048

    @T.prim_func
    def main(output_handle: T.handle, accum_handle: T.handle, rows: T.int64):
        output = T.match_buffer(output_handle, (rows * columns,), "bfloat16")
        accum = T.match_buffer(accum_handle, (2 * rows * columns,), "float32")
        tiles = rows * columns // seg
        with T.Kernel(CORES) as core:
            staged = T.alloc_shared((2 * seg,), "float32")
            result = T.alloc_shared((seg,), "bfloat16")
            T.annotate_buffer_versions({staged: 3, result: 3})
            for step in T.Pipelined(T.ceildiv(tiles, CORES), num_stages=3):
                tile = step * CORES + T.cast(core, "int64")
                if tile < tiles:
                    base = T.cast(tile * seg, "int32")
                    T.copy(accum[2 * base : 2 * (base + seg)], staged)
                    with T.SimdVF():
                        m = T.simd.pset(32)
                        one = T.simd.vdup(1.0, "float32", m)
                        for c in range(seg // LANES):
                            g, u = T.simd.vld2(staged[c * 2 * LANES], "DINTLV_B32")
                            v = _simd_silu(g, u, m, one)
                            T.simd.vsts(result[c * LANES],
                                        T.simd.vcvt(v, "bfloat16", round="ROUND_R"),
                                        dist="PK_B32")
                    T.copy(result, output[base : base + seg])

    return main


def split_finish(columns):
    """Split gate/up halves with the optional BF16 bias before the activation."""
    seg = _segment(columns)
    per_row = columns // seg
    pad = (seg + LANES - 1) // LANES * LANES

    @T.prim_func
    def main(
        output_handle: T.handle,
        accum_handle: T.handle,
        bias_handle: T.handle,
        rows: T.int64,
        has_bias: T.int64,
    ):
        output = T.match_buffer(output_handle, (rows * columns,), "bfloat16")
        accum = T.match_buffer(accum_handle, (2 * rows * columns,), "float32")
        bias = T.match_buffer(bias_handle, (2 * columns,), "bfloat16")
        tiles = rows * per_row
        with T.Kernel(CORES) as core:
            gate_buf = T.alloc_shared((pad,), "float32")
            up_buf = T.alloc_shared((pad,), "float32")
            bias_gate = T.alloc_shared((pad,), "bfloat16")
            bias_up = T.alloc_shared((pad,), "bfloat16")
            result = T.alloc_shared((pad,), "bfloat16")
            T.annotate_buffer_versions(
                {gate_buf: 3, up_buf: 3, bias_gate: 3, bias_up: 3, result: 3}
            )
            for step in T.Pipelined(T.ceildiv(tiles, CORES), num_stages=3):
                tile = step * CORES + T.cast(core, "int64")
                if tile < tiles:
                    row = tile // per_row
                    c0 = (tile % per_row) * seg
                    arow = T.cast(row * 2 * columns, "int32")
                    c0f = T.cast(c0, "int32")
                    base = T.cast(row * columns + c0, "int32")
                    T.copy(accum[arow + c0f : arow + c0f + seg], gate_buf[:seg])
                    T.copy(
                        accum[arow + columns + c0f : arow + columns + c0f + seg],
                        up_buf[:seg],
                    )
                    if has_bias != 0:
                        T.copy(bias[c0f : c0f + seg], bias_gate[:seg])
                        T.copy(
                            bias[columns + c0f : columns + c0f + seg], bias_up[:seg]
                        )
                        with T.SimdVF():
                            m = T.simd.pset(32)
                            one = T.simd.vdup(1.0, "float32", m)
                            for c in range(pad // LANES):
                                g = T.simd.vld(gate_buf[c * LANES])
                                u = T.simd.vld(up_buf[c * LANES])
                                bg = T.simd.vcvt(
                                    T.simd.vld(bias_gate[c * LANES], dist="UNPK_B16"),
                                    "float32", part=0)
                                bu = T.simd.vcvt(
                                    T.simd.vld(bias_up[c * LANES], dist="UNPK_B16"),
                                    "float32", part=0)
                                g = T.simd.vadd(g, bg, m)
                                u = T.simd.vadd(u, bu, m)
                                v = _simd_silu(g, u, m, one)
                                T.simd.vsts(result[c * LANES],
                                            T.simd.vcvt(v, "bfloat16", round="ROUND_R"),
                                            dist="PK_B32")
                    else:
                        with T.SimdVF():
                            m = T.simd.pset(32)
                            one = T.simd.vdup(1.0, "float32", m)
                            for c in range(pad // LANES):
                                g = T.simd.vld(gate_buf[c * LANES])
                                u = T.simd.vld(up_buf[c * LANES])
                                v = _simd_silu(g, u, m, one)
                                T.simd.vsts(result[c * LANES],
                                            T.simd.vcvt(v, "bfloat16", round="ROUND_R"),
                                            dist="PK_B32")
                    T.copy(result[:seg], output[base : base + seg])

    return main


def rowpass_finish(columns):
    """Compute SwiGLU from whole-row tiles with optional BF16 bias and an integer round-to-nearest-even BF16 store."""
    # One staged pass stays <= ~96KB of UB: 2*tile FP32 loads + tile BF16 store.
    tile = columns
    while columns % tile or tile * 10 > 98304:
        tile //= 2
    per_row = columns // tile

    @T.prim_func
    def main(
        output_handle: T.handle,
        accum_handle: T.handle,
        bias_handle: T.handle,
        rows: T.int64,
        has_bias: T.int64,
    ):
        output = T.match_buffer(output_handle, (rows * columns,), "bfloat16")
        accum = T.match_buffer(accum_handle, (2 * rows * columns,), "float32")
        bias = T.match_buffer(bias_handle, (2 * columns,), "bfloat16")
        tiles = rows * per_row
        with T.Kernel(CORES) as core:
            gate = T.alloc_shared((tile,), "float32")
            up = T.alloc_shared((tile,), "float32")
            bias_gate = T.alloc_shared((tile,), "bfloat16")
            bias_up = T.alloc_shared((tile,), "bfloat16")
            result = T.alloc_shared((tile,), "bfloat16")
            T.annotate_buffer_versions(
                {gate: 2, up: 2, bias_gate: 2, bias_up: 2, result: 2}
            )
            for step in T.Pipelined(T.ceildiv(tiles, CORES), num_stages=2):
                tile_idx = step * CORES + T.cast(core, "int64")
                if tile_idx < tiles:
                    row = tile_idx // per_row
                    c0 = (tile_idx % per_row) * tile
                    arow = T.cast(row * 2 * columns, "int32")
                    c0f = T.cast(c0, "int32")
                    base = T.cast(row * columns + c0, "int32")
                    T.copy(accum[arow + c0f : arow + c0f + tile], gate)
                    T.copy(
                        accum[arow + columns + c0f : arow + columns + c0f + tile],
                        up,
                    )
                    if has_bias != 0:
                        T.copy(bias[c0f : c0f + tile], bias_gate)
                        T.copy(bias[columns + c0f : columns + c0f + tile], bias_up)
                        with T.SimtVF(threads=THREADS):
                            for col in T.Parallel(tile):
                                g = gate[col] + T.cast(bias_gate[col], "float32")
                                u = up[col] + T.cast(bias_up[col], "float32")
                                value = T.alloc_var(
                                    "float32", init=(g / (1.0 + T.exp(-g))) * u
                                )
                                result[col] = _store_rne(value, "bfloat16")
                    else:
                        with T.SimtVF(threads=THREADS):
                            for col in T.Parallel(tile):
                                g = gate[col]
                                u = up[col]
                                value = T.alloc_var(
                                    "float32", init=(g / (1.0 + T.exp(-g))) * u
                                )
                                result[col] = _store_rne(value, "bfloat16")
                    T.copy(result, output[base : base + tile])

    return main


def build_programs():
    for columns in (2048, 3424, 3456, 6144, 11008, 9216):
        yield f"gemm_glu_fp32_finish_split_{columns}", split_finish(columns)
    for columns in (2048, 6144):
        yield f"gemm_glu_fp32_finish_paired_{columns}", paired_finish(columns)
    # Vision uses whole-row tiles at 3456 columns; prefill uses half-row tiles at 11008.
    for columns in (3456, 11008):
        yield f"gemm_glu_fp32_rowpass_finish_{columns}", rowpass_finish(columns)
