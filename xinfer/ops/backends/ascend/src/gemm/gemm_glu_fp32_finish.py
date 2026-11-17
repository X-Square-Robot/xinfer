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

"""Vector FP32 SwiGLU finish for the deferred Ascend GLU projection.

The ACLNN Mm projection of ``GemmGluFp32`` leaves a flat row-major FP32
``[rows, 2*columns]`` accumulator (paired interleaved or split gate/up rows).
The finish stages one span per tile through shared memory with pipelined
copies and applies the activation on the CCE vector units:

* ``vexp_1ulp_ftz_false`` + the raw hardware ``vdiv``: an adversarial witness
  probe (278,770 values: the 1+2^-8 BF16 tie, an exhaustive near-tie sweep
  within 4 FP32 ulps of BF16 rounding midpoints, the FP32 subnormal region,
  saturation/specials, and random FP32 sweeps) showed this pair is
  bit-identical to the scalar production path ``g / (1.0f + expf(-g))``.
  The *software* ``vdiv_0ulp_ftz_true`` wrapper is the one that differs by
  one FP32 ulp from the C division -- it must NOT be used here.
* ``vcvt ROUND_R`` + a ``PK_B32`` packed store replaces the integer RNE
  helper (verified exact at the tie witness; 2-3 boundary flips per 3.5M
  random elements, within the FP64 oracle bound).
* Split spans whose width is not a multiple of 64 lanes (3424 -> 1712) pad
  the shared buffers to the next 64-lane multiple; padded lanes never leave
  shared memory.

Buffers are matched flat (the multicast GeGLU finish precedent): the paired
gate/up interleave is 2-periodic, so the accumulator span of an output tile
``[base, base+seg)`` is the contiguous flat range ``[2*base, 2*base+2*seg)``
loaded with ``vld2 DINTLV_B32``.
"""

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
    # divisor exists fall back to the plain largest true divisor (union of the
    # absorb2 divisor guarantee and the glu-r1 alignment requirement).
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
    """Rope-mold whole-row FP32 SwiGLU finish (glu-r1).

    The staged column kernels above top out at the ~2.2TB/s staged-copy
    ceiling (35.2 us measured on the vision [2112, 6912] accumulator), while
    the shipped vision-rope finish (gemm_qkv_vision_rope_finish, one whole
    GM->UB row tile per core-step, scalar SimtVF math) sustains ~4.6TB/s
    effective on the same kind of borrowed FP32 accumulator. This is that
    mold for the engine GLU widths: per-core row tiles with contiguous
    multi-KB DMA bursts, the witness-exact scalar ``g / (1.0f + expf(-g)) * u``
    SimtVF expression (the production row_finish math) and the integer-RNE
    BF16 store, with the optional BF16 bias folded into the same pass.
    """
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
    # Union: 11008 (wall_oss N22016 prefill GLU half-width, staged seg 1376)
    # and 9216 (2610 N18432 prefill) join the split set; the kernel template
    # is N-generic (absorb2 9216 + glu-r1 11008 receipts).
    for columns in (2048, 3424, 3456, 6144, 11008, 9216):
        yield f"gemm_glu_fp32_finish_split_{columns}", split_finish(columns)
    for columns in (2048, 6144):
        yield f"gemm_glu_fp32_finish_paired_{columns}", paired_finish(columns)
    # glu-r1: the rope-mold row finishes for the two engine widths -- vision
    # 3456 (whole-row tiles) and prefill 11008 (half-row tiles).
    for columns in (3456, 11008):
        yield f"gemm_glu_fp32_rowpass_finish_{columns}", rowpass_finish(columns)
