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

"""ubfinish-r5: UB-vectorized bias finish (T.copy DMA recipe).

Revives the libcopy-r3 knife-2 arithmetic (out = rne(acc_f32 + bias_f32),
bitwise with the torch tail ``y + bias.float(); y.to(bf16)`` -- proven
bitwise in r3 across 7 shapes x 3 trials) with the memory pattern of the
production finish family:

* v1 ``bias_finish_ub``: the finish_bf16_ub4096 mold (fixed CORES grid,
  flat 1-D one-row tiles, Pipelined stages=2, int32 bases, scalar SimtVF
  math with the integer-RNE store). Measured 0.9-1.2 TB/s eager on card 7
  (probe-r5.json): bitwise but the 5-8 KB row copies leave MTE efficiency
  and scalar compute on the table.
* v2 ``bias_finish_rows``: the gemm_glu rowpass mold -- one large
  multi-row contiguous span per core-step (rows x cols f32 is contiguous
  row-major), the same scalar integer-RNE math, stages=2.
* v3 ``bias_finish_simd``: the same rowpass mold with the production SIMD
  idioms (vld / UNPK_B16 bias widen / vadd / vcvt ROUND_R + PK_B32 store,
  the gemm_glu_fp32_finish.split_finish recipe). vcvt ROUND_R is documented
  ties-to-even; bitwise vs the torch tail is decided by the witness probe
  (random sweeps + crafted ties/subnormals/overflow), not assumed.

Rows stays a runtime int64; the python wrapper routes only complete
row-tile multiples and int32-eligible counts, everything else falls back
to the torch tail.
"""

import tilelang.ascend.language as T

from gemm.silu_and_mul import _store_rne

CORES = 64
THREADS = 256
LANES = 64  # FP32 lanes per CCE vector register

# Gate-stack shape probe (2026-11-13 19:46): bias-only gemm_epilogue
# population is [2112,1280] (dominant) + [528,2048]. Column widths {1280, 2048}.
SHAPES = [1280, 2048]
ROWS_PER_TILE = {1280: 8, 2048: 4}


def bias_finish_ub(cols, cores=CORES, threads=THREADS):
    @T.prim_func
    def main(oh: T.handle, ah: T.handle, bh: T.handle, rows: T.int64):
        out = T.match_buffer(oh, (rows * cols,), "bfloat16")
        acc = T.match_buffer(ah, (rows * cols,), "float32")
        bias = T.match_buffer(bh, (cols,), "bfloat16")
        with T.Kernel(cores) as core:
            acc_ub = T.alloc_shared((cols,), "float32")
            out_ub = T.alloc_shared((cols,), "bfloat16")
            bias_ub = T.alloc_shared((cols,), "bfloat16")
            T.annotate_buffer_versions({acc_ub: 2, out_ub: 1})
            T.copy(bias, bias_ub)
            for step in T.Pipelined(T.ceildiv(rows, cores), num_stages=2):
                row = step * cores + core
                if row < rows:
                    base = T.cast(row * cols, "int32")
                    T.copy(acc[base : base + cols], acc_ub)
                    with T.SimtVF(threads=threads):
                        for index in T.Parallel(cols):
                            value = T.alloc_var(
                                "float32",
                                init=acc_ub[index] + T.cast(bias_ub[index], "float32"),
                            )
                            out_ub[index] = _store_rne(value, "bfloat16")
                    T.copy(out_ub, out[base : base + cols])

    return main


def bias_finish_rows(cols, rows_per_tile, cores=CORES, threads=THREADS):
    """Rowpass mold: one rows_per_tile x cols contiguous span per core-step."""

    @T.prim_func
    def main(oh: T.handle, ah: T.handle, bh: T.handle, rows: T.int64):
        out = T.match_buffer(oh, (rows * cols,), "bfloat16")
        acc = T.match_buffer(ah, (rows * cols,), "float32")
        bias = T.match_buffer(bh, (cols,), "bfloat16")
        tiles = rows // rows_per_tile
        span = rows_per_tile * cols
        with T.Kernel(cores) as core:
            acc_ub = T.alloc_shared((span,), "float32")
            out_ub = T.alloc_shared((span,), "bfloat16")
            bias_ub = T.alloc_shared((cols,), "bfloat16")
            T.annotate_buffer_versions({acc_ub: 2, out_ub: 1})
            T.copy(bias, bias_ub)
            for step in T.Pipelined(T.ceildiv(tiles, cores), num_stages=2):
                tile = step * cores + core
                if tile < tiles:
                    base = T.cast(tile * span, "int32")
                    T.copy(acc[base : base + span], acc_ub)
                    with T.SimtVF(threads=threads):
                        for r in T.serial(rows_per_tile):
                            row_off = r * cols
                            for index in T.Parallel(cols):
                                value = T.alloc_var(
                                    "float32",
                                    init=acc_ub[row_off + index]
                                    + T.cast(bias_ub[index], "float32"),
                                )
                                out_ub[row_off + index] = _store_rne(value, "bfloat16")
                    T.copy(out_ub, out[base : base + span])

    return main


def bias_finish_simd(cols, rows_per_tile, cores=CORES):
    """Rowpass mold + production SIMD math (split_finish recipe).

    Per 64-lane chunk: bf16 bias widened via UNPK_B16 + vcvt part=0 (exact),
    one fp32 vadd (the torch tail add), one vcvt ROUND_R to bfloat16 with a
    PK_B32 packed store. Bitwise adjudicated by the witness probe.
    """

    @T.prim_func
    def main(oh: T.handle, ah: T.handle, bh: T.handle, rows: T.int64):
        out = T.match_buffer(oh, (rows * cols,), "bfloat16")
        acc = T.match_buffer(ah, (rows * cols,), "float32")
        bias = T.match_buffer(bh, (cols,), "bfloat16")
        tiles = rows // rows_per_tile
        with T.Kernel(cores) as core:
            acc_ub = T.alloc_shared((rows_per_tile * cols,), "float32")
            out_ub = T.alloc_shared((rows_per_tile * cols,), "bfloat16")
            bias_ub = T.alloc_shared((cols,), "bfloat16")
            T.annotate_buffer_versions({acc_ub: 3, out_ub: 3})
            T.copy(bias, bias_ub)
            for step in T.Pipelined(T.ceildiv(tiles, cores), num_stages=3):
                tile = step * cores + core
                if tile < tiles:
                    base = T.cast(tile * rows_per_tile * cols, "int32")
                    T.copy(acc[base : base + rows_per_tile * cols], acc_ub)
                    with T.SimdVF():
                        m = T.simd.pset(32)
                        for r in T.serial(rows_per_tile):
                            row_off = r * cols
                            for c in range(cols // LANES):
                                a = T.simd.vld(acc_ub[row_off + c * LANES])
                                b = T.simd.vcvt(
                                    T.simd.vld(bias_ub[c * LANES], dist="UNPK_B16"),
                                    "float32",
                                    part=0,
                                )
                                s = T.simd.vadd(a, b, m)
                                T.simd.vsts(
                                    out_ub[row_off + c * LANES],
                                    T.simd.vcvt(s, "bfloat16", round="ROUND_R"),
                                    dist="PK_B32",
                                )
                    T.copy(
                        out_ub,
                        out[base : base + rows_per_tile * cols],
                    )

    return main


def build_programs():
    for cols in SHAPES:
        yield f"ubfinish_r5_c{cols}", bias_finish_ub(cols)
        yield (
            f"ubfinish_r5_r{ROWS_PER_TILE[cols]}_c{cols}",
            bias_finish_rows(cols, ROWS_PER_TILE[cols]),
        )
        yield (
            f"ubfinish_r5_s{ROWS_PER_TILE[cols]}_c{cols}",
            bias_finish_simd(cols, ROWS_PER_TILE[cols]),
        )
