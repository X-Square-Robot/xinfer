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

"""RMSFUSE-R4 kernel variants.

Baselines come from the vendored ``rms_consistent`` module (verbatim from the
walloss lib source); variants re-parameterize only scheduling (rows_per_step /
weight staging mode) or data path (UB staging), never the arithmetic statements.
"""
import tilelang.ascend.language as T  # noqa: E402

from .rms_consistent import generic, ubv  # the vendored baseline module


def m16(width=1024):
    """Lean M16/H1024 plan: UB-staged loads, direct FP32 weight, arithmetic
    statements identical to generic(1024) (same fragment loop, same reducer
    construct under the same threads, same rsqrt expression, same output
    association). add_residual/write_output flags mirror the lib ABI; the
    weight buffer is FP32 (the engine stores these norms in FP32)."""
    tile = 1024

    @T.prim_func
    def main(
        xh: T.handle,
        rh: T.handle,
        sh: T.handle,
        wh: T.handle,
        yh: T.handle,
        ih: T.handle,
        rows: T.int64,
        epsilon_bits: T.int64,
        centered: T.int64,
        add_residual: T.int64,
        write_output: T.int64,
        write_stats: T.int64,
    ):
        x = T.match_buffer(xh, (rows * width,), "bfloat16")
        residual = T.match_buffer(rh, (rows * width,), "bfloat16")
        summed = T.match_buffer(sh, (rows * width,), "bfloat16")
        weight = T.match_buffer(wh, (width,), "float32")
        output = T.match_buffer(yh, (rows * width,), "bfloat16")
        stats = T.match_buffer(ih, (rows,), "float32")
        epsilon = T.reinterpret("float32", T.cast(epsilon_bits, "uint32"))
        blocks = T.min(rows, 64)
        with T.Kernel(blocks) as core:
            x_ub = T.alloc_shared((tile,), "bfloat16")
            res_ub = T.alloc_shared((tile,), "bfloat16")
            sum_ub = T.alloc_shared((tile,), "bfloat16")
            out_ub = T.alloc_shared((tile,), "bfloat16")
            w32_ub = T.alloc_shared((tile,), "float32")
            T.copy(weight[:width], w32_ub[:width])
            for step in T.serial(T.ceildiv(rows, blocks)):
                row = step * blocks + core
                if row < rows:
                    base = row * width
                    T.copy(x[base : base + width], x_ub[:width])
                    if add_residual != 0:
                        T.copy(residual[base : base + width], res_ub[:width])
                    with T.SimtVF(threads=256):
                        value = T.alloc_fragment((tile,), "float32")
                        for col in T.Parallel(tile, coalesced_width=T.int32(1)):
                            value[col] = T.float32(0)
                            if col < width:
                                value[col] = T.cast(x_ub[col], "float32")
                                if add_residual != 0:
                                    value[col] = T.cast(
                                        T.cast(
                                            value[col]
                                            + T.cast(res_ub[col], "float32"),
                                            "bfloat16",
                                        ),
                                        "float32",
                                    )
                                    sum_ub[col] = T.cast(value[col], "bfloat16")
                        total = T.alloc_reducer((1,), "float32", op="sum")
                        T.reducer_init(total)
                        for col in T.Parallel(tile, coalesced_width=T.int32(1)):
                            T.reducer_update(total[0], value[col] * value[col])
                        inverse = T.alloc_fragment((1,), "float32")
                        T.finalize_reducer(total, inverse)
                        for i in T.Parallel(1):
                            inverse[i] = T.rsqrt(
                                inverse[i] / T.float32(width) + epsilon
                            )
                            if write_stats != 0 and T.get_thread_binding() == 0:
                                stats[row] = inverse[i]
                        if write_output != 0:
                            for col in T.Parallel(tile, coalesced_width=T.int32(1)):
                                if col < width:
                                    output_col = value[col] * w32_ub[col] * inverse[0]
                                    out_ub[col] = T.cast(output_col, "bfloat16")
                    if add_residual != 0:
                        T.copy(sum_ub[:width], summed[base : base + width])
                    if write_output != 0:
                        T.copy(out_ub[:width], output[base : base + width])

    return main


def build_programs():
    # -- baselines: verbatim lib programs (same source the lib binary built from)
    yield "rf4_g1024_base", generic(1024)
    yield "rf4_ubv1280_base", ubv(1280, weight_mode="fold16")
    yield "rf4_ubv2048_base", ubv(2048)
    # -- lean m16 plan (RF1)
    yield "rf4_m16", m16(1024)
    # -- scheduling-only retunes of the vector plans (RF2)
    yield "rf4_ubv1280_f", ubv(1280, weight_mode="f32direct")
    yield "rf4_ubv2048_f", ubv(2048, weight_mode="f32direct")
    yield "rf4_m16_f32in", m16_f32in(1024)


def m16_f32in(width=1024):
    """RF3: the FP32 accumulator feeds the kernel directly; the RNE round to
    BF16 happens inside it (value = f32(bf16(acc32))), bit-identical to the
    released chain (gemm_epilogue _finish cast followed by the m16 plan)."""
    tile = 1024

    @T.prim_func
    def main(
        xh: T.handle,
        rh: T.handle,
        sh: T.handle,
        wh: T.handle,
        yh: T.handle,
        ih: T.handle,
        rows: T.int64,
        epsilon_bits: T.int64,
        centered: T.int64,
        add_residual: T.int64,
        write_output: T.int64,
        write_stats: T.int64,
    ):
        x = T.match_buffer(xh, (rows * width,), "float32")
        residual = T.match_buffer(rh, (rows * width,), "bfloat16")
        summed = T.match_buffer(sh, (rows * width,), "bfloat16")
        weight = T.match_buffer(wh, (width,), "float32")
        output = T.match_buffer(yh, (rows * width,), "bfloat16")
        stats = T.match_buffer(ih, (rows,), "float32")
        epsilon = T.reinterpret("float32", T.cast(epsilon_bits, "uint32"))
        blocks = T.min(rows, 64)
        with T.Kernel(blocks) as core:
            x_ub = T.alloc_shared((tile,), "float32")
            res_ub = T.alloc_shared((tile,), "bfloat16")
            sum_ub = T.alloc_shared((tile,), "bfloat16")
            out_ub = T.alloc_shared((tile,), "bfloat16")
            w32_ub = T.alloc_shared((tile,), "float32")
            T.copy(weight[:width], w32_ub[:width])
            for step in T.serial(T.ceildiv(rows, blocks)):
                row = step * blocks + core
                if row < rows:
                    base = row * width
                    T.copy(x[base : base + width], x_ub[:width])
                    if add_residual != 0:
                        T.copy(residual[base : base + width], res_ub[:width])
                    with T.SimtVF(threads=256):
                        value = T.alloc_fragment((tile,), "float32")
                        for col in T.Parallel(tile, coalesced_width=T.int32(1)):
                            value[col] = T.float32(0)
                            if col < width:
                                value[col] = T.cast(
                                    T.cast(x_ub[col], "bfloat16"), "float32"
                                )
                                if add_residual != 0:
                                    value[col] = T.cast(
                                        T.cast(
                                            value[col]
                                            + T.cast(res_ub[col], "float32"),
                                            "bfloat16",
                                        ),
                                        "float32",
                                    )
                                    sum_ub[col] = T.cast(value[col], "bfloat16")
                        total = T.alloc_reducer((1,), "float32", op="sum")
                        T.reducer_init(total)
                        for col in T.Parallel(tile, coalesced_width=T.int32(1)):
                            T.reducer_update(total[0], value[col] * value[col])
                        inverse = T.alloc_fragment((1,), "float32")
                        T.finalize_reducer(total, inverse)
                        for i in T.Parallel(1):
                            inverse[i] = T.rsqrt(
                                inverse[i] / T.float32(width) + epsilon
                            )
                            if write_stats != 0 and T.get_thread_binding() == 0:
                                stats[row] = inverse[i]
                        if write_output != 0:
                            for col in T.Parallel(tile, coalesced_width=T.int32(1)):
                                if col < width:
                                    output_col = value[col] * w32_ub[col] * inverse[0]
                                    out_ub[col] = T.cast(output_col, "bfloat16")
                    if add_residual != 0:
                        T.copy(sum_ub[:width], summed[base : base + width])
                    if write_output != 0:
                        T.copy(out_ub[:width], output[base : base + width])

    return main


