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

"""Build-only generic SiLU kernels; no runtime import or compilation.

Six fixed objects cover three dtypes and forward/backward. Native bindings map
contiguous, split and in-place views onto this same strided logical-row ABI.
All arithmetic and dtype rounding execute on the Ascend vector core.
"""

import tilelang.ascend.language as T


def _row_offset(row, dimensions, strides):
    offset = T.int64(0)
    for axis in range(6, -1, -1):
        coordinate = row % dimensions[axis]
        stride = strides[axis]
        # Native code passes the lowest addressed storage position. A
        # negative stride starts this logical axis at its highest position.
        coordinate = T.Select(stride < 0, dimensions[axis] - 1 - coordinate, coordinate)
        magnitude = T.Select(stride < 0, -stride, stride)
        offset = offset + coordinate * magnitude
        row = row // dimensions[axis]
    return offset


def _store_rne(value, dtype):
    if dtype == "float32":
        return value
    bits = T.reinterpret(value, "uint32")
    if dtype == "bfloat16":
        rounded = (bits + T.uint32(0x7FFF) + ((bits >> 16) & T.uint32(1))) & T.uint32(
            0xFFFF0000
        )
        special = (bits & T.uint32(0x7F800000)) == T.uint32(0x7F800000)
        return T.cast(T.reinterpret(T.Select(special, bits, rounded), "float32"), dtype)
    # IEEE binary32 -> binary16 RNE, including half subnormals and overflow.
    # Clamp shift expressions even in branches not selected by T.Select.
    sign = (bits >> 16) & T.uint32(0x8000)
    exponent = T.cast((bits >> 23) & T.uint32(255), "int32")
    fraction = bits & T.uint32(0x7FFFFF)
    normal = (T.cast(T.max(exponent - 112, 0), "uint32") << 10) + (
        (fraction + T.uint32(0xFFF) + ((fraction >> 13) & T.uint32(1))) >> 13
    )
    shift = T.cast(T.max(T.min(126 - exponent, 24), 14), "uint32")
    significand = fraction | T.uint32(0x800000)
    subnormal = (
        significand
        + ((T.uint32(1) << (shift - 1)) - T.uint32(1))
        + ((significand >> shift) & T.uint32(1))
    ) >> shift
    finite = T.Select(
        exponent < 102,
        T.uint32(0),
        T.Select(
            exponent < 113,
            subnormal,
            T.Select(exponent > 142, T.uint32(0x7C00), normal),
        ),
    )
    special = T.Select(
        fraction == 0, T.uint32(0x7C00), T.uint32(0x7E00) | (fraction >> 13)
    )
    encoded = sign | T.Select(exponent == 255, special, finite)
    return T.reinterpret(T.cast(encoded, "uint16"), "float16")


def _one_axis_offset(row, column, rows, stride):
    """Native pointers address the minimum element, including signed strides."""
    coordinate = T.Select(stride < 0, rows - 1 - row, row)
    magnitude = T.Select(stride < 0, -stride, stride)
    return coordinate * magnitude + column


def forward(dtype):
    @T.prim_func
    def main(
        g_handle: T.handle,
        u_handle: T.handle,
        o_handle: T.handle,
        g_span: T.int64,
        u_span: T.int64,
        o_span: T.int64,
        count: T.int64,
        columns: T.int64,
        d0: T.int64,
        d1: T.int64,
        d2: T.int64,
        d3: T.int64,
        d4: T.int64,
        d5: T.int64,
        d6: T.int64,
        gs0: T.int64,
        gs1: T.int64,
        gs2: T.int64,
        gs3: T.int64,
        gs4: T.int64,
        gs5: T.int64,
        gs6: T.int64,
        us0: T.int64,
        us1: T.int64,
        us2: T.int64,
        us3: T.int64,
        us4: T.int64,
        us5: T.int64,
        us6: T.int64,
        os0: T.int64,
        os1: T.int64,
        os2: T.int64,
        os3: T.int64,
        os4: T.int64,
        os5: T.int64,
        os6: T.int64,
    ):
        g = T.match_buffer(g_handle, (g_span,), dtype)
        u = T.match_buffer(u_handle, (u_span,), dtype)
        o = T.match_buffer(o_handle, (o_span,), dtype)
        blocks = T.max(T.int64(1), T.min(T.ceildiv(count, 256), T.int64(64)))
        with T.Kernel(
            blocks
        ) as block:  # noqa: SIM117 - the TileLang parser needs nested contexts
            with T.SimtVF(threads=256):
                if count == columns:
                    for chunk_flat in T.serial(T.ceildiv(count, blocks * 256)):
                        for lane_flat in T.Parallel(256):
                            index_flat = (
                                chunk_flat * blocks + T.cast(block, "int64")
                            ) * 256 + lane_flat
                            if index_flat < count:
                                gate_flat = T.cast(g[index_flat], "float32")
                                up_flat = T.cast(u[index_flat], "float32")
                                result_flat = (
                                    gate_flat / (1.0 + T.exp(-gate_flat))
                                ) * up_flat
                                o[index_flat] = _store_rne(result_flat, dtype)
                else:
                    row_count = count // columns
                    if (
                        T.cast(d0 == row_count, "int32")
                        + T.cast(d1 == row_count, "int32")
                        + T.cast(d2 == row_count, "int32")
                        + T.cast(d3 == row_count, "int32")
                        + T.cast(d4 == row_count, "int32")
                        + T.cast(d5 == row_count, "int32")
                        + T.cast(d6 == row_count, "int32")
                    ) > 0:
                        gs_active = gs0 + gs1 + gs2 + gs3 + gs4 + gs5 + gs6
                        us_active = us0 + us1 + us2 + us3 + us4 + us5 + us6
                        os_active = os0 + os1 + os2 + os3 + os4 + os5 + os6
                        for chunk_simple in T.serial(T.ceildiv(count, blocks * 256)):
                            for lane_simple in T.Parallel(256):
                                index_simple = (
                                    chunk_simple * blocks + T.cast(block, "int64")
                                ) * 256 + lane_simple
                                if index_simple < count:
                                    row_simple = index_simple // columns
                                    col_simple = index_simple % columns
                                    gi_simple = _one_axis_offset(
                                        row_simple, col_simple, row_count, gs_active
                                    )
                                    ui_simple = _one_axis_offset(
                                        row_simple, col_simple, row_count, us_active
                                    )
                                    oi_simple = _one_axis_offset(
                                        row_simple, col_simple, row_count, os_active
                                    )
                                    gate_simple = T.cast(g[gi_simple], "float32")
                                    up_simple = T.cast(u[ui_simple], "float32")
                                    result_simple = (
                                        gate_simple / (1.0 + T.exp(-gate_simple))
                                    ) * up_simple
                                    o[oi_simple] = _store_rne(result_simple, dtype)
                    else:
                        for chunk in T.serial(T.ceildiv(count, blocks * 256)):
                            for lane in T.Parallel(256):
                                index = (
                                    chunk * blocks + T.cast(block, "int64")
                                ) * 256 + lane
                                if index < count:
                                    row = index // columns
                                    col = index % columns
                                    gi = (
                                        _row_offset(
                                            row,
                                            (d0, d1, d2, d3, d4, d5, d6),
                                            (gs0, gs1, gs2, gs3, gs4, gs5, gs6),
                                        )
                                        + col
                                    )
                                    ui = (
                                        _row_offset(
                                            row,
                                            (d0, d1, d2, d3, d4, d5, d6),
                                            (us0, us1, us2, us3, us4, us5, us6),
                                        )
                                        + col
                                    )
                                    oi = (
                                        _row_offset(
                                            row,
                                            (d0, d1, d2, d3, d4, d5, d6),
                                            (os0, os1, os2, os3, os4, os5, os6),
                                        )
                                        + col
                                    )
                                    gate = T.cast(g[gi], "float32")
                                    up = T.cast(u[ui], "float32")
                                    result = (gate / (1.0 + T.exp(-gate))) * up
                                    o[oi] = _store_rne(result, dtype)

    return main


def backward(dtype):
    @T.prim_func
    def main(
        g_handle: T.handle,
        u_handle: T.handle,
        dy_handle: T.handle,
        dg_handle: T.handle,
        du_handle: T.handle,
        g_span: T.int64,
        u_span: T.int64,
        dy_span: T.int64,
        dg_span: T.int64,
        du_span: T.int64,
        count: T.int64,
        columns: T.int64,
        d0: T.int64,
        d1: T.int64,
        d2: T.int64,
        d3: T.int64,
        d4: T.int64,
        d5: T.int64,
        d6: T.int64,
        gs0: T.int64,
        gs1: T.int64,
        gs2: T.int64,
        gs3: T.int64,
        gs4: T.int64,
        gs5: T.int64,
        gs6: T.int64,
        us0: T.int64,
        us1: T.int64,
        us2: T.int64,
        us3: T.int64,
        us4: T.int64,
        us5: T.int64,
        us6: T.int64,
        dys0: T.int64,
        dys1: T.int64,
        dys2: T.int64,
        dys3: T.int64,
        dys4: T.int64,
        dys5: T.int64,
        dys6: T.int64,
        dgs0: T.int64,
        dgs1: T.int64,
        dgs2: T.int64,
        dgs3: T.int64,
        dgs4: T.int64,
        dgs5: T.int64,
        dgs6: T.int64,
        dus0: T.int64,
        dus1: T.int64,
        dus2: T.int64,
        dus3: T.int64,
        dus4: T.int64,
        dus5: T.int64,
        dus6: T.int64,
    ):
        g = T.match_buffer(g_handle, (g_span,), dtype)
        u = T.match_buffer(u_handle, (u_span,), dtype)
        dy = T.match_buffer(dy_handle, (dy_span,), dtype)
        dg = T.match_buffer(dg_handle, (dg_span,), dtype)
        du = T.match_buffer(du_handle, (du_span,), dtype)
        blocks = T.max(T.int64(1), T.min(T.ceildiv(count, 256), T.int64(64)))
        with T.Kernel(
            blocks
        ) as block:  # noqa: SIM117 - the TileLang parser needs nested contexts
            with T.SimtVF(threads=256):
                if count == columns:
                    for chunk_flat in T.serial(T.ceildiv(count, blocks * 256)):
                        for lane_flat in T.Parallel(256):
                            index_flat = (
                                chunk_flat * blocks + T.cast(block, "int64")
                            ) * 256 + lane_flat
                            if index_flat < count:
                                gate_flat = T.cast(g[index_flat], "float32")
                                up_flat = T.cast(u[index_flat], "float32")
                                grad_flat = T.cast(dy[index_flat], "float32")
                                sigmoid_flat = 1.0 / (1.0 + T.exp(-gate_flat))
                                silu_flat = gate_flat * sigmoid_flat
                                derivative_flat = sigmoid_flat * (
                                    1.0 + gate_flat * (1.0 - sigmoid_flat)
                                )
                                dg[index_flat] = _store_rne(
                                    (grad_flat * up_flat) * derivative_flat, dtype
                                )
                                du[index_flat] = _store_rne(
                                    grad_flat * silu_flat, dtype
                                )
                else:
                    row_count = count // columns
                    if (
                        T.cast(d0 == row_count, "int32")
                        + T.cast(d1 == row_count, "int32")
                        + T.cast(d2 == row_count, "int32")
                        + T.cast(d3 == row_count, "int32")
                        + T.cast(d4 == row_count, "int32")
                        + T.cast(d5 == row_count, "int32")
                        + T.cast(d6 == row_count, "int32")
                    ) > 0:
                        gs_active = gs0 + gs1 + gs2 + gs3 + gs4 + gs5 + gs6
                        us_active = us0 + us1 + us2 + us3 + us4 + us5 + us6
                        dys_active = dys0 + dys1 + dys2 + dys3 + dys4 + dys5 + dys6
                        dgs_active = dgs0 + dgs1 + dgs2 + dgs3 + dgs4 + dgs5 + dgs6
                        dus_active = dus0 + dus1 + dus2 + dus3 + dus4 + dus5 + dus6
                        for chunk_simple in T.serial(T.ceildiv(count, blocks * 256)):
                            for lane_simple in T.Parallel(256):
                                index_simple = (
                                    chunk_simple * blocks + T.cast(block, "int64")
                                ) * 256 + lane_simple
                                if index_simple < count:
                                    row_simple = index_simple // columns
                                    col_simple = index_simple % columns
                                    gi_simple = _one_axis_offset(
                                        row_simple, col_simple, row_count, gs_active
                                    )
                                    ui_simple = _one_axis_offset(
                                        row_simple, col_simple, row_count, us_active
                                    )
                                    dyi_simple = _one_axis_offset(
                                        row_simple, col_simple, row_count, dys_active
                                    )
                                    dgi_simple = _one_axis_offset(
                                        row_simple, col_simple, row_count, dgs_active
                                    )
                                    dui_simple = _one_axis_offset(
                                        row_simple, col_simple, row_count, dus_active
                                    )
                                    gate_simple = T.cast(g[gi_simple], "float32")
                                    up_simple = T.cast(u[ui_simple], "float32")
                                    grad_simple = T.cast(dy[dyi_simple], "float32")
                                    sigmoid_simple = 1.0 / (1.0 + T.exp(-gate_simple))
                                    silu_simple = gate_simple * sigmoid_simple
                                    derivative_simple = sigmoid_simple * (
                                        1.0 + gate_simple * (1.0 - sigmoid_simple)
                                    )
                                    dg[dgi_simple] = _store_rne(
                                        (grad_simple * up_simple) * derivative_simple,
                                        dtype,
                                    )
                                    du[dui_simple] = _store_rne(
                                        grad_simple * silu_simple, dtype
                                    )
                    else:
                        for chunk in T.serial(T.ceildiv(count, blocks * 256)):
                            for lane in T.Parallel(256):
                                index = (
                                    chunk * blocks + T.cast(block, "int64")
                                ) * 256 + lane
                                if index < count:
                                    row = index // columns
                                    col = index % columns
                                    gi = (
                                        _row_offset(
                                            row,
                                            (d0, d1, d2, d3, d4, d5, d6),
                                            (gs0, gs1, gs2, gs3, gs4, gs5, gs6),
                                        )
                                        + col
                                    )
                                    ui = (
                                        _row_offset(
                                            row,
                                            (d0, d1, d2, d3, d4, d5, d6),
                                            (us0, us1, us2, us3, us4, us5, us6),
                                        )
                                        + col
                                    )
                                    dyi = (
                                        _row_offset(
                                            row,
                                            (d0, d1, d2, d3, d4, d5, d6),
                                            (dys0, dys1, dys2, dys3, dys4, dys5, dys6),
                                        )
                                        + col
                                    )
                                    dgi = (
                                        _row_offset(
                                            row,
                                            (d0, d1, d2, d3, d4, d5, d6),
                                            (dgs0, dgs1, dgs2, dgs3, dgs4, dgs5, dgs6),
                                        )
                                        + col
                                    )
                                    dui = (
                                        _row_offset(
                                            row,
                                            (d0, d1, d2, d3, d4, d5, d6),
                                            (dus0, dus1, dus2, dus3, dus4, dus5, dus6),
                                        )
                                        + col
                                    )
                                    gate = T.cast(g[gi], "float32")
                                    up = T.cast(u[ui], "float32")
                                    grad = T.cast(dy[dyi], "float32")
                                    sigmoid = 1.0 / (1.0 + T.exp(-gate))
                                    silu = gate * sigmoid
                                    derivative = sigmoid * (
                                        1.0 + gate * (1.0 - sigmoid)
                                    )
                                    dg[dgi] = _store_rne(
                                        (grad * up) * derivative, dtype
                                    )
                                    du[dui] = _store_rne(grad * silu, dtype)

    return main


def build_programs():
    for suffix, dtype in (("f16", "float16"), ("bf16", "bfloat16"), ("f32", "float32")):
        yield "silu_fwd_" + suffix, forward(dtype)
        yield "silu_bwd_" + suffix, backward(dtype)
