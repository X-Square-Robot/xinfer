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

"""Dynamic fused output head and Euler step, with source rounding boundaries."""

# Separate TileLang hardware scopes must remain nested.
# ruff: noqa: SIM117
import tilelang.ascend.language as T

COMPILE_OPTIONS = ["-ffp-contract=off"]


def _finite(value):
    return (T.reinterpret("uint32", value) & 0x7FFFFFFF) < 0x7F800000


# Bisheng may contract SIMT mul+add despite -ffp-contract=off. Keep each
# source RN product observable to the device compiler.
_ROUNDING_PRELUDE = r"""
#define XCOMPUTE_DENOISE_MUL_RN(lhs, rhs) ({ volatile float product = (lhs) * (rhs); product; })
"""


@T.macro
def _mul_rn(lhs, rhs):
    return T.call_extern("float32", "XCOMPUTE_DENOISE_MUL_RN", lhs, rhs)


@T.macro
def _serial_projection(hidden, weight, bias, norm, shift, hbase, action, flags, prenorm, eps, _column_lane):
    dot = T.alloc_var("float32", init=0)
    dot_comp = T.alloc_var("float32", init=0)
    square = T.alloc_var("float32", init=0)
    square_comp = T.alloc_var("float32", init=0)
    shifted = T.alloc_var("float32", init=0)
    shifted_comp = T.alloc_var("float32", init=0)
    for k in T.serial(1024):
        hv = T.cast(hidden[hbase + k], "float32")
        wv = T.cast(weight[action, k], "float32")
        factor = T.alloc_var("float32", init=1)
        if prenorm != 0:
            factor = T.cast(norm[k], "float32")
            if prenorm == 2:
                factor = T.float32(1) + factor
            term = T.alloc_var("float32", init=hv * hv)
            if _finite(term) and _finite(square):
                y = term - square_comp
                total = square + y
                if _finite(total):
                    square_comp = (total - square) - y
                else:
                    square_comp = T.float32(0)
                square = total
            else:
                square = square + term
                square_comp = T.float32(0)
        if prenorm == 2:
            sv = T.cast(shift[k], "float32")
            term = T.alloc_var("float32", init=sv * wv)
            if _finite(term) and _finite(shifted):
                y = term - shifted_comp
                total = shifted + y
                if _finite(total):
                    shifted_comp = (total - shifted) - y
                else:
                    shifted_comp = T.float32(0)
                shifted = total
            else:
                shifted = shifted + term
                shifted_comp = T.float32(0)
        folded = T.alloc_var(
            "float32", init=wv * factor
        )
        term = T.alloc_var("float32", init=hv * folded)
        if _finite(term) and _finite(dot):
            y = term - dot_comp
            total = dot + y
            if _finite(total):
                dot_comp = (total - dot) - y
            else:
                dot_comp = T.float32(0)
            dot = total
        else:
            dot = dot + term
            dot_comp = T.float32(0)
    bias_value = T.alloc_var("float32", init=0)
    if (flags & 1) != 0:
        bias_value = T.cast(bias[action], "float32")
    constant = T.alloc_var(
        "float32", init=bias_value + shifted
    )
    prediction = T.alloc_var("float32", init=dot + constant)
    if prenorm != 0:
        inverse = T.rsqrt(
            square * T.float32(1.0 / 1024.0) + eps
        )
        prediction = T.call_extern(
            "float32", "fmaf", inverse, dot, constant
        )
    return prediction


@T.macro
def _warp_projection(hidden, weight, _bias, _norm, _shift, hbase, action, _flags, _prenorm, _eps, column_lane):
    partial = T.alloc_var("float32", init=0)
    comp = T.alloc_var("float32", init=0)
    for stripe in T.serial(32):
        k = stripe * 32 + column_lane
        hv = T.cast(hidden[hbase + k], "float32")
        term = T.alloc_var("float32", init=hv * weight[action, k])
        if _finite(term) and _finite(partial):
            delta = term - comp
            total = partial + delta
            if _finite(total):
                comp = (total - partial) - delta
            else:
                comp = T.float32(0)
            partial = total
        else:
            partial = partial + term
            comp = T.float32(0)
    return T.warp_reduce_sum(partial)


def kernel(hidden_dtype, weight_dtype, norm_dtype, *, warp=False):
    project = _warp_projection if warp else _serial_projection
    lanes_per_output = 32 if warp else 1
    outputs_per_block = 128 // lanes_per_output

    @T.prim_func
    def main(
        oh: T.handle,
        xh: T.handle,
        hh: T.handle,
        wh: T.handle,
        bh: T.handle,
        nh: T.handle,
        sh: T.handle,
        rh: T.handle,
        d8h: T.handle,
        d32h: T.handle,
        vh: T.handle,
        ph: T.handle,
        pvh: T.handle,
        dth: T.handle,
        batches: T.int64,
        rows: T.int64,
        actions: T.int64,
        hidden_span: T.int64,
        hidden_batch_stride: T.int64,
        hidden_row_stride: T.int64,
        row_mask_batch_stride: T.int64,
        row_mask_row_stride: T.int64,
        dof_batch_stride: T.int64,
        dof_row_stride: T.int64,
        flags: T.int64,
        prenorm: T.int64,
        dt_bits: T.int64,
        eps_bits: T.int64,
        beta_bits: T.int64,
    ):
        count = batches * rows * actions
        out = T.match_buffer(oh, (count,), "float32")
        current = T.match_buffer(xh, (count,), "float32")
        hidden = T.match_buffer(hh, (hidden_span,), hidden_dtype)
        weight = T.match_buffer(wh, (actions, 1024), weight_dtype)
        bias = T.match_buffer(bh, (actions,), weight_dtype)
        norm = T.match_buffer(nh, (1024,), norm_dtype)
        shift = T.match_buffer(sh, (1024,), norm_dtype)
        row_mask = T.match_buffer(
            rh,
            (
                (batches - 1) * row_mask_batch_stride
                + (rows - 1) * row_mask_row_stride
                + 1,
            ),
            "uint8",
        )
        dof8 = T.match_buffer(
            d8h,
            ((batches - 1) * dof_batch_stride + (rows - 1) * dof_row_stride + actions,),
            "uint8",
        )
        dof32 = T.match_buffer(
            d32h,
            ((batches - 1) * dof_batch_stride + (rows - 1) * dof_row_stride + actions,),
            "float32",
        )
        padding = T.match_buffer(vh, (count,), "float32")
        pinned = T.match_buffer(ph, (batches,), "int32")
        fixed = T.match_buffer(pvh, (count,), "float32")
        dt_tensor = T.match_buffer(dth, (batches,), "float32")
        dt = T.reinterpret("float32", T.cast(dt_bits, "uint32"))
        eps = T.reinterpret("float32", T.cast(eps_bits, "uint32"))
        beta = T.reinterpret("float32", T.cast(beta_bits, "uint32"))
        blocks = T.ceildiv(count, 4) if warp else T.min(T.ceildiv(count, 128), T.int64(128))
        steps = 1 if warp else T.ceildiv(count, blocks * 128)
        action_width = T.int64(32) if warp else actions
        with T.Kernel(blocks, prelude=_ROUNDING_PRELUDE) as block:
            with T.SimtVF(threads=128):
                for step in T.serial(steps):
                    for lane in T.Parallel(128):
                        index = (step * blocks + T.cast(block, "int64")) * outputs_per_block + lane // lanes_per_output
                        column_lane = lane % lanes_per_output
                        if index < count:
                            action = T.truncmod(index, action_width)
                            flat_row = T.truncdiv(index, action_width)
                            row = T.truncmod(flat_row, rows)
                            batch = T.truncdiv(flat_row, rows)
                            xv = T.alloc_var("float32", init=0)
                            if (flags & 128) == 0 or beta != 0:
                                xv = current[index]
                            is_fixed = T.alloc_var("bool", init=False)
                            if (flags & 16) != 0:
                                is_fixed = row < T.cast(pinned[batch], "int64")
                            if is_fixed:
                                if column_lane == 0:
                                    if (flags & 32) != 0:
                                        out[index] = fixed[index]
                                    else:
                                        out[index] = xv
                            else:
                                active = T.alloc_var("bool", init=True)
                                if (flags & 2) != 0:
                                    mi = batch * row_mask_batch_stride + row * row_mask_row_stride
                                    active = row_mask[mi] != 0
                                if (flags & 128) != 0 and dt == 0:
                                    active = False
                                prediction = T.alloc_var("float32", init=0)
                                if active:
                                    hbase = batch * hidden_batch_stride + row * hidden_row_stride
                                    prediction = project(
                                        hidden, weight, bias, norm, shift, hbase, action,
                                        flags, prenorm, eps, column_lane,
                                    )
                                if column_lane == 0:
                                    velocity = T.alloc_var("float32", init=prediction)
                                    if (flags & 4) != 0:
                                        di = batch * dof_batch_stride + row * dof_row_stride + action
                                        mask = T.alloc_var("float32", init=0)
                                        if (flags & 8) != 0:
                                            mask = dof32[di]
                                        else:
                                            mask = T.cast(dof8[di] != 0, "float32")
                                        remainder = T.alloc_var("float32", init=T.float32(1) - mask)
                                        pad = T.alloc_var("float32", init=_mul_rn(padding[index], remainder))
                                        pred = T.alloc_var("float32", init=_mul_rn(prediction, mask))
                                        velocity = pad + pred
                                    step_size = T.alloc_var("float32", init=dt)
                                    if (flags & 64) != 0:
                                        step_size = dt_tensor[batch]
                                    update = T.alloc_var("float32", init=_mul_rn(step_size, velocity))
                                    if (flags & 128) != 0:
                                        scaled = T.alloc_var("float32", init=_mul_rn(beta, xv))
                                        out[index] = update + scaled
                                    else:
                                        out[index] = xv + update

    return main


def build_programs():
    from .denoise_head_cube import (
        adarm_head_cube,
        adarm_head_prep,
        bf16_head_cube,
        fp32_head_cube,
        fp32_head_prep,
    )
    from .denoise_input_embed import input_embed

    for ht, hd in (("bf16", "bfloat16"), ("f32", "float32")):
        for wt, wd in (("bf16", "bfloat16"), ("f32", "float32")):
            for nt, nd in (("bf16", "bfloat16"), ("f32", "float32")):
                yield f"denoise_{ht}_{wt}_{nt}", kernel(hd, wd, nd)
    yield "denoise_bf16_f32_warp", kernel("bfloat16", "float32", "float32", warp=True)
    yield "denoise_f32_f32_warp", kernel("float32", "float32", "float32", warp=True)
    yield "denoise_input_embed", input_embed()
    yield "denoise_bf16_cube", bf16_head_cube()
    yield "denoise_fp32head_prep", fp32_head_prep()
    yield "denoise_fp32head_cube", fp32_head_cube()
    yield "denoise_adarm_prep_bf16", adarm_head_prep("bfloat16")
    yield "denoise_adarm_prep_f32", adarm_head_prep("float32")
    yield "denoise_adarm_cube", adarm_head_cube()
