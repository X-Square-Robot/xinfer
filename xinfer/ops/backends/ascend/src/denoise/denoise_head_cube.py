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

"""Cube projections for the qualified K1024 A32 denoise head envelopes.

The serial TileLang candidates spend 1024 scalar iterations per output element,
and the warp candidates re-read the whole weight row per output; both lose to
the CANN chains once rows grow. These programs run the projection on the Cube
unit as 16x16 tiles and keep every source rounding boundary in the SIMT
epilogue:

- ``bf16_head_cube``: BF16 hidden and weight stay BF16 in L1 (products are
  exact in the FP32 accumulator).
- ``fp32_head_prep`` / ``fp32_head_cube``: DMA copies cannot cast, and a
  vector pass inside the Cube kernel forces pipeline drains that cost more
  than the whole projection; a separate prep kernel widens only the live
  rows of the BF16 hidden into an FP32 workspace, and the Cube kernel then
  runs the proven direct-copy FP32 plan.
- ``adarm_head_prep`` / ``adarm_head_cube``: the prep kernel widens and
  modulates the BF16 weight once (``w' = w * (1 + scale)``), reduces
  ``sums = shift . w`` per action and the per-row inverse RMS ``rho``; the
  Cube kernel computes ``prediction = fma(rho, h @ w', bias + sums)``,
  matching the serial candidate's ``fma(inverse, dot, constant)`` boundary.

Qualification envelope (enforced by the native dispatch, not here): A in
[17, 32] for the bf16 / fp32 heads (the weight tile copy zero-pads actions past
A; the flow readout A = 26 rides the same plans), A = 32 for the adarm head
whose prep is width-32, B in [1, 4], M in [1, 32], contiguous rows, no beta
blend.
"""

import tilelang.ascend.language as T

from .denoise_output import _ROUNDING_PRELUDE, _mul_rn

_K = 1024
_CHUNK = 128  # columns per lane in the vector passes (16 rows * 8 lanes)


@T.macro
def _cube_euler_tail(
    out,
    current,
    padding,
    pinned,
    fixed,
    dof8,
    dof32,
    dt_tensor,
    row_mask,
    prediction,
    index,
    batch,
    token,
    action,
    row_mask_batch_stride,
    row_mask_row_stride,
    dof_batch_stride,
    dof_row_stride,
    flags,
    dt,
    beta,
):
    """Euler tail after the projection, with the serial candidate's boundaries."""
    xv = T.alloc_var("float32", init=0)
    if (flags & 128) == 0 or beta != 0:
        xv = current[index]
    is_fixed = T.alloc_var("bool", init=False)
    if (flags & 16) != 0:
        is_fixed = token < T.cast(pinned[batch], "int64")
    if is_fixed:
        if (flags & 32) != 0:
            out[index] = fixed[index]
        else:
            out[index] = xv
    else:
        active = T.alloc_var("bool", init=True)
        if (flags & 2) != 0:
            active = row_mask[
                batch * row_mask_batch_stride + token * row_mask_row_stride
            ] != 0
        if (flags & 128) != 0 and dt == 0:
            active = False
        velocity = T.alloc_var("float32", init=0)
        if active:
            velocity = prediction
        if (flags & 4) != 0:
            di = batch * dof_batch_stride + token * dof_row_stride + action
            mask = T.alloc_var("float32", init=0)
            if (flags & 8) != 0:
                mask = dof32[di]
            else:
                mask = T.cast(dof8[di] != 0, "float32")
            remainder = T.alloc_var("float32", init=T.float32(1) - mask)
            pad = T.alloc_var("float32", init=_mul_rn(padding[index], remainder))
            pred = T.alloc_var("float32", init=_mul_rn(velocity, mask))
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


def bf16_head_cube():
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
        hidden = T.match_buffer(hh, (batches * rows, _K), "bfloat16")
        weight = T.match_buffer(wh, (actions, _K), "bfloat16")
        bias = T.match_buffer(bh, (actions,), "bfloat16")
        norm = T.match_buffer(nh, (_K,), "bfloat16")
        shift = T.match_buffer(sh, (_K,), "bfloat16")
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
        total_rows = batches * rows
        jobs = T.ceildiv(total_rows, 16) * 2
        with T.MixedKernel(jobs, sids=2, prelude=_ROUNDING_PRELUDE) as (bx, sid):
            tile = T.cast(bx, "int32")
            first_row = (tile // 2) * 16
            first_action = (tile % 2) * 16
            h_l1 = T.alloc_l1((16, _K), "bfloat16")
            w_l1 = T.alloc_l1((16, _K), "bfloat16")
            dot = T.alloc_l0c((16, 16), "float32")
            projection = T.alloc_shared((8, 16), "float32")
            T.copy(hidden[first_row : first_row + 16, 0:_K], h_l1, pad_value=0)
            T.copy(weight[first_action : first_action + 16, 0:_K], w_l1, pad_value=0)
            T.gemm(h_l1, w_l1, dot, transpose_B=True, clear_accum=True)
            T.dual_copy(dot, projection)
            with T.SimtVF(threads=128):
                for lane in T.Parallel(128):
                    local_row = lane // 16
                    local_action = lane % 16
                    row = first_row + sid * 8 + local_row
                    action = first_action + local_action
                    if row < total_rows and action < actions:
                        batch = row // rows
                        token = row % rows
                        index = row * actions + action
                        prediction = T.alloc_var(
                            "float32", init=projection[local_row, local_action]
                        )
                        if (flags & 1) != 0:
                            prediction = prediction + T.cast(bias[action], "float32")
                        _cube_euler_tail(
                            out,
                            current,
                            padding,
                            pinned,
                            fixed,
                            dof8,
                            dof32,
                            dt_tensor,
                            row_mask,
                            prediction,
                            index,
                            batch,
                            token,
                            action,
                            row_mask_batch_stride,
                            row_mask_row_stride,
                            dof_batch_stride,
                            dof_row_stride,
                            flags,
                            dt,
                            beta,
                        )

    return main


def fp32_head_prep():
    """Widen the live BF16 hidden rows into the FP32 workspace (vector only)."""

    @T.prim_func
    def main(hh: T.handle, oh: T.handle, total: T.int64):
        hidden = T.match_buffer(hh, (total * _K,), "bfloat16")
        widened = T.match_buffer(oh, (total * _K,), "float32")
        elements = total * _K
        blocks = T.min(T.ceildiv(elements, 1024), 128)
        with T.Kernel(blocks) as block:
            with T.SimtVF(threads=128):
                for lane in T.Parallel(128):
                    for step in T.serial(T.ceildiv(elements, blocks * 1024)):
                        index = (step * blocks + T.cast(block, "int64")) * 1024 + lane * 8
                        for element in T.serial(8):
                            if index + element < elements:
                                widened[index + element] = T.cast(
                                    hidden[index + element], "float32"
                                )

    return main


def fp32_head_cube():
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
        hidden = T.match_buffer(hh, (batches * rows, _K), "float32")
        weight = T.match_buffer(wh, (actions, _K), "float32")
        bias = T.match_buffer(bh, (actions,), "float32")
        norm = T.match_buffer(nh, (_K,), "float32")
        shift = T.match_buffer(sh, (_K,), "float32")
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
        total_rows = batches * rows
        jobs = T.ceildiv(total_rows, 16) * 2
        with T.MixedKernel(jobs, sids=2, prelude=_ROUNDING_PRELUDE) as (bx, sid):
            T.set_hf32_mode(None)
            tile = T.cast(bx, "int32")
            first_row = (tile // 2) * 16
            first_action = (tile % 2) * 16
            h_l1 = T.alloc_l1((16, _K), "float32")
            w_l1 = T.alloc_l1((16, _K), "float32")
            dot = T.alloc_l0c((16, 16), "float32")
            projection = T.alloc_shared((8, 16), "float32")
            T.copy(hidden[first_row : first_row + 16, 0:_K], h_l1, pad_value=0)
            T.copy(weight[first_action : first_action + 16, 0:_K], w_l1, pad_value=0)
            T.gemm(h_l1, w_l1, dot, transpose_B=True, clear_accum=True)
            T.dual_copy(dot, projection)
            with T.SimtVF(threads=128):
                for lane in T.Parallel(128):
                    local_row = lane // 16
                    local_action = lane % 16
                    row = first_row + sid * 8 + local_row
                    action = first_action + local_action
                    if row < total_rows and action < actions:
                        batch = row // rows
                        token = row % rows
                        index = row * actions + action
                        prediction = T.alloc_var(
                            "float32", init=projection[local_row, local_action]
                        )
                        if (flags & 1) != 0:
                            prediction = prediction + bias[action]
                        _cube_euler_tail(
                            out,
                            current,
                            padding,
                            pinned,
                            fixed,
                            dof8,
                            dof32,
                            dt_tensor,
                            row_mask,
                            prediction,
                            index,
                            batch,
                            token,
                            action,
                            row_mask_batch_stride,
                            row_mask_row_stride,
                            dof_batch_stride,
                            dof_row_stride,
                            flags,
                            dt,
                            beta,
                        )

    return main


def adarm_head_prep(norm_dtype):
    """Modulated weight, per-action shift dot and per-row inverse RMS."""

    @T.prim_func
    def main(
        hh: T.handle,
        wh: T.handle,
        nh: T.handle,
        sh: T.handle,
        wph: T.handle,
        rhoh: T.handle,
        sumsh: T.handle,
        batches: T.int64,
        rows: T.int64,
        eps_bits: T.int64,
    ):
        hidden = T.match_buffer(hh, (batches * rows * _K,), "float32")
        weight = T.match_buffer(wh, (32, _K), "bfloat16")
        norm = T.match_buffer(nh, (_K,), norm_dtype)
        shift = T.match_buffer(sh, (_K,), norm_dtype)
        modulated = T.match_buffer(wph, (32 * _K,), "float32")
        rho = T.match_buffer(rhoh, (batches * rows,), "float32")
        sums = T.match_buffer(sumsh, (32,), "float32")
        eps = T.reinterpret("float32", T.cast(eps_bits, "uint32"))
        total_rows = batches * rows
        with T.Kernel(32 + total_rows) as block:
            job = T.cast(block, "int64")
            partials = T.alloc_shared((4,), "float32")
            with T.SimtVF(threads=128):
                for lane in T.Parallel(128):
                    warp = lane // 32
                    partial = T.alloc_var("float32", init=0)
                    if job < 32:
                        for element in T.serial(_K // 128):
                            column = lane * (_K // 128) + element
                            wv = T.cast(weight[job, column], "float32")
                            modulated[job * _K + column] = wv * (
                                T.float32(1) + T.cast(norm[column], "float32")
                            )
                            partial = partial + T.cast(shift[column], "float32") * wv
                    else:
                        row = job - 32
                        for element in T.serial(_K // 128):
                            column = lane * (_K // 128) + element
                            value = hidden[row * _K + column]
                            partial = partial + value * value
                    warp_total = T.warp_reduce_sum(partial)
                    if lane % 32 == 0:
                        partials[warp] = warp_total
                    T.sync_threads()
                    if lane == 0:
                        total = T.alloc_var("float32", init=0)
                        for i in T.serial(4):
                            total = total + partials[i]
                        if job < 32:
                            sums[job] = total
                        else:
                            rho[job - 32] = T.rsqrt(total * T.float32(1.0 / _K) + eps)

    return main


def adarm_head_cube():
    @T.prim_func
    def main(
        oh: T.handle,
        xh: T.handle,
        hh: T.handle,
        wh: T.handle,
        bh: T.handle,
        rhoh: T.handle,
        sumsh: T.handle,
        rmh: T.handle,
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
        hidden = T.match_buffer(hh, (batches * rows, _K), "float32")
        weight = T.match_buffer(wh, (actions, _K), "float32")
        bias = T.match_buffer(bh, (actions,), "bfloat16")
        rho = T.match_buffer(rhoh, (batches * rows,), "float32")
        sums = T.match_buffer(sumsh, (32,), "float32")
        row_mask = T.match_buffer(
            rmh,
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
        total_rows = batches * rows
        jobs = T.ceildiv(total_rows, 16) * 2
        with T.MixedKernel(jobs, sids=2, prelude=_ROUNDING_PRELUDE) as (bx, sid):
            T.set_hf32_mode(None)
            tile = T.cast(bx, "int32")
            first_row = (tile // 2) * 16
            first_action = (tile % 2) * 16
            h_l1 = T.alloc_l1((16, _K), "float32")
            w_l1 = T.alloc_l1((16, _K), "float32")
            dot = T.alloc_l0c((16, 16), "float32")
            projection = T.alloc_shared((8, 16), "float32")
            T.copy(hidden[first_row : first_row + 16, 0:_K], h_l1, pad_value=0)
            T.copy(weight[first_action : first_action + 16, 0:_K], w_l1)
            T.gemm(h_l1, w_l1, dot, transpose_B=True, clear_accum=True)
            T.dual_copy(dot, projection)
            with T.SimtVF(threads=128):
                for lane in T.Parallel(128):
                    local_row = lane // 16
                    local_action = lane % 16
                    row = first_row + sid * 8 + local_row
                    if row < total_rows:
                        action = first_action + local_action
                        batch = row // rows
                        token = row % rows
                        index = row * 32 + action
                        bias_value = T.alloc_var("float32", init=0)
                        if (flags & 1) != 0:
                            bias_value = T.cast(bias[action], "float32")
                        constant = T.alloc_var("float32", init=bias_value + sums[action])
                        prediction = T.call_extern(
                            "float32",
                            "fmaf",
                            rho[row],
                            projection[local_row, local_action],
                            constant,
                        )
                        _cube_euler_tail(
                            out,
                            current,
                            padding,
                            pinned,
                            fixed,
                            dof8,
                            dof32,
                            dt_tensor,
                            row_mask,
                            prediction,
                            index,
                            batch,
                            token,
                            action,
                            row_mask_batch_stride,
                            row_mask_row_stride,
                            dof_batch_stride,
                            dof_row_stride,
                            flags,
                            dt,
                            beta,
                        )

    return main


def build_programs():
    yield "denoise_bf16_cube", bf16_head_cube()
    yield "denoise_fp32head_prep", fp32_head_prep()
    yield "denoise_fp32head_cube", fp32_head_cube()
    yield "denoise_adarm_prep_bf16", adarm_head_prep("bfloat16")
    yield "denoise_adarm_prep_f32", adarm_head_prep("float32")
    yield "denoise_adarm_cube", adarm_head_cube("bfloat16")
