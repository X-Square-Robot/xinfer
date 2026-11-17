"""Pinned FP32 scalar Denoise projection and separately rounded Euler tail."""

# Unused buffer views keep the shared native launch ABI typed.
# ruff: noqa: F841
import tilelang.ascend.language as T
import tilelang.ascend.language.simd as S

from .denoise_output import _ROUNDING_PRELUDE, _mul_rn


def scalar_projection(*, cube=False):
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
        hidden = T.match_buffer(
            hh, (batches * rows, 1024) if cube else (hidden_span,), "float32"
        )
        weight = T.match_buffer(wh, (actions, 1024), "float32")
        bias = T.match_buffer(bh, (actions,), "float32")
        norm = T.match_buffer(nh, (1024,), "float32")
        shift = T.match_buffer(sh, (1024,), "float32")
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
        if cube:
            total_rows = batches * rows
            jobs = T.ceildiv(total_rows, 16) * 2
            with T.MixedKernel(jobs, sids=2, prelude=_ROUNDING_PRELUDE) as (bx, sid):
                T.set_hf32_mode(None)
                tile = T.cast(bx, "int32")
                first_row = (tile // 2) * 16
                first_action = (tile % 2) * 16
                h_l1 = T.alloc_l1((16, 1024), "float32")
                w_l1 = T.alloc_l1((16, 1024), "float32")
                dot = T.alloc_l0c((16, 16), "float32")
                projection = T.alloc_shared((8, 16), "float32")
                T.copy(hidden[first_row : first_row + 16, 0:1024], h_l1, pad_value=0)
                T.copy(weight[first_action : first_action + 16, 0:1024], w_l1)
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
                            mask = dof32[
                                batch * dof_batch_stride
                                + token * dof_row_stride
                                + action
                            ]
                            pad = _mul_rn(padding[index], T.float32(1) - mask)
                            pred = _mul_rn(projection[local_row, local_action], mask)
                            out[index] = current[index] + _mul_rn(dt, pad + pred)
        else:
            with T.Kernel(count, prelude=_ROUNDING_PRELUDE) as block:
                index = T.cast(block, "int64")
                batch = index // 32
                action = index % 32
                hbase = batch * hidden_batch_stride
                h_ub = T.alloc_shared((1024,), "float32")
                w_ub = T.alloc_shared((1024,), "float32")
                prediction = T.alloc_shared((64,), "float32")
                T.copy(hidden[hbase : hbase + 1024], h_ub)
                T.copy(weight[action, 0:1024], w_ub)
                with T.SimdVF():
                    allp = S.pset(32, "PAT_ALL")
                    one = S.pset(32, "PAT_VL1")
                    zero = S.vdup(T.float32(0), "float32", allp)
                    acc = S.alloc_local((1,), "float32")
                    comp = S.alloc_local((1,), "float32")
                    acc[0] = zero
                    comp[0] = zero
                    for stripe in range(16):
                        term = S.vmul(
                            S.vld(h_ub[stripe * 64], "NORM"),
                            S.vld(w_ub[stripe * 64], "NORM"),
                            allp,
                        )
                        finite_term = S.vcmps(
                            S.vabs(term, allp), T.infinity("float32"), allp, "lt"
                        )
                        finite_acc = S.vcmps(
                            S.vabs(acc[0], allp), T.infinity("float32"), allp, "lt"
                        )
                        compensated = S.pand(finite_term, finite_acc, allp)
                        delta = S.vsub(term, comp[0], allp)
                        total = S.vadd(acc[0], delta, allp)
                        finite_total = S.vcmps(
                            S.vabs(total, allp), T.infinity("float32"), allp, "lt"
                        )
                        keep_comp = S.pand(compensated, finite_total, allp)
                        comp[0] = S.vsel(
                            S.vsub(S.vsub(total, acc[0], allp), delta, allp),
                            zero,
                            keep_comp,
                        )
                        acc[0] = S.vsel(total, S.vadd(acc[0], term, allp), compensated)
                    S.vsts(prediction[0], S.vcadd(acc[0], allp), one, "ONEPT_B32")
                with T.SimtVF(threads=32):
                    for lane in T.Parallel(32):
                        if lane == 0:
                            mask = dof32[batch * dof_batch_stride + action]
                            pad = _mul_rn(padding[index], T.float32(1) - mask)
                            pred = _mul_rn(prediction[0], mask)
                            out[index] = current[index] + _mul_rn(dt, pad + pred)

    return main
