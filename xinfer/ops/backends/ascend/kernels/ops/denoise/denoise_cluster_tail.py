"""Build-only K1280 AdaRMS, FP32 projection and Euler update.

Each block shares one BF16-rounded AdaRMS row across a tile of outputs.
One warp computes each FP32 projection and Euler update.
"""

import tilelang.ascend.language as T

from .silu_and_mul import _one_axis_offset

COMPILE_OPTIONS = ["-ffp-contract=off"]
K = 1280
THREADS = 256
WARP_SIZE = 32


def _bfr(value):
    """Original cluster_tail bfr bits, including its nonfinite behavior."""
    bits = T.reinterpret("uint32", value)
    rounded = (bits + T.uint32(0x7FFF) + ((bits >> 16) & T.uint32(1))) & T.uint32(
        0xFFFF0000
    )
    return T.reinterpret("float32", rounded)


def cluster_tail():
    @T.prim_func
    def main(
        oh: T.handle,
        xh: T.handle,
        hh: T.handle,
        wh: T.handle,
        hih: T.handle,
        sch: T.handle,
        shh: T.handle,
        aih: T.handle,
        d8h: T.handle,
        d32h: T.handle,
        pvh: T.handle,
        dth: T.handle,
        tth: T.handle,
        rows: T.int64,
        m: T.int64,
        actions: T.int64,
        heads: T.int64,
        tables: T.int64,
        hspan: T.int64,
        hs: T.int64,
        scspan: T.int64,
        scs: T.int64,
        shspan: T.int64,
        shs: T.int64,
        flags: T.int64,
        eps_bits: T.int64,
        dt_bits: T.int64,
        tau_bits: T.int64,
        floor_bits: T.int64,
    ):
        count = rows * actions
        out = T.match_buffer(oh, (count,), "float32")
        x = T.match_buffer(xh, (count,), "float32")
        hidden = T.match_buffer(hh, (hspan,), "bfloat16")
        weight = T.match_buffer(wh, (heads, actions, K), "float32")
        head_index = T.match_buffer(hih, (rows,), "int32")
        scale = T.match_buffer(sch, (scspan,), "bfloat16")
        shift = T.match_buffer(shh, (shspan,), "bfloat16")
        ada_index = T.match_buffer(aih, (rows,), "int32")
        dof8 = T.match_buffer(d8h, (count,), "uint8")
        dof32 = T.match_buffer(d32h, (count,), "float32")
        padding = T.match_buffer(pvh, (count,), "float32")
        dt_tensor = T.match_buffer(dth, (T.truncdiv(rows, m),), "float32")
        tau_tensor = T.match_buffer(tth, (T.truncdiv(rows, m),), "float32")
        eps = T.reinterpret("float32", T.cast(eps_bits, "uint32"))
        dt = T.reinterpret("float32", T.cast(dt_bits, "uint32"))
        tau = T.reinterpret("float32", T.cast(tau_bits, "uint32"))
        tau_floor = T.reinterpret("float32", T.cast(floor_bits, "uint32"))
        outputs_per_block = THREADS // WARP_SIZE
        tiles = T.ceildiv(actions, outputs_per_block)
        with T.Kernel(rows * tiles) as block:
            normalized = T.alloc_shared((K,), "float32")
            warp_squares = T.alloc_shared((outputs_per_block,), "float32")
            with T.SimtVF(threads=THREADS):
                for lane in T.Parallel(THREADS):
                    row = T.truncdiv(T.cast(block, "int64"), tiles)
                    tile = T.truncmod(T.cast(block, "int64"), tiles)
                    warp = lane // WARP_SIZE
                    column_lane = lane % WARP_SIZE
                    col = tile * outputs_per_block + warp
                    index = row * actions + col
                    batch = T.truncdiv(row, m)
                    head = T.alloc_var("int64", init=0)
                    table = T.alloc_var("int64", init=0)
                    if (flags & 1) != 0:
                        head = T.cast(head_index[row], "int64")
                    if (flags & 2) != 0:
                        table = T.cast(ada_index[row], "int64")
                    # Every lane in the block reads the same head and table indices.
                    if head >= 0 and head < heads and table >= 0 and table < tables:
                        hb = _one_axis_offset(row, 0, rows, hs)
                        sb = _one_axis_offset(table, 0, tables, scs)
                        tb = _one_axis_offset(table, 0, tables, shs)
                        squares = T.alloc_var("float32", init=0)
                        for stripe in T.serial(K // THREADS):
                            k = stripe * THREADS + lane
                            value = T.cast(hidden[hb + k], "float32")
                            squares = squares + value * value
                        sum_warp = T.warp_reduce_sum(squares)
                        if column_lane == 0:
                            warp_squares[warp] = sum_warp
                        T.sync_threads()
                        total = T.alloc_var("float32", init=0)
                        for i in T.unroll(outputs_per_block):
                            total = total + warp_squares[i]
                        inverse = T.rsqrt(total * T.float32(1.0 / K) + eps)
                        for stripe in T.serial(K // THREADS):
                            k = stripe * THREADS + lane
                            value = T.cast(hidden[hb + k], "float32") * inverse
                            factor = T.float32(1) + T.cast(scale[sb + k], "float32")
                            ada = T.call_extern(
                                "float32",
                                "fmaf",
                                value,
                                factor,
                                T.cast(shift[tb + k], "float32"),
                            )
                            normalized[k] = _bfr(ada)
                        T.sync_threads()
                        if col < actions:
                            partial = T.alloc_var("float32", init=0)
                            for stripe in T.serial(K // WARP_SIZE):
                                k = stripe * WARP_SIZE + column_lane
                                partial = partial + normalized[k] * weight[head, col, k]
                            projected = T.warp_reduce_sum(partial)
                            if column_lane == 0:
                                old = T.alloc_var("float32", init=x[index])
                                velocity = T.alloc_var("float32", init=projected)
                                if (flags & 128) != 0:
                                    time = T.alloc_var("float32", init=tau)
                                    if (flags & 64) != 0:
                                        time = tau_tensor[batch]
                                    denominator = T.call_extern(
                                        "float32",
                                        "fmaxf",
                                        T.float32(1) - time,
                                        tau_floor,
                                    )
                                    velocity = (velocity - old) / denominator
                                if (flags & 4) != 0:
                                    mi = T.alloc_var("int64", init=index)
                                    if (flags & 256) != 0:
                                        mi = batch * actions + col
                                    mask = T.alloc_var("float32", init=0)
                                    if (flags & 8) != 0:
                                        mask = dof32[mi]
                                    else:
                                        mask = T.cast(dof8[mi] != 0, "float32")
                                    pad = T.alloc_var("float32", init=0)
                                    if (flags & 16) != 0:
                                        pad = padding[index]
                                    left = T.alloc_var(
                                        "float32", init=pad * (T.float32(1) - mask)
                                    )
                                    right = T.alloc_var("float32", init=velocity * mask)
                                    velocity = left + right
                                step = T.alloc_var("float32", init=dt)
                                if (flags & 32) != 0:
                                    step = dt_tensor[batch]
                                update = T.alloc_var("float32", init=step * velocity)
                                out[index] = old + update

    return main


def build_programs():
    yield "denoise_cluster_tail", cluster_tail()
