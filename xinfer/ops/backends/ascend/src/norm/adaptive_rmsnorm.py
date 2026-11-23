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

"""Vector adaptive RMS normalization for the Wall-X 2B action stream.

Gated residual addition and BF16 rounding precede the norm. Compute FP32
statistics, apply the step's scale and shift, and return the gate. Pinned
RTC rows select the pinned modulation; double-norm variants retain the
intermediate rounding boundary between the two norms."""

import tilelang.ascend.language as T
from gemm.silu_and_mul import _store_rne

ROWS, HIDDEN, WARP, LANES = 16, 1024, 32, 64


def action_fast(double):
    """One block per token; pure-VEC elementwise + statistics (no SIMT)."""

    @T.prim_func
    def main(
        rh: T.handle, bh: T.handle, gh: T.handle, roh: T.handle, boh: T.handle,
        goh: T.handle, m1h: T.handle, m2h: T.handle, epsilon_bits: T.int64,
    ):
        residual = T.match_buffer(rh, (ROWS, HIDDEN), "bfloat16")
        branch = T.match_buffer(bh, (ROWS, HIDDEN), "bfloat16")
        previous_gate = T.match_buffer(gh, (HIDDEN,), "bfloat16")
        residual_out = T.match_buffer(roh, (ROWS, HIDDEN), "bfloat16")
        branch_out = T.match_buffer(boh, (ROWS, HIDDEN), "bfloat16")
        next_gate = T.match_buffer(goh, (HIDDEN,), "bfloat16")
        modulation1 = T.match_buffer(m1h, (3 * HIDDEN,), "float32")
        modulation2 = T.match_buffer(m2h, (3 * HIDDEN,), "float32")
        epsilon = T.reinterpret("float32", T.cast(epsilon_bits, "uint32"))
        with T.Kernel(ROWS) as block:
            row = T.cast(block, "int32")
            r = T.alloc_shared((HIDDEN,), "bfloat16")
            b = T.alloc_shared((HIDDEN,), "bfloat16")
            gate = T.alloc_shared((HIDDEN,), "bfloat16")
            first = T.alloc_shared((3 * HIDDEN,), "float32")
            second = T.alloc_shared((3 * HIDDEN,), "float32")
            n16 = T.alloc_shared((HIDDEN,), "bfloat16")
            g16 = T.alloc_shared((HIDDEN,), "bfloat16")
            sq = T.alloc_shared((HIDDEN,), "float32")
            T.copy(residual[row, :], r)
            T.copy(branch[row, :], b)
            T.copy(previous_gate, gate)
            T.copy(modulation1, first)
            if double:
                T.copy(modulation2, second)
            with T.SimdVF():
                m = T.simd.pset(32)
                one = T.simd.vdup(1.0, "float32")
                wide = T.simd.vdup(1.0 / HIDDEN, "float32")
                eps_v = T.simd.vdup(epsilon, "float32")
                # ---- VEC A: r = RNE(r_f32 + RNE(b_f32 * g_f32))  [_action_add]
                #      squares of the rounded row staged in the same pass
                for c in range(HIDDEN // LANES):
                    o = c * LANES
                    rv = T.simd.vcvt(
                        T.simd.vld(r[o], dist="UNPK_B16"), "float32", part=0
                    )
                    bv = T.simd.vcvt(
                        T.simd.vld(b[o], dist="UNPK_B16"), "float32", part=0
                    )
                    gv = T.simd.vcvt(
                        T.simd.vld(gate[o], dist="UNPK_B16"), "float32", part=0
                    )
                    prod = T.simd.vmul(bv, gv, m)
                    T.simd.vsts(
                        n16[o], T.simd.vcvt(prod, "bfloat16", round="ROUND_R"),
                        dist="PK_B32",
                    )
                    pf = T.simd.vcvt(
                        T.simd.vld(n16[o], dist="UNPK_B16"), "float32", part=0
                    )
                    s = T.simd.vadd(rv, pf, m)
                    T.simd.vsts(
                        r[o], T.simd.vcvt(s, "bfloat16", round="ROUND_R"),
                        dist="PK_B32",
                    )
                    rr = T.simd.vcvt(
                        T.simd.vld(r[o], dist="UNPK_B16"), "float32", part=0
                    )
                    T.simd.vsts(sq[o], T.simd.vmul(rr, rr, m))
                # ---- statistics 1: 16-segment vadd tree (balanced)
                # VST_VLD fence: the scheduler may otherwise hoist the tree
                # loads above the staging stores (opaque intrinsics, no dep).
                T.simd.mem_bar("VST_VLD")
                q0 = T.simd.vld(sq[0])
                q1 = T.simd.vld(sq[LANES])
                q2 = T.simd.vld(sq[2 * LANES])
                q3 = T.simd.vld(sq[3 * LANES])
                q4 = T.simd.vld(sq[4 * LANES])
                q5 = T.simd.vld(sq[5 * LANES])
                q6 = T.simd.vld(sq[6 * LANES])
                q7 = T.simd.vld(sq[7 * LANES])
                q8 = T.simd.vld(sq[8 * LANES])
                q9 = T.simd.vld(sq[9 * LANES])
                q10 = T.simd.vld(sq[10 * LANES])
                q11 = T.simd.vld(sq[11 * LANES])
                q12 = T.simd.vld(sq[12 * LANES])
                q13 = T.simd.vld(sq[13 * LANES])
                q14 = T.simd.vld(sq[14 * LANES])
                q15 = T.simd.vld(sq[15 * LANES])
                t0 = T.simd.vadd(q0, q1, m)
                t1 = T.simd.vadd(q2, q3, m)
                t2 = T.simd.vadd(q4, q5, m)
                t3 = T.simd.vadd(q6, q7, m)
                t4 = T.simd.vadd(q8, q9, m)
                t5 = T.simd.vadd(q10, q11, m)
                t6 = T.simd.vadd(q12, q13, m)
                t7 = T.simd.vadd(q14, q15, m)
                u0 = T.simd.vadd(t0, t1, m)
                u1 = T.simd.vadd(t2, t3, m)
                u2 = T.simd.vadd(t4, t5, m)
                u3 = T.simd.vadd(t6, t7, m)
                w0 = T.simd.vadd(u0, u1, m)
                w1 = T.simd.vadd(u2, u3, m)
                tree = T.simd.vadd(w0, w1, m)
                # ---- statistics 1: vcpadd chain 64 -> 1 lane, broadcast
                cp1 = T.simd.vcpadd(tree, m)
                cp2 = T.simd.vcpadd(cp1, m)
                cp3 = T.simd.vcpadd(cp2, m)
                cp4 = T.simd.vcpadd(cp3, m)
                cp5 = T.simd.vcpadd(cp4, m)
                total = T.simd.vdupv(T.simd.vcpadd(cp5, m), m)
                denom = T.simd.vadd(T.simd.vmul(total, wide, m), eps_v, m)
                inv1 = T.simd.vdiv(one, T.simd.vsqrt(denom, m), m)
                if not double:
                    # ---- VEC B (gated): b = norm(r, first, inv1); gate row 0
                    wv = T.simd.alloc_local((1,), "float32")
                    for c in range(HIDDEN // LANES):
                        o = c * LANES
                        rf = T.simd.vcvt(
                            T.simd.vld(r[o], dist="UNPK_B16"), "float32",
                            part=0,
                        )
                        sc = T.simd.vld(first[o])
                        sh = T.simd.vld(first[HIDDEN + o])
                        wv[0] = T.simd.vmul(rf, inv1, m)
                        u = T.simd.vadd(one, sc, m)
                        T.simd.vmadd(wv[0], u, sh, m)
                        T.simd.vsts(
                            b[o],
                            T.simd.vcvt(wv[0], "bfloat16", round="ROUND_R"),
                            dist="PK_B32",
                        )
                        gt = T.simd.vld(first[2 * HIDDEN + o])
                        T.simd.vsts(
                            gate[o],
                            T.simd.vcvt(gt, "bfloat16", round="ROUND_R"),
                            dist="PK_B32",
                        )
                else:
                    # ---- VEC B (double): gate1 pack, then norm1 + gated add
                    wv = T.simd.alloc_local((1,), "float32")
                    for c in range(HIDDEN // LANES):
                        o = c * LANES
                        gt = T.simd.vld(first[2 * HIDDEN + o])
                        T.simd.vsts(
                            g16[o],
                            T.simd.vcvt(gt, "bfloat16", round="ROUND_R"),
                            dist="PK_B32",
                        )
                    for c in range(HIDDEN // LANES):
                        o = c * LANES
                        rf = T.simd.vcvt(
                            T.simd.vld(r[o], dist="UNPK_B16"), "float32",
                            part=0,
                        )
                        sc = T.simd.vld(first[o])
                        sh = T.simd.vld(first[HIDDEN + o])
                        wv[0] = T.simd.vmul(rf, inv1, m)
                        u = T.simd.vadd(one, sc, m)
                        T.simd.vmadd(wv[0], u, sh, m)
                        T.simd.vsts(
                            n16[o],
                            T.simd.vcvt(wv[0], "bfloat16", round="ROUND_R"),
                            dist="PK_B32",
                        )
                        nf = T.simd.vcvt(
                            T.simd.vld(n16[o], dist="UNPK_B16"), "float32",
                            part=0,
                        )
                        g1v = T.simd.vcvt(
                            T.simd.vld(g16[o], dist="UNPK_B16"), "float32",
                            part=0,
                        )
                        prod2 = T.simd.vmul(nf, g1v, m)
                        T.simd.vsts(
                            n16[o],
                            T.simd.vcvt(prod2, "bfloat16", round="ROUND_R"),
                            dist="PK_B32",
                        )
                        pf2 = T.simd.vcvt(
                            T.simd.vld(n16[o], dist="UNPK_B16"), "float32",
                            part=0,
                        )
                        s2 = T.simd.vadd(rf, pf2, m)
                        T.simd.vsts(
                            r[o],
                            T.simd.vcvt(s2, "bfloat16", round="ROUND_R"),
                            dist="PK_B32",
                        )
                        rr2 = T.simd.vcvt(
                            T.simd.vld(r[o], dist="UNPK_B16"), "float32",
                            part=0,
                        )
                        T.simd.vsts(sq[o], T.simd.vmul(rr2, rr2, m))
                    # ---- statistics 2 (same vector reduction, second stats)
                    T.simd.mem_bar("VST_VLD")
                    v0 = T.simd.vld(sq[0])
                    v1 = T.simd.vld(sq[LANES])
                    v2 = T.simd.vld(sq[2 * LANES])
                    v3 = T.simd.vld(sq[3 * LANES])
                    v4 = T.simd.vld(sq[4 * LANES])
                    v5 = T.simd.vld(sq[5 * LANES])
                    v6 = T.simd.vld(sq[6 * LANES])
                    v7 = T.simd.vld(sq[7 * LANES])
                    v8 = T.simd.vld(sq[8 * LANES])
                    v9 = T.simd.vld(sq[9 * LANES])
                    v10 = T.simd.vld(sq[10 * LANES])
                    v11 = T.simd.vld(sq[11 * LANES])
                    v12 = T.simd.vld(sq[12 * LANES])
                    v13 = T.simd.vld(sq[13 * LANES])
                    v14 = T.simd.vld(sq[14 * LANES])
                    v15 = T.simd.vld(sq[15 * LANES])
                    x0 = T.simd.vadd(v0, v1, m)
                    x1 = T.simd.vadd(v2, v3, m)
                    x2 = T.simd.vadd(v4, v5, m)
                    x3 = T.simd.vadd(v6, v7, m)
                    x4 = T.simd.vadd(v8, v9, m)
                    x5 = T.simd.vadd(v10, v11, m)
                    x6 = T.simd.vadd(v12, v13, m)
                    x7 = T.simd.vadd(v14, v15, m)
                    y0 = T.simd.vadd(x0, x1, m)
                    y1 = T.simd.vadd(x2, x3, m)
                    y2 = T.simd.vadd(x4, x5, m)
                    y3 = T.simd.vadd(x6, x7, m)
                    z0 = T.simd.vadd(y0, y1, m)
                    z1 = T.simd.vadd(y2, y3, m)
                    tree2 = T.simd.vadd(z0, z1, m)
                    dq1 = T.simd.vcpadd(tree2, m)
                    dq2 = T.simd.vcpadd(dq1, m)
                    dq3 = T.simd.vcpadd(dq2, m)
                    dq4 = T.simd.vcpadd(dq3, m)
                    dq5 = T.simd.vcpadd(dq4, m)
                    total2 = T.simd.vdupv(T.simd.vcpadd(dq5, m), m)
                    denom2 = T.simd.vadd(
                        T.simd.vmul(total2, wide, m), eps_v, m
                    )
                    inv2 = T.simd.vdiv(one, T.simd.vsqrt(denom2, m), m)
                    # ---- VEC C (double): b = norm(r, second, inv2); gate row 0
                    wv2 = T.simd.alloc_local((1,), "float32")
                    for c in range(HIDDEN // LANES):
                        o = c * LANES
                        rf = T.simd.vcvt(
                            T.simd.vld(r[o], dist="UNPK_B16"), "float32",
                            part=0,
                        )
                        sc = T.simd.vld(second[o])
                        sh = T.simd.vld(second[HIDDEN + o])
                        wv2[0] = T.simd.vmul(rf, inv2, m)
                        u = T.simd.vadd(one, sc, m)
                        T.simd.vmadd(wv2[0], u, sh, m)
                        T.simd.vsts(
                            b[o],
                            T.simd.vcvt(wv2[0], "bfloat16", round="ROUND_R"),
                            dist="PK_B32",
                        )
                        gt = T.simd.vld(second[2 * HIDDEN + o])
                        T.simd.vsts(
                            gate[o],
                            T.simd.vcvt(gt, "bfloat16", round="ROUND_R"),
                            dist="PK_B32",
                        )
            T.copy(r, residual_out[row, :])
            T.copy(b, branch_out[row, :])
            if row == 0:
                T.copy(gate, next_gate)

    return main


@T.macro
def _action_add(residual, branch, gate):
    product = _store_rne(T.cast(branch, "float32") * T.cast(gate, "float32"), "bfloat16")
    return _store_rne(T.cast(residual, "float32") + T.cast(product, "float32"), "bfloat16")


@T.macro
def _action_norm(residual, modulation, inverse, col):
    value = T.alloc_var(
        "float32",
        init=T.cast(residual, "float32") * inverse * (1.0 + modulation[col])
        + modulation[1024 + col],
    )
    return _store_rne(value, "bfloat16")


@T.macro
def _action_norm_pin(residual, modulation, pinned_modulation, selected, inverse, col):
    scale = T.Select(selected != 0, pinned_modulation[col], modulation[col])
    shift = T.Select(selected != 0, pinned_modulation[1024 + col], modulation[1024 + col])
    value = T.alloc_var(
        "float32",
        init=T.cast(residual, "float32") * inverse * (1.0 + scale) + shift,
    )
    return _store_rne(value, "bfloat16")


def action_adarms_bf16_fp32_pinned(double):
    """Pinned (RTC) variant of :func:`action_adarms_bf16_fp32`.

    The Wall-X 2B RTC denoise pins its first ``pinned_rows[0]`` tokens to the
    pinned modulations.  The generic kernels then read ``pinned_rows`` per row
    from global memory while their row-per-thread grid degenerates on the
    16-token action shape (one block, 16 of 128 threads active), which made
    every pinned norm call ~100x the fused one and collapsed serving capacity.
    This variant keeps the fused kernel's one-warp-per-token layout: both the
    shared and the pinned modulation are staged into unified buffers once and
    selected per token with ``T.Select``, exactly the selection the composed
    generic path performs.  Gates are per token, as pinned requests require.
    """
    rows, hidden, warp = 16, 1024, 32

    @T.prim_func
    def main(
        rh: T.handle, bh: T.handle, gh: T.handle, roh: T.handle, boh: T.handle,
        goh: T.handle, m1h: T.handle, m2h: T.handle, pm1h: T.handle, pm2h: T.handle,
        prh: T.handle, epsilon_bits: T.int64,
    ):
        residual = T.match_buffer(rh, (rows, hidden), "bfloat16")
        branch = T.match_buffer(bh, (rows, hidden), "bfloat16")
        previous_gate = T.match_buffer(gh, (rows, hidden), "bfloat16")
        residual_out = T.match_buffer(roh, (rows, hidden), "bfloat16")
        branch_out = T.match_buffer(boh, (rows, hidden), "bfloat16")
        next_gate = T.match_buffer(goh, (rows, hidden), "bfloat16")
        modulation1 = T.match_buffer(m1h, (3 * hidden,), "float32")
        modulation2 = T.match_buffer(m2h, (3 * hidden,), "float32")
        pinned_mod1 = T.match_buffer(pm1h, (3 * hidden,), "float32")
        pinned_mod2 = T.match_buffer(pm2h, (3 * hidden,), "float32")
        pinned_rows = T.match_buffer(prh, (rows,), "int32")
        epsilon = T.reinterpret("float32", T.cast(epsilon_bits, "uint32"))
        with T.Kernel(rows) as block:
            row = T.cast(block, "int32")
            selected = T.alloc_var("int32", init=0)
            if pinned_rows[0] > row:
                selected = T.int32(1)
            r = T.alloc_shared((hidden,), "bfloat16")
            b = T.alloc_shared((hidden,), "bfloat16")
            gate = T.alloc_shared((hidden,), "bfloat16")
            first = T.alloc_shared((3 * hidden,), "float32")
            second = T.alloc_shared((3 * hidden,), "float32")
            pinned_first = T.alloc_shared((3 * hidden,), "float32")
            pinned_second = T.alloc_shared((3 * hidden,), "float32")
            T.copy(residual[row, :], r)
            T.copy(branch[row, :], b)
            T.copy(previous_gate[row, :], gate)
            T.copy(modulation1, first)
            T.copy(pinned_mod1, pinned_first)
            if double:
                T.copy(modulation2, second)
                T.copy(pinned_mod2, pinned_second)
            with T.SimtVF(threads=warp):
                for lane in T.Parallel(warp):
                    square1 = T.alloc_var("float32", init=0)
                    for step in T.serial(hidden // warp):
                        col = step * warp + lane
                        r[col] = _action_add(r[col], b[col], gate[col])
                        value = T.cast(r[col], "float32")
                        square1 += value * value
                    inv1 = T.rsqrt(T.warp_reduce_sum(square1) / T.float32(hidden) + epsilon)
                    if double:
                        square2 = T.alloc_var("float32", init=0)
                        for step in T.serial(hidden // warp):
                            col = step * warp + lane
                            norm = _action_norm_pin(r[col], first, pinned_first, selected, inv1, col)
                            gate1 = _store_rne(
                                T.Select(
                                    selected != 0,
                                    pinned_first[2 * hidden + col],
                                    first[2 * hidden + col],
                                ),
                                "bfloat16",
                            )
                            r[col] = _action_add(r[col], norm, gate1)
                            value = T.cast(r[col], "float32")
                            square2 += value * value
                        inv2 = T.rsqrt(T.warp_reduce_sum(square2) / T.float32(hidden) + epsilon)
                        for step in T.serial(hidden // warp):
                            col = step * warp + lane
                            b[col] = _action_norm_pin(r[col], second, pinned_second, selected, inv2, col)
                            gate[col] = _store_rne(
                                T.Select(
                                    selected != 0,
                                    pinned_second[2 * hidden + col],
                                    second[2 * hidden + col],
                                ),
                                "bfloat16",
                            )
                    else:
                        for step in T.serial(hidden // warp):
                            col = step * warp + lane
                            b[col] = _action_norm_pin(r[col], first, pinned_first, selected, inv1, col)
                            gate[col] = _store_rne(
                                T.Select(
                                    selected != 0,
                                    pinned_first[2 * hidden + col],
                                    first[2 * hidden + col],
                                ),
                                "bfloat16",
                            )
            T.copy(r, residual_out[row, :])
            T.copy(b, branch_out[row, :])
            T.copy(gate, next_gate[row, :])

    return main



def build_programs():
    yield "adarmsnorm_action_gated_bf16_f32", action_fast(False)
    yield "adarmsnorm_action_double_bf16_f32", action_fast(True)
    # RTC selects the pinned modulation in-kernel for its first pinned_rows[0] tokens.
    yield "adarmsnorm_action_gated_bf16_f32_pinned", action_adarms_bf16_fp32_pinned(False)
    yield "adarmsnorm_action_double_bf16_f32_pinned", action_adarms_bf16_fp32_pinned(True)
