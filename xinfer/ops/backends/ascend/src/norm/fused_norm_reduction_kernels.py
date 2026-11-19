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

"""Fused row reductions for the 2048-wide residual RMSNorm and 1024-wide vision LayerNorm.

RMSNorm rounds the residual sum to BF16, accumulates its squares in FP32,
refines the reciprocal square root and applies the zero-centered weight.
LayerNorm computes the mean, variance and affine output in FP32 before BF16
rounding. The vector reductions have a fixed tree; its ordering can differ
from the vendor reductions."""

import tilelang.ascend.language as T

CORES = 64
LANES = 64
WARP = 32


def _rsqrt_f32(t):
    """Refine the hardware reciprocal-square-root estimate with two FP32 Newton steps."""
    half_t = T.float32(0.5) * t
    y = T.rsqrt(t)
    y = y * (T.float32(1.5) - half_t * y * y)
    y = y * (T.float32(1.5) - half_t * y * y)
    return y


def rmsnorm_full_c2048(cols=2048, cores=CORES):
    """(s, normed) of add_zero_centered_rms_norm in a single fused launch."""

    @T.prim_func
    def main(sh: T.handle, oh: T.handle, xh: T.handle, rh: T.handle, wh: T.handle,
             epsilon_bits: T.int64, rows: T.int64):
        s = T.match_buffer(sh, (rows * cols,), "bfloat16")
        out = T.match_buffer(oh, (rows * cols,), "bfloat16")
        x = T.match_buffer(xh, (rows * cols,), "bfloat16")
        r = T.match_buffer(rh, (rows * cols,), "bfloat16")
        w = T.match_buffer(wh, (cols,), "float32")
        epsilon = T.reinterpret("float32", T.cast(epsilon_bits, "uint32"))
        with T.Kernel(cores) as core:
            x_ub = T.alloc_shared((cols,), "bfloat16")
            r_ub = T.alloc_shared((cols,), "bfloat16")
            s_ub = T.alloc_shared((cols,), "bfloat16")
            o_ub = T.alloc_shared((cols,), "bfloat16")
            w_ub = T.alloc_shared((cols,), "float32")
            red_ub = T.alloc_shared((64,), "float32")
            inv_ub = T.alloc_shared((1,), "float32")
            T.annotate_buffer_versions({x_ub: 2, r_ub: 2, s_ub: 2, o_ub: 2})
            T.copy(w, w_ub)
            for step in T.Pipelined(T.ceildiv(rows, cores), num_stages=2):
                row = step * cores + core
                if row < rows:
                    base = T.cast(row * cols, "int32")
                    T.copy(x[base : base + cols], x_ub)
                    T.copy(r[base : base + cols], r_ub)
                    # VEC A: s = rne(f32(x)+f32(res)) (normprep math verbatim,
                    # store-reload roundtrip for the bf16-rounded s) while the
                    # 64-lane fp32 accumulator gathers f32(s)^2
                    with T.SimdVF():
                        m = T.simd.pset(32)
                        acc = T.simd.alloc_local((1,), "float32")
                        acc[0] = T.simd.vdup(0.0, "float32", m)
                        for i in range(cols // LANES):
                            a = T.simd.vcvt(
                                T.simd.vld(x_ub[i * LANES], dist="UNPK_B16"),
                                "float32", part=0)
                            b = T.simd.vcvt(
                                T.simd.vld(r_ub[i * LANES], dist="UNPK_B16"),
                                "float32", part=0)
                            T.simd.vsts(
                                s_ub[i * LANES],
                                T.simd.vcvt(T.simd.vadd(a, b, m),
                                            "bfloat16", round="ROUND_R"),
                                dist="PK_B32")
                            sf = T.simd.vcvt(
                                T.simd.vld(s_ub[i * LANES], dist="UNPK_B16"),
                                "float32", part=0)
                            acc[0] = T.simd.vadd(
                                acc[0], T.simd.vmul(sf, sf, m), m)
                        T.simd.vsts(red_ub[0], acc[0])
                    with T.SimtVF(threads=WARP):
                        for lane in T.Parallel(WARP):
                            val = red_ub[lane] + red_ub[lane + 32]
                            total = T.warp_reduce_sum(val)
                            if lane == 0:
                                inv_ub[0] = _rsqrt_f32(total / T.float32(cols) + epsilon)
                    # VEC B: normed = rne((f32(s) * rstd) * (1 + w)) -- the
                    # same association as apply_rstd_c2048
                    with T.SimdVF():
                        m = T.simd.pset(32)
                        one = T.simd.vdup(1.0, "float32", m)
                        invv = T.simd.vdup(inv_ub[0], "float32", m)
                        for i in range(cols // LANES):
                            a = T.simd.vcvt(
                                T.simd.vld(s_ub[i * LANES], dist="UNPK_B16"),
                                "float32", part=0)
                            wv = T.simd.vld(w_ub[i * LANES])
                            t = T.simd.vmul(a, invv, m)
                            u = T.simd.vadd(one, wv, m)
                            T.simd.vsts(
                                o_ub[i * LANES],
                                T.simd.vcvt(T.simd.vmul(t, u, m),
                                            "bfloat16", round="ROUND_R"),
                                dist="PK_B32")
                    T.copy(s_ub, s[base : base + cols])
                    T.copy(o_ub, out[base : base + cols])

    return main


def layernorm_c1024(cols=1024, cores=CORES):
    """Two-pass FP32 mean/variance, reciprocal square root and affine transform with one final BF16 rounding."""

    @T.prim_func
    def main(oh: T.handle, xh: T.handle, wh: T.handle, bh: T.handle,
             epsilon_bits: T.int64, rows: T.int64):
        out = T.match_buffer(oh, (rows * cols,), "bfloat16")
        x = T.match_buffer(xh, (rows * cols,), "bfloat16")
        w = T.match_buffer(wh, (cols,), "bfloat16")
        b = T.match_buffer(bh, (cols,), "bfloat16")
        epsilon = T.reinterpret("float32", T.cast(epsilon_bits, "uint32"))
        with T.Kernel(cores) as core:
            x_ub = T.alloc_shared((cols,), "bfloat16")
            xf_ub = T.alloc_shared((cols,), "float32")
            o_ub = T.alloc_shared((cols,), "bfloat16")
            w_ub = T.alloc_shared((cols,), "bfloat16")
            b_ub = T.alloc_shared((cols,), "bfloat16")
            red_ub = T.alloc_shared((64,), "float32")
            mu_ub = T.alloc_shared((1,), "float32")
            inv_ub = T.alloc_shared((1,), "float32")
            T.annotate_buffer_versions({x_ub: 2, xf_ub: 2, o_ub: 2})
            T.copy(w, w_ub)
            T.copy(b, b_ub)
            for step in T.Pipelined(T.ceildiv(rows, cores), num_stages=2):
                row = step * cores + core
                if row < rows:
                    base = T.cast(row * cols, "int32")
                    T.copy(x[base : base + cols], x_ub)
                    # VEC A: xf = f32(x) (staged for passes B/C), mean accumulator
                    with T.SimdVF():
                        m = T.simd.pset(32)
                        acc = T.simd.alloc_local((1,), "float32")
                        acc[0] = T.simd.vdup(0.0, "float32", m)
                        for i in range(cols // LANES):
                            xf = T.simd.vcvt(
                                T.simd.vld(x_ub[i * LANES], dist="UNPK_B16"),
                                "float32", part=0)
                            T.simd.vsts(xf_ub[i * LANES], xf)
                            acc[0] = T.simd.vadd(acc[0], xf, m)
                        T.simd.vsts(red_ub[0], acc[0])
                    with T.SimtVF(threads=WARP):
                        for lane in T.Parallel(WARP):
                            val = red_ub[lane] + red_ub[lane + 32]
                            total = T.warp_reduce_sum(val)
                            if lane == 0:
                                mu_ub[0] = total / T.float32(cols)
                    # VEC B: variance of the centered values (two-pass; no
                    # catastrophic cancellation)
                    with T.SimdVF():
                        m = T.simd.pset(32)
                        muv = T.simd.vdup(mu_ub[0], "float32", m)
                        acc = T.simd.alloc_local((1,), "float32")
                        acc[0] = T.simd.vdup(0.0, "float32", m)
                        for i in range(cols // LANES):
                            d = T.simd.vsub(T.simd.vld(xf_ub[i * LANES]), muv, m)
                            acc[0] = T.simd.vadd(
                                acc[0], T.simd.vmul(d, d, m), m)
                        T.simd.vsts(red_ub[0], acc[0])
                    with T.SimtVF(threads=WARP):
                        for lane in T.Parallel(WARP):
                            val = red_ub[lane] + red_ub[lane + 32]
                            total = T.warp_reduce_sum(val)
                            if lane == 0:
                                inv_ub[0] = _rsqrt_f32(total / T.float32(cols) + epsilon)
                    # VEC C: out = rne(((xf - mean) * rstd) * w + b), one RNE
                    with T.SimdVF():
                        m = T.simd.pset(32)
                        muv = T.simd.vdup(mu_ub[0], "float32", m)
                        invv = T.simd.vdup(inv_ub[0], "float32", m)
                        for i in range(cols // LANES):
                            xf = T.simd.vld(xf_ub[i * LANES])
                            wv = T.simd.vcvt(
                                T.simd.vld(w_ub[i * LANES], dist="UNPK_B16"),
                                "float32", part=0)
                            bv = T.simd.vcvt(
                                T.simd.vld(b_ub[i * LANES], dist="UNPK_B16"),
                                "float32", part=0)
                            d = T.simd.vsub(xf, muv, m)
                            t = T.simd.vmul(d, invv, m)
                            u = T.simd.vmul(t, wv, m)
                            T.simd.vsts(
                                o_ub[i * LANES],
                                T.simd.vcvt(T.simd.vadd(u, bv, m),
                                            "bfloat16", round="ROUND_R"),
                                dist="PK_B32")
                    T.copy(o_ub, out[base : base + cols])

    return main


def build_programs():
    yield "rmsnorm_full_c2048", rmsnorm_full_c2048()
    yield "layernorm_c1024", layernorm_c1024()
