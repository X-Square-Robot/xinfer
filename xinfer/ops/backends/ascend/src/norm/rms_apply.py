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

"""Apply a supplied per-row FP32 reciprocal RMS to H2048 BF16 input.
Compute (x * rstd) * (1 + weight) in FP32 vector lanes and round once to
BF16 on store. The caller owns the statistics reduction."""

import tilelang.ascend.language as T

WIDTH = 2048
CORES = 64
ROWS = 4
LANES = 64


def apply_rstd(width=WIDTH):
    W = width
    R = ROWS
    L = LANES

    body_lines = []
    for c in range(W // L):
        off = f"r * {W} + {c * L}"
        body_lines.append(f's0_{c} = T.simd.vld(x_ub[{off}], dist="UNPK_B16")')
    apply_lines = []
    for c in range(W // L):
        apply_lines.append(f'sr_{c} = T.simd.vcvt(s0_{c}, "float32", part=0)')
        apply_lines.append(f'wv_{c} = T.simd.vld(w32_ub[{c * L}])')
        apply_lines.append(
            f'v_{c} = T.simd.vmul(T.simd.vmul(sr_{c}, inv, m), wv_{c}, m)')
    store_lines = []
    for c in range(W // L):
        store_lines.append(
            f'T.simd.vsts(out_ub[r * {W} + {c * L}],'
            f' T.simd.vcvt(v_{c}, "bfloat16", round="ROUND_R"), dist="PK_B32")')
    loads_x = "\n".join(f"                            {ln}" for ln in body_lines)
    applies_text = "\n".join(f"                            {ln}" for ln in apply_lines)
    stores_text = "\n".join(f"                            {ln}" for ln in store_lines)

    template = f"""
    @T.prim_func
    def main(
        xh: T.handle,
        rh: T.handle,
        sh: T.handle,
        wh: T.handle,
        yh: T.handle,
        ih: T.handle,
        rows: T.int64,
        x_stride: T.int64,
        residual_stride: T.int64,
        sum_stride: T.int64,
        output_stride: T.int64,
        stats_stride: T.int64,
        weight_words: T.int64,
        epsilon_bits: T.int64,
        centered: T.int64,
        add_residual: T.int64,
        write_output: T.int64,
        write_stats: T.int64,
    ):
        x = T.match_buffer(xh, (rows * x_stride,), "bfloat16")
        residual = T.match_buffer(rh, (rows * residual_stride,), "bfloat16")
        summed = T.match_buffer(sh, (rows * sum_stride,), "bfloat16")
        weight = T.match_buffer(wh, (W * weight_words,), "uint16")
        output = T.match_buffer(yh, (rows * output_stride,), "bfloat16")
        stats = T.match_buffer(ih, (rows, stats_stride), "float32")
        with T.Kernel(CORES) as core:
            x_ub = T.alloc_shared((R * W,), "bfloat16")
            out_ub = T.alloc_shared((R * W,), "bfloat16")
            w_raw = T.alloc_shared((W * 2,), "uint16")
            w32_ub = T.alloc_shared((W,), "float32")
            r_ub = T.alloc_shared((R,), "float32")
            T.annotate_buffer_versions({{x_ub: 2, out_ub: 1}})
            if weight_words == 1:
                T.copy(weight[:W], w_raw[:W])
            else:
                T.copy(weight[: W * 2], w_raw)
            with T.SimtVF(threads=256):
                for col in T.Parallel(W):
                    if weight_words == 1:
                        w32_ub[col] = T.reinterpret(
                            "float32", T.cast(w_raw[col], "uint32") << 16
                        )
                    else:
                        w32_ub[col] = T.reinterpret(
                            "float32",
                            T.cast(w_raw[col * 2], "uint32")
                            | (T.cast(w_raw[col * 2 + 1], "uint32") << 16),
                        )
                    if centered != 0:
                        w32_ub[col] = w32_ub[col] + T.float32(1)
            for step in T.Pipelined(T.ceildiv(rows, CORES * R), num_stages=2):
                r0 = (step * CORES + core) * R
                if r0 < rows:
                    base = r0 * W
                    T.copy(x[base : base + R * W], x_ub)
                    with T.SimtVF(threads=256):
                        for r in T.Parallel(R):
                            r_ub[r] = stats[r0 + r, 0]
                    with T.SimdVF():
                        m = T.simd.pset(32)
                        for r in range(R):
                            inv = T.simd.vdup(r_ub[r], "float32")
{loads_x}
{applies_text}
{stores_text}
                    T.copy(out_ub, output[base : base + R * W])
"""

    template = chr(10).join(
        line[4:] if line.startswith("    ") else line for line in template.split(chr(10))
    )
    namespace = {"T": T, "W": W, "R": R, "L": L, "CORES": CORES}
    source_name = f"<rms-apply-vec-{width}>"
    import linecache
    linecache.cache[source_name] = (
        len(template), None,
        [line + chr(10) for line in template.split(chr(10))], source_name
    )
    exec(compile(template, source_name, "exec"), namespace)
    return namespace["main"]


def build_programs():
    yield "rms_apply_rstd_h2048", apply_rstd(2048)
