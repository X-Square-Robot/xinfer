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

"""Vector residual-add RMS normalization with configurable staging and reduction.

Buffer count, rows per step and pipeline depth control staging. Rolling
reduction keeps lane partials in registers and schedules additions in the
same balanced-tree order. The residual sum is rounded to BF16 before it
contributes to the FP32 statistics. FP32 weights can be loaded directly."""
import tilelang.ascend.language as T  # noqa: E402

from . import rms_consistent as _orig  # noqa: E402  (the vendored baseline)

CORES = 64


def _tree_ops(n):
    """The released tree_statements association as an ordered op list
    (dst, a, b): every vadd of the balanced tree, carries as identities."""
    ops = []
    cur = [f"s{i}" for i in range(n)]
    level = 1
    while len(cur) > 1:
        pairs = [(cur[i], cur[i + 1]) for i in range(0, len(cur) - 1, 2)]
        outs = []
        for j, (a, b) in enumerate(pairs):
            dst = f"t{level}_{j}"
            ops.append((dst, a, b))
            outs.append(dst)
        if len(cur) % 2:
            outs.append(cur[-1])
        cur = outs
        level += 1
    return ops, cur[0]


def _roll_schedule(n):
    """Interleave the tree adds with segment production: after segment c's
    square q{c}, fire every op whose operands are both ready, in the released
    op order. Returns {c: [(dst, a, b)]} and the final name."""
    ops, final = _tree_ops(n)
    remaining = list(ops)

    def reg(name):
        return f"q{name[1:]}" if name.startswith("s") else name

    ready = set()
    schedule = {c: [] for c in range(n)}
    for c in range(n):
        ready.add(f"q{c}")
        progress = True
        while progress:
            progress = False
            for op in remaining:
                dst, a, b = op
                if reg(a) in ready and reg(b) in ready:
                    schedule[c].append((dst, a, b))
                    ready.add(dst)
                    remaining.remove(op)
                    progress = True
                    break
    assert not remaining, remaining
    assert len([o for lst in schedule.values() for o in lst]) == len(ops)
    return schedule, final


def _tree_statements_text(names, level):
    """Verbatim copy of the lib tree generator (same names, same statements)."""
    if len(names) == 1:
        return "", names[0]
    pairs = [(names[i], names[i + 1]) for i in range(0, len(names) - 1, 2)]
    outs = [f"t{level}_{j}" for j in range(len(pairs))]
    lines = [
        f"                            {outs[j]} = T.simd.vadd({a}, {b}, m)"
        for j, (a, b) in enumerate(pairs)
    ]
    carry = [names[-1]] if len(names) % 2 else []
    text = "\n".join(lines)
    if carry:
        text += f"\n                            c{level}_0 = {carry[0]}"
        outs.append(f"c{level}_0")
    rest, final = _tree_statements_text(outs, level + 1)
    return text + ("\n" + rest if rest else ""), final


def buffered_ub_rmsnorm(width=1280, rows_per_step=4, lanes=64, weight_mode="fold16",
         buf2=False, roll=False, sr_live=False, stages=2, f32in=False):
    W, R, L = width, rows_per_step, lanes
    nseg = W // L
    assert not sr_live or roll, "sr_live rides the roll structure"
    assert weight_mode in ("fold16", "f32direct")
    assert not f32in or weight_mode == "f32direct", "f32in rides f32direct weights"

    if roll:
        schedule, tree_final = _roll_schedule(nseg)
        body = []
        for c in range(nseg):
            o = f"r * W + {c} * L"
            if f32in:
                body.append(f"                            av = T.simd.vld(x_ub[{o}])")
                body.append(f"                            T.simd.vsts(")
                body.append(f"                                x16_ub[{o}],")
                body.append(f"                                T.simd.vcvt(av, \"bfloat16\", round=\"ROUND_R\"),")
                body.append(f"                                dist=\"PK_B32\",")
                body.append(f"                            )")
            body.append(f"                            xv = T.simd.vcvt(")
            if f32in:
                body.append(f"                                T.simd.vld(x16_ub[{o}], dist=\"UNPK_B16\"),")
            else:
                body.append(f"                                T.simd.vld(x_ub[{o}], dist=\"UNPK_B16\"),")
            body.append(f"                                \"float32\", part=0,")
            body.append(f"                            )")
            body.append(f"                            rv = T.simd.vcvt(")
            body.append(f"                                T.simd.vld(res_ub[{o}], dist=\"UNPK_B16\"),")
            body.append(f"                                \"float32\", part=0,")
            body.append(f"                            )")
            body.append(f"                            s = T.simd.vadd(xv, rv, m)")
            body.append(f"                            T.simd.vsts(")
            body.append(f"                                sum_ub[{o}],")
            body.append(f"                                T.simd.vcvt(s, \"bfloat16\", round=\"ROUND_R\"),")
            body.append(f"                                dist=\"PK_B32\",")
            body.append(f"                            )")
            sr_name = f"sr{c}"
            body.append(f"                            {sr_name} = T.simd.vcvt(")
            body.append(f"                                T.simd.vld(sum_ub[{o}], dist=\"UNPK_B16\"),")
            body.append(f"                                \"float32\", part=0,")
            body.append(f"                            )")
            body.append(f"                            q{c} = T.simd.vmul({sr_name}, {sr_name}, m)")
            for dst, a, b in schedule[c]:
                a = f"q{a[1:]}" if a.startswith("s") else a
                b = f"q{b[1:]}" if b.startswith("s") else b
                body.append(f"                            {dst} = T.simd.vadd({a}, {b}, m)")
        body.append(f"                            p1 = T.simd.vcpadd({tree_final}, m)")
        for i in range(2, 6):
            body.append(f"                            p{i} = T.simd.vcpadd(p{i - 1}, m)")
        body.append(f"                            total = T.simd.vdupv(T.simd.vcpadd(p5, m), m)")
        body.append(f"                            denom = T.simd.vadd(")
        body.append(f"                                T.simd.vmul(total, wide, m), eps_v, m")
        body.append(f"                            )")
        body.append(f"                            inv = T.simd.vdiv(one, T.simd.vsqrt(denom, m), m)")
        if sr_live:
            for c in range(nseg):
                o = f"r * W + {c} * L"
                if weight_mode == "fold16":
                    body.append(f"                            wv = T.simd.vcvt(")
                    body.append(f"                                T.simd.vld(w_raw[{c} * L], dist=\"UNPK_B16\"),")
                    body.append(f"                                \"float32\", part=0,")
                    body.append(f"                            )")
                else:
                    body.append(f"                            wv = T.simd.vld(w32_ub[{c} * L])")
                body.append(f"                            v = T.simd.vmul(T.simd.vmul(sr{c}, wv, m), inv, m)")
                body.append(f"                            T.simd.vsts(")
                body.append(f"                                out_ub[{o}],")
                body.append(f"                                T.simd.vcvt(v, \"bfloat16\", round=\"ROUND_R\"),")
                body.append(f"                                dist=\"PK_B32\",")
                body.append(f"                            )")
        else:
            body.append(f"                            for c in range(W // L):")
            body.append(f"                                o = r * W + c * L")
            body.append(f"                                sr = T.simd.vcvt(")
            body.append(f"                                    T.simd.vld(sum_ub[o], dist=\"UNPK_B16\"),")
            body.append(f"                                    \"float32\", part=0,")
            body.append(f"                                )")
            if weight_mode == "fold16":
                body.append(f"                                wv = T.simd.vcvt(")
                body.append(f"                                    T.simd.vld(w_raw[c * L], dist=\"UNPK_B16\"),")
                body.append(f"                                    \"float32\", part=0,")
                body.append(f"                                )")
            else:
                body.append(f"                                wv = T.simd.vld(w32_ub[c * L])")
            body.append(f"                                v = T.simd.vmul(T.simd.vmul(sr, wv, m), inv, m)")
            body.append(f"                                T.simd.vsts(")
            body.append(f"                                    out_ub[o],")
            body.append(f"                                    T.simd.vcvt(v, \"bfloat16\", round=\"ROUND_R\"),")
            body.append(f"                                    dist=\"PK_B32\",")
            body.append(f"                                )")
        row_body = "\n".join(body)
    else:
        loads = [f"s0_{c}" for c in range(nseg)]
        load_lines = "\n".join(
            f"                            s0_{c} = T.simd.vld(sqlane_ub[r * W + {c} * L])"
            for c in range(nseg)
        )
        tree_text, tree_final = _tree_statements_text(loads, 1)
        if weight_mode == "f32direct":
            wv_line = """wv = T.simd.vld(w32_ub[c * L])"""
        else:
            wv_line = """wv = T.simd.vcvt(
                                    T.simd.vld(w_raw[c * L], dist="UNPK_B16"),
                                    "float32", part=0,
                                )"""
        if f32in:
            xv_load = """av = T.simd.vld(x_ub[o])
                                T.simd.vsts(
                                    x16_ub[o],
                                    T.simd.vcvt(av, "bfloat16", round="ROUND_R"),
                                    dist="PK_B32",
                                )
                                xv = T.simd.vcvt(
                                    T.simd.vld(x16_ub[o], dist="UNPK_B16"),
                                    "float32", part=0,
                                )"""
        else:
            xv_load = """xv = T.simd.vcvt(
                                    T.simd.vld(x_ub[o], dist="UNPK_B16"),
                                    "float32", part=0,
                                )"""
        # Stage lane-wise squares for the balanced row reduction.
        row_body = f"""                            for c in range(W // L):
                                o = r * W + c * L
                                {xv_load}
                                rv = T.simd.vcvt(
                                    T.simd.vld(res_ub[o], dist="UNPK_B16"),
                                    "float32", part=0,
                                )
                                s = T.simd.vadd(xv, rv, m)
                                T.simd.vsts(
                                    sum_ub[o],
                                    T.simd.vcvt(s, "bfloat16", round="ROUND_R"),
                                    dist="PK_B32",
                                )
                                sr = T.simd.vcvt(
                                    T.simd.vld(sum_ub[o], dist="UNPK_B16"),
                                    "float32", part=0,
                                )
                                T.simd.vsts(
                                    sqlane_ub[o],
                                    T.simd.vmul(sr, sr, m),
                                )
{load_lines}
{tree_text}
                            p1 = T.simd.vcpadd({tree_final}, m)
                            p2 = T.simd.vcpadd(p1, m)
                            p3 = T.simd.vcpadd(p2, m)
                            p4 = T.simd.vcpadd(p3, m)
                            p5 = T.simd.vcpadd(p4, m)
                            total = T.simd.vdupv(T.simd.vcpadd(p5, m), m)
                            denom = T.simd.vadd(
                                T.simd.vmul(total, wide, m), eps_v, m
                            )
                            inv = T.simd.vdiv(one, T.simd.vsqrt(denom, m), m)
                            for c in range(W // L):
                                o = r * W + c * L
                                sr = T.simd.vcvt(
                                    T.simd.vld(sum_ub[o], dist="UNPK_B16"),
                                    "float32", part=0,
                                )
                                {wv_line}
                                v = T.simd.vmul(T.simd.vmul(sr, wv, m), inv, m)
                                T.simd.vsts(
                                    out_ub[o],
                                    T.simd.vcvt(v, "bfloat16", round="ROUND_R"),
                                    dist="PK_B32",
                                )"""

    if weight_mode == "f32direct":
        weight_decl = """    weight = T.match_buffer(wh, (W,), "float32")"""
        weight_bufs = """        w32_ub = T.alloc_shared((W,), "float32")"""
        weight_stage = """        T.copy(weight[:W], w32_ub)"""
        final_w = "w32_ub"
    else:
        weight_decl = """    weight = T.match_buffer(wh, (W,), "bfloat16")"""
        weight_bufs = """        w_raw = T.alloc_shared((W,), "bfloat16")"""
        weight_stage = """        T.copy(weight, w_raw)"""
        final_w = "w_raw"
    _ = final_w  # pass D text already branches on weight_mode above

    sqlane_alloc = "" if roll else """        sqlane_ub = T.alloc_shared((R * W,), "float32")"""
    sqlane_ver = "" if roll else ", sqlane_ub: 1"
    sum_v = 2 if buf2 else 1
    out_v = 2 if buf2 else 1

    # The generated row body nests one level less deeply in this template.
    row_body = "\n".join(
        line[4:] if line.startswith("    ") else line for line in row_body.split("\n")
    )

    x_dtype = "float32" if f32in else "bfloat16"
    x_ub_line = (f"        x_ub = T.alloc_shared((R * W,), \"{x_dtype}\")")
    x16_alloc = "" if not f32in else """        x16_ub = T.alloc_shared((R * W,), "bfloat16")"""
    x16_ver = "" if not f32in else ", x16_ub: 1"

    template = f"""@T.prim_func
def main(
    xh: T.handle,
    rh: T.handle,
    sh: T.handle,
    wh: T.handle,
    yh: T.handle,
    rows: T.int64,
    weight_words: T.int64,
    epsilon_bits: T.int64,
):
    x = T.match_buffer(xh, (rows * W,), "{x_dtype}")
    residual = T.match_buffer(rh, (rows * W,), "bfloat16")
    summed = T.match_buffer(sh, (rows * W,), "bfloat16")
{weight_decl}
    output = T.match_buffer(yh, (rows * W,), "bfloat16")
    epsilon = T.reinterpret("float32", T.cast(epsilon_bits, "uint32"))
    with T.Kernel(CORES) as core:
{x_ub_line}
        res_ub = T.alloc_shared((R * W,), "bfloat16")
        sum_ub = T.alloc_shared((R * W,), "bfloat16")
        out_ub = T.alloc_shared((R * W,), "bfloat16")
{x16_alloc}
{weight_bufs}
{sqlane_alloc}
        T.annotate_buffer_versions({{x_ub: {stages}, res_ub: {stages}, sum_ub: {sum_v}, out_ub: {out_v}{sqlane_ver}{x16_ver}}})
{weight_stage}
        for step in T.Pipelined(T.ceildiv(rows, CORES * R), num_stages={stages}):
            r0 = (step * CORES + core) * R
            if r0 < rows:
                base = r0 * W
                T.copy(x[base : base + R * W], x_ub)
                T.copy(residual[base : base + R * W], res_ub)
                with T.SimdVF():
                    m = T.simd.pset(32)
                    one = T.simd.vdup(1.0, "float32")
                    wide = T.simd.vdup(1.0 / W, "float32")
                    eps_v = T.simd.vdup(epsilon, "float32")
                    for r in range(R):
{row_body}
                T.copy(sum_ub, summed[base : base + R * W])
                T.copy(out_ub, output[base : base + R * W])
"""

    namespace = {"T": T, "W": W, "R": R, "L": L, "CORES": CORES, "WM": weight_mode}
    source_name = f"<buffered_ub_rmsnorm-{width}-{R}-{weight_mode}-{'roll' if roll else 'tree'}>"
    import linecache
    linecache.cache[source_name] = (
        len(template), None,
        [line + chr(10) for line in template.split(chr(10))], source_name
    )
    exec(compile(template, source_name, "exec"), namespace)
    return namespace["main"]


def build_programs():
    # -- baseline: the verbatim lib program (same source the lib built from)
    yield "rmsnorm_vector_h1280_folded_weight", _orig.ubv(1280, weight_mode="fold16")
    # -- scheduling only: MTE3 double-buffered stores
    yield "rmsnorm_vector_h1280_double_buffer", buffered_ub_rmsnorm(buf2=True)
    # -- register-resident inline reduction (same adds, no sqlane round trip)
    yield "rmsnorm_vector_h1280_rolling_reduce", buffered_ub_rmsnorm(roll=True)
    yield "rmsnorm_vector_h1280_double_buffer_rolling_reduce", buffered_ub_rmsnorm(buf2=True, roll=True)
    # -- pass D reuses the pass-A registers
    yield "rmsnorm_vector_h1280_rolling_reduce_keep_sum", buffered_ub_rmsnorm(roll=True, sr_live=True)
    yield "rmsnorm_vector_h1280_double_buffer_rolling_reduce_keep_sum", buffered_ub_rmsnorm(buf2=True, roll=True, sr_live=True)
    # -- knobs on the leading structure
    yield "rmsnorm_vector_h1280_double_buffer_rolling_reduce_keep_sum_rows2", buffered_ub_rmsnorm(buf2=True, roll=True, sr_live=True,
                                            rows_per_step=2)
    yield "rmsnorm_vector_h1280_double_buffer_rolling_reduce_keep_sum_fp32_weight", buffered_ub_rmsnorm(buf2=True, roll=True, sr_live=True,
                                           weight_mode="f32direct")
    yield "rmsnorm_vector_h1280_double_buffer_rolling_reduce_keep_sum_stages3", buffered_ub_rmsnorm(buf2=True, roll=True, sr_live=True,
                                            stages=3)
    # Double-buffered stores with direct weights, deeper staging or more rows per step.
    yield "rmsnorm_vector_h1280_double_buffer_fp32_weight", buffered_ub_rmsnorm(buf2=True, weight_mode="f32direct")
    yield "rmsnorm_vector_h1280_double_buffer_stages3", buffered_ub_rmsnorm(buf2=True, stages=3)
    yield "rmsnorm_vector_h1280_double_buffer_rows8", buffered_ub_rmsnorm(buf2=True, rows_per_step=8)
    yield "rmsnorm_vector_h1280_double_buffer_rows8_fp32_weight", buffered_ub_rmsnorm(buf2=True, rows_per_step=8,
                                      weight_mode="f32direct")
    # -- side pool: prefill h2048 f32in (the gemm FP32 accumulator feeds the
    #    plan directly; the RNE round to BF16 happens in-kernel via the
    #    PK-store + UNPK-reload round trip, bit-identical to the gemm_epilogue
    #    finish cast followed by the ubv plan -- the m16_f32in semantics)
    yield "rmsnorm_vector_h2048_baseline", _orig.ubv(2048)
    yield "rmsnorm_vector_h2048_fp32_input", buffered_ub_rmsnorm(width=2048, weight_mode="f32direct",
                                    buf2=True, f32in=True)
