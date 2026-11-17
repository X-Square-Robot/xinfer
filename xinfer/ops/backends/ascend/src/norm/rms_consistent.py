"""Private H1280 BF16 RMS plan shared by plain/add/saved/recomputed paths."""

# Kernel and SIMT scopes are separate constructs.
# ruff: noqa: SIM117
import tilelang.ascend.language as T

WIDTH = 1280
TILE = 2048
THREADS = 256
CORES = 64
ROWS = 4


def weight_value(weight, column, words):
    return T.if_then_else(
        words == 1,
        T.reinterpret("float32", T.cast(weight[column], "uint32") << 16),
        T.reinterpret(
            "float32",
            T.cast(weight[column * 2], "uint32")
            | (T.cast(weight[column * 2 + 1], "uint32") << 16),
        ),
    )


def generic(width=WIDTH):
    tile = 1024 if width == 1024 else TILE

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
        x = T.match_buffer(xh, (rows, x_stride), "bfloat16")
        residual = T.match_buffer(rh, (rows, residual_stride), "bfloat16")
        summed = T.match_buffer(sh, (rows, sum_stride), "bfloat16")
        weight = T.match_buffer(wh, (width * weight_words,), "uint16")
        output = T.match_buffer(yh, (rows, output_stride), "bfloat16")
        stats = T.match_buffer(ih, (rows, stats_stride), "float32")
        epsilon = T.reinterpret("float32", T.cast(epsilon_bits, "uint32"))
        blocks = T.min(rows, CORES)
        with T.Kernel(blocks) as core:
            with T.SimtVF(threads=THREADS):
                for step in T.serial(T.ceildiv(rows, blocks)):
                    row = step * blocks + core
                    if row < rows:
                        value = T.alloc_fragment((tile,), "float32")
                        for col in T.Parallel(tile, coalesced_width=T.int32(1)):
                            value[col] = T.float32(0)
                            if col < width:
                                value[col] = T.cast(x[row, col], "float32")
                                if add_residual != 0:
                                    value[col] = T.cast(
                                        T.cast(
                                            value[col]
                                            + T.cast(residual[row, col], "float32"),
                                            "bfloat16",
                                        ),
                                        "float32",
                                    )
                                    summed[row, col] = T.cast(value[col], "bfloat16")
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
                                stats[row, 0] = inverse[i]
                        if write_output != 0:
                            for col in T.Parallel(tile, coalesced_width=T.int32(1)):
                                if col < width:
                                    scale = T.alloc_var(
                                        "float32",
                                        init=weight_value(weight, col, weight_words),
                                    )
                                    if centered != 0:
                                        scale = scale + T.float32(1)
                                    output[row, col] = T.cast(
                                        value[col] * scale * inverse[0], "bfloat16"
                                    )

    return main


def generic_f32in(width=WIDTH):
    """r3 (agent H): the K1-class fused cast+add+rms form of the generic plan.

    ``x`` arrives as the FP32 mm output; the kernel rounds it to BF16 first
    (RNE -- bit-identical to the chain's separate ``mm32.to(bf16)`` cast
    kernel) and then runs the released add+rms math unchanged, so the chain
    ``[cast kernel] + [add_rmsnorm]`` collapses into one launch.
    """
    tile = 1024 if width == 1024 else TILE

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
        x = T.match_buffer(xh, (rows, x_stride), "float32")
        residual = T.match_buffer(rh, (rows, residual_stride), "bfloat16")
        summed = T.match_buffer(sh, (rows, sum_stride), "bfloat16")
        weight = T.match_buffer(wh, (width * weight_words,), "uint16")
        output = T.match_buffer(yh, (rows, output_stride), "bfloat16")
        stats = T.match_buffer(ih, (rows, stats_stride), "float32")
        epsilon = T.reinterpret("float32", T.cast(epsilon_bits, "uint32"))
        blocks = T.min(rows, CORES)
        with T.Kernel(blocks) as core:
            with T.SimtVF(threads=THREADS):
                for step in T.serial(T.ceildiv(rows, blocks)):
                    row = step * blocks + core
                    if row < rows:
                        value = T.alloc_fragment((tile,), "float32")
                        for col in T.Parallel(tile, coalesced_width=T.int32(1)):
                            value[col] = T.float32(0)
                            if col < width:
                                # RNE to BF16 first: identical bits to the
                                # chain's separate cast of the mm output.
                                value[col] = T.cast(
                                    T.cast(x[row, col], "bfloat16"), "float32"
                                )
                                if add_residual != 0:
                                    value[col] = T.cast(
                                        T.cast(
                                            value[col]
                                            + T.cast(residual[row, col], "float32"),
                                            "bfloat16",
                                        ),
                                        "float32",
                                    )
                                    summed[row, col] = T.cast(value[col], "bfloat16")
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
                                stats[row, 0] = inverse[i]
                        if write_output != 0:
                            for col in T.Parallel(tile, coalesced_width=T.int32(1)):
                                if col < width:
                                    scale = T.alloc_var(
                                        "float32",
                                        init=weight_value(weight, col, weight_words),
                                    )
                                    if centered != 0:
                                        scale = scale + T.float32(1)
                                    output[row, col] = T.cast(
                                        value[col] * scale * inverse[0], "bfloat16"
                                    )

    return main


def ub(width=WIDTH):
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
        weight = T.match_buffer(wh, (width * weight_words,), "uint16")
        output = T.match_buffer(yh, (rows * output_stride,), "bfloat16")
        stats = T.match_buffer(ih, (rows, stats_stride), "float32")
        epsilon = T.reinterpret("float32", T.cast(epsilon_bits, "uint32"))
        with T.Kernel(CORES) as core:
            x_ub = T.alloc_shared((ROWS * width,), "bfloat16")
            residual_ub = T.alloc_shared((ROWS * width,), "bfloat16")
            sum_ub = T.alloc_shared((ROWS * width,), "bfloat16")
            output_ub = T.alloc_shared((ROWS * width,), "bfloat16")
            weight_ub = T.alloc_shared((width * 2,), "uint16")
            T.annotate_buffer_versions(
                {x_ub: 2, residual_ub: 2, sum_ub: 1, output_ub: 1}
            )
            if weight_words == 1:
                T.copy(weight[:width], weight_ub[:width])
            else:
                T.copy(weight[: width * 2], weight_ub)
            for step in T.Pipelined(T.ceildiv(rows, CORES * ROWS), num_stages=2):
                row = (step * CORES + core) * ROWS
                if row < rows:
                    base = row * width
                    T.copy(x[base : base + ROWS * width], x_ub)
                    if add_residual != 0:
                        T.copy(residual[base : base + ROWS * width], residual_ub)
                    with T.SimtVF(threads=THREADS):
                        value = T.alloc_fragment((ROWS, TILE), "float32")
                        for r, col in T.Parallel(
                            ROWS, TILE, coalesced_width=T.int32(1)
                        ):
                            value[r, col] = T.float32(0)
                            if col < width:
                                value[r, col] = T.cast(x_ub[r * width + col], "float32")
                                if add_residual != 0:
                                    value[r, col] = T.cast(
                                        T.cast(
                                            value[r, col]
                                            + T.cast(
                                                residual_ub[r * width + col], "float32"
                                            ),
                                            "bfloat16",
                                        ),
                                        "float32",
                                    )
                                    sum_ub[r * width + col] = T.cast(
                                        value[r, col], "bfloat16"
                                    )
                        total = T.alloc_reducer((ROWS,), "float32", op="sum")
                        T.reducer_init(total)
                        for r, col in T.Parallel(
                            ROWS, TILE, coalesced_width=T.int32(1)
                        ):
                            T.reducer_update(total[r], value[r, col] * value[r, col])
                        inverse = T.alloc_fragment((ROWS,), "float32")
                        T.finalize_reducer(total, inverse)
                        for r in T.Parallel(ROWS):
                            inverse[r] = T.rsqrt(
                                inverse[r] / T.float32(width) + epsilon
                            )
                            if write_stats != 0 and T.get_thread_binding() == 0:
                                stats[row + r, 0] = inverse[r]
                        if write_output != 0:
                            for r, col in T.Parallel(
                                ROWS, TILE, coalesced_width=T.int32(1)
                            ):
                                if col < width:
                                    scale = T.alloc_var(
                                        "float32",
                                        init=weight_value(weight_ub, col, weight_words),
                                    )
                                    if centered != 0:
                                        scale = scale + T.float32(1)
                                    output_ub[r * width + col] = T.cast(
                                        value[r, col] * scale * inverse[r], "bfloat16"
                                    )
                    if add_residual != 0:
                        T.copy(sum_ub, summed[base : base + ROWS * width])
                    if write_output != 0:
                        T.copy(output_ub, output[base : base + ROWS * width])

    return main




def ubv(width=WIDTH, rows_per_step=4, lanes=64, weight_mode="staged"):
    """Vector-engine add plan: DMA rows, CCE-vector elementwise, lane partials.

    The ub() plan runs its elementwise math as scalar SIMT code over the staged
    rows (~0.5 TB/s effective on the wall_oss vision rows). This plan keeps the
    same GM<->UB staging but moves the elementwise work onto the CCE vector
    units: each 64-column segment unpacks x/residual, adds, rounds the sum to
    BF16 (PK store) and stores the squares of the re-upcast sum as one FP32
    lane vector; a balanced vadd tree over the segment vectors (explicit
    distinct-name statements -- the eager tracer rejects list indexing inside
    the traced body) plus a vcpadd chain and vdupv broadcast produce the row
    total / rsqrt-class inverse (hardware vsqrt + the same correctly-rounded
    vdiv the production GLU finish uses) without any cross-unit bridge; the
    final pass multiplies the staged sum by weight * inverse with one BF16
    round. The sum write stays bitwise (elementwise FP32 add + RNE); the norm
    keeps the 2-BF16-ulp class of the released plans (the lane-partial tree
    reorders the reduction). add_residual=1, write_output=1, no stats, not
    centered -- exactly the inference add_rms_norm call.
    """
    W = width
    R = rows_per_step
    L = lanes

    def tree_statements(names, level):
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
        rest, final = tree_statements(outs, level + 1)
        return text + ("\n" + rest if rest else ""), final

    loads = [f"s0_{c}" for c in range(W // L)]
    load_lines = "\n".join(
        f"                            s0_{c} = T.simd.vld(sqlane_ub[r * W + {c} * L])"
        for c in range(W // L)
    )
    tree_text, tree_final = tree_statements(loads, 1)

    template = f"""
    @T.prim_func
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
        x = T.match_buffer(xh, (rows * W,), "bfloat16")
        residual = T.match_buffer(rh, (rows * W,), "bfloat16")
        summed = T.match_buffer(sh, (rows * W,), "bfloat16")
        if WM == "staged":
            weight = T.match_buffer(wh, (W * 2,), "uint16")
        elif WM == "f32direct":
            weight = T.match_buffer(wh, (W,), "float32")
        else:
            weight = T.match_buffer(wh, (W,), "bfloat16")
        output = T.match_buffer(yh, (rows * W,), "bfloat16")
        epsilon = T.reinterpret("float32", T.cast(epsilon_bits, "uint32"))
        with T.Kernel(CORES) as core:
            x_ub = T.alloc_shared((R * W,), "bfloat16")
            res_ub = T.alloc_shared((R * W,), "bfloat16")
            sum_ub = T.alloc_shared((R * W,), "bfloat16")
            out_ub = T.alloc_shared((R * W,), "bfloat16")
            if WM == "staged":
                w_raw = T.alloc_shared((W * 2,), "uint16")
                w32_ub = T.alloc_shared((W,), "float32")
            elif WM == "f32direct":
                w32_ub = T.alloc_shared((W,), "float32")
            else:
                w_raw = T.alloc_shared((W,), "bfloat16")
            sqlane_ub = T.alloc_shared((R * W,), "float32")
            T.annotate_buffer_versions({{x_ub: 2, res_ub: 2, sum_ub: 1, out_ub: 1}})
            if WM == "staged":
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
            elif WM == "f32direct":
                T.copy(weight[:W], w32_ub)
            else:
                T.copy(weight, w_raw)
            for step in T.Pipelined(T.ceildiv(rows, CORES * R), num_stages=2):
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
                            for c in range(W // L):
                                o = r * W + c * L
                                xv = T.simd.vcvt(
                                    T.simd.vld(x_ub[o], dist="UNPK_B16"),
                                    "float32", part=0,
                                )
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
                                if WM == "fold16":
                                    wv = T.simd.vcvt(
                                        T.simd.vld(w_raw[c * L], dist="UNPK_B16"),
                                        "float32", part=0,
                                    )
                                else:
                                    wv = T.simd.vld(w32_ub[c * L])
                                v = T.simd.vmul(T.simd.vmul(sr, wv, m), inv, m)
                                T.simd.vsts(
                                    out_ub[o],
                                    T.simd.vcvt(v, "bfloat16", round="ROUND_R"),
                                    dist="PK_B32",
                                )
                    T.copy(sum_ub, summed[base : base + R * W])
                    T.copy(out_ub, output[base : base + R * W])
"""

    template = chr(10).join(
        line[4:] if line.startswith("    ") else line for line in template.split(chr(10))
    )
    namespace = {"T": T, "W": W, "R": R, "L": L, "CORES": CORES, "WM": weight_mode}
    source_name = f"<ubv-{width}>"
    import linecache
    linecache.cache[source_name] = (
        len(template), None,
        [line + chr(10) for line in template.split(chr(10))], source_name
    )
    exec(compile(template, source_name, "exec"), namespace)
    return namespace["main"]


def build_programs():
    yield "rms_consistent_generic_h1280", generic()
    yield "rms_consistent_ub_h1280", ub()
    yield "rms_consistent_generic_h1024", generic(1024)
    yield "rms_consistent_generic_h1024_f32in", generic_f32in(1024)
    yield "rms_consistent_ub_h2048", ub(2048)
    yield "rms_consistent_ubv_h1280", ubv(weight_mode="fold16")
    yield "rms_consistent_ubv_h2048", ubv(2048)
