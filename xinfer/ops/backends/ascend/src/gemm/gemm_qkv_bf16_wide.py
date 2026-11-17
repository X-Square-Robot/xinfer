"""Wide-tile BF16 QKV Cube kernels and the fused standard-order RoPE finish.

`cube_bf16_kn128x256` multiplies x [M, K] by a persistent KN weight [K, N]:
the GM->L1 weight copy lands already transposed into the [N, K] tile the NT
Cube consumes, so the large element-wise pad-transpose of the old path is
gone. `qkv_std_rope_finish` reads the FP32 accumulator, adds the FP32
bias and applies rotate-half RoPE from a pairs4 table in FP32 with a single
final BF16 store; columns at and after ``qk_width`` take the projection only.
"""

import tilelang.ascend.language as T

from .silu_and_mul import _store_rne


def cube_bf16_kn128x256():
    """accum [M, N] = x [M, K] @ weight [K, N]; M % 128 == N % 256 == K % 128 == 0."""

    @T.prim_func
    def main(
        x_handle: T.handle,
        weight_handle: T.handle,
        accum_handle: T.handle,
        rows: T.int64,
        columns: T.int64,
        reduction: T.int64,
    ):
        x = T.match_buffer(x_handle, (rows, reduction), "bfloat16")
        weight = T.match_buffer(weight_handle, (reduction, columns), "bfloat16")
        accum = T.match_buffer(accum_handle, (rows, columns), "float32")
        with T.Kernel((rows // 128) * (columns // 256)) as block:
            acc = T.alloc_l0c((128, 256), "float32")
            x_l1 = T.alloc_l1((128, 128), "bfloat16")
            w_l1 = T.alloc_l1((256, 128), "bfloat16")
            n_tiles = columns // 256
            k_tiles = reduction // 128
            m0 = T.cast((block // n_tiles) * 128, "int32")
            n0 = T.cast((block % n_tiles) * 256, "int32")
            for kt in T.Pipelined(k_tiles, num_stages=2):
                k0 = T.cast(kt * 128, "int32")
                T.copy(x[m0 : m0 + 128, k0 : k0 + 128], x_l1)
                T.copy(weight[k0 : k0 + 128, n0 : n0 + 256], w_l1, transpose=True)
                T.gemm(
                    x_l1,
                    w_l1,
                    acc,
                    transpose_B=True,
                    clear_accum=kt == 0,
                    unit_flag_ctrl=T.Select(kt == k_tiles - 1, 3, 2),
                )
            T.copy(acc, accum[m0 : m0 + 128, n0 : n0 + 256], unit_flag_ctrl=3)

    return main


def cube_bf16_kn_rope_wide128x256():
    """Fused wide NT Cube + standard RoPE epilogue (mode-5 QKV: N=3072,
    K=1024, pairs4 table [16, M, 4], M % 128 == 0).

    Replaces the gemm_cube_bf16_wide128x256 + gemm_qkv_std_rope_finish
    pair for 128-aligned row blocks: the FP32 accumulator never touches GM
    (saves the M*3072*4 write + read plus one launch). The weight is the
    RAW NK [N, K] buffer exactly as the launcher's contract check pins it
    (weight[n0:n0+256, k0:k0+128] lands directly in the [256, 128] L1 tile -
    the cube_wide_f16 pattern; gamma-1's first cut matched the weight as a
    prepacked KN [K, N] slab, misreading the raw buffer and failing the row
    bound gate at ~6769). The epilogue rides the
    two-AIV N-split dual copy (E's deep_stream_gate pattern): each AIV runs
    the finish math - identical formulas and element order - over its
    128-column half. Rotate partners stay local (the pair offset is +-32
    inside one 64-column group, and 2048 is a multiple of both 256 and the
    128-column AIV split, so the rope/V boundary is block-uniform); the bias
    rides the 1-D N-split dual (gate precedent), the pairs4 slice is a plain
    per-AIV full load staged per 64-row chunk, and the BF16 tile returns
    through the N-split UB->GM dual store (il_residual precedent)."""

    @T.prim_func
    def main(
        x_handle: T.handle,
        weight_handle: T.handle,
        bias_handle: T.handle,
        table_handle: T.handle,
        output_handle: T.handle,
        rows: T.int64,
        columns: T.int64,
        reduction: T.int64,
    ):
        x = T.match_buffer(x_handle, (rows, reduction), "bfloat16")
        weight = T.match_buffer(weight_handle, (columns, reduction), "bfloat16")
        bias = T.match_buffer(bias_handle, (columns,), "bfloat16")
        table = T.match_buffer(table_handle, (16, rows * 4), "float32")
        output = T.match_buffer(output_handle, (rows, columns), "bfloat16")
        with T.Kernel((rows // 128) * (columns // 256)) as block:
            acc = T.alloc_l0c((128, 256), "float32")
            x_l1 = T.alloc_l1((128, 128), "bfloat16")
            w_l1 = T.alloc_l1((256, 128), "bfloat16")
            accum_ub = T.alloc_shared((128, 128), "float32")
            bias_ub = T.alloc_shared((128,), "bfloat16")
            cs_ub = T.alloc_shared((16, 256), "float32")
            out_ub = T.alloc_shared((128, 128), "bfloat16")
            n_tiles = columns // 256
            k_tiles = reduction // 128
            m0 = T.cast((block // n_tiles) * 128, "int32")
            n0 = T.cast((block % n_tiles) * 256, "int32")
            for kt in T.Pipelined(k_tiles, num_stages=2):
                k0 = T.cast(kt * 128, "int32")
                T.copy(x[m0 : m0 + 128, k0 : k0 + 128], x_l1)
                T.copy(weight[n0 : n0 + 256, k0 : k0 + 128], w_l1)
                T.gemm(
                    x_l1,
                    w_l1,
                    acc,
                    transpose_B=True,
                    clear_accum=kt == 0,
                    unit_flag_ctrl=T.Select(kt == k_tiles - 1, 3, 2),
                )
            T.dual_copy(acc, accum_ub, unit_flag_ctrl=3)
            T.dual_copy(bias[n0 : n0 + 256], bias_ub)
            for chunk in T.serial(2):
                row0 = m0 + chunk * 64
                T.copy(table[:, 4 * row0 : 4 * row0 + 256], cs_ub)
                with T.SimtVF(threads=128):
                    for i, c in T.Parallel(64, 128):
                        value = T.alloc_var(
                            "float32",
                            init=accum_ub[chunk * 64 + i, c]
                            + T.cast(bias_ub[c], "float32"),
                        )
                        if n0 < 2048:
                            head_col = c % 64
                            pair = T.Select(head_col < 32, head_col, head_col - 32)
                            partner_c = T.Select(head_col < 32, c + 32, c - 32)
                            partner = T.alloc_var(
                                "float32",
                                init=accum_ub[chunk * 64 + i, partner_c]
                                + T.cast(bias_ub[partner_c], "float32"),
                            )
                            offset = (pair % 2) * 2
                            cosine = cs_ub[pair // 2, 4 * i + offset]
                            sine = cs_ub[pair // 2, 4 * i + offset + 1]
                            value = T.Select(
                                head_col < 32,
                                value * cosine - partner * sine,
                                value * cosine + partner * sine,
                            )
                        out_ub[chunk * 64 + i, c] = T.cast(value, "bfloat16")
            T.dual_copy(out_ub, output[m0 : m0 + 128, n0 : n0 + 256])

    return main


def qkv_std_rope_finish():
    """out = bf16(rotate_half(accum + bias)) on [0, 2048), bf16(accum + bias) after.

    Specialized for the wide-cube QKV shapes the launcher pins (columns 3072,
    qk_width 2048, rotary 64, pairs4 table [16, rows, 4]). Follows the
    paired_rope_ub structure that carries the hd80 wins: a fixed 64-core grid
    (one wave on the device, no per-block launch latency), one-row tiles DMA'd
    GM->UB with a two-stage pipeline, elementwise math from UB, one DMA store
    (row tiles keep every M in [1, 4096] legal without partial-tile writes).
    The flat per-element kernels (and the first chunk-per-block rewrite) were
    bounded by block dispatch / per-block load latency at ~21 ns x 49K blocks,
    not by bandwidth.
    """

    cores, width = 64, 3072

    @T.prim_func
    def main(
        output_handle: T.handle,
        accum_handle: T.handle,
        bias_handle: T.handle,
        table_handle: T.handle,
        rows: T.int64,
        columns: T.int64,
        qk_width: T.int64,
        rotary_dim: T.int64,
    ):
        output = T.match_buffer(output_handle, (rows, width), "bfloat16")
        accum = T.match_buffer(accum_handle, (rows, width), "float32")
        bias = T.match_buffer(bias_handle, (width,), "bfloat16")
        table = T.match_buffer(table_handle, (16, rows * 4), "float32")
        with T.Kernel(cores) as core:
            accum_ub = T.alloc_shared((1, width), "float32")
            cs_ub = T.alloc_shared((16, 4), "float32")
            bias_ub = T.alloc_shared((width,), "bfloat16")
            out_ub = T.alloc_shared((1, width), "bfloat16")
            T.annotate_buffer_versions({accum_ub: 2, cs_ub: 2, bias_ub: 1, out_ub: 1})
            T.copy(bias, bias_ub)
            for step in T.Pipelined(rows, num_stages=2):
                block = step * cores + core
                if block < rows:
                    start = T.cast(block, "int32")
                    T.copy(accum[start : start + 1, :], accum_ub)
                    T.copy(table[:, 4 * start : 4 * start + 4], cs_ub)
                    with T.SimtVF(threads=256):
                        for col in T.Parallel(width):
                            value = T.alloc_var(
                                "float32",
                                init=accum_ub[0, col]
                                + T.cast(bias_ub[col], "float32"),
                            )
                            if col < 2048:
                                head_col = col % 64
                                pair = T.Select(head_col < 32, head_col, head_col - 32)
                                partner_col = T.Select(
                                    head_col < 32, col + 32, col - 32
                                )
                                partner = (
                                    accum_ub[0, partner_col]
                                    + T.cast(bias_ub[partner_col], "float32")
                                )
                                offset = (pair % 2) * 2
                                c = cs_ub[pair // 2, offset]
                                s = cs_ub[pair // 2, offset + 1]
                                value = T.Select(
                                    head_col < 32,
                                    value * c - partner * s,
                                    value * c + partner * s,
                                )
                            out_ub[0, col] = T.cast(value, "bfloat16")
                    T.copy(out_ub, output[start : start + 1, :])

    return main


def qkv_compat_rope_finish():
    """hd64 compatibility finish (gemm_compat kind 3, N=3072, rotary 64).

    out = bf16(rotate(accum + bias)) on [0, 2048) and bf16(accum + bias) after.
    The packed KN weight lands the Q/K projection in interleaved (lo, hi)
    pairs at head columns 2*p / 2*p+1; bias is added in the OUTPUT column
    order and rotate-half stores the lo results on head columns [0, 32) and
    the hi results on [32, 64) - exactly the flat gemm_compat_finish kind-3
    arithmetic (single-rounded FP32 multiplies, RN add/subtract, one BF16
    store). Same ub-pipeline structure as qkv_std_rope_finish: fixed 64-core
    grid, one-row GM<->UB DMA tiles, two-stage pipeline.
    """

    cores, width = 64, 3072

    @T.prim_func
    def main(
        output_handle: T.handle,
        accum_handle: T.handle,
        bias_handle: T.handle,
        cosine_handle: T.handle,
        sine_handle: T.handle,
        rows: T.int64,
        columns: T.int64,
    ):
        output = T.match_buffer(output_handle, (rows, width), "bfloat16")
        accum = T.match_buffer(accum_handle, (rows, width), "float32")
        bias = T.match_buffer(bias_handle, (width,), "bfloat16")
        cosine = T.match_buffer(cosine_handle, (rows, 32), "float32")
        sine = T.match_buffer(sine_handle, (rows, 32), "float32")
        with T.Kernel(cores) as core:
            accum_ub = T.alloc_shared((1, width), "float32")
            cs_ub = T.alloc_shared((1, 32), "float32")
            sn_ub = T.alloc_shared((1, 32), "float32")
            bias_ub = T.alloc_shared((width,), "bfloat16")
            out_ub = T.alloc_shared((1, width), "bfloat16")
            T.annotate_buffer_versions({accum_ub: 2, cs_ub: 2, sn_ub: 2, bias_ub: 1, out_ub: 1})
            T.copy(bias, bias_ub)
            for step in T.Pipelined(rows, num_stages=2):
                block = step * cores + core
                if block < rows:
                    start = T.cast(block, "int32")
                    T.copy(accum[start : start + 1, :], accum_ub)
                    T.copy(cosine[start : start + 1, :], cs_ub)
                    T.copy(sine[start : start + 1, :], sn_ub)
                    with T.SimtVF(threads=256):
                        for col in T.Parallel(width):
                            value = T.alloc_var(
                                "float32",
                                init=accum_ub[0, col]
                                + T.cast(bias_ub[col], "float32"),
                            )
                            if col < 2048:
                                head_col = col % 64
                                base = col - head_col
                                pair = T.Select(head_col < 32, head_col, head_col - 32)
                                lo = (
                                    accum_ub[0, base + 2 * pair]
                                    + T.cast(bias_ub[base + pair], "float32")
                                )
                                hi = (
                                    accum_ub[0, base + 2 * pair + 1]
                                    + T.cast(bias_ub[base + pair + 32], "float32")
                                )
                                c = cs_ub[0, pair]
                                s = sn_ub[0, pair]
                                value = T.Select(
                                    head_col < 32,
                                    lo * c - hi * s,
                                    hi * c + lo * s,
                                )
                            out_ub[0, col] = T.cast(value, "bfloat16")
                    T.copy(out_ub, output[start : start + 1, :])

    return main


def cube_bf16_nk64x256():
    """accum [M, N] = x [M, K] @ weight[N, K]^T; M % 64 == N % 256 == K % 128 == 0.

    Small-M variant of the wide Cube: a 64-row tile halves the padded MACs of
    the cluster / norm-gate paths whose real M never exceeds ~80 rows.
    """

    @T.prim_func
    def main(
        x_handle: T.handle,
        weight_handle: T.handle,
        accum_handle: T.handle,
        rows: T.int64,
        columns: T.int64,
        reduction: T.int64,
    ):
        x = T.match_buffer(x_handle, (rows, reduction), "bfloat16")
        weight = T.match_buffer(weight_handle, (columns, reduction), "bfloat16")
        accum = T.match_buffer(accum_handle, (rows, columns), "float32")
        with T.Kernel((rows // 64) * (columns // 256)) as block:
            acc = T.alloc_l0c((64, 256), "float32")
            x_l1 = T.alloc_l1((64, 128), "bfloat16")
            w_l1 = T.alloc_l1((256, 128), "bfloat16")
            n_tiles = columns // 256
            k_tiles = reduction // 128
            m0 = T.cast((block // n_tiles) * 64, "int32")
            n0 = T.cast((block % n_tiles) * 256, "int32")
            for kt in T.Pipelined(k_tiles, num_stages=2):
                k0 = T.cast(kt * 128, "int32")
                T.copy(x[m0 : m0 + 64, k0 : k0 + 128], x_l1)
                T.copy(weight[n0 : n0 + 256, k0 : k0 + 128], w_l1)
                T.gemm(
                    x_l1,
                    w_l1,
                    acc,
                    transpose_B=True,
                    clear_accum=kt == 0,
                    unit_flag_ctrl=T.Select(kt == k_tiles - 1, 3, 2),
                )
            T.copy(acc, accum[m0 : m0 + 64, n0 : n0 + 256], unit_flag_ctrl=3)

    return main



def qkv_vision_rope_finish():
    """out = bf16(rotate_half(accum + bias)) on the Q/K heads, bf16 past them.

    The wall_oss vision QKV shape (columns 3840 = 48 heads x 80, qk_width 2560,
    rotary 80, half 40, cos/sin [rows, 40] FP32) in the qkv_std_rope_finish
    mold: a fixed 64-core grid, one-row GM->UB tiles with a two-stage pipeline,
    scalar SimtVF rope math from UB, one DMA store. The torch chain this
    replaces moves the [2112, 3840] FP32 accumulator through bias-add plus a
    five-tensor cat/cast epilogue (~113 us/call); this is one pass.
    """

    cores, width, qk_width, head_dim, half = 64, 3840, 2560, 80, 40

    @T.prim_func
    def main(
        output_handle: T.handle,
        accum_handle: T.handle,
        bias_handle: T.handle,
        cosine_handle: T.handle,
        sine_handle: T.handle,
        rows: T.int64,
    ):
        output = T.match_buffer(output_handle, (rows, width), "bfloat16")
        accum = T.match_buffer(accum_handle, (rows, width), "float32")
        bias = T.match_buffer(bias_handle, (width,), "bfloat16")
        cosine = T.match_buffer(cosine_handle, (rows, half), "float32")
        sine = T.match_buffer(sine_handle, (rows, half), "float32")
        with T.Kernel(cores) as core:
            accum_ub = T.alloc_shared((1, width), "float32")
            cs_ub = T.alloc_shared((1, half), "float32")
            sn_ub = T.alloc_shared((1, half), "float32")
            bias_ub = T.alloc_shared((width,), "bfloat16")
            out_ub = T.alloc_shared((1, width), "bfloat16")
            T.annotate_buffer_versions(
                {accum_ub: 2, cs_ub: 2, sn_ub: 2, out_ub: 2}
            )
            T.copy(bias, bias_ub)
            for step in T.Pipelined(rows, num_stages=2):
                row = step * cores + core
                if row < rows:
                    start = T.cast(row, "int32")
                    T.copy(accum[start : start + 1, :], accum_ub)
                    T.copy(cosine[start : start + 1, :], cs_ub)
                    T.copy(sine[start : start + 1, :], sn_ub)
                    with T.SimtVF(threads=256):
                        for col in T.Parallel(width):
                            base_v = accum_ub[0, col] + T.cast(
                                bias_ub[col], "float32"
                            )
                            head_col = col % head_dim
                            pair = T.Select(
                                head_col < half, head_col, head_col - half
                            )
                            partner_col = T.Select(
                                head_col < half, col + half, col - half
                            )
                            partner = accum_ub[0, partner_col] + T.cast(
                                bias_ub[partner_col], "float32"
                            )
                            c = cs_ub[0, pair]
                            s = sn_ub[0, pair]
                            rotated = T.Select(
                                head_col < half,
                                base_v * c - partner * s,
                                base_v * c + partner * s,
                            )
                            value = T.Select(col < qk_width, rotated, base_v)
                            out_ub[0, col] = T.cast(value, "bfloat16")
                    T.copy(out_ub, output[start : start + 1, :])

    return main


def qkv_vision_rope_cube():
    """out = bf16(rotate_half(x @ w^T + bias)) on the QK heads, bf16(x @ w^T + bias) on the V heads.

    The wall_oss vision QKV shape in ONE cube launch: [M, 1280] x NK weight [3840, 1280]
    (the launcher pads M=2112 to the 128-row tile grid and slices the tail rows away).
    r11b retiling: N tiles of 240 (16 tiles, 272 blocks vs the r10 160-tiling's 408;
    the bare wide gemm's own quantization is 255) with an M-split epilogue -- each
    of the two AIVs finishes one 64-row half of the full 240-wide tile, so every
    +/-40 rotary partner stays inside its AIV row and every 80-wide head stays
    intact. The 240 tiling crosses qk_width=2560 mid-tile (2400-2640), so the QK/V
    branch selects per element on n0 + j; the per-row cos/sin tables M-split through
    dual_copy so each AIV's table rows align with its accumulator half. The
    rotate-half reads the FP32 L0C accumulator delivered to UB, adds the FP32
    bias, and stores with a single RNE to BF16 -- the finish contract, with the
    FP32 accumulator roundtrip gone.
    """

    width, qk_width, half = 3840, 2560, 40
    m_tile, n_tile = 128, 240

    @T.prim_func
    def main(
        x_handle: T.handle,
        weight_handle: T.handle,
        bias_handle: T.handle,
        cosine_handle: T.handle,
        sine_handle: T.handle,
        out_handle: T.handle,
        rows: T.int64,
        columns: T.int64,
        reduction: T.int64,
    ):
        x = T.match_buffer(x_handle, (rows, reduction), "bfloat16")
        weight = T.match_buffer(weight_handle, (columns, reduction), "bfloat16")
        bias = T.match_buffer(bias_handle, (columns,), "bfloat16")
        cosine = T.match_buffer(cosine_handle, (rows, half), "float32")
        sine = T.match_buffer(sine_handle, (rows, half), "float32")
        out = T.match_buffer(out_handle, (rows, columns), "bfloat16")
        n_tiles = columns // n_tile
        k_tiles = reduction // 128
        with T.Kernel((rows // m_tile) * n_tiles) as block:
            accum = T.alloc_l0c((m_tile, n_tile), "float32")
            x_l1 = T.alloc_l1((m_tile, 128), "bfloat16")
            w_l1 = T.alloc_l1((n_tile, 128), "bfloat16")
            accum_ub = T.alloc_shared((m_tile // 2, n_tile), "float32")
            cs_ub = T.alloc_shared((m_tile // 2, half), "float32")
            sn_ub = T.alloc_shared((m_tile // 2, half), "float32")
            bias_ub = T.alloc_shared((n_tile,), "bfloat16")
            out_ub = T.alloc_shared((m_tile // 2, n_tile), "bfloat16")
            m0 = T.cast((block // n_tiles) * m_tile, "int32")
            n0 = T.cast((block % n_tiles) * n_tile, "int32")
            for kt in T.Pipelined(k_tiles, num_stages=2):
                k0 = T.cast(kt * 128, "int32")
                T.copy(x[m0 : m0 + m_tile, k0 : k0 + 128], x_l1)
                T.copy(weight[n0 : n0 + n_tile, k0 : k0 + 128], w_l1)
                T.gemm(
                    x_l1,
                    w_l1,
                    accum,
                    transpose_B=True,
                    clear_accum=kt == 0,
                    unit_flag_ctrl=T.Select(kt == k_tiles - 1, 3, 2),
                )
            # M-split: each AIV's accum_ub/cs_ub/sn_ub/out_ub view is its 64-row
            # half of the full 240-wide tile.
            T.dual_copy(accum, accum_ub, unit_flag_ctrl=3)
            T.copy(bias[n0 : n0 + n_tile], bias_ub)
            T.dual_copy(cosine[m0 : m0 + m_tile, :], cs_ub)
            T.dual_copy(sine[m0 : m0 + m_tile, :], sn_ub)
            with T.SimtVF(threads=256):
                for i, j in T.Parallel(m_tile // 2, n_tile):
                    value = T.alloc_var(
                        "float32",
                        init=accum_ub[i, j] + T.cast(bias_ub[j], "float32"),
                    )
                    c80 = j % (2 * half)
                    partner_j = T.Select(c80 < half, j + half, j - half)
                    partner = T.alloc_var(
                        "float32",
                        init=accum_ub[i, partner_j]
                        + T.cast(bias_ub[partner_j], "float32"),
                    )
                    c = cs_ub[i, c80 % half]
                    s = sn_ub[i, c80 % half]
                    rotated = T.Select(
                        c80 < half, value * c - partner * s, value * c + partner * s
                    )
                    value = T.Select(n0 + j < qk_width, rotated, value)
                    out_ub[i, j] = _store_rne(value, "bfloat16")
            T.dual_copy(out_ub, out[m0 : m0 + m_tile, n0 : n0 + n_tile])

    return main

def build_programs():
    yield "gemm_cube_bf16_kn_wide128x256", cube_bf16_kn128x256()
    yield "gemm_cube_bf16_kn_rope_wide128x256", cube_bf16_kn_rope_wide128x256()
    yield "gemm_qkv_std_rope_finish", qkv_std_rope_finish()
    yield "gemm_qkv_compat_rope_finish", qkv_compat_rope_finish()
    yield "gemm_cube_bf16_nk64x256", cube_bf16_nk64x256()
    yield "gemm_qkv_vision_rope_finish", qkv_vision_rope_finish()
    yield "gemm_qkv_vision_rope_cube", qkv_vision_rope_cube()


def qkv_class_prepare():
    """Packed [128, K] BF16 activation: rows 0..M-1 = bf16(x * (1 + scale[class])),
    rows M..M+R-1 = the class shift rows, the tail zero.

    The scale / shift class tables are strided [R, K] rows (unit column
    stride); row_class picks the class row of each token.
    """

    @T.prim_func
    def main(
        packed_handle: T.handle,
        x_handle: T.handle,
        scale_handle: T.handle,
        shift_handle: T.handle,
        row_class_handle: T.handle,
        rows: T.int64,
        reduction: T.int64,
        classes: T.int64,
        scale_stride: T.int64,
        shift_stride: T.int64,
    ):
        packed = T.match_buffer(packed_handle, (128 * reduction,), "bfloat16")
        x = T.match_buffer(x_handle, (rows * reduction,), "bfloat16")
        scale = T.match_buffer(scale_handle, (classes * scale_stride,), "bfloat16")
        shift = T.match_buffer(shift_handle, (classes * shift_stride,), "bfloat16")
        row_class = T.match_buffer(row_class_handle, (rows,), "int32")
        count = 128 * reduction
        blocks = T.max(T.int64(1), T.min(T.ceildiv(count, 256), T.int64(128)))
        with T.Kernel(blocks) as block:
            with T.SimtVF(threads=256):
                for step in T.serial(T.ceildiv(count, blocks * 256)):
                    for lane in T.Parallel(256):
                        index = (step * blocks + T.cast(block, "int64")) * 256 + lane
                        if index < count:
                            row, col = index // reduction, index % reduction
                            value = T.alloc_var("bfloat16", init=T.cast(0, "bfloat16"))
                            if row < rows:
                                cls = row_class[row]
                                scale_value = scale[cls * scale_stride + col]
                                value = T.cast(
                                    T.cast(x[index], "float32")
                                    * (1.0 + T.cast(scale_value, "float32")),
                                    "bfloat16",
                                )
                            elif row < rows + classes:
                                cls = row - rows
                                value = shift[cls * shift_stride + col]
                            packed[index] = value

    return main


def qkv_row_rstd():
    """rstd[row] = rsqrt(mean(x[row]^2) + 1e-6) in FP32; one warp per row.

    32 lanes stripe the reduction columns coalesced (40 serial steps at
    K=1280) and fold the partials with ``T.warp_reduce_sum`` - the original
    one-lane-per-row serial walk cost 63-89 us of pure load latency at these
    M (it dominated the whole cluster chain at M<=20).
    """

    @T.prim_func
    def main(rstd_handle: T.handle, x_handle: T.handle, rows: T.int64, reduction: T.int64):
        rstd = T.match_buffer(rstd_handle, (128,), "float32")
        x = T.match_buffer(x_handle, (rows * reduction,), "bfloat16")
        blocks = T.ceildiv(rows, 8)
        with T.Kernel(blocks) as block:
            with T.SimtVF(threads=256):
                for lane in T.Parallel(256):
                    warp = T.cast(lane // 32, "int64")
                    pos = T.cast(lane % 32, "int64")
                    row = T.cast(block, "int64") * 8 + warp
                    if row < rows:
                        total = T.alloc_var("float32", init=0.0)
                        for stripe in T.serial(T.ceildiv(reduction, 32)):
                            col = stripe * 32 + pos
                            if col < reduction:
                                value = T.cast(x[row * reduction + col], "float32")
                                total = total + value * value
                        sumsq = T.warp_reduce_sum(total)
                        rstd[row] = T.rsqrt(
                            sumsq / T.cast(reduction, "float32") + T.float32(1e-6)
                        )
                    elif row < 128:
                        rstd[row] = T.float32(0.0)

    return main


def qkv_cluster_finish():
    """Deferred finish for 16/4 heads of 256: y = bf16(dot*rstd + shift_dot[class]),
    zero-centered per-head RMSNorm to BF16, rotate-half RoPE on the first 64 dims,
    Q | gate to out and K / V to head-major caches. V and gate keep y.

    Warp-per-job layout: each (row, head) stats pass stripes 256 columns over
    32 lanes (8 serial 2-load steps) and folds with ``T.warp_reduce_sum``;
    the apply pass reuses the hot accumulators. The gate (4096 cols) and V
    (1024 cols) tails ride idle warps: 128 + 32 elementwise warps per row.
    Divisions are by literals only (20/4096/1024/256), so no per-element int64
    division survives. The grid is capped at 256 blocks with a serial job
    loop per warp: above ~2K jobs the one-job-per-warp grid paid ~21 ns per
    block in dispatch alone (1733 blocks = ~36 us at M=77).
    """

    @T.prim_func
    def main(
        output_handle: T.handle,
        key_handle: T.handle,
        value_handle: T.handle,
        accum_handle: T.handle,
        rstd_handle: T.handle,
        row_class_handle: T.handle,
        q_weight_handle: T.handle,
        k_weight_handle: T.handle,
        cosine_handle: T.handle,
        sine_handle: T.handle,
        rows: T.int64,
        columns: T.int64,
        key_head_stride: T.int64,
        key_row_stride: T.int64,
        value_head_stride: T.int64,
        value_row_stride: T.int64,
        start: T.int64,
    ):
        output = T.match_buffer(output_handle, (rows * 8192,), "bfloat16")
        key = T.match_buffer(key_handle, (4 * key_head_stride,), "bfloat16")
        value = T.match_buffer(value_handle, (4 * value_head_stride,), "bfloat16")
        accum = T.match_buffer(accum_handle, (128, columns), "float32")
        rstd = T.match_buffer(rstd_handle, (128,), "float32")
        row_class = T.match_buffer(row_class_handle, (rows,), "int32")
        q_weight = T.match_buffer(q_weight_handle, (256,), "bfloat16")
        k_weight = T.match_buffer(k_weight_handle, (256,), "bfloat16")
        cosine = T.match_buffer(cosine_handle, (rows * 32,), "float32")
        sine = T.match_buffer(sine_handle, (rows * 32,), "float32")
        jobs = rows * 180
        blocks = T.min(T.ceildiv(jobs, 8), 256)
        with T.Kernel(blocks) as block:
            with T.SimtVF(threads=256):
                for lane in T.Parallel(256):
                    warp = T.cast(lane // 32, "int64")
                    pos = T.cast(lane % 32, "int64")
                    for step in T.serial(T.ceildiv(jobs, blocks * 8)):
                        job = (step * blocks + T.cast(block, "int64")) * 8 + warp
                        if job < rows * 20:
                            row = job // 20
                            head = job % 20
                            base = head * 256
                            cls = T.cast(row_class[row], "int64")
                            scale = rstd[row]
                            total = T.alloc_var("float32", init=0.0)
                            for stripe in T.serial(8):
                                j = stripe * 32 + pos
                                first = T.alloc_var(
                                    "float32",
                                    init=accum[row, base + j] * scale
                                    + accum[rows + cls, base + j],
                                )
                                rounded = T.cast(first, "bfloat16")
                                v = T.cast(rounded, "float32")
                                total = total + v * v
                            sumsq = T.warp_reduce_sum(total)
                            inverse = T.rsqrt(sumsq / T.float32(256.0) + T.float32(1e-6))
                            for stripe in T.serial(8):
                                j = stripe * 32 + pos
                                first = T.alloc_var(
                                    "float32",
                                    init=accum[row, base + j] * scale
                                    + accum[rows + cls, base + j],
                                )
                                w = T.Select(
                                    head < 16,
                                    T.cast(q_weight[j], "float32"),
                                    T.cast(k_weight[j], "float32"),
                                )
                                z = T.cast(
                                    T.cast(first, "bfloat16") * inverse * (1.0 + w),
                                    "bfloat16",
                                )
                                result = T.alloc_var("float32", init=T.cast(z, "float32"))
                                if j < 64:
                                    pair = T.Select(j < 32, j, j - 32)
                                    partner_j = T.Select(j < 32, j + 32, j - 32)
                                    second = T.alloc_var(
                                        "float32",
                                        init=accum[row, base + partner_j] * scale
                                        + accum[rows + cls, base + partner_j],
                                    )
                                    w_partner = T.Select(
                                        head < 16,
                                        T.cast(q_weight[partner_j], "float32"),
                                        T.cast(k_weight[partner_j], "float32"),
                                    )
                                    z_partner = T.cast(
                                        T.cast(second, "bfloat16") * inverse * (1.0 + w_partner),
                                        "bfloat16",
                                    )
                                    c = cosine[row * 32 + pair]
                                    s = sine[row * 32 + pair]
                                    result = T.Select(
                                        j < 32,
                                        result * c - T.cast(z_partner, "float32") * s,
                                        result * c + T.cast(z_partner, "float32") * s,
                                    )
                                if head < 16:
                                    output[row * 8192 + head * 256 + j] = T.cast(result, "bfloat16")
                                else:
                                    key[(head - 16) * key_head_stride + (start + row) * key_row_stride + j] = T.cast(result, "bfloat16")
                        elif job < rows * 148:
                            flat = (job - rows * 20) * 32 + pos
                            row = flat // 4096
                            col = flat % 4096
                            cls = T.cast(row_class[row], "int64")
                            y = T.cast(
                                accum[row, 6144 + col] * rstd[row]
                                + accum[rows + cls, 6144 + col],
                                "bfloat16",
                            )
                            output[row * 8192 + 4096 + col] = y
                        elif job < jobs:
                            flat = (job - rows * 148) * 32 + pos
                            row = flat // 1024
                            col = flat % 1024
                            cls = T.cast(row_class[row], "int64")
                            y = T.cast(
                                accum[row, 5120 + col] * rstd[row]
                                + accum[rows + cls, 5120 + col],
                                "bfloat16",
                            )
                            value[(col // 256) * value_head_stride + (start + row) * value_row_stride + col % 256] = y

    return main


def qkv_norm_gate_finish():
    """bias finish for 8/2 heads of 256 with partial RoPE on 64 dims and gate columns:
    y = bf16(accum + bias), zero-centered per-head RMSNorm to BF16, rotate-half RoPE on
    the first 64 dims of every Q/K head; V and gate keep y.

    Each (row, head) job is one 32-lane warp: the stats pass stripes the 256
    columns across the lanes (8 serial loads of 2 each, coalesced) and
    ``T.warp_reduce_sum`` folds the partials, so no single lane walks 2 x 256
    serial GM loads. The apply pass re-reads the same (now hot) accumulators
    and stores per lane. The gate tail rides the idle warps: 80 elementwise
    warps per row (2560 gate columns / 32 lanes).
    """

    @T.prim_func
    def main(
        output_handle: T.handle,
        accum_handle: T.handle,
        bias_handle: T.handle,
        q_weight_handle: T.handle,
        k_weight_handle: T.handle,
        cosine_handle: T.handle,
        sine_handle: T.handle,
        rows: T.int64,
        columns: T.int64,
    ):
        output = T.match_buffer(output_handle, (rows * columns,), "bfloat16")
        accum = T.match_buffer(accum_handle, (128, columns), "float32")
        bias = T.match_buffer(bias_handle, (columns,), "bfloat16")
        q_weight = T.match_buffer(q_weight_handle, (256,), "bfloat16")
        k_weight = T.match_buffer(k_weight_handle, (256,), "bfloat16")
        cosine = T.match_buffer(cosine_handle, (rows * 32,), "float32")
        sine = T.match_buffer(sine_handle, (rows * 32,), "float32")
        jobs = rows * 90
        blocks = T.ceildiv(jobs, 8)
        with T.Kernel(blocks) as block:
            with T.SimtVF(threads=256):
                for lane in T.Parallel(256):
                    warp = T.cast(lane // 32, "int64")
                    pos = T.cast(lane % 32, "int64")
                    job = T.cast(block, "int64") * 8 + warp
                    if job < rows * 10:
                        row = job // 10
                        head = job % 10
                        base = head * 256
                        total = T.alloc_var("float32", init=0.0)
                        for stripe in T.serial(8):
                            j = stripe * 32 + pos
                            y = T.cast(
                                accum[row, base + j] + T.cast(bias[base + j], "float32"),
                                "bfloat16",
                            )
                            v = T.cast(y, "float32")
                            total = total + v * v
                        sumsq = T.warp_reduce_sum(total)
                        inverse = T.rsqrt(sumsq / T.float32(256.0) + T.float32(1e-6))
                        for stripe in T.serial(8):
                            j = stripe * 32 + pos
                            y = T.cast(
                                accum[row, base + j] + T.cast(bias[base + j], "float32"),
                                "bfloat16",
                            )
                            w = T.Select(
                                head < 8,
                                T.cast(q_weight[j], "float32"),
                                T.cast(k_weight[j], "float32"),
                            )
                            z = T.cast(
                                T.cast(y, "float32") * inverse * (1.0 + w),
                                "bfloat16",
                            )
                            result = T.alloc_var("float32", init=T.cast(z, "float32"))
                            if j < 64:
                                pair = T.Select(j < 32, j, j - 32)
                                partner_j = T.Select(j < 32, j + 32, j - 32)
                                y_partner = T.cast(
                                    accum[row, base + partner_j]
                                    + T.cast(bias[base + partner_j], "float32"),
                                    "bfloat16",
                                )
                                w_partner = T.Select(
                                    head < 8,
                                    T.cast(q_weight[partner_j], "float32"),
                                    T.cast(k_weight[partner_j], "float32"),
                                )
                                z_partner = T.cast(
                                    T.cast(y_partner, "float32") * inverse * (1.0 + w_partner),
                                    "bfloat16",
                                )
                                c = cosine[row * 32 + pair]
                                s = sine[row * 32 + pair]
                                result = T.Select(
                                    j < 32,
                                    result * c - T.cast(z_partner, "float32") * s,
                                    result * c + T.cast(z_partner, "float32") * s,
                                )
                            output[row * columns + base + j] = T.cast(result, "bfloat16")
                    elif job < jobs:
                        flat = (job - rows * 10) * 32 + pos
                        row = flat // 2560
                        col = flat % 2560 + 2560
                        output[row * columns + col] = T.cast(
                            accum[row, col] + T.cast(bias[col], "float32"), "bfloat16"
                        )

    return main


def wide_build_programs():
    yield "gemm_qkv_class_prepare", qkv_class_prepare()
    yield "gemm_qkv_row_rstd", qkv_row_rstd()
    yield "gemm_qkv_cluster_finish", qkv_cluster_finish()
    yield "gemm_qkv_norm_gate_finish", qkv_norm_gate_finish()
