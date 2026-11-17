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

"""decodeattn-r4: specialized small-M cache-attention for the wall-oss decode
family (B1 H16 KV2 M16 D128 L784; fia-r3: 180 calls x 12.6us FIA per request,
fixed overhead dominated).

Two TileLang kernels over the splitk-v3 idioms
(walloss/src/src/attention/npu/attention_splitk_decode.py) specialized to
static shapes. Structural fixes vs the shipped generic splitkv (ms-class on
this family): the lts/ute flash mask is folded into the SIMD softmax pass as
per-lane constants (one 128-lane scalar setup per block-chunk instead of a
per-row-per-lane scalar pass), partials move as whole-tile T.copy block
stores instead of scalar walks, and the reduce is a vectorized vaxpy merge.

Layout contract (the production bridge shapes, zero extra copies):
  Q  [256, 128] bf16, row = head*16 + query (H-major [H, M, D] memory; the
     production caller hands a strided [1, M, H, D] view over it, whose
     transpose(1,2) is exactly this contiguous buffer)
  K  [1568, 128] bf16, row = kvhead*784 + position (cache [KV, L, D] major)
  V  same as K
  FM [784, 2] int32: lts, ute per position (flash_mask [1,1,L,2])
  PL [1] int32: prefix_len (device data)
  P  [28, 65, 128] fp32 partials: pair(kv, tile) x chunk; rows 0..63 = o
     [64, 128] (sid*32 rows per sub-core), row 64 cols 0..63 = m, 64..127 = l
  OUT [256, 128] bf16, row = head*16 + query (BNSD [H, M, D], the o_proj A)

Mask semantics = the bridge make_drop_mask:
  visible(pos, m) = ((pos < prefix) or (pos >= new_pos and pos < new_pos+16))
                    and (m < lts[pos]) and (m >= ute[pos])
The scan grid is (kvhead, tile, chunk): 28 blocks = 2 kv x 2 Q-tiles x 7
128-key chunks (the last chunk slides to [L-128, L); a coverage bound keeps
chunk windows disjoint, splitk-v3 rule). Each block loads its K/V tile once
and its 64-row Q tile once: a wide A-tile of 4 heads x 16 queries, so K/V
bytes move exactly once per (kv, tile) pair - no mgroups re-read
amplification (ascendc-decode r5 lesson). Variant A (14 blocks, tile looped
inside) is kept for the bandwidth comparison.
"""

import functools

import tilelang.ascend.language as T
import tilelang.ascend.language.simd as S

HEADS, KVHEADS, QUERIES, WIDTH, KEYS = 16, 2, 16, 128, 784
KEY_TILE = 128
CHUNKS = 7
TILES = 2  # 64-row Q tiles per kv group (tile = head half)
PAIRS = KVHEADS * TILES * CHUNKS  # 28
PAIR_ROWS = 65  # 64 o rows + 1 row packing m (cols 0..63) and l (64..127)


def _pair(kv, tile, chunk):
    return (kv * TILES + tile) * CHUNKS + chunk


def scan(tile_per_block: bool):
    """tile_per_block=True: 28 blocks (kv, tile, chunk); False: 14 (kv, chunk)."""

    @T.prim_func
    def main(
        qh: T.handle,
        kh: T.handle,
        vh: T.handle,
        fmh: T.handle,
        plh: T.handle,
        ph: T.handle,
        new_pos: T.int64,
        scale_bits: T.int64,
    ):
        Q = T.match_buffer(qh, (HEADS * QUERIES, WIDTH), "bfloat16")
        K = T.match_buffer(kh, (KVHEADS * KEYS, WIDTH), "bfloat16")
        V = T.match_buffer(vh, (KVHEADS * KEYS, WIDTH), "bfloat16")
        FM = T.match_buffer(fmh, (KEYS, 2), "int32")
        PL = T.match_buffer(plh, (1,), "int32")
        P = T.match_buffer(ph, (PAIRS, PAIR_ROWS, WIDTH), "float32")
        scale = T.reinterpret("float32", T.cast(scale_bits, "uint32"))
        groups = KVHEADS * (TILES if tile_per_block else 1)
        with T.MixedKernel(groups * CHUNKS, sids=2) as (bx, sid):
            T.set_hf32_mode(None)
            q_l1 = T.alloc_l1((64, WIDTH), "bfloat16")
            k_l1 = T.alloc_l1((KEY_TILE, WIDTH), "bfloat16")
            v_l1 = T.alloc_l1((WIDTH, KEY_TILE), "bfloat16")
            p_l1 = T.alloc_l1((64, KEY_TILE), "bfloat16")
            qk_cc = T.alloc_l0c((64, KEY_TILE), "float32")
            pv_cc = T.alloc_l0c((64, WIDTH), "float32")
            s_ub = T.alloc_shared((32, KEY_TILE), "float32")
            visible_ub = T.alloc_shared((32, KEY_TILE), "float32")
            p_ub = T.alloc_shared((32, KEY_TILE), "bfloat16")
            o_ub = T.alloc_shared((32, WIDTH), "float32")
            m_ub = T.alloc_shared((32,), "float32")
            sums = T.alloc_shared((32,), "float32")
            mask16_ub = T.alloc_shared((QUERIES, KEY_TILE), "float32")
            kv = bx // (CHUNKS * (TILES if tile_per_block else 1))
            rest = bx - kv * CHUNKS * (TILES if tile_per_block else 1)
            tile_fixed = rest // CHUNKS
            chunk = rest - tile_fixed * CHUNKS
            prefix = T.cast(PL[0], "int64")
            # Chunk window (splitk-v3 slide rule for KEYS % KEY_TILE != 0):
            # chunks 0..5 own [c*128, min((c+1)*128, KEYS-128)), the last
            # chunk slides to [KEYS-128, KEYS).
            base_row = T.alloc_var("int64", init=chunk * KEY_TILE)
            coverage = T.alloc_var("int64", init=KEYS)
            if chunk == CHUNKS - 1:
                base_row = KEYS - KEY_TILE
            if chunk < CHUNKS - 1:
                coverage = T.min((chunk + 1) * KEY_TILE, KEYS - KEY_TILE)
            k_row0 = T.cast(kv * KEYS + base_row, "int32")
            T.copy(K[k_row0 : k_row0 + KEY_TILE, 0:WIDTH], k_l1)
            T.copy(
                V[k_row0 : k_row0 + KEY_TILE, 0:WIDTH],
                v_l1,
                transpose=True,
            )
            # Visibility table, once per block: mask16_ub[m, lane] = 1.0 iff
            # column base_row+lane is visible to query row m under the bridge
            # keep rule (prefix/new_pos window) and the flash lts/ute bounds.
            # The SIMD softmax phases then only use constant-scalar compares
            # and dynamic-address loads (both legal in r-loops; dynamic-scalar
            # vcmps/vdup operands are not hoistable and fail the backend).
            with T.SimtVF(threads=KEY_TILE):
                for lane in T.Parallel(KEY_TILE):
                    kr = base_row + lane
                    keep_col = (kr < prefix) or (
                        kr >= new_pos and kr < new_pos + QUERIES
                    )
                    lts_col = FM[kr, 0]
                    ute_col = FM[kr, 1]
                    for mm in range(QUERIES):
                        vis = keep_col and (mm < lts_col) and (mm >= ute_col)
                        mask16_ub[mm, lane] = T.Select(
                            vis and kr < coverage, T.float32(1), T.float32(0)
                        )
            for tile in T.serial(TILES if not tile_per_block else 1):
                tl = tile_fixed if tile_per_block else tile
                q_row0 = T.cast((kv * 8 + tl * 4) * QUERIES, "int32")
                T.copy(Q[q_row0 : q_row0 + 64, 0:WIDTH], q_l1)
                T.gemm(q_l1, k_l1, qk_cc, transpose_B=True, clear_accum=True)
                T.dual_copy(qk_cc, s_ub)
                with T.SimdVF():
                    allp = S.pset(32, "PAT_ALL")
                    one = S.pset(32, "PAT_VL1")
                    ninf = S.vdup(-T.infinity("float32"), "float32", allp)
                    onev = S.vdup(T.float32(1), "float32", allp)
                    zerov = S.vdup(T.float32(0), "float32", allp)
                    for r in range(32):
                        mi = (sid * 32 + r) % QUERIES
                        s0 = S.vmuls(S.vld(s_ub[r, 0], "NORM"), scale, allp)
                        v0 = S.vcmps(
                            S.vld(mask16_ub[mi, 0], "NORM"), T.float32(0), allp, "gt"
                        )
                        s0 = S.vsel(s0, ninf, v0)
                        S.vsts(s_ub[r, 0], s0, allp, "NORM_B32")
                        S.vsts(visible_ub[r, 0], S.vsel(onev, zerov, v0), allp, "NORM_B32")
                        s64 = S.vmuls(S.vld(s_ub[r, 64], "NORM"), scale, allp)
                        v64 = S.vcmps(
                            S.vld(mask16_ub[mi, 64], "NORM"), T.float32(0), allp, "gt"
                        )
                        s64 = S.vsel(s64, ninf, v64)
                        S.vsts(s_ub[r, 64], s64, allp, "NORM_B32")
                        S.vsts(visible_ub[r, 64], S.vsel(onev, zerov, v64), allp, "NORM_B32")
                        S.vsts(
                            m_ub[r],
                            S.vcmax(S.vmax(s0, s64, allp), allp),
                            one,
                            "ONEPT_B32",
                        )
                    S.mem_bar("VST_VLD")
                with T.SimdVF():
                    allp = S.pset(32, "PAT_ALL")
                    one = S.pset(32, "PAT_VL1")
                    zero = S.vdup(T.float32(0), "float32", allp)
                    for r in range(32):
                        row_max = S.vld(m_ub[r], "BRC_B32")
                        prob0 = S.vexpdif(S.vld(s_ub[r, 0], "NORM"), row_max, allp)
                        prob0 = S.vsel(
                            prob0,
                            zero,
                            S.vcmps(S.vld(visible_ub[r, 0], "NORM"), T.float32(0), allp, "gt"),
                        )
                        S.vsts(p_ub[r, 0], S.vcvt(prob0, "bfloat16", allp), allp, "PK_B32")
                        prob64 = S.vexpdif(S.vld(s_ub[r, 64], "NORM"), row_max, allp)
                        prob64 = S.vsel(
                            prob64,
                            zero,
                            S.vcmps(S.vld(visible_ub[r, 64], "NORM"), T.float32(0), allp, "gt"),
                        )
                        S.vsts(p_ub[r, 64], S.vcvt(prob64, "bfloat16", allp), allp, "PK_B32")
                        S.vsts(
                            sums[r],
                            S.vcadd(S.vadd(prob0, prob64, allp), allp),
                            one,
                            "ONEPT_B32",
                        )
                    S.mem_bar("VST_VLD")
                T.dual_copy(p_ub, p_l1)
                T.gemm(p_l1, v_l1, pv_cc, transpose_B=True, clear_accum=True)
                T.dual_copy(pv_cc, o_ub[0:32, 0:WIDTH])
                pair = _pair(kv, tl, chunk)
                T.copy(o_ub, P[pair, sid * 32 : sid * 32 + 32, 0:WIDTH])
                T.copy(m_ub, P[pair, 64, sid * 32 : sid * 32 + 32])
                T.copy(sums, P[pair, 64, 64 + sid * 32 : 64 + sid * 32 + 32])

    return main


def reduce():
    """16 blocks = (kv, tile, head-quarter); one output head x 16 queries each."""

    @T.prim_func
    def main(ph: T.handle, oh: T.handle):
        P = T.match_buffer(ph, (PAIRS, PAIR_ROWS, WIDTH), "float32")
        OUT = T.match_buffer(oh, (HEADS * QUERIES, WIDTH), "bfloat16")
        with T.Kernel(16) as block:
            kv = block // 8
            tile = (block % 8) // 4
            h_local = block % 4
            lr0 = h_local * 16
            o7 = T.alloc_shared((CHUNKS * 16, WIDTH), "float32")
            ml7 = T.alloc_shared((CHUNKS, WIDTH), "float32")
            w_ub = T.alloc_shared((16, CHUNKS), "float32")
            inv_ub = T.alloc_shared((16,), "float32")
            out_ub = T.alloc_shared((16, WIDTH), "bfloat16")
            base = _pair(kv, tile, 0)
            for c in range(CHUNKS):
                T.copy(P[base + c, lr0 : lr0 + 16, 0:WIDTH], o7[c * 16 : (c + 1) * 16, 0:WIDTH])
                T.copy(P[base + c, 64, 0:WIDTH], ml7[c, 0:WIDTH])
            with T.SimtVF(threads=16):
                for m in T.Parallel(16):
                    lr = lr0 + m
                    m_star = T.alloc_var("float32", init=-T.infinity("float32"))
                    for c in T.serial(CHUNKS):
                        m_star = T.Select(ml7[c, lr] > m_star, ml7[c, lr], m_star)
                    l_star = T.alloc_var("float32", init=0)
                    for c in T.serial(CHUNKS):
                        weight = T.exp(ml7[c, lr] - m_star)
                        w_ub[m, c] = weight
                        l_star = l_star + weight * ml7[c, 64 + lr]
                    inv_ub[m] = T.float32(1) / l_star
            with T.SimdVF():
                S.mem_bar("VST_VLD")
                allp = S.pset(32, "PAT_ALL")
                for m in range(16):
                    # Python-folded expression trees: TIR let variables must
                    # not be reassigned across loop iterations (lesson 10).
                    weights = [S.vld(w_ub[m, c], "BRC_B32") for c in range(CHUNKS)]
                    terms0 = [
                        S.vmul(S.vld(o7[c * 16 + m, 0], "NORM"), weights[c])
                        for c in range(CHUNKS)
                    ]
                    terms64 = [
                        S.vmul(S.vld(o7[c * 16 + m, 64], "NORM"), weights[c])
                        for c in range(CHUNKS)
                    ]
                    acc0 = functools.reduce(
                        lambda a, b: S.vadd(a, b, allp), terms0
                    )
                    acc64 = functools.reduce(
                        lambda a, b: S.vadd(a, b, allp), terms64
                    )
                    inv = S.vld(inv_ub[m], "BRC_B32")
                    S.vsts(out_ub[m, 0], S.vcvt(S.vmul(acc0, inv), "bfloat16", allp), allp, "PK_B32")
                    S.vsts(out_ub[m, 64], S.vcvt(S.vmul(acc64, inv), "bfloat16", allp), allp, "PK_B32")
                S.mem_bar("VST_VLD")
            T.copy(out_ub, OUT[block * 16 : block * 16 + 16, 0:WIDTH])

    return main


def reduce_row():
    """Token-major twin of reduce(): same 16-block math, OUT is [M, H, D].

    Layout tax knife K1: the o-projection A is consumed as [M, H*D] rows, so
    the shipped head-major store makes the consumer read 16 strided 256B
    segments per row. Writing OUT[m, h, d] here keeps every arithmetic
    statement of reduce() byte-identical (same weights, same vaxpy fold, same
    RNE pack); only the destination address of the final GM store changes.
    """

    @T.prim_func
    def main(ph: T.handle, oh: T.handle):
        P = T.match_buffer(ph, (PAIRS, PAIR_ROWS, WIDTH), "float32")
        OUT = T.match_buffer(oh, (QUERIES, HEADS, WIDTH), "bfloat16")
        with T.Kernel(16) as block:
            kv = block // 8
            tile = (block % 8) // 4
            h_local = block % 4
            lr0 = h_local * 16
            o7 = T.alloc_shared((CHUNKS * 16, WIDTH), "float32")
            ml7 = T.alloc_shared((CHUNKS, WIDTH), "float32")
            w_ub = T.alloc_shared((16, CHUNKS), "float32")
            inv_ub = T.alloc_shared((16,), "float32")
            out_ub = T.alloc_shared((16, WIDTH), "bfloat16")
            base = _pair(kv, tile, 0)
            for c in range(CHUNKS):
                T.copy(P[base + c, lr0 : lr0 + 16, 0:WIDTH], o7[c * 16 : (c + 1) * 16, 0:WIDTH])
                T.copy(P[base + c, 64, 0:WIDTH], ml7[c, 0:WIDTH])
            with T.SimtVF(threads=16):
                for m in T.Parallel(16):
                    lr = lr0 + m
                    m_star = T.alloc_var("float32", init=-T.infinity("float32"))
                    for c in T.serial(CHUNKS):
                        m_star = T.Select(ml7[c, lr] > m_star, ml7[c, lr], m_star)
                    l_star = T.alloc_var("float32", init=0)
                    for c in T.serial(CHUNKS):
                        weight = T.exp(ml7[c, lr] - m_star)
                        w_ub[m, c] = weight
                        l_star = l_star + weight * ml7[c, 64 + lr]
                    inv_ub[m] = T.float32(1) / l_star
            with T.SimdVF():
                S.mem_bar("VST_VLD")
                allp = S.pset(32, "PAT_ALL")
                for m in range(16):
                    # Python-folded expression trees: TIR let variables must
                    # not be reassigned across loop iterations (lesson 10).
                    weights = [S.vld(w_ub[m, c], "BRC_B32") for c in range(CHUNKS)]
                    terms0 = [
                        S.vmul(S.vld(o7[c * 16 + m, 0], "NORM"), weights[c])
                        for c in range(CHUNKS)
                    ]
                    terms64 = [
                        S.vmul(S.vld(o7[c * 16 + m, 64], "NORM"), weights[c])
                        for c in range(CHUNKS)
                    ]
                    acc0 = functools.reduce(
                        lambda a, b: S.vadd(a, b, allp), terms0
                    )
                    acc64 = functools.reduce(
                        lambda a, b: S.vadd(a, b, allp), terms64
                    )
                    inv = S.vld(inv_ub[m], "BRC_B32")
                    S.vsts(out_ub[m, 0], S.vcvt(S.vmul(acc0, inv), "bfloat16", allp), allp, "PK_B32")
                    S.vsts(out_ub[m, 64], S.vcvt(S.vmul(acc64, inv), "bfloat16", allp), allp, "PK_B32")
                S.mem_bar("VST_VLD")
            T.copy(out_ub, OUT[0:QUERIES, block, 0:WIDTH])

    return main


def build_programs():
    yield "decodeattn_r4_scan_a", scan(False)
    yield "decodeattn_r4_scan_b", scan(True)
    yield "decodeattn_r4_reduce", reduce()
    yield "decodeattn_r4_reduce_row", reduce_row()
