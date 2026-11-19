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

"""Split-K cache attention with a gated output for Wall-X 2B (B1, H8, KV2, M16, D256).

Each scan block loads a 64-key chunk and a 32-row query tile. For cache lengths
432, 448, 464 and 480, the last tile slides back to the last full chunk; the
visibility predicate excludes keys already covered by an earlier tile. A key
must also belong to the prefix or new-key window and meet its LTS/UTE bounds.

The reducer combines FP32 softmax partials, rounds attention to BF16, applies the
FP32 sigmoid gate with the same BF16 rounding boundaries as gated_attn_bnsd_pack_pow2,
and stores contiguous token-major output. A separate reducer can write head-major
attention for callers that apply the gate later."""

import functools

import tilelang.ascend.language as T
import tilelang.ascend.language.simd as S

HEADS, KVHEADS, QUERIES, WIDTH, KEY_TILE = 8, 2, 16, 256, 64
TILE_ROWS = 32  # 2 heads x 16 queries per (kv, tile)
TILES = 2  # per kv group: HEADS/KVHEADS/2 = 2
PAIR_ROWS = TILE_ROWS + 1  # 32 o rows + 1 row packing m (0..31) and l (32..63)


def _chunks(keys):
    return (keys + KEY_TILE - 1) // KEY_TILE


def _pair(kv, tile, chunk, chunks):
    return (kv * TILES + tile) * chunks + chunk


def _rne_bf16_bits(value):
    """Manual RNE fp32->bf16 of a scalar, the pack kernel's exact helper."""
    bits = T.reinterpret(value, "uint32")
    rounded = (bits + T.uint32(0x7FFF) + ((bits >> 16) & T.uint32(1))) & T.uint32(0xFFFF0000)
    special = (bits & T.uint32(0x7F800000)) == T.uint32(0x7F800000)
    return T.Select(special, bits, rounded)


def scan(keys: int):
    chunks = _chunks(keys)

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
        K = T.match_buffer(kh, (KVHEADS * keys, WIDTH), "bfloat16")
        V = T.match_buffer(vh, (KVHEADS * keys, WIDTH), "bfloat16")
        FM = T.match_buffer(fmh, (keys, 2), "int32")
        PL = T.match_buffer(plh, (1,), "int32")
        P = T.match_buffer(ph, (KVHEADS * TILES * chunks, PAIR_ROWS, WIDTH), "float32")
        scale = T.reinterpret("float32", T.cast(scale_bits, "uint32"))
        with T.MixedKernel(KVHEADS * TILES * chunks, sids=2) as (bx, sid):
            T.set_hf32_mode(None)
            q_l1 = T.alloc_l1((TILE_ROWS, WIDTH), "bfloat16")
            k_l1 = T.alloc_l1((KEY_TILE, WIDTH), "bfloat16")
            v_l1 = T.alloc_l1((WIDTH, KEY_TILE), "bfloat16")
            p_l1 = T.alloc_l1((TILE_ROWS, KEY_TILE), "bfloat16")
            qk_cc = T.alloc_l0c((TILE_ROWS, KEY_TILE), "float32")
            pv_cc = T.alloc_l0c((TILE_ROWS, WIDTH), "float32")
            s_ub = T.alloc_shared((16, KEY_TILE), "float32")
            visible_ub = T.alloc_shared((16, KEY_TILE), "float32")
            p_ub = T.alloc_shared((16, KEY_TILE), "bfloat16")
            o_ub = T.alloc_shared((16, WIDTH), "float32")
            m_ub = T.alloc_shared((16,), "float32")
            sums = T.alloc_shared((16,), "float32")
            mask16_ub = T.alloc_shared((QUERIES, KEY_TILE), "float32")
            kv = bx // (TILES * chunks)
            rest = bx - kv * TILES * chunks
            tile = rest // chunks
            chunk = rest - tile * chunks
            prefix = T.cast(PL[0], "int64")
            # The last window slides back to avoid a partial tile: chunks 0..C-2 own [c*T, min((c+1)*T, L-T)),
            # the last chunk slides to [L-T, L); the windows stay disjoint.
            base_row = T.alloc_var("int64", init=chunk * KEY_TILE)
            coverage = T.alloc_var("int64", init=keys)
            if chunk == chunks - 1:
                base_row = keys - KEY_TILE
            if chunk < chunks - 1:
                coverage = T.min((chunk + 1) * KEY_TILE, keys - KEY_TILE)
            k_row0 = T.cast(kv * keys + base_row, "int32")
            T.copy(K[k_row0 : k_row0 + KEY_TILE, 0:WIDTH], k_l1)
            T.copy(V[k_row0 : k_row0 + KEY_TILE, 0:WIDTH], v_l1, transpose=True)
            # Visibility table, once per block: mask16_ub[m, lane] = 1.0 iff
            # column base_row+lane is visible to query row m under the bridge
            # keep rule (prefix/new_pos window) and the flash lts/ute bounds.
            with T.SimtVF(threads=KEY_TILE):
                for lane in T.Parallel(KEY_TILE):
                    kr = base_row + lane
                    keep_col = (kr < prefix) or (kr >= new_pos and kr < new_pos + QUERIES)
                    lts_col = FM[kr, 0]
                    ute_col = FM[kr, 1]
                    for mm in range(QUERIES):
                        vis = keep_col and (mm < lts_col) and (mm >= ute_col)
                        mask16_ub[mm, lane] = T.Select(
                            vis and kr < coverage, T.float32(1), T.float32(0)
                        )
            q_row0 = T.cast((kv * 4 + tile * 2) * QUERIES, "int32")
            T.copy(Q[q_row0 : q_row0 + TILE_ROWS, 0:WIDTH], q_l1)
            T.gemm(q_l1, k_l1, qk_cc, transpose_B=True, clear_accum=True)
            T.dual_copy(qk_cc, s_ub)
            with T.SimdVF():
                allp = S.pset(32, "PAT_ALL")
                one = S.pset(32, "PAT_VL1")
                ninf = S.vdup(-T.infinity("float32"), "float32", allp)
                onev = S.vdup(T.float32(1), "float32", allp)
                zerov = S.vdup(T.float32(0), "float32", allp)
                for r in range(16):
                    # sid owns rows sid*16..sid*16+16 of the tile: one head,
                    # queries 0..15, so the mask row is r for both sids.
                    s0 = S.vmuls(S.vld(s_ub[r, 0], "NORM"), scale, allp)
                    v0 = S.vcmps(S.vld(mask16_ub[r, 0], "NORM"), T.float32(0), allp, "gt")
                    s0 = S.vsel(s0, ninf, v0)
                    S.vsts(s_ub[r, 0], s0, allp, "NORM_B32")
                    S.vsts(visible_ub[r, 0], S.vsel(onev, zerov, v0), allp, "NORM_B32")
                    S.vsts(m_ub[r], S.vcmax(s0, allp), one, "ONEPT_B32")
                S.mem_bar("VST_VLD")
            with T.SimdVF():
                allp = S.pset(32, "PAT_ALL")
                one = S.pset(32, "PAT_VL1")
                zero = S.vdup(T.float32(0), "float32", allp)
                for r in range(16):
                    row_max = S.vld(m_ub[r], "BRC_B32")
                    prob0 = S.vexpdif(S.vld(s_ub[r, 0], "NORM"), row_max, allp)
                    prob0 = S.vsel(
                        prob0, zero, S.vcmps(S.vld(visible_ub[r, 0], "NORM"), T.float32(0), allp, "gt")
                    )
                    S.vsts(p_ub[r, 0], S.vcvt(prob0, "bfloat16", allp), allp, "PK_B32")
                    S.vsts(sums[r], S.vcadd(prob0, allp), one, "ONEPT_B32")
                S.mem_bar("VST_VLD")
            T.dual_copy(p_ub, p_l1)
            T.gemm(p_l1, v_l1, pv_cc, transpose_B=True, clear_accum=True)
            T.dual_copy(pv_cc, o_ub)
            pair = _pair(kv, tile, chunk, chunks)
            T.copy(o_ub, P[pair, sid * 16 : sid * 16 + 16, 0:WIDTH])
            T.copy(m_ub, P[pair, PAIR_ROWS - 1, sid * 16 : sid * 16 + 16])
            T.copy(sums, P[pair, PAIR_ROWS - 1, 32 + sid * 16 : 32 + sid * 16 + 16])

    return main


def reduce_packed(keys: int):
    """16 blocks = (head, 128-column half); merged attention -> BF16 -> the
    pack_pow2 gate tail -> packed [M, H*D] store."""
    chunks = _chunks(keys)

    @T.prim_func
    def main(ph: T.handle, oh: T.handle, gh: T.handle, pitch: T.int64):
        P = T.match_buffer(ph, (KVHEADS * TILES * chunks, PAIR_ROWS, WIDTH), "float32")
        OUT = T.match_buffer(oh, (QUERIES, HEADS * WIDTH), "bfloat16")
        GATE = T.match_buffer(gh, (16 * pitch + HEADS * WIDTH,), "bfloat16")
        with T.Kernel(HEADS * 2) as block:
            h = block // 2
            c2 = block - h * 2
            kv = h // 4
            tile = (h - kv * 4) // 2
            lr0 = (h - (kv * 4 + tile * 2)) * 16
            col0 = c2 * 128
            o7 = T.alloc_shared((chunks * 16, 128), "float32")
            ml_m = T.alloc_shared((chunks, 16), "float32")
            ml_l = T.alloc_shared((chunks, 16), "float32")
            w_ub = T.alloc_shared((16, chunks), "float32")
            inv_ub = T.alloc_shared((16,), "float32")
            attn_ub = T.alloc_shared((16, 128), "float32")
            base = _pair(kv, tile, 0, chunks)
            for c in range(chunks):
                T.copy(P[base + c, lr0 : lr0 + 16, col0 : col0 + 128], o7[c * 16 : (c + 1) * 16, 0:128])
                T.copy(P[base + c, PAIR_ROWS - 1, lr0 : lr0 + 16], ml_m[c, 0:16])
                T.copy(P[base + c, PAIR_ROWS - 1, 32 + lr0 : 32 + lr0 + 16], ml_l[c, 0:16])
            with T.SimtVF(threads=16):
                for m in T.Parallel(16):
                    m_star = T.alloc_var("float32", init=-T.infinity("float32"))
                    for c in T.serial(chunks):
                        m_star = T.Select(ml_m[c, m] > m_star, ml_m[c, m], m_star)
                    l_star = T.alloc_var("float32", init=0)
                    for c in T.serial(chunks):
                        weight = T.exp(ml_m[c, m] - m_star)
                        w_ub[m, c] = weight
                        l_star = l_star + weight * ml_l[c, m]
                    inv_ub[m] = T.float32(1) / l_star
            with T.SimdVF():
                S.mem_bar("VST_VLD")
                allp = S.pset(32, "PAT_ALL")
                for m in range(16):
                    weights = [S.vld(w_ub[m, c], "BRC_B32") for c in range(chunks)]
                    terms0 = [S.vmul(S.vld(o7[c * 16 + m, 0], "NORM"), weights[c]) for c in range(chunks)]
                    terms64 = [S.vmul(S.vld(o7[c * 16 + m, 64], "NORM"), weights[c]) for c in range(chunks)]
                    acc0 = functools.reduce(lambda a, b: S.vadd(a, b, allp), terms0)
                    acc64 = functools.reduce(lambda a, b: S.vadd(a, b, allp), terms64)
                    inv = S.vld(inv_ub[m], "BRC_B32")
                    S.vsts(attn_ub[m, 0], S.vmul(acc0, inv, allp), allp, "NORM_B32")
                    S.vsts(attn_ub[m, 64], S.vmul(acc64, inv, allp), allp, "NORM_B32")
                S.mem_bar("VST_VLD")
            # gate tail: the gated_attn_bnsd_pack_pow2 scalar arithmetic on
            # the merged attention -- RNE to BF16 (the FIA output rounding),
            # FP32 sigmoid of the gate with manual RNE, one more RNE product.
            colbase = h * WIDTH + col0
            with T.SimtVF(threads=256):
                for seg in T.serial(8):
                    for lane in T.Parallel(256):
                        idx = seg * 256 + lane
                        if idx < 16 * 128:
                            m = idx // 128
                            e = idx - m * 128
                            attn = attn_ub[m, e]
                            o_bits = _rne_bf16_bits(attn)
                            o = T.cast(T.reinterpret(o_bits, "float32"), "bfloat16")
                            g = T.cast(GATE[m * pitch + colbase + e], "float32")
                            sig_bits = _rne_bf16_bits(1.0 / (1.0 + T.exp(-g)))
                            sig = T.cast(T.reinterpret(sig_bits, "float32"), "bfloat16")
                            res = _rne_bf16_bits(T.cast(o, "float32") * T.cast(sig, "float32"))
                            OUT[m, colbase + e] = T.cast(T.reinterpret(res, "float32"), "bfloat16")

    return main


def reduce_bnsd(keys: int):
    """Merge softmax partials into BF16 head-major attention, leaving the gate to the caller."""
    chunks = _chunks(keys)

    @T.prim_func
    def main(ph: T.handle, oh: T.handle):
        P = T.match_buffer(ph, (KVHEADS * TILES * chunks, PAIR_ROWS, WIDTH), "float32")
        OUT = T.match_buffer(oh, (HEADS * QUERIES, WIDTH), "bfloat16")
        with T.Kernel(HEADS) as block:
            h = block
            kv = h // 4
            tile = (h - kv * 4) // 2
            lr0 = (h - (kv * 4 + tile * 2)) * 16
            o7 = T.alloc_shared((chunks * 16, WIDTH), "float32")
            ml_m = T.alloc_shared((chunks, 16), "float32")
            ml_l = T.alloc_shared((chunks, 16), "float32")
            w_ub = T.alloc_shared((16, chunks), "float32")
            inv_ub = T.alloc_shared((16,), "float32")
            out_ub = T.alloc_shared((16, WIDTH), "bfloat16")
            base = _pair(kv, tile, 0, chunks)
            for c in range(chunks):
                T.copy(P[base + c, lr0 : lr0 + 16, 0:WIDTH], o7[c * 16 : (c + 1) * 16, 0:WIDTH])
                T.copy(P[base + c, PAIR_ROWS - 1, lr0 : lr0 + 16], ml_m[c, 0:16])
                T.copy(P[base + c, PAIR_ROWS - 1, 32 + lr0 : 32 + lr0 + 16], ml_l[c, 0:16])
            with T.SimtVF(threads=16):
                for m in T.Parallel(16):
                    m_star = T.alloc_var("float32", init=-T.infinity("float32"))
                    for c in T.serial(chunks):
                        m_star = T.Select(ml_m[c, m] > m_star, ml_m[c, m], m_star)
                    l_star = T.alloc_var("float32", init=0)
                    for c in T.serial(chunks):
                        weight = T.exp(ml_m[c, m] - m_star)
                        w_ub[m, c] = weight
                        l_star = l_star + weight * ml_l[c, m]
                    inv_ub[m] = T.float32(1) / l_star
            with T.SimdVF():
                S.mem_bar("VST_VLD")
                allp = S.pset(32, "PAT_ALL")
                for m in range(16):
                    weights = [S.vld(w_ub[m, c], "BRC_B32") for c in range(chunks)]
                    terms0 = [S.vmul(S.vld(o7[c * 16 + m, 0], "NORM"), weights[c]) for c in range(chunks)]
                    terms64 = [S.vmul(S.vld(o7[c * 16 + m, 64], "NORM"), weights[c]) for c in range(chunks)]
                    terms128 = [S.vmul(S.vld(o7[c * 16 + m, 128], "NORM"), weights[c]) for c in range(chunks)]
                    terms192 = [S.vmul(S.vld(o7[c * 16 + m, 192], "NORM"), weights[c]) for c in range(chunks)]
                    acc0 = functools.reduce(lambda a, b: S.vadd(a, b, allp), terms0)
                    acc64 = functools.reduce(lambda a, b: S.vadd(a, b, allp), terms64)
                    acc128 = functools.reduce(lambda a, b: S.vadd(a, b, allp), terms128)
                    acc192 = functools.reduce(lambda a, b: S.vadd(a, b, allp), terms192)
                    inv = S.vld(inv_ub[m], "BRC_B32")
                    S.vsts(out_ub[m, 0], S.vcvt(S.vmul(acc0, inv, allp), "bfloat16", allp), allp, "PK_B32")
                    S.vsts(out_ub[m, 64], S.vcvt(S.vmul(acc64, inv, allp), "bfloat16", allp), allp, "PK_B32")
                    S.vsts(out_ub[m, 128], S.vcvt(S.vmul(acc128, inv, allp), "bfloat16", allp), allp, "PK_B32")
                    S.vsts(out_ub[m, 192], S.vcvt(S.vmul(acc192, inv, allp), "bfloat16", allp), allp, "PK_B32")
                S.mem_bar("VST_VLD")
            T.copy(out_ub, OUT[h * 16 : h * 16 + 16, 0:WIDTH])

    return main


COMPILE_OPTIONS = ()

_BUILD_KEYS = (432, 448, 464, 480)


def build_programs():
    for keys in _BUILD_KEYS:
        yield f"fia_fuse_scan_l{keys}", scan(keys)
    for keys in _BUILD_KEYS:
        yield f"fia_fuse_reduce_l{keys}", reduce_packed(keys)
    for keys in _BUILD_KEYS:
        yield f"fia_fuse_reduce_bnsd_l{keys}", reduce_bnsd(keys)
