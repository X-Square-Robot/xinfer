"""R6B-BRIDGE B2 r2: pow2-specialized gating pack for the decode shape.

The r4 streaming kernel decomposes every element index with six runtime
int64 divmods (row/head/batch/dim splits); at the decode shape every
divisor is a power of two (H=8, M=16, D=256, HD=2048), so this variant
bakes those constants and replaces the decomposition with shifts/masks.
The per-element scalar arithmetic is unchanged -- the same FP32 sigmoid
``1/(1+exp(-x))``, the same two RNE roundings, the same store order -- so
the outputs stay bitwise with the r4 kernel and the torch tail on this
envelope (probe requirement).  Envelope: B*M*H*D elements with M=16, H=8,
D=256, any B; callers fall back to the r4 kernel otherwise.
"""

import tilelang.ascend.language as T


def _store_rne_bf16(value):
    bits = T.reinterpret(value, "uint32")
    rounded = (bits + T.uint32(0x7FFF) + ((bits >> 16) & T.uint32(1))) & T.uint32(0xFFFF0000)
    special = (bits & T.uint32(0x7F800000)) == T.uint32(0x7F800000)
    return T.cast(T.reinterpret(T.Select(special, bits, rounded), "float32"), "bfloat16")


def gated_attn_bnsd_pack_pow2():
    @T.prim_func
    def main(
        rh: T.handle,
        oh: T.handle,
        gh: T.handle,
        count: T.int64,
        gspan: T.int64,
        hd: T.int64,
        rows: T.int64,
        d: T.int64,
        pitch: T.int64,
        offset: T.int64,
    ):
        res = T.match_buffer(rh, (count,), "bfloat16")
        out = T.match_buffer(oh, (count,), "bfloat16")
        gate = T.match_buffer(gh, (gspan,), "bfloat16")
        blocks = T.max(T.int64(1), T.min(T.ceildiv(count, 256), T.int64(64)))
        with T.Kernel(blocks) as block:
            with T.SimtVF(threads=256):
                for chunk in T.serial(T.ceildiv(count, blocks * 256)):
                    for lane in T.Parallel(256):
                        idx = (chunk * blocks + T.cast(block, "int64")) * 256 + lane
                        if idx < count:
                            # M=16, H=8, D=256, HD=2048 (compile-time constants):
                            # r = idx >> 11, j = idx & 2047, m = r & 15,
                            # out slot = ((b*8 + h)*16 + m)*256 + e with
                            # b = r >> 4, h = j >> 8, e = j & 255.
                            r = idx >> 11
                            j = idx - (r << 11)
                            m = r - ((r >> 4) << 4)
                            h = j >> 8
                            out_idx = ((((((r >> 4) << 3) + h) << 4) + m) << 8) + (j - (h << 8))
                            g_idx = r * pitch + offset + j
                            o = T.cast(out[out_idx], "float32")
                            g = T.cast(gate[g_idx], "float32")
                            sig = _store_rne_bf16(1.0 / (1.0 + T.exp(-g)))
                            res[idx] = _store_rne_bf16(o * T.cast(sig, "float32"))

    return main


def build_programs():
    yield "gated_attn_bnsd_pack_pow2", gated_attn_bnsd_pack_pow2()
