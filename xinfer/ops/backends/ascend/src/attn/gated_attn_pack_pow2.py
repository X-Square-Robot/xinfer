"""Gated attention packing for M=16, H=8 and D=256.

The power-of-two dimensions allow shifts and masks for index decomposition.
The FP32 sigmoid, both round-to-nearest-even conversions and the store order
match the generic packing kernel.
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
