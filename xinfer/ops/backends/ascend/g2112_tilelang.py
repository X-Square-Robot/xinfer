"""g2112-r3 flag-isolated TileLang Cube routes for the 2112 family + prefill QKV.

Default off (every XINFER_G2112_* flag unset keeps the released production
routes). The delivery library is untouched: kernels compile lazily at first
warm-up through the CANN 9.3.0 bisheng toolchain (TILELANG_CACHE_DIR holds
the campaign cache) and launch onto Torch's current NPU stream via the
tilelang torch-exchange install; each route warms once per stream before
graph capture (the bnsd/mx8 discipline).

Numerics (op precheck, card 1, 2026-11-13): K is walked in sequential
128-wide steps inside one L0C FP32 accumulator, so the FP32 kernels are
BITWISE vs the production cann_mm_fp32 buffer (A/B/D: 0 diff cells) and the
BF16 down kernel is BITWISE vs the FP32+cast chain (C: 0 diff cells). The
GLU knife keeps the production finish's per-element op order: BF16 biases
added at FP32, silu g/(1+exp(-g)) in FP32, one RNE round at the store.
"""

import os
from contextlib import contextmanager

import torch

KSTEP = 128

# (bm, bn, stages) per route -- the proto sweep winners (proto-g2112.json,
# proto_samu results); (64, 384) divides both 2112 rows and 3456 cols.
GLU_CFG = (192, 128, 2)
VITQKV_CFG = (192, 256, 2)
DOWN_CFG = (192, 256, 2)
PREFQKV_CFG = (256, 256, 2)
SAMU_CFG = (64, 384)

_FLAG_NAMES = ("GLU", "VITQKV", "DOWN", "PREFQKV", "TORCH_A", "TORCH_C")
G2112_FLAGS = {name: os.environ.get(f"XINFER_G2112_{name}", "") not in ("", "0", "off")
               for name in _FLAG_NAMES}

# The torch-stream exchange must be installed BEFORE the delivery library's
# first NPUGraph capture: a later install is swallowed by the stream provider
# the FFI capture path registers, and every TileLang launch then misses the
# capture (dbg-exch6/7, 2026-11-13). Engine startup imports this module well
# before any device work, so arming it here is early enough; flags-off (base)
# imports nothing.
if any(G2112_FLAGS.values()):
    from tilelang.ascend.torch_exchange import install_torch_npu_stream_exchange as _install_exchange

    _install_exchange()


def _store_rne_bf16(value):
    import tilelang.ascend.language as T

    bits = T.reinterpret(value, "uint32")
    rounded = (bits + T.uint32(0x7FFF) + ((bits >> 16) & T.uint32(1))) & T.uint32(0xFFFF0000)
    special = (bits & T.uint32(0x7F800000)) == T.uint32(0x7F800000)
    return T.cast(T.reinterpret(T.Select(special, bits, rounded), "float32"), "bfloat16")


def _blocked_mm(pad_m, n, k, bm, bn, stages, out_dtype):
    import tilelang.ascend.language as T

    steps = k // KSTEP

    @T.prim_func
    def main(a_handle: T.handle, weight_handle: T.handle, output_handle: T.handle):
        A = T.match_buffer(a_handle, (pad_m, k), "bfloat16")
        W = T.match_buffer(weight_handle, (n, k), "bfloat16")
        OUT = T.match_buffer(output_handle, (pad_m, n), out_dtype)
        nblocks = n // bn
        with T.Kernel((pad_m // bm) * nblocks) as block:
            accum = T.alloc_l0c((bm, bn), "float32")
            a_l1 = T.alloc_l1((bm, KSTEP), "bfloat16")
            w_l1 = T.alloc_l1((bn, KSTEP), "bfloat16")
            m0 = T.cast((block // nblocks) * bm, "int32")
            n0 = T.cast((block % nblocks) * bn, "int32")
            for kk in T.Pipelined(steps, num_stages=stages):
                T.copy(A[m0 : m0 + bm, kk * KSTEP : (kk + 1) * KSTEP], a_l1)
                T.copy(W[n0 : n0 + bn, kk * KSTEP : (kk + 1) * KSTEP], w_l1)
                T.gemm(a_l1, w_l1, accum, transpose_B=True, clear_accum=kk == 0,
                       unit_flag_ctrl=T.Select(kk == steps - 1, 3, 2))
            T.copy(accum, OUT[m0 : m0 + bm, n0 : n0 + bn], unit_flag_ctrl=3)

    return main


def _glu_fused(pad_m, n2, k, bm, bn, stages):
    import tilelang.ascend.language as T

    n = n2 // 2
    steps = k // KSTEP

    @T.prim_func
    def main(a_handle: T.handle, weight_handle: T.handle, bias_handle: T.handle, output_handle: T.handle):
        A = T.match_buffer(a_handle, (pad_m, k), "bfloat16")
        W = T.match_buffer(weight_handle, (n2, k), "bfloat16")
        B = T.match_buffer(bias_handle, (n2,), "bfloat16")
        OUT = T.match_buffer(output_handle, (pad_m, n), "bfloat16")
        nblocks = n // bn
        with T.Kernel((pad_m // bm) * nblocks) as block:
            acc_g = T.alloc_l0c((bm, bn), "float32")
            acc_u = T.alloc_l0c((bm, bn), "float32")
            a_l1 = T.alloc_l1((bm, KSTEP), "bfloat16")
            wg_l1 = T.alloc_l1((bn, KSTEP), "bfloat16")
            wu_l1 = T.alloc_l1((bn, KSTEP), "bfloat16")
            m0 = T.cast((block // nblocks) * bm, "int32")
            n0 = T.cast((block % nblocks) * bn, "int32")
            for kk in T.Pipelined(steps, num_stages=stages):
                T.copy(A[m0 : m0 + bm, kk * KSTEP : (kk + 1) * KSTEP], a_l1)
                T.copy(W[n0 : n0 + bn, kk * KSTEP : (kk + 1) * KSTEP], wg_l1)
                T.copy(W[n + n0 : n + n0 + bn, kk * KSTEP : (kk + 1) * KSTEP], wu_l1)
                T.gemm(a_l1, wg_l1, acc_g, transpose_B=True, clear_accum=kk == 0,
                       unit_flag_ctrl=T.Select(kk == steps - 1, 3, 2))
                T.gemm(a_l1, wu_l1, acc_u, transpose_B=True, clear_accum=kk == 0,
                       unit_flag_ctrl=T.Select(kk == steps - 1, 3, 2))
            ag_ub = T.alloc_shared((bm, bn // 2), "float32")
            au_ub = T.alloc_shared((bm, bn // 2), "float32")
            bg_ub = T.alloc_shared((bn // 2,), "bfloat16")
            bu_ub = T.alloc_shared((bn // 2,), "bfloat16")
            out_ub = T.alloc_shared((bm, bn // 2), "bfloat16")
            T.dual_copy(acc_g, ag_ub, unit_flag_ctrl=3)
            T.dual_copy(acc_u, au_ub, unit_flag_ctrl=3)
            T.dual_copy(B[n0 : n0 + bn], bg_ub)
            T.dual_copy(B[n + n0 : n + n0 + bn], bu_ub)
            with T.SimtVF(threads=128):
                for i, j in T.Parallel(bm, bn // 2):
                    g = ag_ub[i, j] + T.cast(bg_ub[j], "float32")
                    u = au_ub[i, j] + T.cast(bu_ub[j], "float32")
                    out_ub[i, j] = _store_rne_bf16(g / (1.0 + T.exp(-g)) * u)
            T.dual_copy(out_ub, OUT[m0 : m0 + bm, n0 : n0 + bn])

    return main




SAMU_SIMD = os.environ.get("XINFER_SAMU_SIMD", "") not in ("", "0", "off")
CORES = 64
LANES = 64


def _samu_simd(rows, cols, bm=2):
    """GAP-WOSS-NONGE: rowpass CCE-vector silu*mul (production op order).

    One contiguous bm x 2*cols GM->UB span, UNPK_B16 widen of gate and up,
    vneg+vadd(1,.)+vexp+exact vdiv+vmul in FP32, one vcvt ROUND_R with a
    PK_B32 store, one contiguous UB->GM span. Arithmetic order is the
    production finish's ``g / (1 + exp(-g)) * u`` with the single final
    round; the vector exp/div replace the scalar expf with the same FP32
    semantics (measured: bitwise vs the torch FP32 chain on 7.3M cells).
    """
    import tilelang.ascend.language as T

    @T.prim_func
    def main(g_handle: T.handle, out_handle: T.handle):
        G = T.match_buffer(g_handle, (rows * 2 * cols,), "bfloat16")
        OUT = T.match_buffer(out_handle, (rows * cols,), "bfloat16")
        tiles = rows // bm
        span2 = bm * 2 * cols
        span1 = bm * cols
        with T.Kernel(CORES) as core:
            gu_ub = T.alloc_shared((span2,), "bfloat16")
            out_ub = T.alloc_shared((span1,), "bfloat16")
            T.annotate_buffer_versions({gu_ub: 2, out_ub: 2})
            for step in T.Pipelined(T.ceildiv(tiles, CORES), num_stages=2):
                tile = step * CORES + core
                if tile < tiles:
                    base = T.cast(tile * span2, "int32")
                    T.copy(G[base : base + span2], gu_ub)
                    with T.SimdVF():
                        m = T.simd.pset(32)
                        one = T.simd.vdup(1.0, "float32")
                        for r in T.serial(bm):
                            row2 = r * 2 * cols
                            row1 = r * cols
                            for c in range(cols // LANES):
                                g = T.simd.vcvt(
                                    T.simd.vld(gu_ub[row2 + c * LANES], dist="UNPK_B16"),
                                    "float32", part=0,
                                )
                                u = T.simd.vcvt(
                                    T.simd.vld(gu_ub[row2 + cols + c * LANES], dist="UNPK_B16"),
                                    "float32", part=0,
                                )
                                eg = T.simd.vexp(T.simd.vneg(g), m)
                                den = T.simd.vadd(one, eg, m)
                                s = T.simd.vdiv(g, den, m, precision="exact")
                                v = T.simd.vmul(s, u, m)
                                T.simd.vsts(
                                    out_ub[row1 + c * LANES],
                                    T.simd.vcvt(v, "bfloat16", round="ROUND_R"),
                                    dist="PK_B32",
                                )
                    obase = T.cast(tile * span1, "int32")
                    T.copy(out_ub, OUT[obase : obase + span1])

    return main


class _KernelStore:
    """Lazy compile + per-stream warm tracking + route-owned FP32 buffers."""

    def __init__(self):
        self._kernels = {}
        self._warmed = set()
        self._buffers = {}
        self._exchange_done = False

    def _ensure_exchange(self):
        if not self._exchange_done:
            from tilelang.ascend.torch_exchange import install_torch_npu_stream_exchange

            install_torch_npu_stream_exchange()
            self._exchange_done = True

    def kernel(self, key, factory):
        k = self._kernels.get(key)
        if k is None:
            self._ensure_exchange()
            import tilelang

            k = tilelang.compile(factory(), target="ascend", out_idx=None)
            self._kernels[key] = k
        return k

    def gate(self, key, device):
        stream = torch.npu.current_stream(device).npu_stream
        if torch.npu.is_current_stream_capturing() and (stream, key) not in self._warmed:
            raise RuntimeError(f"warm up the g2112 {key} route on this stream before graph capture")
        return stream

    def mark(self, key, stream):
        self._warmed.add((stream, key))

    def fp32_buffer(self, key, device, m, n, stream):
        buf_key = (device.index, stream, key, m, n)
        buf = self._buffers.get(buf_key)
        if buf is None:
            if torch.npu.is_current_stream_capturing():
                raise RuntimeError(f"warm up the g2112 {key} route before graph capture")
            buf = torch.empty(m, n, dtype=torch.float32, device=device)
            self._buffers[buf_key] = buf
        return buf


_K = _KernelStore()


def _padded(x, pad):
    rows = x.shape[0]
    if pad == rows:
        return x
    ap = torch.empty(pad, x.shape[1], dtype=x.dtype, device=x.device)
    ap[:rows].copy_(x)
    return ap


def glu_route(x, weight, bias):
    """[2112,1280] x [6912,1280] + BF16 bias -> [2112,3456] BF16, one launch."""
    m, k = x.shape
    n2 = weight.shape[0]
    bm, bn, stages = GLU_CFG
    pad = m if m % bm == 0 else (m + bm - 1) // bm * bm
    key = ("glu", pad, n2, k, bm, bn, stages)
    stream = _K.gate(key, x.device)
    kernel = _K.kernel(key, lambda: _glu_fused(pad, n2, k, bm, bn, stages))
    a = _padded(x, pad)
    out = x.new_empty((pad, n2 // 2), dtype=torch.bfloat16)
    kernel(a, weight, bias, out)
    _K.mark(key, stream)
    return out if pad == m else out[:m]


def mm_fp32_route(x, weight, tag):
    """The production FP32 [M,N] accumulator, produced by the Cube kernel."""
    m, k = x.shape
    n = weight.shape[0]
    cfg = {"vitqkv": VITQKV_CFG, "prefqkv": PREFQKV_CFG}[tag]
    bm, bn, stages = cfg
    pad = m if m % bm == 0 else (m + bm - 1) // bm * bm
    key = (tag, pad, n, k, bm, bn, stages)
    stream = _K.gate(key, x.device)
    kernel = _K.kernel(key, lambda: _blocked_mm(pad, n, k, bm, bn, stages, "float32"))
    a = _padded(x, pad)
    buf = _K.fp32_buffer(key, x.device, pad, n, stream)
    kernel(a, weight, buf)
    _K.mark(key, stream)
    return buf if pad == m else buf[:m]


@contextmanager
def accumulator_route(x, weight, tag):
    """The CannProjection.accumulator contract over the Cube kernel."""
    yield mm_fp32_route(x, weight, tag)


def down_route(x, weight):
    """[2112,3456] x [1280,3456] -> [2112,1280] BF16 (bitwise vs FP32+cast)."""
    m, k = x.shape
    n = weight.shape[0]
    bm, bn, stages = DOWN_CFG
    pad = m if m % bm == 0 else (m + bm - 1) // bm * bm
    key = ("down", pad, n, k, bm, bn, stages)
    stream = _K.gate(key, x.device)
    kernel = _K.kernel(key, lambda: _blocked_mm(pad, n, k, bm, bn, stages, "bfloat16"))
    a = _padded(x, pad)
    out = x.new_empty((pad, n), dtype=torch.bfloat16)
    kernel(a, weight, out)
    _K.mark(key, stream)
    return out if pad == m else out[:m]


def _samu(rows, cols, bm, bi):
    import tilelang.ascend.language as T

    @T.prim_func
    def main(g_handle: T.handle, out_handle: T.handle):
        G = T.match_buffer(g_handle, (rows, 2 * cols), "bfloat16")
        OUT = T.match_buffer(out_handle, (rows, cols), "bfloat16")
        iblocks = cols // bi
        with T.Kernel((rows // bm) * iblocks) as block:
            g_bf = T.alloc_shared((bm, bi), "bfloat16")
            u_bf = T.alloc_shared((bm, bi), "bfloat16")
            o_bf = T.alloc_shared((bm, bi), "bfloat16")
            m0 = T.cast((block // iblocks) * bm, "int32")
            i0 = T.cast((block % iblocks) * bi, "int32")
            T.copy(G[m0 : m0 + bm, i0 : i0 + bi], g_bf)
            T.copy(G[m0 : m0 + bm, cols + i0 : cols + i0 + bi], u_bf)
            with T.SimtVF(threads=128):
                for i, j in T.Parallel(bm, bi):
                    g = T.cast(g_bf[i, j], "float32")
                    u = T.cast(u_bf[i, j], "float32")
                    o_bf[i, j] = _store_rne_bf16(g / (1.0 + T.exp(-g)) * u)
            T.copy(o_bf, OUT[m0 : m0 + bm, i0 : i0 + bi])

    return main


def samu_route(gate_up):
    """[M, 2I] BF16 -> [M, I] BF16: silu(g)*u in FP32, one RNE store.

    Completes the A knife (aclnn addmm folds the bias for free; the lib
    activ_and_mul measured 443.7 us on [2112, 6912] -- unusable; the plain
    torch tail 175.7 us -- a regression). 43.0 us at (64, 384).
    """
    rows, twon = gate_up.shape
    cols = twon // 2
    bm, bi = SAMU_CFG
    if rows % bm or cols % bi:
        raise RuntimeError(f"samu route requires {bm}|rows and {bi}|cols, got {rows}x{cols}")
    if SAMU_SIMD and (rows, cols) == (2112, 3456) and gate_up.is_contiguous():
        key = ("samu_simd", rows, cols)
        stream = _K.gate(key, gate_up.device)
        kernel = _K.kernel(key, lambda: _samu_simd(rows, cols))
        out = gate_up.new_empty((rows, cols), dtype=torch.bfloat16)
        kernel(gate_up.view(-1), out.view(-1))
        _K.mark(key, stream)
        return out
    key = ("samu", rows, cols, bm, bi)
    stream = _K.gate(key, gate_up.device)
    kernel = _K.kernel(key, lambda: _samu(rows, cols, bm, bi))
    out = gate_up.new_empty((rows, cols), dtype=torch.bfloat16)
    kernel(gate_up, out)
    _K.mark(key, stream)
    return out
