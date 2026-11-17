"""smallm-r3 opt-in routes: M16 action-loop kernels from libsmallm_r3.so.

Round-3 SMALLM-K. Off by default (XINFER_SMALLM_R3=off). Families:
oproj, gateup, down, qkv (comma list, or 1/all/on for all four). The library
path defaults to the campaign artifact and can be overridden with
XINFER_SMALLM_R3_LIB. All launches resolve the current stream per call so
NPUGraph capture records the node (gdn-gamma ctypes rule).
"""
import ctypes
import os
from threading import Lock

import torch

_ENABLED = None
_LIB = None
_ENTRIES = {}
_LOCK = Lock()
_PACKED = {}

from xinfer.ops.backends.ascend import _native
_DEFAULT_LIB = _native._default_library_path("libsmallm_r3.so")
_FAMILIES = ("oproj", "gateup", "down", "qkv")


def _parse():
    global _ENABLED
    raw = os.environ.get("XINFER_SMALLM_R3", "off").strip().lower()
    if raw in ("off", "", "0"):
        _ENABLED = frozenset()
    elif raw in ("1", "all", "on"):
        _ENABLED = frozenset(_FAMILIES)
    else:
        wanted = {t.strip() for t in raw.split(",")}
        unknown = wanted - set(_FAMILIES)
        if unknown:
            raise RuntimeError(f"XINFER_SMALLM_R3 unknown families: {sorted(unknown)}")
        _ENABLED = frozenset(wanted)


def enabled(family):
    if _ENABLED is None:
        _parse()
    return family in _ENABLED


def _library():
    global _LIB
    if _LIB is None:
        path = os.environ.get("XINFER_SMALLM_R3_LIB", _DEFAULT_LIB)
        _LIB = ctypes.CDLL(path)
    return _LIB


def _entry(name):
    with _LOCK:
        pair = _ENTRIES.get(name)
        if pair is None:
            lib = _library()
            call = getattr(lib, f"xinfer_aot_{name}_call")
            call.restype = ctypes.c_int32
            call.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_void_p]
            err = getattr(lib, f"xinfer_aot_{name}_get_last_error")
            err.restype = ctypes.c_char_p
            pair = (call, err)
            _ENTRIES[name] = pair
    return pair


def _launch(name, tensors):
    if _native.launch_kernel(name, tensors):
        return
    call, err = _entry(name)
    stream = torch.npu.current_stream(tensors[0].device).npu_stream
    status = call(
        *[ctypes.c_void_p(t.data_ptr()) for t in tensors], ctypes.c_void_p(stream)
    )
    if status:
        raise RuntimeError(
            f"smallm-r3 {name}: " + (err() or b"?").decode("utf-8", "replace")
        )


def o_proj(a_bnsd, weight):
    """[16,16,128] BNSD x [1024,2048] -> BF16 [16,1024] (store-round RNE)."""
    out = a_bnsd.new_empty((16, 1024), dtype=torch.bfloat16)
    _launch("smallm_r3_o_proj", (a_bnsd, weight, out))
    return out


def o_proj_row(x, weight):
    """[16,2048] row-major x [1024,2048] -> BF16 [16,1024] (LAYOUT-OPROJ K1).

    The o-projection and the down projection are the same GEMM (M=16, K=2048,
    N=1024, W [N,K]); this launches the released smallm_r3_down body, so the
    k order (h*128+d ascending) and the RNE store round are the ones the
    bitwise o_proj chain already validated. Only the A load changes: one
    contiguous 64KB DMA instead of 16 strided 4KB head copies.
    """
    out = x.new_empty((16, 1024), dtype=torch.bfloat16)
    _launch("smallm_r3_down", (x, weight, out))
    return out


def down(x, weight):
    """[16,2048] x [1024,2048] -> BF16 [16,1024]."""
    out = x.new_empty((16, 1024), dtype=torch.bfloat16)
    _launch("smallm_r3_down", (x, weight, out))
    return out


def qkv_fp32(x, weight):
    """[16,1024] x [2560,1024] -> FP32 [16,2560] (accumulator contract)."""
    out = x.new_empty((16, 2560), dtype=torch.float32)
    _launch("smallm_r3_qkv", (x, weight, out))
    return out


def gateup(x, weight):
    """[16,1024] x [4096,1024] -> silu(g)*u BF16 [16,2048]; one-time prepack."""
    key = (weight.data_ptr(), tuple(weight.shape), weight.dtype)
    packed = _PACKED.get(key)
    if packed is None:
        if torch.npu.is_current_stream_capturing():
            raise RuntimeError(
                "warm up the smallm-r3 gateup route on this stream before graph capture"
            )
        half = weight.shape[0] // 2
        interleaved = weight.new_empty(weight.shape)
        interleaved[0::2] = weight[:half]
        interleaved[1::2] = weight[half:]
        packed = (weight, interleaved.contiguous())
        _PACKED[key] = packed
    out = x.new_empty((16, 2048), dtype=torch.bfloat16)
    _launch("smallm_r3_gateup_glu", (x, packed[1], out))
    return out
