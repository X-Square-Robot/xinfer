"""ubfinish-r5 opt-in route: UB-vectorized bias finish kernels from
libubfinish_r5.so.

Off by default (XINFER_UBFINISH_R5 unset/0). Ship variant: the rowpass-SIMD
mold (``ubfinish_r5_s{rpt}_c{cols}`` -- one large multi-row T.copy span per
core-step, UNPK_B16 bias widen + vadd + vcvt ROUND_R + PK_B32 store). The
launcher resolves the current stream per call so NPUGraph capture records
the node (gdn-gamma ctypes rule). Per-column-width compiled entries with
their row-tile height; unsupported widths or row counts fall back to the
torch tail upstream (no behavior change).
"""
from xinfer.ops.backends.ascend import _native
import ctypes
import os
from threading import Lock

import torch

_LIB = None
_ENTRIES = {}
_REGISTRY = None
_LOCK = Lock()



def enabled():
    return os.environ.get("XINFER_UBFINISH_R5", "0") == "1"


def _registry():
    return {1280: ("ubfinish_r5_s8_c1280", 8), 2048: ("ubfinish_r5_s4_c2048", 4)}


def supported_cols():
    return frozenset(_registry())


def routes(rows, cols):
    """True when (rows, cols) has a compiled entry and a complete row tiling."""
    entry = _registry().get(cols)
    if entry is None:
        return False
    _, rows_per_tile = entry
    return rows % rows_per_tile == 0 and rows * cols < 2**31


def _entry(cols):
    with _LOCK:
        pair = _ENTRIES.get(cols)
        if pair is None:
            name, _ = _registry()[cols]
            lib = _native.core_library()
            call = getattr(lib, f"xinfer_aot_{name}_call")
            call.restype = ctypes.c_int32
            call.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int64, ctypes.c_void_p]
            err = getattr(lib, f"xinfer_aot_{name}_get_last_error")
            err.restype = ctypes.c_char_p
            pair = (call, err, name)
            _ENTRIES[cols] = pair
    return pair


def finish(out, acc, bias, rows, cols):
    """out[r,c] = rne(acc_f32[r,c] + bias_f32[c]); all tensors contiguous."""
    if not routes(rows, cols):
        raise ValueError(f"ubfinish-r5: no route for {rows}x{cols}")
    name = _registry()[cols][0]
    if _native.launch_kernel(name, (out, acc, bias), (rows,)):
        return
    call, err, name = _entry(cols)
    stream = torch.npu.current_stream(out.device).npu_stream
    status = call(
        ctypes.c_void_p(out.data_ptr()),
        ctypes.c_void_p(acc.data_ptr()),
        ctypes.c_void_p(bias.data_ptr()),
        ctypes.c_int64(rows),
        ctypes.c_void_p(stream),
    )
    if status:
        message = err()
        raise RuntimeError(
            "ubfinish-r5 " + name + ": " + (message.decode("utf-8", "replace") if message else "unknown failure")
        )
