"""conv-gemm knife mount: dconv4_c6144 through the AOT packed-launch ABI.

XINFER_CONVGEMM=1 (default off) activates the kernel route inside
``glue.cann_gated_delta_net.CannGatedDeltaNet.__call__``; every
other code path (route disabled, envelope miss, launcher failure) keeps the
released F.conv1d chain bit for bit. The library lives in its own .so
(XINFER_CONVGEMM_LIB) loaded through ctypes with the unified AOT packed-launch
ABI (pointers, int64 scalars, stream) -- the soupkill precedent, graph-capture
safe because the launch goes to the live torch stream.
"""
from __future__ import annotations

import atexit
import ctypes
import json
import os
import threading

import torch

from xinfer.ops.backends.ascend import _native

CHANNELS = 6144

_LOCK = threading.Lock()
_LIB = None
_ENTRY = None
_PAD = {}
_STATS = {"routed": 0, "fell_back": 0, "errors": []}


def enabled() -> bool:
    return os.environ.get("XINFER_CONVGEMM", "0") == "1"


def stats() -> dict:
    """Route counters for probes and the smoke receipts."""
    out = dict(_STATS)
    out["errors"] = list(_STATS["errors"][:8])
    return out


def _library():
    global _LIB, _ENTRY
    if _LIB is not None:
        return _ENTRY
    with _LOCK:
        if _LIB is None:
            _LIB = _native.core_library()
            call = _LIB.xinfer_aot_dconv4_c6144_call
            call.restype = ctypes.c_int32
            call.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int64] * 2 + [ctypes.c_void_p]
            err = _LIB.xinfer_aot_dconv4_c6144_get_last_error
            err.restype = ctypes.c_char_p
            _ENTRY = (call, err)
    return _ENTRY


def _pad(device, dtype):
    key = (str(device), dtype)
    got = _PAD.get(key)
    if got is None or got.device != device:
        got = torch.zeros((3, CHANNELS), dtype=dtype, device=device)
        _PAD[key] = got
    return got


def _weight_rows(conv_weight):
    """[C, 4] BF16 depthwise filters -> cached [4, C] tap-major copy (values unchanged).

    The copy is cached as an attribute of the weight tensor itself: lifetime and
    identity follow the parameter, so a different tensor landing on a recycled
    device address can never be served a stale entry (the data_ptr-keyed dict of
    the first draft failed exactly that way in probe3).
    """
    got = getattr(conv_weight, "_conv_gemm_wt", None)
    if got is None or got.shape != (conv_weight.shape[1], conv_weight.shape[0]) \
            or got.device != conv_weight.device:
        got = conv_weight.t().contiguous()
        conv_weight._conv_gemm_wt = got
    return got


def _envelope(qkv, conv_weight):
    """True when the kernel route covers this call exactly."""
    return (
        qkv.ndim == 3
        and qkv.shape[0] == 1
        and qkv.dtype == torch.bfloat16
        and qkv.shape[2] == CHANNELS
        and qkv.stride(-1) == 1
        and qkv.stride(1) % 8 == 0          # 16-byte aligned row starts
        and qkv.stride(1) >= CHANNELS
        and conv_weight.dtype == torch.bfloat16
        and conv_weight.shape == (CHANNELS, 4)
        and conv_weight.is_contiguous()
        and conv_weight.device == qkv.device
        and qkv.data_ptr() % 16 == 0
        and 1 <= qkv.shape[1] <= (1 << 26)
    )


def conv_mixed_tokenmajor(qkv, conv_weight):
    """``mixed`` rows [1, T, C] BF16 token-major = causal depthwise conv, pre-SiLU.

    Returns None (caller falls back) when the envelope misses or the launch fails.
    """
    try:
        if not _envelope(qkv, conv_weight):
            _STATS["fell_back"] += 1
            return None
        tokens = qkv.shape[1]
        w = _weight_rows(conv_weight)
        pad = _pad(qkv.device, qkv.dtype)
        out = qkv.new_empty((1, tokens, CHANNELS))
        call, error = _library()
        stream = torch.npu.current_stream(qkv.device).npu_stream
        # the single library's TVM-FFI face serves first (the unified launch boundary);
        # the ctypes launcher of the family library is the fallback below
        if _native.launch_kernel("dconv4_c6144", (out, qkv, w, pad),
                                 (tokens, qkv.stride(1))):
            _STATS["routed"] += 1
            return out
        status = call(
            ctypes.c_void_p(out.data_ptr()),
            ctypes.c_void_p(qkv.data_ptr()),
            ctypes.c_void_p(w.data_ptr()),
            ctypes.c_void_p(pad.data_ptr()),
            ctypes.c_int64(tokens),
            ctypes.c_int64(qkv.stride(1)),
            ctypes.c_void_p(stream),
        )
        if status:
            _STATS["fell_back"] += 1
            message = error()
            _STATS["errors"].append(message.decode("utf-8", "replace") if message else f"status {status}")
            return None
        _STATS["routed"] += 1
        return out
    except Exception as exc:  # noqa: BLE001 - fail-safe to the released chain
        _STATS["fell_back"] += 1
        _STATS["errors"].append(repr(exc))
        return None


def _dump_stats():
    if not os.environ.get("XINFER_CONVGEMM_STATS"):
        return
    try:
        path = os.environ.get("XINFER_CONVGEMM_STATS")
        with open(path, "w") as fh:
            json.dump(stats(), fh, indent=1)
    except Exception:  # noqa: BLE001 - stats must never fail a run
        pass


atexit.register(_dump_stats)
