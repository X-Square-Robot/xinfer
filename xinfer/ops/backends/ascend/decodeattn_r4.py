"""Optional small-M cache-attention scan/reduce pair in the native library.

``XINFER_DECODEATTN_R4`` is off by default. The supported Wall-OSS call is
B1/H16/KV2/M16/D128/L784 with new_pos=768. Prefix length, the new-key
window and flash-mask LTS/UTE bounds retain their existing semantics.
This route is tolerance-level relative to FIA, not bitwise equivalent.
Launches resolve the current stream for graph capture.
"""
from xinfer.ops.backends.ascend import _native
import ctypes
import os
import struct
from threading import Lock

import torch

_ENABLED = None
_SCAN_VARIANT = None
_LIB = None
_ENTRIES = {}
_LOCK = Lock()

_H, _KVH, _M, _D, _L = 16, 2, 16, 128, 784
_P_SHAPE = (28, 65, 128)


def _row_enabled():
    """LAYOUT-OPROJ K1: token-major output (XINFER_DECODEATTN_R4_ROW=1).

    When on, the reduce twin stores OUT[m, h, d] and the route returns a
    contiguous [1, M, H*D] buffer (zero-copy view) instead of BNSD. The
    arithmetic of the kernel is unchanged -- only the GM store address.
    """
    return os.environ.get(_ROW_ENV, "off").strip().lower() not in ("off", "", "0")


def _parse():
    global _ENABLED, _SCAN_VARIANT
    raw = os.environ.get("XINFER_DECODEATTN_R4", "off").strip().lower()
    _ENABLED = raw not in ("off", "", "0")
    _SCAN_VARIANT = os.environ.get("XINFER_DECODEATTN_R4_SCAN", "b").strip().lower()
    if _SCAN_VARIANT not in ("a", "b"):
        raise RuntimeError("XINFER_DECODEATTN_R4_SCAN must be a or b")


def enabled():
    if _ENABLED is None:
        _parse()
    return _ENABLED


def _library():
    return _native.core_library()


def _entry(name, n_ptr, scalar_types):
    with _LOCK:
        pair = _ENTRIES.get(name)
        if pair is None:
            lib = _library()
            call = getattr(lib, f"xinfer_aot_{name}_call")
            call.restype = ctypes.c_int32
            call.argtypes = [ctypes.c_void_p] * n_ptr + list(scalar_types) + [ctypes.c_void_p]
            err = getattr(lib, f"xinfer_aot_{name}_get_last_error")
            err.restype = ctypes.c_char_p
            pair = (call, err)
            _ENTRIES[name] = pair
    return pair


_ROW_ENV = "XINFER_DECODEATTN_R4_ROW"
_DEBUG_PATH = os.environ.get("XINFER_DECODEATTN_R4_DEBUG")


def _debug(reason, q, k_cache, flash_mask, new_pos, scale):
    if not _DEBUG_PATH:
        return
    try:
        with open(_DEBUG_PATH, "a") as fh:
            fh.write(reason
                     + f" q={tuple(q.shape)} qtc={q.transpose(1, 2).is_contiguous()}"
                     + f" k={tuple(k_cache.shape)} kback={k_cache.transpose(1, 2).is_contiguous()}"
                     + f""" fm={None if flash_mask is None else (tuple(flash_mask.shape), str(flash_mask.dtype),
                                                                flash_mask.is_contiguous())}"""
                     + f" new_pos={new_pos} scale={scale}" + chr(10))
    except OSError:
        pass


def applies(q, k_cache, v_cache, prefix_len, new_pos, flash_mask, scale):
    """The decode family this kernel pair serves, checked per call."""
    if not enabled():
        return False
    if (
        flash_mask is None
        or not (scale is None or float(scale) == _D**-0.5)
        or q.shape != (1, _M, _H, _D)
        or k_cache.shape != (1, _L, _KVH, _D)
        or v_cache.shape != k_cache.shape
        or new_pos != _L - _M
    ):
        return False
    # Input may have packed query rows or native head-major strides; the scan consumes dense head-major Q.
    if not (
        q.dtype == k_cache.dtype == v_cache.dtype == torch.bfloat16
        and q.device.type == "npu"
        and k_cache.device == q.device
        and v_cache.device == q.device
        and (q.is_contiguous() or q.transpose(1, 2).is_contiguous())
        and k_cache.transpose(1, 2).is_contiguous()
        and v_cache.transpose(1, 2).is_contiguous()
        and prefix_len.dtype == torch.int32
        and prefix_len.shape == (1,)
        and prefix_len.device == q.device
        and flash_mask.dtype == torch.int32
        and flash_mask.shape == (1, 1, _L, 2)
        and flash_mask.device == q.device
        and flash_mask.is_contiguous()
    ):
        _debug("REJECT-props", q, k_cache, flash_mask, new_pos, scale)
        return False
    if (
        flash_mask is None
        or not (scale is None or float(scale) == _D**-0.5)
        or q.shape != (1, _M, _H, _D)
        or k_cache.shape != (1, _L, _KVH, _D)
        or v_cache.shape != k_cache.shape
        or new_pos != _L - _M
    ):
        _debug("REJECT-family", q, k_cache, flash_mask, new_pos, scale)
        return False
    return True


def prefix_kvcache_attention(q, k_cache, v_cache, prefix_len, new_pos, flash_mask, scale, return_bnsd):
    """The decode-family route; ``applies`` must have accepted the call."""
    if torch.npu.is_current_stream_capturing():
        # the engine warms every route eagerly before capture; a first sight
        # during capture means the warmup discipline was broken
        if ("warm",) not in _SEEN:
            raise RuntimeError("warm up the decodeattn-r4 route before graph capture")
    variant = _SCAN_VARIANT
    scan_name = f"decodeattn_r4_scan_{variant}"
    call, err = _entry(scan_name, 6, (ctypes.c_int64, ctypes.c_int64))
    row = not return_bnsd and _row_enabled()
    reduce_call, reduce_err = _entry(
        "decodeattn_r4_reduce_row" if row else "decodeattn_r4_reduce", 2, ())
    stream = torch.npu.current_stream(q.device).npu_stream
    pbuf = torch.empty(_P_SHAPE, dtype=torch.float32, device=q.device)
    out = torch.empty((_M, _H, _D) if row else (_H, _M, _D), dtype=q.dtype, device=q.device)
    q_flat = q.transpose(1, 2).reshape(_H * _M, _D)
    keys = k_cache.transpose(1, 2).reshape(_KVH * _L, _D)
    values = v_cache.transpose(1, 2).reshape(_KVH * _L, _D)
    fm = flash_mask.reshape(_L, 2)
    scale_bits = struct.unpack("<q", struct.pack("<f", _D**-0.5) + bytes(4))[0]
    # the single library's TVM-FFI faces serve first (the unified launch boundary); the
    # C-ABI launchers from the same library are the fallback below
    if not _native.launch_kernel(scan_name, (q_flat, keys, values, fm, prefix_len, pbuf),
                                 (int(new_pos), scale_bits)):
        status = call(
            ctypes.c_void_p(q_flat.data_ptr()),
            ctypes.c_void_p(keys.data_ptr()),
            ctypes.c_void_p(values.data_ptr()),
            ctypes.c_void_p(fm.data_ptr()),
            ctypes.c_void_p(prefix_len.data_ptr()),
            ctypes.c_void_p(pbuf.data_ptr()),
            ctypes.c_int64(int(new_pos)),
            ctypes.c_int64(scale_bits),
            ctypes.c_void_p(stream),
        )
        if status:
            raise RuntimeError("decodeattn-r4 scan: " + (err() or b"?").decode("utf-8", "replace"))
    reduce_name = "decodeattn_r4_reduce_row" if row else "decodeattn_r4_reduce"
    if not _native.launch_kernel(reduce_name, (pbuf, out)):
        status = reduce_call(
            ctypes.c_void_p(pbuf.data_ptr()),
            ctypes.c_void_p(out.data_ptr()),
            ctypes.c_void_p(stream),
        )
        if status:
            raise RuntimeError("decodeattn-r4 reduce: " + (reduce_err() or b"?").decode("utf-8", "replace"))
    _SEEN.add(("warm",))
    _debug("HIT", q, k_cache, flash_mask, new_pos, scale)
    if row:
        # token-major [M, H, D] storage: [1, M, H*D] is a free view and k
        # order (h*D + d ascending) matches the BNSD A-load exactly.
        return out.view(1, _M, _H * _D)
    out = out.view(1, _H, _M, _D)
    if return_bnsd:
        return out
    return out.transpose(1, 2).reshape(1, _M, _H * _D)


_SEEN = set()
