"""Split-K cache attention and gated output packing for WALL-OSS-1.0-Nano.

``XINFER_FIA_FUSE`` is on by default; ``off`` disables the route. Scan and
reduce entries come from the in-tree native library. They round attention
to BF16 and apply the pack_pow2 tail: FP32 ``1/(1+exp(-x))`` followed by
two RNE BF16 roundings, producing packed [B,M,H*D] rows.

Launches use the current stream inside the caller's action scope.
``XINFER_FIA_FUSE_STATS`` optionally records per-process route counts.
"""
import atexit
import ctypes
import json
import os
import struct
from threading import Lock

import torch

from xinfer.ops.backends.ascend import _native

_H, _KVH, _M, _D = 8, 2, 16, 256
_BUILD_KEYS = (432, 448, 464, 480)
_CHUNKS = {L: (L + 63) // 64 for L in _BUILD_KEYS}
_PAIR_ROWS = 33

_ENABLED = None
_LIB = None
_ENTRIES = {}
_LOCK = Lock()
_STATS = {"routed": 0, "fell_back": 0, "errors": [], "by_shape": {}}
_SEEN = set()



def _parse():
    global _ENABLED
    raw = os.environ.get("XINFER_FIA_FUSE", "on").strip().lower()
    _ENABLED = raw not in ("off", "", "0")


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


def _note(key, field):
    _STATS["by_shape"].setdefault(key, {"routed": 0, "fell_back": 0})
    _STATS["by_shape"][key][field] += 1


def _dump_stats():
    base = os.environ.get("XINFER_FIA_FUSE_STATS", "")
    if base:
        try:
            payload = dict(_STATS)
            payload["by_shape"] = {str(k): v for k, v in _STATS["by_shape"].items()}
            with open(f"{base}-pid{os.getpid()}.json", "w") as fh:
                json.dump(payload, fh, indent=1)
        except OSError:
            pass


atexit.register(_dump_stats)


def applies(q, k_cache, v_cache, prefix_len, new_pos, gate, flash_mask, scale):
    """The 2B decode family this kernel pair serves, checked per call."""
    if not enabled():
        return False
    L = k_cache.shape[1] if k_cache.ndim == 4 else -1
    key = (tuple(q.shape), L, None if flash_mask is None else tuple(flash_mask.shape))
    if (
        flash_mask is None
        or not (scale is None or float(scale) == _D**-0.5)
        or q.shape != (1, _H, _M, _D)
        or not q.is_contiguous()
        or L not in _BUILD_KEYS
        or k_cache.shape != (1, L, _KVH, _D)
        or v_cache.shape != k_cache.shape
        or new_pos != L - _M
        or gate.shape != (1, _M, _H * _D)
        or gate.dtype != torch.bfloat16
        or gate.stride(2) != 1 or gate.stride(1) < _H * _D
    ):
        _note(key, "fell_back")
        return False
    if not (
        q.dtype == k_cache.dtype == v_cache.dtype == torch.bfloat16
        and q.device.type == "npu"
        and k_cache.device == q.device
        and v_cache.device == q.device
        and k_cache.transpose(1, 2).is_contiguous()
        and v_cache.transpose(1, 2).is_contiguous()
        and prefix_len.dtype == torch.int32
        and prefix_len.shape == (1,)
        and prefix_len.device == q.device
        and flash_mask.dtype == torch.int32
        and flash_mask.shape == (1, 1, L, 2)
        and flash_mask.device == q.device
        and flash_mask.is_contiguous()
    ):
        _note(key, "fell_back")
        return False
    return True


def _warmup_launches(device):
    """One throwaway launch per kernel variant, eager and outside capture.

    The first MixedKernel launch of a fresh process can be dropped by the
    AOT runtime (probe evidence: first scan call writes nothing, rc=0; every
    later launch is healthy). The engine warms routes eagerly before graph
    capture, and this explicit warmup consumes that first launch with dummy
    operands so no production call ever sees it.
    """
    for L in _BUILD_KEYS:
        chunks = _CHUNKS[L]
        q = torch.zeros((_H * _M, _D), dtype=torch.bfloat16, device=device)
        k = torch.zeros((_KVH * L, _D), dtype=torch.bfloat16, device=device)
        fm = torch.zeros((L, 2), dtype=torch.int32, device=device)
        pl = torch.zeros((1,), dtype=torch.int32, device=device)
        pbuf = torch.empty((4 * chunks, _PAIR_ROWS, _D), dtype=torch.float32, device=device)
        out = torch.empty((_M, _H * _D), dtype=torch.bfloat16, device=device)
        stream = ctypes.c_void_p(torch.npu.current_stream(device).npu_stream)
        scale_bits = struct.unpack("<q", struct.pack("<f", _D**-0.5) + bytes(4))[0]
        scan_call, _ = _entry(f"fia_fuse_scan_l{L}", 6, (ctypes.c_int64, ctypes.c_int64))
        reduce_call, _ = _entry(f"fia_fuse_reduce_l{L}", 3, (ctypes.c_int64,))
        # the warmup launches must take the same path the production calls will (the
        # single library's TVM-FFI face when it is loaded - the AOT first-launch loss is
        # per library and per process, so warming one library does not warm the other)
        for _ in range(2):
            if not _native.launch_kernel(f"fia_fuse_scan_l{L}",
                                         (q, k, k, fm, pl, pbuf), (L - _M, scale_bits)):
                scan_call(
                    ctypes.c_void_p(q.data_ptr()), ctypes.c_void_p(k.data_ptr()),
                    ctypes.c_void_p(k.data_ptr()), ctypes.c_void_p(fm.data_ptr()),
                    ctypes.c_void_p(pl.data_ptr()), ctypes.c_void_p(pbuf.data_ptr()),
                    ctypes.c_int64(L - _M), ctypes.c_int64(scale_bits), stream)
            if not _native.launch_kernel(f"fia_fuse_reduce_l{L}",
                                         (pbuf, out, out), (_H * _D,)):
                reduce_call(
                    ctypes.c_void_p(pbuf.data_ptr()), ctypes.c_void_p(out.data_ptr()),
                    ctypes.c_void_p(out.data_ptr()), ctypes.c_int64(_H * _D), stream)
    torch.npu.synchronize()


def try_route(q, k_cache, v_cache, prefix_len, new_pos, gate, flash_mask, scale):
    """The decode-family route; returns the packed gated [1, M, H*D] tensor or
    None when the family does not apply (caller falls back to the released
    chain)."""
    if (q.ndim == 4 and torch.is_tensor(gate)
            and gate.shape == (q.shape[0], q.shape[1], q.shape[2] * q.shape[3])):
        q = q.transpose(1, 2)  # Canonical BMHD input can be a view of the native BNSD buffer.
    if not applies(q, k_cache, v_cache, prefix_len, new_pos, gate, flash_mask, scale):
        return None
    L = k_cache.shape[1]
    key = (tuple(q.shape), L, tuple(flash_mask.shape))
    try:
        if ("warm",) not in _SEEN:
            if torch.npu.is_current_stream_capturing():
                raise RuntimeError("warm up the fia-fuse route before graph capture")
            _warmup_launches(q.device)
            _SEEN.add(("warm",))
        chunks = _CHUNKS[L]
        scan_call, scan_err = _entry(f"fia_fuse_scan_l{L}", 6, (ctypes.c_int64, ctypes.c_int64))
        reduce_call, reduce_err = _entry(f"fia_fuse_reduce_l{L}", 3, (ctypes.c_int64,))
        stream = ctypes.c_void_p(torch.npu.current_stream(q.device).npu_stream)
        pbuf = torch.empty((4 * chunks, _PAIR_ROWS, _D), dtype=torch.float32, device=q.device)
        out = torch.empty((1, _M, _H * _D), dtype=q.dtype, device=q.device)
        q_flat = q.view(_H * _M, _D)
        keys = k_cache.transpose(1, 2).view(_KVH * L, _D)
        values = v_cache.transpose(1, 2).view(_KVH * L, _D)
        fm = flash_mask.view(L, 2)
        scale_bits = struct.unpack("<q", struct.pack("<f", _D**-0.5) + bytes(4))[0]
        # the single library's TVM-FFI faces serve first (the unified launch boundary);
        # the same library's C-ABI launchers are the fallback below
        if not _native.launch_kernel(f"fia_fuse_scan_l{L}",
                                     (q_flat, keys, values, fm, prefix_len, pbuf),
                                     (int(new_pos), scale_bits)):
            status = scan_call(
                ctypes.c_void_p(q_flat.data_ptr()),
                ctypes.c_void_p(keys.data_ptr()),
                ctypes.c_void_p(values.data_ptr()),
                ctypes.c_void_p(fm.data_ptr()),
                ctypes.c_void_p(prefix_len.data_ptr()),
                ctypes.c_void_p(pbuf.data_ptr()),
                ctypes.c_int64(int(new_pos)),
                ctypes.c_int64(scale_bits),
                stream,
            )
            if status:
                raise RuntimeError(f"fia-fuse scan l{L}: " + (scan_err() or b"?").decode("utf-8", "replace"))
        if not _native.launch_kernel(f"fia_fuse_reduce_l{L}", (pbuf, out, gate),
                                     (gate.stride(1),)):
            status = reduce_call(
                ctypes.c_void_p(pbuf.data_ptr()),
                ctypes.c_void_p(out.data_ptr()),
                ctypes.c_void_p(gate.data_ptr()),
                ctypes.c_int64(gate.stride(1)),
                stream,
            )
            if status:
                raise RuntimeError(f"fia-fuse reduce l{L}: " + (reduce_err() or b"?").decode("utf-8", "replace"))
        _SEEN.add(("warm",))
        _STATS["routed"] += 1
        _note(key, "routed")
        return out
    except Exception as exc:  # noqa: BLE001 - an admitted native failure cannot select another implementation
        _STATS["errors"].append(f"l{L}: {type(exc).__name__}: {exc}"[:200])
        raise RuntimeError(f"fia-fuse l{L} failed: {exc}") from exc
