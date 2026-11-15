"""FIA-FUSE opt-in route: specialized split-K scan+reduce cache-attention pair
for the wall-x 2B decode family, fusing the FIA call AND the gated pack tail.

Off by default (XINFER_FIA_FUSE=off). Replaces the per-call decode chain
(_r6b_prefix_kvcache_bnsd FusedInferAttentionScore + gated_attn_bnsd_pack_pow2,
17.2us measured) by libfia_fuse.so scan + reduce (TileLang AOT, bisheng
9.3.0 / cann 9.1.1 runtime) which merge the split-K partials, round the
attention to BF16 (the FIA output rounding) and apply the exact pack_pow2
gate tail (FP32 sigmoid 1/(1+exp(-x)), two manual-RNE BF16 roundings) while
storing the packed [B, M, H*D] rows the o_proj GEMM reads.

Launches resolve the current stream per call so NPUGraph capture records the
node. The route stays inside the folded() call's action scope.

Route stats: when XINFER_FIA_FUSE_STATS names a path, every interpreter that
routes writes <path>-pid<pid>.json at exit (routed / fell_back / errors /
by_shape) so serve subprocesses give their own receipts.
"""
import atexit
import ctypes
import json
import os
import struct
from threading import Lock

import torch

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
    raw = os.environ.get("XINFER_FIA_FUSE", "off").strip().lower()
    _ENABLED = raw not in ("off", "", "0")


def enabled():
    if _ENABLED is None:
        _parse()
    return _ENABLED


def _library():
    global _LIB
    if _LIB is None:
        path = os.environ.get("XINFER_FIA_FUSE_LIB")
        if not path:
            raise OSError("fia-fuse: XINFER_FIA_FUSE_LIB is not set")
        _LIB = ctypes.CDLL(path)
    return _LIB


def _entry(name, n_ptr, scalar_types):
    with _LOCK:
        pair = _ENTRIES.get(name)
        if pair is None:
            lib = _library()
            call = getattr(lib, f"xcompute_aot_{name}_call")
            call.restype = ctypes.c_int32
            call.argtypes = [ctypes.c_void_p] * n_ptr + list(scalar_types) + [ctypes.c_void_p]
            err = getattr(lib, f"xcompute_aot_{name}_get_last_error")
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
        for _ in range(2):
            scan_call(
                ctypes.c_void_p(q.data_ptr()), ctypes.c_void_p(k.data_ptr()),
                ctypes.c_void_p(k.data_ptr()), ctypes.c_void_p(fm.data_ptr()),
                ctypes.c_void_p(pl.data_ptr()), ctypes.c_void_p(pbuf.data_ptr()),
                ctypes.c_int64(L - _M), ctypes.c_int64(scale_bits), stream)
            reduce_call(
                ctypes.c_void_p(pbuf.data_ptr()), ctypes.c_void_p(out.data_ptr()),
                ctypes.c_void_p(out.data_ptr()), ctypes.c_int64(_H * _D), stream)
    torch.npu.synchronize()


def try_route(q, k_cache, v_cache, prefix_len, new_pos, gate, flash_mask, scale):
    """The decode-family route; returns the packed gated [1, M, H*D] tensor or
    None when the family does not apply (caller falls back to the released
    chain)."""
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
    except Exception as exc:  # noqa: BLE001 - the released chain stays reachable
        _STATS["errors"].append(f"l{L}: {type(exc).__name__}: {exc}"[:200])
        _note(key, "fell_back")
        return None
