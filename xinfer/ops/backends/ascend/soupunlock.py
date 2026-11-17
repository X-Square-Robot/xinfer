"""Optional fused reductions for prefill normalization and vision LayerNorm.

``XINFER_SOUPUNLOCK`` defaults to ``rmsfull`` from layer 8. ``off`` disables
it; ``on`` enables both ``rmsfull`` and ``visionln``. The existing
``XINFER_SOUPUNLOCK_FROM_LAYER`` selects the first RMS layer.

``rmsfull`` combines the residual sum, deterministic FP32 tree statistics
and zero-centered RMS apply. ``visionln`` uses a two-pass LayerNorm for
1024-wide BF16 inputs. These are tolerance-level reduction implementations;
the in-tree library retains the original operation order and envelopes.
Unsupported calls return None. ``XINFER_SOUPUNLOCK_STATS`` records optional
per-process route counts.
"""

from __future__ import annotations
from xinfer.ops.backends.ascend import _native

import ctypes
import json
import os
import struct
import time

import torch

_FAMILIES = ("rmsfull", "visionln")
_LOCK = None
_LIB = None
_ENTRIES: dict = {}
_ENABLED = None
_STATS = {"routed": 0, "fell_back": 0, "rows": 0}


def _parse() -> None:
    global _ENABLED
    raw = os.environ.get("XINFER_SOUPUNLOCK", "rmsfull").strip().lower()
    if raw in ("", "off", "0", "false", "no"):
        _ENABLED = frozenset()
    elif raw in ("1", "all", "on"):
        _ENABLED = frozenset(_FAMILIES)
    else:
        wanted = {t.strip() for t in raw.split(",")}
        unknown = wanted - set(_FAMILIES)
        if unknown:
            raise RuntimeError(f"XINFER_SOUPUNLOCK unknown families: {sorted(unknown)}")
        _ENABLED = frozenset(wanted)


def enabled(family: str) -> bool:
    if _ENABLED is None:
        _parse()
    return family in _ENABLED


def from_layer() -> int:
    """Deterministic site gate: rmsfull arms only at layer indices >= this."""
    try:
        return max(0, int(os.environ.get("XINFER_SOUPUNLOCK_FROM_LAYER", "8")))
    except ValueError:
        return 0


def _flush_stats(final: bool = False) -> None:
    base = os.environ.get("XINFER_SOUPUNLOCK_STATS")
    if not base:
        return
    try:
        path = base if base.endswith(f"pid{os.getpid()}") else f"{base}-pid{os.getpid()}"
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"final": final, "pid": os.getpid(), "ts": time.time(),
                                 **_STATS}) + "\n")
    except Exception:  # noqa: BLE001 - receipts must never break serving
        pass


def _note(route: str, rows: int) -> None:
    if route == "routed":
        _STATS["routed"] += 1
        _STATS["rows"] += rows
    else:
        _STATS["fell_back"] += 1
    if _STATS["routed"] + _STATS["fell_back"] >= 64:
        _flush_stats()
        _STATS["routed"] = _STATS["fell_back"] = _STATS["rows"] = 0


def _library():
    return _native.core_library()


def _entry(name, pointers, scalars):
    key = (name, pointers, scalars)
    pair = _ENTRIES.get(key)
    if pair is None:
        lib = _library()
        call = getattr(lib, f"xinfer_aot_{name}_call")
        call.restype = ctypes.c_int32
        call.argtypes = [ctypes.c_void_p] * pointers + [ctypes.c_int64] * scalars + [ctypes.c_void_p]
        err = getattr(lib, f"xinfer_aot_{name}_get_last_error")
        err.restype = ctypes.c_char_p
        pair = (call, err)
        _ENTRIES[key] = pair
    return pair


def _launch(name, tensors, scalars):
    if _native.launch_kernel(name, tensors, scalars):
        return
    call, err = _entry(name, len(tensors), len(scalars))
    stream = torch.npu.current_stream(tensors[0].device).npu_stream
    status = call(
        *[ctypes.c_void_p(t.data_ptr()) for t in tensors],
        *[ctypes.c_int64(s) for s in scalars],
        ctypes.c_void_p(stream),
    )
    if status:
        raise RuntimeError(f"{name} failed: {err().decode('utf-8', 'replace')}")


def _f32_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(value)))[0]


def _flat(t) -> bool:
    return t is not None and t.is_contiguous() and t.device.type == "npu"


def _rows(x) -> int:
    rows = 1
    for dim in x.shape[:-1]:
        rows *= dim
    return rows


def rmsnorm_full(x, residual, weight, eps):
    """(s, normed) of add_zero_centered_rms_norm in one fused launch.

    Envelope mirrors soupform normprep (flat BF16 [.., 2048], FP32 weight,
    any positive rows -- rows_per_tile is 1 here, so no %4 constraint).
    """
    if not enabled("rmsfull"):
        return None
    rows = _rows(x)
    if not (_flat(x) and x.dtype == torch.bfloat16 and x.shape[-1] == 2048
            and _flat(residual) and residual.dtype == torch.bfloat16
            and residual.shape == x.shape and residual.device == x.device
            and _flat(weight) and weight.dtype == torch.float32
            and weight.shape == (2048,) and weight.device == x.device
            and 0 < float(eps) <= torch.finfo(torch.float32).max
            and rows > 0):
        _note("fell_back", rows)
        return None
    try:
        s = torch.empty_like(x)
        normed = torch.empty_like(x)
        _launch("rmsnorm_full_c2048", (s, normed, x, residual, weight),
                (_f32_bits(eps), rows))
    except (ValueError, RuntimeError):
        _note("fell_back", rows)
        return None
    _note("routed", rows)
    return s, normed


def layernorm_c1024(x, weight, bias, eps):
    """Vision LayerNorm (hidden 1024, BF16 x/w/b) in one fused launch."""
    if not enabled("visionln"):
        return None
    rows = _rows(x)
    if not (_flat(x) and x.dtype == torch.bfloat16 and x.shape[-1] == 1024
            and _flat(weight) and weight.dtype == torch.bfloat16
            and weight.shape == (1024,) and weight.device == x.device
            and _flat(bias) and bias.dtype == torch.bfloat16
            and bias.shape == (1024,) and bias.device == x.device
            and 0 < float(eps) <= torch.finfo(torch.float32).max
            and rows > 0):
        _note("fell_back", rows)
        return None
    try:
        out = torch.empty_like(x)
        _launch("layernorm_c1024", (out, x, weight, bias), (_f32_bits(eps), rows))
    except (ValueError, RuntimeError):
        _note("fell_back", rows)
        return None
    _note("routed", rows)
    return out


import atexit  # noqa: E402 - final receipt line per process


atexit.register(lambda: _flush_stats(final=True))
