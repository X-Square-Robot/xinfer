"""Gated DeltaNet layout and elementwise adapters for the in-tree native library.

``XINFER_MOVER2`` is off by default and accepts these families:
- ``gdrapply``: write TND Q/K packs with one RNE and copy V after the existing
  convolution, SiLU and normalization statistics.
- ``gdrgates``: compute beta and gate pre-activations from strided ZBA rows.
- ``gdrtailmix``: apply the elementwise tail with its three RNE roundings;
  normalization statistics remain in the existing Torch/CANN chain.

Chunk inputs keep their layouts and dtypes. The gate route may differ in
FP32 transcendental approximations; the layout/tail routes retain the same
arithmetic. Optional route counts use ``XINFER_MOVER2_STATS``.
"""

from __future__ import annotations

import atexit

import ctypes
import json
import os

import torch

from xinfer.ops.backends.ascend import _native

FAMILIES = ("gdrapply", "gdrtailmix")
_ENABLED = None
_LIB = None
_GDP = None
_BROKEN = set()
_STATS = {"gdrapply": [0, 0], "gdrtailmix": [0, 0]}


def enabled(family: str) -> bool:
    global _ENABLED
    if _ENABLED is None:
        raw = os.environ.get("XINFER_MOVER2", "off").strip().lower()
        if raw in ("", "off", "0", "false", "no"):
            _ENABLED = frozenset()
        elif raw in ("1", "all", "on"):
            _ENABLED = frozenset(FAMILIES)
        else:
            wanted = {t.strip() for t in raw.split(",")}
            unknown = wanted - set(FAMILIES)
            if unknown:
                raise RuntimeError(f"XINFER_MOVER2 unknown families: {sorted(unknown)}")
            _ENABLED = frozenset(wanted)
    return family in _ENABLED and family not in _BROKEN


def _break(family):
    _BROKEN.add(family)
    _STATS[family][1] += 1


def _lib():
    return _native.core_library()


def _entry(name, pointers, scalars):
    call = getattr(_lib(), f"xinfer_aot_{name}_call")
    call.restype = ctypes.c_int32
    call.argtypes = [ctypes.c_void_p] * pointers + [ctypes.c_int64] * scalars + [
        ctypes.c_void_p
    ]
    err = getattr(_lib(), f"xinfer_aot_{name}_get_last_error")
    err.restype = ctypes.c_char_p
    return call, err


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


def _routed(family):
    _STATS[family][0] += 1


def dump_stats():
    base = os.environ.get("XINFER_MOVER2_STATS")
    if not base:
        return
    try:
        with open(base + f"-pid{os.getpid()}.json", "w") as fh:
            json.dump({k: dict(routed=v[0], fell_back=v[1]) for k, v in _STATS.items()}, fh)
    except Exception:
        pass


# ---------------------------------------------------------------- kernels --


def gdrapply(x_t, rstd, tokens, hk, dk):
    """(q, k, v) TND packs: q/k = RNE(f32(x) * rstd), v = x copy."""
    dev = x_t.device
    q = torch.empty(tokens, hk, dk, dtype=torch.bfloat16, device=dev)
    k = torch.empty(tokens, hk, dk, dtype=torch.bfloat16, device=dev)
    v = torch.empty(tokens, hk, dk, dtype=torch.bfloat16, device=dev)
    _launch(
        "gdrapply_bf16",
        (x_t, rstd, q, k, v),
        (tokens,),
    )
    _routed("gdrapply")
    return q, k, v


def gdrtailmix(raw, rstd, gate, w, tokens, hv, dv):
    """result = RNE(RNE_bf16(w * RNE_bf16(raw*rstd)) * gate), one kernel."""
    dev = raw.device
    result = torch.empty(tokens, hv * dv, dtype=torch.bfloat16, device=dev)
    _launch(
        "gdrtailmix_bf16",
        (result, raw, rstd, gate, w),
        (tokens,),
    )
    _routed("gdrtailmix")
    return result


atexit.register(dump_stats)
