"""MOVE-R2 mount (kernel launchers for the XINFER_MOVER2 class route).

The route lives in ``glue.cann_gated_delta_net.CannGatedDeltaNet.__call__``
(conv-gemm precedent); this module only loads libmove_r2.so and launches the
two delivered kernels, counting per-family route stats.

Flag ``XINFER_MOVER2`` (default off), comma list:
  ``gdrapply``  — post-conv elementwise fuse: the full-width transpose copy
                  (kills the conv-output width slice), aclnn SiLU + the aclnn
                  square/sum/rsqrt stats stay, then ONE kernel writes the TND
                  q/k packs (x * rstd, single RNE) and the v copy. Replaces
                  Slice + Mul_Cast(n) + 2x mul-by-one + AsStrided(v).
  ``gdrgates``  — beta/g gate chain in one pass via the production library's
                  ``gdp_gates_fwd_bf16`` (theta-lib line-equal math), reading
                  the strided zba rows directly.
  ``gdrtailmix``— the raw-side elementwise tail in ONE kernel (nbf/wn/gate
                  muls with the chain's exact three RNE roundings) from the
                  aclnn rstd and gate tensors. Replaces the 6-op aclnn
                  elementwise tail (raw cast, mul, cast, w-mul, gate-mul,
                  cast); the stats (pow/mean/rsqrt) stay on aclnn.

All three keep the chunk kernel's operand contracts (q,k,v,beta,g layouts
and dtypes unchanged), so the GDN-K3 chunk domain is untouched. The
gdrapply/gdrtailmix kernels are mul/cast-only on aclnn-identical inputs
(bitwise-expected); gdrgates differs only in vendor transcendental
approximations (probe-adjudicated <= 3 fp32 ulp on g, beta bitwise).
Every route is envelope-guarded with per-family fail-safe fallback.
Route counters land in $XINFER_MOVER2_STATS[-pidN].json at shutdown.
"""

from __future__ import annotations

import atexit

import ctypes
import json
import os

import torch
import torch.nn.functional as F

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
    global _LIB
    if _LIB is None:
        path = os.environ.get("XINFER_MOVER2_LIB")
        if not path:
            raise OSError("move-r2: XINFER_MOVER2_LIB is not set")
        _LIB = ctypes.CDLL(path)
    return _LIB


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
