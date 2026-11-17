"""GDN-VENDOR mount (kernel launcher for the XINFER_GDNV class route).

The route lives in ``glue.cann_gated_delta_net.CannGatedDeltaNet.__call__``
(the move-r2/conv-gemm precedent); this module loads libgdnv.so and computes
the fused gates prefix, counting per-family route stats.

Flag ``XINFER_GDNV`` (default off), comma list:
  ``gdrprep`` — the per-layer gates prefix in ONE kernel:
                beta[t,h] = RNE_bf16(1/(1+exp(-f32(zba[t,2048+h]))))
                gpre[t,h] = f32(zba[t,2064+h]) + dt[h]
                then g = premul * F.softplus(gpre) with premul = -exp(A_log)
                cached at warmup (the same aclnn Exp/Neg on the same [16]
                input -- bitwise) and dt pre-cast once.
                Replaces Sigmoid_Slice + Sigmoid + Exp + Neg + Slice + Cast +
                Add (7 launches); Softplus and the broadcast Mul stay on
                aclnn with bit-identical inputs.

Bitwise evidence (R/gdn-vendor/results/):
  probe-tr-bitwise.json  -- sigmoid v0 composition: n_diff=0 over ALL 65536
                            bf16 patterns; softplus log1p tail NOT bitwise
                            (hence Softplus stays on aclnn);
  witness-gdrprep.json   -- the composite (kernel + aclnn Softplus + aclnn
                            Mul + cached premul) vs the released 9-op chain:
                            bitwise on the production T family, a 4096-row
                            bf16-domain sweep, extremes (+-inf/0/denormal) and
                            NaN (bit equality).
The g-amp advisory (R/gdn-k3-r2): g feeds the vendor chunk recurrence and
amplifies ulp differences past the action gate -- this route is landable
precisely because it is bitwise end to end.
"""

from __future__ import annotations

import atexit
import ctypes
import json
import os

import torch
import torch.nn.functional as F

FAMILIES = ("gdrprep",)
_ENABLED = None
_LIB = None
_BROKEN = set()
_STATS = {"gdrprep": [0, 0]}
_CACHE = {}


def enabled(family: str) -> bool:
    global _ENABLED
    if _ENABLED is None:
        raw = os.environ.get("XINFER_GDNV", "off").strip().lower()
        if raw in ("", "off", "0", "false", "no"):
            _ENABLED = frozenset()
        elif raw in ("1", "all", "on"):
            _ENABLED = frozenset(FAMILIES)
        else:
            wanted = {t.strip() for t in raw.split(",")}
            unknown = wanted - set(FAMILIES)
            if unknown:
                raise RuntimeError(f"XINFER_GDNV unknown families: {sorted(unknown)}")
            _ENABLED = frozenset(wanted)
    return family in _ENABLED and family not in _BROKEN


def _lib():
    global _LIB
    if _LIB is None:
        path = os.environ.get("XINFER_GDNV_LIB")
        if not path:
            raise OSError("gdn-vendor: XINFER_GDNV_LIB is not set")
        _LIB = ctypes.CDLL(path)
    return _LIB


def _launch(name, tensors, scalars):
    call = getattr(_lib(), f"xinfer_aot_{name}_call")
    call.restype = ctypes.c_int32
    call.argtypes = [ctypes.c_void_p] * len(tensors) + [ctypes.c_int64] * len(scalars) + [
        ctypes.c_void_p
    ]
    err = getattr(_lib(), f"xinfer_aot_{name}_get_last_error")
    err.restype = ctypes.c_char_p
    stream = torch.npu.current_stream(tensors[0].device).npu_stream
    status = call(
        *[ctypes.c_void_p(t.data_ptr()) for t in tensors],
        *[ctypes.c_int64(s) for s in scalars],
        ctypes.c_void_p(stream),
    )
    if status:
        raise RuntimeError(f"{name} failed: {err().decode('utf-8', 'replace')}")


def dump_stats():
    base = os.environ.get("XINFER_GDNV_STATS")
    if not base:
        return
    try:
        with open(base + f"-pid{os.getpid()}.json", "w") as fh:
            json.dump({k: dict(routed=v[0], fell_back=v[1]) for k, v in _STATS.items()}, fh)
    except Exception:
        pass


atexit.register(dump_stats)


def gdrprep(zba, A_log, dt_bias, tokens, hv, dv, dtype):
    """Return ``(beta [T,hv] bf16, g [T,hv] fp32)`` or ``None`` to fall back.

    ``zba`` is the packed ``[1, T, hv*(dv+2)]`` projection; ``A_log`` and
    ``dt_bias`` are the per-head decay parameters. The premul ``-exp(A_log)``
    and the fp32 dt cast are cached per parameter tensor (computed eagerly by
    the same aclnn ops, so the captured graph reads bitwise-identical bits).
    """
    if hv != 16 or dv != 128 or dtype != torch.bfloat16:
        return None
    try:
        key = (A_log.data_ptr(), dt_bias.data_ptr(), A_log.dtype, dt_bias.dtype)
        cached = _CACHE.get(key)
        if cached is None:
            premul = -(A_log.float().exp())
            dtf = dt_bias if dt_bias.dtype == torch.float32 else dt_bias.float()
            if premul.dtype != torch.float32 or dtf.dtype != torch.float32:
                return None
            cached = (premul, dtf)
            _CACHE[key] = cached
            if len(_CACHE) > 512:
                _CACHE.clear()
                _CACHE[key] = cached
        premul, dtf = cached
        beta = torch.empty(tokens, hv, dtype=torch.bfloat16, device=zba.device)
        gpre = torch.empty(tokens, hv, dtype=torch.float32, device=zba.device)
        _launch("gdrprep_bf16", [zba, dtf, beta, gpre], [tokens])
        g = premul * F.softplus(gpre)
        _STATS["gdrprep"][0] += 1
        return beta, g
    except (ImportError, OSError, RuntimeError):
        _STATS["gdrprep"][1] += 1
        return None
