"""SOUP2 mount (kernel launchers + the plain-finish Matmul route for the XINFER_SOUP2 class).

Route points (all envelope-guarded with fail-safe fallback to the released chains):

``qksq``    — GDN prefill qk stats pre-pass in glue.cann_gated_delta_net
              (mover2 branch): ONE kernel writes f32(RNE_bf16(qk*qk)) over the
              strided qk window of x_t; the aclnn ReduceSum/Adds/Rsqrt that
              follow read bit-identical bits (replaces aclnnMul + the sum's
              implicit bf16->f32 cast kernel).
``rawsq``   — GDN gdrtailmix raw stats pre-pass: Cast(raw f32) + Square in ONE
              kernel writing RNE_f32(a*a); aclnn Mean/Adds/Rsqrt unchanged and
              bit-identical (replaces aclnnInplaceCopy_Cast + aclnnPow).
``fc1gelu`` — vision fc1 finish tail: bias-add (same fp32 vadd, bitwise) +
              tanh-form gelu + single bf16 RNE in ONE kernel (replaces aclnnAdd
              + aclnnGeluV2 + aclnnCast). NOT bitwise vs aclnnGeluV2 (vendor
              transcendental approximation); shipped only behind the action
              gate, default off in the lib route until adjudicated.
``finplain``— the plain-finish prefill GEMMs ([P,K]x[N,K], no bias/residual/
              activation) ride a single bf16 aclnnMatmul instead of the FP32
              Mm + Cast pair. Probed bitwise on all production (m, k, n)
              shapes (12/12 zero flips): both kernels accumulate FP32 and
              round once in the final store (the 2b-m16 precedent, widened).

Bitwise contract: qksq/rawsq/finplain are bit-identical to the released
chains (witness-adjudicated); fc1gelu differs only in the vendor gelu
approximation (op-level flip rate quantified per site, action gate is the
adjudicator). Every route keeps the consumers' operand contracts unchanged
(the chunk kernel, gdrapply and gdrtailmix see the same bits as before).
Route counters land in $XINFER_SOUP2_STATS[-pidN].json at shutdown.
"""

from __future__ import annotations

import atexit
import ctypes
import json
import os

import torch

from xinfer.ops.backends.ascend import _native

FAMILIES = ("qksq", "rawsq", "fc1gelu", "finplain", "quant")
_ENABLED = None
_LIB = None
_BROKEN = set()
_STATS = {f: [0, 0] for f in FAMILIES}


def enabled(family: str) -> bool:
    global _ENABLED
    if _ENABLED is None:
        raw = os.environ.get("XINFER_SOUP2", "off").strip().lower()
        if raw in ("", "off", "0", "false", "no"):
            _ENABLED = frozenset()
        elif raw in ("1", "all", "on"):
            _ENABLED = frozenset(FAMILIES)
        else:
            wanted = {t.strip() for t in raw.split(",")}
            unknown = wanted - set(FAMILIES)
            if unknown:
                raise RuntimeError(f"XINFER_SOUP2 unknown families: {sorted(unknown)}")
            _ENABLED = frozenset(wanted)
    return family in _ENABLED and family not in _BROKEN


def _break(family):
    _BROKEN.add(family)
    _STATS[family][1] += 1


def _lib():
    global _LIB
    if _LIB is None:
        path = os.environ.get("XINFER_SOUP2_LIB")
        if not path:
            raise OSError("soup2: XINFER_SOUP2_LIB is not set")
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
    base = os.environ.get("XINFER_SOUP2_STATS")
    if not base:
        return
    try:
        with open(base + f"-pid{os.getpid()}.json", "w") as fh:
            json.dump({k: dict(routed=v[0], fell_back=v[1]) for k, v in _STATS.items()}, fh)
    except Exception:
        pass


# ---------------------------------------------------------------- kernels --


def qksq(x_t, tokens, two_hk, dk):
    """sq[1, tokens, two_hk, dk] f32 = f32(RNE_bf16(qk*qk)); qk = x_t rows' window.

    Bitwise equal to cast(mul(qk, qk)) -- the aclnn ReduceSum input bits do
    not change. Returns None (caller falls back) on any envelope miss.
    """
    width = x_t.shape[-1]
    qcols = two_hk * dk
    if not (
        x_t.dtype == torch.bfloat16 and x_t.is_contiguous() and x_t.dim() == 3
        and x_t.shape[0] == 1 and x_t.shape[1] == tokens and qcols < width
        and tokens * width < 2**31 and tokens * qcols < 2**31
        and qcols % 64 == 0 and x_t.device.type == "npu"
    ):
        return None
    try:
        sq = torch.empty(tokens * qcols, dtype=torch.float32, device=x_t.device)
        _launch("qksq_c6144", (sq, x_t.view(-1)), (tokens,))
        _routed("qksq")
        return sq.view(1, tokens, two_hk, dk)
    except (OSError, RuntimeError, AttributeError):
        _break("qksq")
        return None


def rawsq(out, tokens, hv, dv):
    """sq[tokens*hv, dv] f32 = RNE_f32(raw_f32 ** 2); raw = out.view(-1, dv).

    Bitwise equal to out.view(-1, dv).float().pow(2) elementwise. The aclnn
    Mean/Adds/Rsqrt that follow read the same bits.
    """
    total = tokens * hv
    if not (
        out.dtype == torch.bfloat16 and out.is_contiguous()
        and total * dv == out.numel() and total * dv < 2**31
        and dv % 64 == 0 and out.device.type == "npu"
        and total % 32 == 0
    ):
        return None
    try:
        sq = torch.empty(total * dv, dtype=torch.float32, device=out.device)
        _launch("rawsq_r128", (sq, out.view(-1)), (total,))
        _routed("rawsq")
        return sq.view(total, dv)
    except (OSError, RuntimeError, AttributeError):
        _break("rawsq")
        return None


def fc1gelu(acc, bias, out):
    """out bf16 = rne(0.5*t*(1+tanh(...))), t = acc + bias; None on any miss."""
    rows, cols = acc.shape
    if not (
        acc.dtype == torch.float32 and acc.is_contiguous() and acc.dim() == 2
        and cols == 4096 and bias.dtype == torch.float32 and bias.is_contiguous()
        and bias.shape == (cols,) and rows < 2**31 // 2 and acc.device.type == "npu"
    ):
        return None
    try:
        _launch("fc1_gelu_c4096", (out, acc, bias), (rows,))
        _routed("fc1gelu")
        return out
    except (OSError, RuntimeError, AttributeError):
        _break("fc1gelu")
        return None


atexit.register(dump_stats)


def finplain(x, weight_t):
    """Single bf16 Matmul for the plain-finish envelope (hook D route)."""
    _routed("finplain")
    return torch.matmul(x, weight_t)


def quant_u8(x):
    """out_u8 = uint8(clamp(floor(x+0.5), 0, 255)) in one pass (bitwise tail)."""
    n = x.numel()
    if not (
        x.dtype == torch.float32 and x.is_contiguous() and n % 1024 == 0
        and n < 2**31 and x.device.type == "npu"
    ):
        return None
    try:
        out = torch.empty(n, dtype=torch.uint8, device=x.device)
        _launch("quant_u8", (out, x.view(-1)), (n // 1024,))
        _routed("quant")
        return out.view(x.shape)
    except (OSError, RuntimeError, AttributeError):
        _break("quant")
        return None
