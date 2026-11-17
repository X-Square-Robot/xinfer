"""soupkill-r1 mount: single-pass BF16 finishes over the FP32 ACLNN accumulator.

Flag ``XINFER_SOUPKILL`` (default off): ``1``/``all``/a comma list of the
families ``finish`` (bias+residual finish), ``vrope`` (vision QKV + RoPE
finish), ``glu`` (prompt SwiGLU split finish through the library's own
``gemm_glu_fp32_finish_split_6144`` entry), ``gelu`` (vision fc1
bias+gelu_tanh finish).

Each route keeps the reference composition's FP32 arithmetic and rounds once
with RNE; the envelope checks mirror the library's own (contiguous BF16
operands, aligned dense rows, exact column widths), and any miss returns None
so the caller keeps the released torch chain.
"""

from __future__ import annotations

import ctypes
import os

import torch

_FAMILIES = ("finish", "vrope", "glu", "gelu")
from xinfer.ops.backends.ascend import _native
_DEFAULT_LIB = _native._default_library_path("libsoupkill.so")
_LOCK = None
_LIB = None
_ENTRIES: dict = {}
_ENABLED = None


def _parse() -> None:
    global _ENABLED
    raw = os.environ.get("XINFER_SOUPKILL", "off").strip().lower()
    if raw in ("", "off", "0", "false", "no"):
        _ENABLED = frozenset()
    elif raw in ("1", "all", "on"):
        _ENABLED = frozenset(_FAMILIES)
    else:
        wanted = {t.strip() for t in raw.split(",")}
        unknown = wanted - set(_FAMILIES)
        if unknown:
            raise RuntimeError(f"XINFER_SOUPKILL unknown families: {sorted(unknown)}")
        _ENABLED = frozenset(wanted)


def enabled(family: str) -> bool:
    if _ENABLED is None:
        _parse()
    return family in _ENABLED


def _library():
    global _LIB, _LOCK
    if _LOCK is None:
        import threading

        _LOCK = threading.Lock()
    if _LIB is None:
        with _LOCK:
            if _LIB is None:
                _LIB = ctypes.CDLL(os.environ.get("XINFER_SOUPKILL_LIB", _DEFAULT_LIB))
    return _LIB


_XC_LOCK = None
_XC_LIB = None
_XC_ENTRIES: dict = {}


def _xc_library():
    """The unified core library: the glu route calls its own AOT kernel."""
    global _XC_LIB, _XC_LOCK
    if _XC_LOCK is None:
        import threading

        _XC_LOCK = threading.Lock()
    if _XC_LIB is None:
        with _XC_LOCK:
            if _XC_LIB is None:
                _XC_LIB = _native.core_library()
                if _XC_LIB is None:
                    raise OSError("soupkill glu: the unified core library is not loaded")
    return _XC_LIB


def _entry_xc(name, pointers, scalars):
    key = (name, pointers, scalars)
    pair = _XC_ENTRIES.get(key)
    if pair is None:
        lib = _xc_library()
        call = getattr(lib, f"xinfer_aot_{name}_call")
        call.restype = ctypes.c_int32
        call.argtypes = [ctypes.c_void_p] * pointers + [ctypes.c_int64] * scalars + [ctypes.c_void_p]
        err = getattr(lib, f"xinfer_aot_{name}_get_last_error")
        err.restype = ctypes.c_char_p
        pair = (call, err)
        _XC_ENTRIES[key] = pair
    return pair


def _launch_xc(name, tensors, scalars):
    call, err = _entry_xc(name, len(tensors), len(scalars))
    stream = torch.npu.current_stream(tensors[0].device).npu_stream
    status = call(
        *[ctypes.c_void_p(t.data_ptr()) for t in tensors],
        *[ctypes.c_int64(s) for s in scalars],
        ctypes.c_void_p(stream),
    )
    if status:
        raise RuntimeError(f"{name} failed: {err().decode('utf-8', 'replace')}")


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
    call, err = _entry(name, len(tensors), len(scalars))
    stream = torch.npu.current_stream(tensors[0].device).npu_stream
    status = call(
        *[ctypes.c_void_p(t.data_ptr()) for t in tensors],
        *[ctypes.c_int64(s) for s in scalars],
        ctypes.c_void_p(stream),
    )
    if status:
        raise RuntimeError(f"{name} failed: {err().decode('utf-8', 'replace')}")


def _flat(t) -> bool:
    return t is not None and t.is_contiguous() and t.device.type == "npu"


def finish_bias_res(acc, bias, residual, out):
    """out = rne(acc + bias + residual); None when the envelope is not ours."""
    if not enabled("finish"):
        return None
    if not (_flat(acc) and acc.dtype == torch.float32 and acc.dim() == 2
            and _flat(bias) and bias.dtype == torch.bfloat16 and bias.dim() == 1
            and _flat(residual) and residual.dtype == torch.bfloat16
            and residual.shape == acc.shape and _flat(out) and out.dtype == torch.bfloat16
            and out.shape == acc.shape and acc.shape[1] == 1024 and bias.shape[0] == 1024
            and acc.shape[0] % 4 == 0 and acc.shape[0] > 0):
        return None
    _launch("finish_br_c1024", (out, acc, bias, residual), (acc.shape[0],))
    return out


def vision_qkv_rope(acc, bias, cos2, sin2, out, heads=16, dim=64):
    """out = rne(rope_half(acc + bias)); cos2/sin2 are the row's table duplicated to ``dim`` columns."""
    if not enabled("vrope"):
        return None
    if not (heads == 16 and dim == 64 and _flat(acc) and acc.dtype == torch.float32
            and acc.dim() == 2 and acc.shape[1] == 3 * heads * dim
            and _flat(bias) and bias.dtype == torch.bfloat16 and bias.numel() == acc.shape[1]
            and _flat(cos2) and cos2.dtype == torch.float32
            and tuple(cos2.shape) == (acc.shape[0], dim)
            and _flat(sin2) and sin2.dtype == torch.float32
            and tuple(sin2.shape) == (acc.shape[0], dim)
            and _flat(out) and out.dtype == torch.bfloat16
            and out.numel() == acc.shape[0] * 3 * heads * dim
            and acc.shape[0] % 2 == 0 and acc.shape[0] > 0):
        return None
    _launch("vision_qkv_rope_v3", (out, acc, bias, cos2, sin2), (acc.shape[0],))
    return out


def glu_split(acc, out, half=6144):
    """out = rne(silu(acc[:, :half]) * acc[:, half:]) via the library split_6144 entry.

    The library ABI is ``(out, accum, bias, rows:int64, has_bias:int64, stream)``;
    with ``has_bias == 0`` the bias handle is never dereferenced, so ``out``
    stands in for it exactly as the library's own ``EpilogueGlu`` does.
    """
    if not enabled("glu"):
        return None
    if not (_flat(acc) and acc.dtype == torch.float32 and acc.dim() == 2
            and acc.shape[1] == 2 * half and _flat(out) and out.dtype == torch.bfloat16
            and out.shape == (acc.shape[0], half) and acc.shape[0] > 0):
        return None
    _launch_xc("gemm_glu_fp32_finish_split_6144", (out, acc, out), (acc.shape[0], 0))
    return out


def gelu_finish(acc, bias, out, cols=4096):
    """out = rne(gelu_tanh(acc + widen(bias))); None when the envelope is not ours.

    Replaces the released ``output + bias.float()`` / ``F.gelu(tanh)`` / cast
    chain on the vision fc1 shape with one UB-staged pass.
    """
    if not enabled("gelu"):
        return None
    if not (_flat(acc) and acc.dtype == torch.float32 and acc.dim() == 2
            and acc.shape[1] == cols and acc.shape[0] % 2 == 0 and acc.shape[0] > 0
            and _flat(bias) and bias.dtype == torch.bfloat16 and bias.numel() == cols
            and _flat(out) and out.dtype == torch.bfloat16
            and out.shape == acc.shape):
        return None
    _launch("gelu_g1", (out, acc, bias), (acc.shape[0],))
    return out
