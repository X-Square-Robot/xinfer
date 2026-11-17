"""finish-bf16 mounts: single-pass BF16 finishes over the aclnnMatmul BF16 store.

Flag ``XINFER_FINISHBF16`` (default off): a comma list of the families
``gateup`` (prefill SwiGLU [464, 2048] x [12288, 2048]; the bf16-reading
split_6144 finish), ``zba`` (GDN qkv_zba [464, 2048] x [8224, 2048]; the
bf16-reading zbasplit), ``vrope`` (vision QKV + RoPE [868, 1024] x
[3072, 1024]; the bf16-reading packed v3 rope), ``finres`` (vision
proj/fc2 bias+residual finishes, N=1024; the bf16-reading finish_br) and
``gelu`` (vision fc1 bias+gelu_tanh, N=4096; the bf16-reading gelu).

Every route replaces the aclnnMm FP32 accumulator + fp32-reading finish
pair with one aclnnMatmul BF16 store (FP32 accumulate, single RNE --
bitwise the value the old chain rounded, GEMM-AUDIT iso v2) and the
ported finish kernel reading that store. Envelope misses return None so
the caller keeps the released chain.
"""

from __future__ import annotations

import atexit
import ctypes
import json
import os
from threading import Lock

import torch

_FAMILIES = ("gateup", "zba", "vrope", "finres", "gelu")
from xinfer.ops.backends.ascend import _native
_DEFAULT_LIB = _native._default_library_path("libfinishbf16.so")
_LOCK = Lock()
_LIB = None
_ENTRIES: dict = {}
_ENABLED = None

_STATS = {family: 0 for family in _FAMILIES}
_STATS_PATH = os.environ.get("XINFER_FINISHBF16_STATS")


def _stats_dump():
    if _STATS_PATH:
        try:
            with open(f"{_STATS_PATH}-pid{os.getpid()}", "w") as fh:
                fh.write(json.dumps(_STATS))
        except Exception:  # noqa: BLE001 - a stats receipt must never break serving
            pass


atexit.register(_stats_dump)


def _parse() -> None:
    global _ENABLED
    raw = os.environ.get("XINFER_FINISHBF16", "off").strip().lower()
    if raw in ("", "off", "0", "false", "no"):
        _ENABLED = frozenset()
    elif raw in ("1", "all", "on"):
        _ENABLED = frozenset(_FAMILIES)
    else:
        wanted = {t.strip() for t in raw.split(",")}
        unknown = wanted - set(_FAMILIES)
        if unknown:
            raise RuntimeError(f"XINFER_FINISHBF16 unknown families: {sorted(unknown)}")
        _ENABLED = frozenset(wanted)


def enabled(family: str) -> bool:
    if _ENABLED is None:
        _parse()
    return family in _ENABLED


def _library():
    global _LIB
    if _LIB is None:
        with _LOCK:
            if _LIB is None:
                _LIB = ctypes.CDLL(os.environ.get("XINFER_FINISHBF16_LIB", _DEFAULT_LIB))
    return _LIB


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


def _flat(t) -> bool:
    return t is not None and t.is_contiguous() and t.device.type == "npu"


def _bf16_store(x, weight):
    """One aclnnMatmul BF16 store (FP32 accumulate, single RNE), NT view."""
    w = weight if weight.is_contiguous() else weight.contiguous()
    return torch.matmul(x, w.t())


def glu_split_bf16(x, weight, half=6144):
    """gateup: rne(silu(f32(g16)) * f32(u16)) over the [m, 12288] BF16 store."""
    if not (enabled("gateup") and _flat(x) and _flat(weight)
            and x.dtype == weight.dtype == torch.bfloat16 and x.dim() == 2
            and tuple(weight.shape) == (2 * half, 2048) and x.shape[1] == 2048
            and weight.device == x.device and x.shape[0] > 0):
        return None
    rows = x.shape[0]
    acc16 = _bf16_store(x, weight)
    out = x.new_empty((rows, half), dtype=torch.bfloat16)
    _launch("glu_split_bf16_c6144", (out, acc16), (rows,))
    _STATS["gateup"] += 1
    return out


def zbasplit_bf16(x, weight, keep, conv=6144):
    """zba: (rne(f32(a16)*keep), rne(f32(b16)*keep)) over the [m, 8224] BF16 store."""
    if not (enabled("zba") and _flat(x) and _flat(weight)
            and x.dtype == weight.dtype == torch.bfloat16 and x.dim() == 2
            and tuple(weight.shape) == (8224, 2048) and x.shape[1] == 2048
            and weight.device == x.device
            and _flat(keep) and keep.dtype == torch.float32
            and tuple(keep.shape) == (x.shape[0],) and keep.device == x.device
            and x.shape[0] > 0):
        return None
    rows = x.shape[0]
    acc16 = _bf16_store(x, weight)
    conv_buf = x.new_empty((rows, conv), dtype=torch.bfloat16)
    zba_buf = x.new_empty((rows, 8224 - conv), dtype=torch.bfloat16)
    _launch("zbasplit_bf16_c8224", (conv_buf, zba_buf, acc16, keep), (rows,))
    _STATS["zba"] += 1
    return conv_buf, zba_buf


def vrope_bf16(x, weight, bias, cos2, sin2, heads=16, dim=64):
    """vrope: rne(rope_half(f32(a16) + f32(bias))) stored packed [3, rows, heads, dim]."""
    if not (enabled("vrope") and heads == 16 and dim == 64
            and _flat(x) and _flat(weight)
            and x.dtype == weight.dtype == torch.bfloat16 and x.dim() == 2
            and tuple(weight.shape) == (3 * heads * dim, 1024) and x.shape[1] == 1024
            and weight.device == x.device
            and _flat(bias) and bias.dtype == torch.bfloat16
            and bias.numel() == 3 * heads * dim and bias.device == x.device
            and _flat(cos2) and cos2.dtype == torch.float32
            and tuple(cos2.shape) == (x.shape[0], dim) and cos2.device == x.device
            and _flat(sin2) and sin2.dtype == torch.float32
            and tuple(sin2.shape) == (x.shape[0], dim) and sin2.device == x.device
            and x.shape[0] % 2 == 0 and x.shape[0] > 0):
        return None
    rows = x.shape[0]
    acc16 = _bf16_store(x, weight)
    out = x.new_empty((3, rows, heads, dim), dtype=torch.bfloat16)
    _launch("vision_qkv_rope_bf16_v3p", (out, acc16, bias, cos2, sin2), (rows,))
    _STATS["vrope"] += 1
    return out


def finish_br_bf16(x, weight, bias, residual):
    """finres: rne((f32(a16) + f32(bias)) + f32(res)) over the [m, 1024] BF16 store."""
    if not (enabled("finres") and _flat(x) and _flat(weight)
            and x.dtype == weight.dtype == torch.bfloat16 and x.dim() == 2
            and tuple(weight.shape) in ((1024, 1024), (1024, 4096))
            and x.shape[1] == weight.shape[1] and weight.device == x.device
            and _flat(bias) and bias.dtype == torch.bfloat16
            and tuple(bias.shape) == (1024,) and bias.device == x.device
            and _flat(residual) and residual.dtype == torch.bfloat16
            and tuple(residual.shape) == (x.shape[0], 1024) and residual.device == x.device
            and x.shape[0] % 4 == 0 and x.shape[0] > 0):
        return None
    rows = x.shape[0]
    acc16 = _bf16_store(x, weight)
    out = x.new_empty((rows, 1024), dtype=torch.bfloat16)
    _launch("finish_br_bf16_c1024", (out, acc16, bias, residual), (rows,))
    _STATS["finres"] += 1
    return out


def gelu_bf16(x, weight, bias, cols=4096):
    """gelu: rne(gelu_tanh(f32(a16) + f32(bias))) over the [m, 4096] BF16 store."""
    if not (enabled("gelu") and _flat(x) and _flat(weight)
            and x.dtype == weight.dtype == torch.bfloat16 and x.dim() == 2
            and tuple(weight.shape) == (cols, 1024) and x.shape[1] == 1024
            and weight.device == x.device
            and _flat(bias) and bias.dtype == torch.bfloat16
            and bias.numel() == cols and bias.device == x.device
            and x.shape[0] % 2 == 0 and x.shape[0] > 0):
        return None
    rows = x.shape[0]
    acc16 = _bf16_store(x, weight)
    out = x.new_empty((rows, cols), dtype=torch.bfloat16)
    _launch("gelu_bf16_c4096", (out, acc16, bias), (rows,))
    _STATS["gelu"] += 1
    return out
