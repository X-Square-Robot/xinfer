"""The v22 rmsnorm FFI's plain-form plan dispatch over the unified core library.

The 38.82 stack's ``rms_norm_cast`` fell through its RMSFUSE m16 knife to the
v22 library's ``rmsnorm`` FFI, whose ``Forward()`` routed the served plain
norms onto the rms_consistent plans: the ViT patch norm ``[rows, 1280]``
(BF16 or FP32 weight; the UB plan when the envelope holds, the generic plan
otherwise), the prefill input norm ``[rows, 2048]`` (the UB plan) and the
``[rows in {16, 32, 64}, 1024]`` action rows (the generic h1024 plan over an
FP32 weight). The unified core library carries the same kernels
(``src/norm/rms_consistent.py``, the ``xinfer_aot_rms_consistent_*``
launchers); this module mirrors that dispatch call for call so the served
sites keep the 38.82 bits. An envelope miss returns ``None`` and the caller
keeps its composition (the reference for the binding's general tail); the
v22 tree's k-pair generic fallback for other shapes is not carried, so those
sites keep the reference numerics instead.
"""

from __future__ import annotations

import ctypes
import struct

import torch

from xinfer.ops.backends.ascend import _native

_MAX_I32 = 2**31 - 1
_FLOAT32_MAX = torch.finfo(torch.float32).max


def _bits(value: float) -> int:
    """The float's IEEE-754 bit pattern in the AOT scalar ABI (low word)."""
    return struct.unpack("<I", struct.pack("<f", float(value)))[0]


_ENTRIES = {}


def _entry(name):
    """The ``xinfer_aot_<name>_call`` / ``..._get_last_error`` pair, or None."""
    pair = _ENTRIES.get(name)
    if pair is None:
        lib = _native.core_library()
        if lib is None:
            return None
        try:
            call = getattr(lib, f"xinfer_aot_{name}_call")
            error = getattr(lib, f"xinfer_aot_{name}_get_last_error")
        except AttributeError:
            return None
        call.restype = ctypes.c_int32
        # six buffer handles, twelve int64 scalars, the stream
        call.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int64] * 12 + [ctypes.c_void_p]
        error.restype = ctypes.c_char_p
        pair = (call, error)
        _ENTRIES[name] = pair
    return pair


def _launch(name, x2, weight, out, rows, width, x_stride, weight_words, eps):
    """One plain-form rms_consistent launch: centered/add/write_stats off.

    rh/sh/ih are never dereferenced with add_residual=0 and write_stats=0;
    pass x to keep the packed launcher's non-null handle contract (the v22
    RMSNormApplyRstd precedent). The single library's TVM-FFI face serves the
    same packed arguments first (the unified launch boundary); the ctypes
    launcher of the family library is the fallback."""
    tensors = (x2, x2, x2, weight, out, x2)
    scalars = (rows, x_stride, x_stride, x_stride, width, 1, weight_words, _bits(eps), 0, 0, 1, 0)
    if _native.launch_kernel(name, tensors, scalars):
        return
    call, error = _entry(name)
    stream = torch.npu.current_stream(x2.device).npu_stream
    status = call(ctypes.c_void_p(x2.data_ptr()), ctypes.c_void_p(x2.data_ptr()),
                  ctypes.c_void_p(x2.data_ptr()), ctypes.c_void_p(weight.data_ptr()),
                  ctypes.c_void_p(out.data_ptr()), ctypes.c_void_p(x2.data_ptr()),
                  *[ctypes.c_int64(int(s)) for s in
                    (rows, x_stride, x_stride, x_stride, width, 1, weight_words,
                     _bits(eps), 0, 0, 1, 0)],
                  ctypes.c_void_p(stream))
    if status:
        message = error()
        raise RuntimeError(
            f"rms_consistent {name}: {message.decode('utf-8', 'replace') if message else 'unknown failure'}")


def plain_rms(x, weight, eps):
    """The v22 ``Forward()`` plain-form dispatch for ``x``; ``None`` on an envelope miss.

    ``x`` is BF16 rows on the NPU with a packed ``[rows, width]`` layout (any
    leading dimensions flatten along rows, as the v22 ``Rows()`` contract did);
    ``weight`` is a ``[width]`` BF16/FP32 vector on the same device.
    """
    if not isinstance(x, torch.Tensor) or not torch.is_tensor(weight):
        return None
    if x.device.type != "npu" or weight.device != x.device:
        return None
    if x.dtype != torch.bfloat16 or weight.dtype not in (torch.bfloat16, torch.float32):
        return None
    width = x.shape[-1] if x.dim() else 0
    if not width or weight.shape != (width,):
        return None
    if x.dim() < 2 or not x.is_contiguous() or not weight.is_contiguous():
        return None
    try:
        epsilon = float(eps)
    except (TypeError, ValueError):
        return None
    if not (0 < epsilon <= _FLOAT32_MAX):
        return None
    rows = x.numel() // width
    if not 0 < rows <= _MAX_I32:
        return None
    x2 = x.reshape(rows, width)
    x_stride = x2.stride(0)
    weight_words = 1 if weight.dtype == torch.bfloat16 else 2
    out = torch.empty((rows, width), dtype=torch.bfloat16, device=x2.device)

    # The h1024 fast plan: the v22 "r2 (agent H)" plain route -- batch row
    # counts over an FP32 weight only (the v22 required w.bytes == 4).
    if width == 1024:
        if rows not in (16, 32, 64) or weight.dtype != torch.float32:
            return None
        _launch("rms_consistent_generic_h1024", x2, weight, out, rows, width, x_stride,
                weight_words, epsilon)
        return out.view(x.shape)

    if width == 1280:
        # The UB plan's envelope: a multiple-of-4 row count, packed rows and
        # 32-byte-aligned input, output and weight; a miss rides the generic
        # plan (the same numerics-contract class as the v22 dispatch).
        use_ub = (rows % 4 == 0 and rows <= _MAX_I32 // 1280 and x_stride == 1280
                  and x2.data_ptr() % 32 == 0 and weight.data_ptr() % 32 == 0
                  and out.data_ptr() % 32 == 0)
        _launch("rms_consistent_ub_h1280" if use_ub else "rms_consistent_generic_h1280",
                x2, weight, out, rows, width, x_stride, weight_words, epsilon)
        return out.view(x.shape)

    if width == 2048:
        # The v22 UB plan only; its generic-pair fallback is not carried, so
        # an envelope miss keeps the caller's composition (the reference).
        if (rows % 4 == 0 and rows <= _MAX_I32 // 2048 and x_stride == 2048
                and x2.data_ptr() % 32 == 0 and weight.data_ptr() % 32 == 0
                and out.data_ptr() % 32 == 0):
            _launch("rms_consistent_ub_h2048", x2, weight, out, rows, width, x_stride,
                    weight_words, epsilon)
            return out.view(x.shape)
        return None
    return None
