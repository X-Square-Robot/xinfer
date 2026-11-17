"""2B-COVERAGE: rotary_dim-64 partial rope through the AOT launcher (ctypes).

The launcher symbol ``xinfer_aot_rope_half_d64_call`` follows the
established internal AOT-launcher precedent (no FFI face change): the kernel
rotates the first 64 channels of every head with individually rounded vector
multiplies (no FMA contraction) and one RNE BF16 round, and passes channels
64..255 through an exact bf16->f32->bf16 round trip, bitwise-equal to the
torch composition of ``fused.rope_full`` at ``rotary_dim == 64`` (the only
bit-level deviations are signed zeros, +0 vs -0, numerically identical).
The angle tables are duplicated to 64 columns with a fresh concat per call:
stateless, so a graph capture always bakes the concat into the graph and
every replay recomputes it from the current angles.
"""

import ctypes
import os

import torch

from xinfer.ops.backends.ascend import _native

_CALL = None
_ERROR = None
_WARMED = set()


def _library():
    global _CALL, _ERROR
    if _CALL is None:
        lib = _native.core_library()
        if lib is None:
            raise OSError("rope_d64: the unified core library is not loaded")
        call = getattr(lib, "xinfer_aot_rope_half_d64_call")
        call.restype = ctypes.c_int
        call.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int64] + [ctypes.c_void_p]
        error = getattr(lib, "xinfer_aot_rope_half_d64_get_last_error")
        error.restype = ctypes.c_char_p
        _CALL, _ERROR = call, error
    return _CALL, _ERROR


def rope_half_d64(q, k, cos, sin):
    """Rotate q/k [.., tokens, heads, 256] on the first 64 channels; return new tensors.

    ``q`` is ``[b, n, 8, 256]`` and ``k`` ``[b, n, 2, 256]`` BF16 contiguous
    with ``b*n`` a positive multiple of 4; ``cos``/``sin`` are FP32
    ``[.., b*n, 32]`` contiguous.
    """
    tokens = q.shape[0] * q.shape[1]
    if tokens <= 0 or tokens % 4:
        raise ValueError("rope_half_d64 requires a positive token count divisible by 4")
    stream_obj = torch.npu.current_stream(q.device)
    if torch.npu.is_current_stream_capturing() and not _WARMED:
        raise RuntimeError("warm up rope_half_d64 on this stream before graph capture")
    call, get_error = _library()
    cos2 = torch.cat((cos, cos), dim=-1).contiguous()
    sin2 = torch.cat((sin, sin), dim=-1).contiguous()
    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)
    rc = call(ctypes.c_void_p(q.data_ptr()), ctypes.c_void_p(k.data_ptr()),
              ctypes.c_void_p(cos2.data_ptr()), ctypes.c_void_p(sin2.data_ptr()),
              ctypes.c_void_p(q_out.data_ptr()), ctypes.c_void_p(k_out.data_ptr()),
              tokens, ctypes.c_void_p(stream_obj.npu_stream))
    _WARMED.add(tokens)
    if rc != 0:
        message = get_error()
        text = message.decode("utf-8", "replace") if message else "unknown failure"
        raise RuntimeError(f"rope_half_d64: {text}")
    return q_out, k_out
