"""Partial rotary embedding for the first 64 channels of 256-wide heads.

The in-tree launcher uses individually rounded vector multiplications
without FMA contraction, then one RNE BF16 conversion. Remaining channels
pass through an exact BF16/FP32/BF16 roundtrip. The reference agrees except
for numerically equivalent signed zeros.

Compact angle tables are expanded anew per call, so graph replay recomputes
them from the current angles.
"""

import ctypes

import torch

from xinfer.ops.backends.ascend import _native

_CALL = None
_ERROR = None
_WARMED = set()


def _library():
    global _CALL, _ERROR
    if _CALL is None:
        lib = _native.core_library()
        if not _native.has_symbol(lib, "xinfer_aot_rope_half_d64_call"):
            raise ValueError("token-major rotary is not included in this native build")
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
    # the single library's TVM-FFI face serves first (the unified launch boundary);
    # the same library's C-ABI launcher is the fallback below
    if not _native.launch_kernel("rope_half_d64",
                                 (q, k, cos2, sin2, q_out, k_out), (tokens,)):
        rc = call(ctypes.c_void_p(q.data_ptr()), ctypes.c_void_p(k.data_ptr()),
                  ctypes.c_void_p(cos2.data_ptr()), ctypes.c_void_p(sin2.data_ptr()),
                  ctypes.c_void_p(q_out.data_ptr()), ctypes.c_void_p(k_out.data_ptr()),
                  tokens, ctypes.c_void_p(stream_obj.npu_stream))
        if rc != 0:
            message = get_error()
            text = message.decode("utf-8", "replace") if message else "unknown failure"
            raise RuntimeError(f"rope_half_d64: {text}")
    _WARMED.add(tokens)
    return q_out, k_out
