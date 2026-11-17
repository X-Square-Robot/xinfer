"""Head-major partial rotary embedding with compact angle tables.

The rotation semantics match ``rope_half_d64``. Q/K use [H,rows,256] and
the in-tree kernel expands width-32 cosine/sine tables to the 64 rotated
channels. Caller-allocated outputs allow K to write directly to its cache slot.
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
        if lib is None:
            raise OSError("rope_d64_bhsd: the unified core library is not loaded")
        call = getattr(lib, "xinfer_aot_rope_half_d64_bhsd_call")
        call.restype = ctypes.c_int
        call.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int64] + [ctypes.c_void_p]
        error = getattr(lib, "xinfer_aot_rope_half_d64_bhsd_get_last_error")
        error.restype = ctypes.c_char_p
        _CALL, _ERROR = call, error
    return _CALL, _ERROR


def rope_d64_bhsd(q, k, cos, sin, out_q=None, out_k=None):
    """Rotate q/k ``[b, H, rows, 256]`` BHSD on the first 64 channels.

    ``cos``/``sin`` are the compact FP32 ``[.., b*rows, 32]`` tables (no
    duplication); ``b*rows`` must be a positive multiple of 4. Returns
    ``(q_out, k_out)`` in the same BHSD layout; caller buffers are written
    in place.
    """
    tokens = q.shape[0] * q.shape[2]
    if tokens <= 0 or tokens % 8:
        raise ValueError("rope_d64_bhsd requires a positive token count divisible by 8")
    stream_obj = torch.npu.current_stream(q.device)
    if torch.npu.is_current_stream_capturing() and tokens not in _WARMED:
        raise RuntimeError("warm up rope_d64_bhsd on this stream before graph capture")
    call, get_error = _library()
    if out_q is None:
        out_q = torch.empty_like(q)
    if out_k is None:
        out_k = torch.empty_like(k)
    # the single library's TVM-FFI face serves first (the unified launch boundary);
    # the same library's C-ABI launcher is the fallback below
    if not _native.launch_kernel("rope_half_d64_bhsd",
                                 (q, k, cos, sin, out_q, out_k), (tokens,)):
        rc = call(ctypes.c_void_p(q.data_ptr()), ctypes.c_void_p(k.data_ptr()),
                  ctypes.c_void_p(cos.data_ptr()), ctypes.c_void_p(sin.data_ptr()),
                  ctypes.c_void_p(out_q.data_ptr()), ctypes.c_void_p(out_k.data_ptr()),
                  tokens, ctypes.c_void_p(stream_obj.npu_stream))
        if rc != 0:
            message = get_error()
            text = message.decode("utf-8", "replace") if message else "unknown failure"
            raise RuntimeError(f"rope_d64_bhsd: {text}")
    _WARMED.add(tokens)
    return out_q, out_k
