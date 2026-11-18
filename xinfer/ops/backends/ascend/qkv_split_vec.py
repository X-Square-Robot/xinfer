"""2B-COVERAGE: vector-engine qkv split + norm apply via the AOT launcher.

Replaces the scalar qkv_split_rstd path for the prefill shapes: same math
(``((x32 * factor) * rstd)``, one RNE BF16 round -- the _zero_centered_heads
association), 64-lane vector chunks, bitwise-equal to the torch composition
on identical rstd inputs. Internal AOT launcher only (no FFI face change).
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
        if not _native.has_symbol(lib, "xinfer_aot_qkv_split_vec_h256_call"):
            raise ValueError("token-major QKV split is not included in this native build")
        call = getattr(lib, "xinfer_aot_qkv_split_vec_h256_call")
        call.restype = ctypes.c_int
        call.argtypes = [ctypes.c_void_p] * 9 + [ctypes.c_int64] + [ctypes.c_void_p]
        error = getattr(lib, "xinfer_aot_qkv_split_vec_h256_get_last_error")
        error.restype = ctypes.c_char_p
        _CALL, _ERROR = call, error
    return _CALL, _ERROR


def qkv_split_vec(packed, rstd_q, rstd_k, q_factor, k_factor):
    """Split ``packed [b, n, 5120]`` and apply the q/k norms; return q/k/v/gate.

    ``rstd_q``/``rstd_k`` are the torch-chain FP32 statistics; ``q_factor``/
    ``k_factor`` are ``(weight.float() + 1)``. Rows (``b*n``) must be
    positive (any parity; the kernel tails partial tiles).
    """
    rows = packed.shape[0] * packed.shape[1]
    if rows <= 0:
        raise ValueError("qkv_split_vec requires a positive token count")
    stream_obj = torch.npu.current_stream(packed.device)
    if torch.npu.is_current_stream_capturing() and not _WARMED:
        raise RuntimeError("warm up qkv_split_vec on this stream before graph capture")
    call, get_error = _library()
    b, n = packed.shape[0], packed.shape[1]
    q = torch.empty((b, n, 8, 256), dtype=packed.dtype, device=packed.device)
    k = torch.empty((b, n, 2, 256), dtype=packed.dtype, device=packed.device)
    v = torch.empty_like(k)
    gate = torch.empty((b, n, 2048), dtype=packed.dtype, device=packed.device)
    # the single library's TVM-FFI face serves first when it carries one (the unified
    # launch boundary); the ctypes launcher is the fallback below
    if not _native.launch_kernel("qkv_split_vec_h256",
                                 (packed, rstd_q, rstd_k, q_factor, k_factor,
                                  q, k, v, gate), (rows,)):
        rc = call(ctypes.c_void_p(packed.data_ptr()),
                  ctypes.c_void_p(rstd_q.data_ptr()), ctypes.c_void_p(rstd_k.data_ptr()),
                  ctypes.c_void_p(q_factor.data_ptr()), ctypes.c_void_p(k_factor.data_ptr()),
                  ctypes.c_void_p(q.data_ptr()), ctypes.c_void_p(k.data_ptr()),
                  ctypes.c_void_p(v.data_ptr()), ctypes.c_void_p(gate.data_ptr()),
                  rows, ctypes.c_void_p(stream_obj.npu_stream))
        if rc != 0:
            message = get_error()
            text = message.decode("utf-8", "replace") if message else "unknown failure"
            raise RuntimeError(f"qkv_split_vec: {text}")
    _WARMED.add(rows)
    return q, k, v, gate
