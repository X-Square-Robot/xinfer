"""GQA-GLUE knife 1: BHSD direct-production qkv split + norm apply.

Same math and launcher ABI as the in-service ``qkv_split_vec_h256`` (nine
pointers + rows + stream) with the q/k/v outputs written head-major BHSD
``[H, rows, 256]`` (``[1, H, rows, 256]`` tensors): the prefill chain's five
strided transposes disappear because the FIA BNSD inputs and the
``prompt_kv`` cache slots consume this layout directly. Outputs are
caller-allocated (``out_q``/``out_k``/``out_v``/``out_gate``) so the KV-cache
slots can be handed straight to the kernel. Internal AOT launcher only.
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
            raise OSError("qkv_split_bhsd: the unified core library is not loaded")
        call = getattr(lib, "xinfer_aot_qkv_split_vec_bhsd_h256_call")
        call.restype = ctypes.c_int
        call.argtypes = [ctypes.c_void_p] * 9 + [ctypes.c_int64] + [ctypes.c_void_p]
        error = getattr(lib, "xinfer_aot_qkv_split_vec_bhsd_h256_get_last_error")
        error.restype = ctypes.c_char_p
        _CALL, _ERROR = call, error
    return _CALL, _ERROR


def qkv_split_bhsd(packed, rstd_q, rstd_k, q_factor, k_factor,
                   out_q=None, out_k=None, out_v=None, out_gate=None):
    """Split ``packed [b, n, 5120]``, apply the q/k norms, write BHSD outputs.

    ``rstd_q``/``rstd_k`` are the torch-chain FP32 statistics and
    ``q_factor``/``k_factor`` are ``(weight.float() + 1)`` (identical inputs
    to ``qkv_split_vec``). Returns ``(q, k, v, gate)`` where q is
    ``[b, 8, n, 256]``, k/v ``[b, 2, n, 256]`` and gate ``[b, n, 2048]``;
    caller-provided buffers (e.g. ``prompt_kv`` slots, which are contiguous
    ``[1, 2, P, 256]`` views) are written in place.
    """
    b, n = packed.shape[0], packed.shape[1]
    rows = b * n
    if rows <= 0 or rows % 8:
        raise ValueError("qkv_split_bhsd requires a positive token count divisible by 8")
    stream_obj = torch.npu.current_stream(packed.device)
    if torch.npu.is_current_stream_capturing() and rows not in _WARMED:
        raise RuntimeError("warm up qkv_split_bhsd on this stream before graph capture")
    call, get_error = _library()
    if out_q is None:
        out_q = torch.empty((b, 8, n, 256), dtype=packed.dtype, device=packed.device)
    if out_k is None:
        out_k = torch.empty((b, 2, n, 256), dtype=packed.dtype, device=packed.device)
    if out_v is None:
        out_v = torch.empty((b, 2, n, 256), dtype=packed.dtype, device=packed.device)
    if out_gate is None:
        out_gate = torch.empty((b, n, 2048), dtype=packed.dtype, device=packed.device)
    # the single library's TVM-FFI face serves first (the unified launch boundary);
    # the ctypes launcher of the family library is the fallback below
    if not _native.launch_kernel("qkv_split_vec_bhsd_h256",
                                 (packed, rstd_q, rstd_k, q_factor, k_factor,
                                  out_q, out_k, out_v, out_gate), (rows,)):
        rc = call(ctypes.c_void_p(packed.data_ptr()),
                  ctypes.c_void_p(rstd_q.data_ptr()), ctypes.c_void_p(rstd_k.data_ptr()),
                  ctypes.c_void_p(q_factor.data_ptr()), ctypes.c_void_p(k_factor.data_ptr()),
                  ctypes.c_void_p(out_q.data_ptr()), ctypes.c_void_p(out_k.data_ptr()),
                  ctypes.c_void_p(out_v.data_ptr()), ctypes.c_void_p(out_gate.data_ptr()),
                  rows, ctypes.c_void_p(stream_obj.npu_stream))
    _WARMED.add(rows)
    if rc != 0:
        message = get_error()
        text = message.decode("utf-8", "replace") if message else "unknown failure"
        raise RuntimeError(f"qkv_split_bhsd: {text}")
    return out_q, out_k, out_v, out_gate
