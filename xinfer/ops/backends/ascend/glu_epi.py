"""GLU-EPILOGUE knife mounts (env XINFER_GLUEPI, default off).

K-B prefillglu: the prefill SwiGLU [M, 2048] x [12288, 2048] finish rides the
shipped AOT kernel ``gemm_glu_fp32_finish_split_6144`` (one staged silu*up +
BF16 round over the borrowed FP32 accumulator) instead of the torch tail
(silu_slice + silu + mul + cast). ABI per lib-glu-gelu §1:
``(out, accum, bias, rows:int64, has_bias:int64, stream)``; ``has_bias=0``
never dereferences bias.
"""
import ctypes
import os
from threading import Lock

import torch

from xinfer.ops.backends.ascend import _native

_LOCK = Lock()
_ENTRY = None


def _parse():
    raw = os.environ.get("XINFER_GLUEPI", "off").strip().lower()
    if raw in ("off", "", "0"):
        return frozenset()
    if raw in ("1", "all", "on"):
        return frozenset({"prefillglu"})
    return frozenset(t.strip() for t in raw.split(","))


def enabled(family):
    return family in _parse()


def _entry_pair():
    global _ENTRY
    if _ENTRY is None:
        from xinfer.ops.backends.ascend import _native

        lib = _native.core_library()
        call = getattr(lib, "xinfer_aot_gemm_glu_fp32_finish_split_6144_call")
        call.restype = ctypes.c_int32
        call.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p]
        err = getattr(lib, "xinfer_aot_gemm_glu_fp32_finish_split_6144_get_last_error")
        err.restype = ctypes.c_char_p
        _ENTRY = (call, err)
    return _ENTRY


def _count():
    path = os.environ.get("GLUEPI_COUNT_FILE")
    if path:
        with open(path, "a") as fh:
            fh.write("prefillglu\n")


def split_6144_finish(out, accum, rows):
    _count()
    # the single library's TVM-FFI face serves first (the unified launch boundary); its
    # bias slot takes a live tensor for the unread pointer (the null-optional convention
    # the packed-launch callers use), with has_bias=0 as the ctypes path passes
    if _native.launch_kernel("gemm_glu_fp32_finish_split_6144",
                             (out, accum, accum), (rows, 0)):
        return
    call, err = _entry_pair()
    stream = torch.npu.current_stream(accum.device).npu_stream
    status = call(out.data_ptr(), accum.data_ptr(), 0, rows, 0, stream)
    if status:
        raise RuntimeError("glu_epi split_6144: " + (err() or b"?").decode("utf-8", "replace"))
