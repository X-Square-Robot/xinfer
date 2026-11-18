"""Optional add-RMSNorm plans with buffered stores and direct FP32 weights.

``XINFER_SIMDEPI`` is off by default. ``v1280`` serves [rows,1280] BF16
vision/merger inputs; ``p2048`` serves [1,P,2048] BF16 prefill inputs.
The arithmetic is unchanged from the corresponding normalization plans.
The in-tree launchers use the current stream; FP32 weight representations
are cached before graph capture.
"""
import array
import ctypes
import os
from threading import Lock

import torch

from xinfer.ops.backends.ascend import _native

_ENABLED = None
_LIB = None
_ENTRIES = {}
_LOCK = Lock()

_FAMILIES = ("v1280", "p2048")
_KERNELS = {"v1280": "se1_ubv1280_buf2f", "p2048": "se1_ubv2048_buf2f"}


def _parse():
    global _ENABLED
    raw = os.environ.get("XINFER_SIMDEPI", "off").strip().lower()
    if raw in ("off", "", "0"):
        _ENABLED = frozenset()
    elif raw in ("1", "all", "on"):
        _ENABLED = frozenset(_FAMILIES)
    else:
        wanted = {t.strip() for t in raw.split(",")}
        unknown = wanted - set(_FAMILIES)
        if unknown:
            raise RuntimeError(f"XINFER_SIMDEPI unknown families: {sorted(unknown)}")
        _ENABLED = frozenset(wanted)


def enabled(family):
    if _ENABLED is None:
        _parse()
    return family in _ENABLED


def _library():
    return _native.core_library()


def _entry(name):
    with _LOCK:
        pair = _ENTRIES.get(name)
        if pair is None:
            lib = _library()
            call = getattr(lib, f"xinfer_aot_{name}_call")
            call.restype = ctypes.c_int32
            err = getattr(lib, f"xinfer_aot_{name}_get_last_error")
            err.restype = ctypes.c_char_p
            pair = (call, err)
            _ENTRIES[name] = pair
    return pair


def _eps_bits(epsilon):
    return array.array("i", array.array("f", [float(epsilon)]).tobytes())[0]


def _weight32(weight):
    """The contiguous FP32 row of the norm weight."""
    if weight.dtype == torch.float32:
        return weight if weight.is_contiguous() else weight.contiguous()
    return _native.derived("norm FP32 weight", weight, lambda: weight.detach().float().contiguous())


def _launch(name, tensors, scalars):
    if _native.launch_kernel(name, tensors, scalars):
        return
    call, err = _entry(name)
    stream = torch.npu.current_stream(tensors[0].device).npu_stream
    args = [ctypes.c_void_p(t.data_ptr()) for t in tensors]
    args += [ctypes.c_int64(s) for s in scalars]
    status = call(*args, ctypes.c_void_p(stream))
    if status:
        raise RuntimeError(
            "simdepi-r1 " + name + ": " + (err() or b"?").decode("utf-8", "replace")
        )


def _add_rms(name, x, residual, weight, eps):
    rows, width = x.shape[0], x.shape[1]
    wt = _weight32(weight)
    sum_out = torch.empty((rows, width), dtype=torch.bfloat16, device=x.device)
    norm_out = torch.empty((rows, width), dtype=torch.bfloat16, device=x.device)
    _launch(name, [x, residual, sum_out, wt, norm_out], [rows, 1, _eps_bits(eps)])
    return sum_out, norm_out


def add_rms_v1280(x, residual, weight, eps):
    """[rows,1280] BF16 vision/merger add rows -> (sum, normed), both BF16.

    Bitwise to the released ubv_h1280 plan on the same inputs (probe-se1)."""
    return _add_rms(_KERNELS["v1280"], x, residual, weight, eps)


def add_rms_p2048(x, residual, weight, eps):
    """[1,P,2048] BF16 prefill add rows -> (sum, normed), both BF16 [1,P,2048].

    Bitwise to the released ubv_h2048 plan on the same inputs (probe-se1)."""
    x2 = x[0]
    r2 = residual[0]
    sum_out, norm_out = _add_rms(_KERNELS["p2048"], x2, r2, weight, eps)
    return sum_out.view(x.shape), norm_out.view(x.shape)
