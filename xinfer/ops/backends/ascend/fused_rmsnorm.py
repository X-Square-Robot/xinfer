"""Optional RMS normalization plans in the in-tree native library.

``XINFER_ASCEND_FUSED_RMSNORM`` is off by default. ``m16`` handles M16/H1024
normalization, ``h2048w`` handles [1,P,2048] add-normalization with direct
FP32 weights, and ``f32in`` exposes the FP32-accumulator finish helper.
That helper rounds the product to BF16 before the M16 normalization and
defers to an enabled small-M projection route. Launches use the current
stream; weight conversions must be prepared before graph capture.
"""
import os
from xinfer.ops.backends.ascend import _native
import array
import ctypes
from threading import Lock

import torch

_ENABLED = None
_LIB = None
_ENTRIES = {}
_LOCK = Lock()

_FAMILIES = ("m16", "f32in", "h2048w")


def _parse():
    global _ENABLED
    raw = os.environ.get("XINFER_ASCEND_FUSED_RMSNORM", "off").strip().lower()
    if raw in ("off", "", "0"):
        _ENABLED = frozenset()
    elif raw in ("1", "all", "on"):
        _ENABLED = frozenset(_FAMILIES)
    else:
        wanted = {t.strip() for t in raw.split(",")}
        unknown = wanted - set(_FAMILIES)
        if unknown:
            raise RuntimeError(f"XINFER_ASCEND_FUSED_RMSNORM unknown families: {sorted(unknown)}")
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
            "fused-rmsnorm " + name + ": " + (err() or b"?").decode("utf-8", "replace")
        )


def _weight32(weight):
    """The contiguous FP32 row of the norm weight."""
    if weight.dtype == torch.float32:
        return weight if weight.is_contiguous() else weight.contiguous()
    return _native.derived("norm FP32 weight", weight, lambda: weight.detach().float().contiguous())


_STATS = {}


def _stats_scratch(device):
    cached = _STATS.get(device)
    if cached is None:
        cached = torch.empty(16, dtype=torch.float32, device=device)
        _STATS[device] = cached
    return cached


def add_rms_m16(x, residual, weight, eps):
    """[1,16,1024] BF16 x/residual + FP32 weight -> (sum, normed), both BF16.

    Bitwise to the lib generic_h1024 plan on the same inputs."""
    w32 = _weight32(weight)
    rows, width = 16, 1024
    shape = residual.shape
    x2 = x.reshape(rows, width)
    r2 = residual.reshape(rows, width)
    sum_out = torch.empty((rows, width), dtype=torch.bfloat16, device=x.device)
    norm_out = torch.empty((rows, width), dtype=torch.bfloat16, device=x.device)
    _launch("rmsnorm_m16_h1024", [x2, r2, sum_out, w32, norm_out, _stats_scratch(x.device)],
            [rows, _eps_bits(eps), 0, 1, 1, 0])
    return sum_out.view(shape), norm_out.view(shape)


def rms_m16(x, weight, eps):
    """[1,16,1024] BF16 plain RMSNorm -> normed BF16 (no residual)."""
    w32 = _weight32(weight)
    rows, width = 16, 1024
    shape = x.shape
    x2 = x.reshape(rows, width)
    sum_out = torch.empty((rows, width), dtype=torch.bfloat16, device=x.device)
    norm_out = torch.empty((rows, width), dtype=torch.bfloat16, device=x.device)
    _launch("rmsnorm_m16_h1024", [x2, x2, sum_out, w32, norm_out, _stats_scratch(x.device)],
            [rows, _eps_bits(eps), 0, 0, 1, 0])
    return norm_out.view(shape)


def add_rms_m16_f32in(acc32, residual, weight, eps):
    """FP32 [16,1024] accumulator + BF16 residual -> (sum, normed) BF16.

    The in-kernel RNE round to BF16 reproduces the gemm_epilogue finish cast
    bit-for-bit before the m16-plan arithmetic."""
    w32 = _weight32(weight)
    rows, width = 16, 1024
    shape = residual.shape
    r2 = residual.reshape(rows, width)
    acc2 = acc32.reshape(rows, width)
    sum_out = torch.empty((rows, width), dtype=torch.bfloat16, device=acc32.device)
    norm_out = torch.empty((rows, width), dtype=torch.bfloat16, device=acc32.device)
    _launch("rmsnorm_m16_h1024_fp32_input",
            [acc2, r2, sum_out, w32, norm_out, _stats_scratch(acc32.device)],
            [rows, _eps_bits(eps), 0, 1, 1, 0])
    return sum_out.view(shape), norm_out.view(shape)


def add_rms_h2048w(x, residual, weight, eps):
    """[1,P,2048] BF16 prefill add rows on the f32-direct-weight ubv copy."""
    w32 = _weight32(weight)
    rows, width = x.shape[1], x.shape[2]
    x2 = x.reshape(rows, width)
    r2 = residual.reshape(rows, width)
    sum_out = torch.empty((rows, width), dtype=torch.bfloat16, device=x.device)
    norm_out = torch.empty((rows, width), dtype=torch.bfloat16, device=x.device)
    _launch("rmsnorm_unified_buffer_h2048_fp32_weight", [x2, r2, sum_out, w32, norm_out],
            [rows, 1, _eps_bits(eps)])
    return sum_out.view(x.shape), norm_out.view(x.shape)


def gemm_add_rmsnorm(x, weight, residual, norm_weight, epsilon, projection):
    """The fused gemm+add+rms entry for the action m16 sites.

    ``x`` is the attention output BNSD [16,16,128] (o-projection) or the GLU
    hidden [1,16,2048]/[16,2048] (down projection). Borrows the projection's
    FP32 accumulator, launches the f32in plan inside the borrow scope, and
    returns (sum, normed) BF16 -- or None when the route does not apply (the
    caller falls back to the released chain). Defers to the action-gemm-m16
    oproj/down kernels when those families are enabled."""
    try:
        from xinfer.ops.backends.ascend import action_gemm_m16
        if action_gemm_m16.enabled("oproj") and x.ndim == 3 and tuple(x.shape) == (16, 16, 128):
            return None
        if action_gemm_m16.enabled("down") and x.ndim >= 2 and tuple(weight.shape) == (1024, 2048) \
                and tuple(x.shape[-2:]) == (16, 2048):
            return None
    except Exception:
        pass
    if x.ndim == 3 and x.shape[0] == 1:
        x = x[0]
    bnsd = x.ndim == 3 and tuple(x.shape) == (16, 16, 128)
    if (weight.ndim != 2 or x.device.type != "npu" or weight.device != x.device
            or x.dtype != weight.dtype or weight.dtype != torch.bfloat16):
        return None
    if bnsd:
        if tuple(weight.shape) != (1024, 2048) or not x.is_contiguous():
            return None
    elif (x.ndim != 2 or x.shape[0] != 16 or x.shape[1] != 2048
              or tuple(weight.shape) != (1024, 2048) or not x.is_contiguous()):
        return None
    if residual.dtype != torch.bfloat16 or residual.device != x.device \
            or residual.numel() != 16 * 1024 or not residual.is_contiguous():
        return None
    with projection.accumulator(x, weight) as acc:
        if tuple(acc.shape) != (16, 1024) or acc.dtype != torch.float32:
            return None
        return add_rms_m16_f32in(acc, residual, norm_weight, epsilon)
