"""soupform mount: single-pass soup kills over the wall-x 2B final stack.

Flag ``XINFER_SOUPFORM`` (default off): a comma list of the families
``normprep`` (add_zero_centered_rms_norm prefill sites: one pass for
s = rne(residual+x) and sq = f32(s)^2, plus the elementwise apply redstone),
``zbasplit`` (the DeltaNet qkv_zba row_scale finish split into contiguous
conv/zba buffers), ``glumulcast`` (the prompt SwiGLU mul+cast single pass over
the borrowed FP32 accumulator; parked -- the kernel lost its race with the
aclnn chain at 45.9 vs 38.4 us/site, kept for the record and for the envelope
audit) and ``vropepack`` (the vision QKV+RoPE finish storing the packed
[3, rows, heads, dim] layout directly).

The knife shared object is delivered with the model stack, not with this
repository, so its path is configuration: ``XINFER_SOUPFORM_LIB``. An unset or
unloadable path disables every route instead of raising into the caller.

Every route keeps the released torch chain's FP32 arithmetic and rounds once
with RNE; envelopes mirror the landed soupkill rules (contiguous operands,
aligned rows, exact widths) and any miss returns None so the caller keeps the
released chain. The kernels are the TileLang AOT objects of
``src/soupform/npu/soupform_kernels.py`` (the ``soupform`` CMake module),
reached through ctypes exactly like the other internal AOT launchers
(``xinfer_aot_<name>_call`` / ``_get_last_error``), so the FFI face does not
move.

Evidence (SOUP-FORM, card 7, final-consolidate stack): witness 0 flips across
all four families (random + crafted tie/subnormal/overflow inputs); action
gate vs GBASE 0.0050923228 PASS with the margin 0.0011076772 unchanged;
e2e 50x3 dual order 27.735 -> 27.000 ms total (engine 25.045 -> 24.300);
ledger busy -471.7 us and launches 3743 -> 3591. Receipts:
``npu/knives/2b-soupform/MANIFEST.json``.
"""

from __future__ import annotations

import ctypes
import os
from threading import Lock

import torch

from xinfer.ops.backends.ascend import _native

_FAMILIES = ("normprep", "zbasplit", "glumulcast", "vropepack")
_LOCK = Lock()
_LIB = None
_ENTRIES: dict = {}
_ENABLED = None


def _parse() -> None:
    global _ENABLED
    raw = os.environ.get("XINFER_SOUPFORM", "off").strip().lower()
    if raw in ("", "off", "0", "false", "no"):
        _ENABLED = frozenset()
    elif raw in ("1", "all", "on"):
        _ENABLED = frozenset(_FAMILIES)
    else:
        wanted = {token.strip() for token in raw.split(",") if token.strip()}
        unknown = wanted - set(_FAMILIES)
        if unknown:
            raise RuntimeError(
                f"XINFER_SOUPFORM unknown families: {sorted(unknown)}"
            )
        _ENABLED = frozenset(wanted)


def enabled(family: str) -> bool:
    if _ENABLED is None:
        _parse()
    return family in _ENABLED


def _library():
    return _native.core_library()


def _entry(name, pointers, scalars):
    key = (name, pointers, scalars)
    pair = _ENTRIES.get(key)
    if pair is None:
        lib = _library()
        if lib is None:
            return None
        try:
            call = getattr(lib, f"xinfer_aot_{name}_call")
            call.restype = ctypes.c_int32
            call.argtypes = (
                [ctypes.c_void_p] * pointers
                + [ctypes.c_int64] * scalars
                + [ctypes.c_void_p]
            )
            error = getattr(lib, f"xinfer_aot_{name}_get_last_error")
            error.restype = ctypes.c_char_p
        except (AttributeError, OSError):
            # A library without this entry (a different knife build, or a stale
            # XINFER_SOUPFORM_LIB) must not break the model: decline the route.
            _ENTRIES[key] = None
            return None
        pair = (call, error)
        _ENTRIES[key] = pair
    return pair


def _launch(name, tensors, scalars):
    if _native.launch_kernel(name, tensors, scalars):
        return True
    pair = _entry(name, len(tensors), len(scalars))
    if pair is None:
        return False
    call, error = pair
    stream = torch.npu.current_stream(tensors[0].device).npu_stream
    status = call(
        *[ctypes.c_void_p(tensor.data_ptr()) for tensor in tensors],
        *[ctypes.c_int64(scalar) for scalar in scalars],
        ctypes.c_void_p(stream),
    )
    if status:
        message = error()
        raise RuntimeError(
            f"{name} failed: {message.decode('utf-8', 'replace') if message else 'unknown'}"
        )
    return True


def _flat(t) -> bool:
    return t is not None and t.is_contiguous() and t.device.type == "npu"


def _rows(x) -> int:
    rows = 1
    for dim in x.shape[:-1]:
        rows *= dim
    return rows


def normprep_norm(x, residual, weight, eps):
    """(s, normed) of add_zero_centered_rms_norm in two fused passes + aclnn stats.

    s = rne(f32(x) + f32(residual)) and sq = f32(s)^2 in one pass; the stats
    keep the aclnn Mean/Add/Rsqrt chain bit for bit (mean over the identical
    sq values); the apply is the elementwise (x*rstd)*(1+w) with one RNE.
    Returns None when the envelope is not ours.
    """
    if not enabled("normprep"):
        return None
    if not (_flat(x) and x.dtype == torch.bfloat16 and x.shape[-1] == 2048
            and _flat(residual) and residual.dtype == torch.bfloat16
            and residual.shape == x.shape and residual.device == x.device
            and _flat(weight) and weight.shape == (2048,)
            and weight.dtype == torch.float32 and weight.device == x.device
            and _rows(x) % 4 == 0 and _rows(x) > 0):
        return None
    rows = _rows(x)
    try:
        s = torch.empty_like(x)
        sq = torch.empty((rows, 2048), dtype=torch.float32, device=x.device)
        if not _launch("normprep_c2048", (s, sq, x, residual), (rows,)):
            return None
        rstd = torch.rsqrt(sq.mean(-1, keepdim=True) + float(eps))
        normed = torch.empty_like(x)
        if not _launch("apply_rstd_c2048", (normed, s, rstd.reshape(-1), weight), (rows,)):
            return None
    except (ValueError, RuntimeError):
        return None
    return s, normed


def zbasplit(acc, keep, conv=6144):
    """(conv_buf, zba_buf) = (rne(acc[:, :conv]*keep), rne(acc[:, conv:]*keep)).

    Both halves land in contiguous BF16 buffers so the model's column slices
    disappear; mul-by-keep is exact (0/1), one RNE per element.
    """
    if not enabled("zbasplit"):
        return None
    if not (_flat(acc) and acc.dtype == torch.float32 and acc.dim() == 2
            and acc.shape[1] == 8224 and _flat(keep) and keep.dtype == torch.float32
            and keep.shape == (acc.shape[0],) and acc.shape[0] > 0):
        return None
    rows = acc.shape[0]
    conv_buf = acc.new_empty((rows, conv), dtype=torch.bfloat16)
    zba_buf = acc.new_empty((rows, acc.shape[1] - conv), dtype=torch.bfloat16)
    try:
        if not _launch("zbasplit_c8224", (conv_buf, zba_buf, acc, keep), (rows,)):
            return None
    except (ValueError, RuntimeError):
        return None
    return conv_buf, zba_buf


def mulcast(silu_out, acc, half=6144):
    """out = rne(silu_out * acc[:, half:]) over the borrowed GLU accumulator."""
    if not enabled("glumulcast"):
        return None
    if not (_flat(silu_out) and silu_out.dtype == torch.float32
            and silu_out.dim() == 2 and silu_out.shape[1] == half
            and _flat(acc) and acc.dtype == torch.float32
            and acc.shape == (silu_out.shape[0], 2 * half)
            and silu_out.shape[0] > 0):
        return None
    rows = silu_out.shape[0]
    out = acc.new_empty((rows, half), dtype=torch.bfloat16)
    try:
        if not _launch("mulcast_c6144", (out, silu_out, acc), (rows,)):
            return None
    except (ValueError, RuntimeError):
        return None
    return out


def vision_qkv_rope_packed(acc, bias, cos2, sin2, heads=16, dim=64):
    """The vrope v3 math stored as packed [3, rows, heads, dim] BF16."""
    if not enabled("vropepack"):
        return None
    if not (heads == 16 and dim == 64 and _flat(acc) and acc.dtype == torch.float32
            and acc.dim() == 2 and acc.shape[1] == 3 * heads * dim
            and _flat(bias) and bias.dtype == torch.bfloat16
            and bias.numel() == acc.shape[1]
            and _flat(cos2) and cos2.dtype == torch.float32
            and tuple(cos2.shape) == (acc.shape[0], dim)
            and _flat(sin2) and sin2.dtype == torch.float32
            and tuple(sin2.shape) == (acc.shape[0], dim)
            and acc.shape[0] % 2 == 0 and acc.shape[0] > 0):
        return None
    rows = acc.shape[0]
    out = acc.new_empty((3, rows, heads, dim), dtype=torch.bfloat16)
    try:
        if not _launch("vision_qkv_rope_v3p", (out, acc, bias, cos2, sin2), (rows,)):
            return None
    except (ValueError, RuntimeError):
        return None
    return out
