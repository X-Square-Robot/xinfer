"""Separable antialiased bicubic downsampling through the in-tree native library.

``XINFER_AA_RESAMPLE`` is on by default. The supported steps are
720x1280 -> 128x224 and 480x640 -> 160x224; other sizes use the caller's
interpolation path. Vertical accumulation uses FP32 and the horizontal
pass includes the uint8 round-half-up conversion.

Host coefficients are computed in FP64 and stored in FP32. Horizontal
uint32 index windows are clamped, with zero coefficients outside the input.
Tables and intermediate buffers retain stable addresses for graph replay.
"""

from __future__ import annotations

import atexit
import ctypes
import json
import os
import time
from threading import Lock

import numpy as np

from xinfer.ops.backends.ascend import _native
import torch

_LOCK = Lock()
_LIB = None
_ENTRIES: dict = {}
_ENABLED = None
_SHAPES: dict = {}
_STATS = {"routed": 0, "declined": 0, "t0": time.time()}


def _dump_stats():
    path = os.environ.get("XINFER_AA_RESAMPLE_STATS", "").strip()
    if path:
        try:
            _STATS["elapsed_s"] = round(time.time() - _STATS["t0"], 3)
            with open(f"{path}-pid{os.getpid()}", "w") as fh:
                json.dump(_STATS, fh)
        except OSError:
            pass


atexit.register(_dump_stats)

A = -0.5
# (HIN, WIN, HOUT, WOUT, TAPS): only these downsampling steps use native entries.
STEPS = {
    (720, 1280, 128, 224): ("aa_v_720_1280_128_224", "aa_h_720_1280_128_224", 25),
    (480, 640, 160, 224): ("aa_v_480_640_160_224", "aa_h_480_640_160_224", 13),
}


def enabled() -> bool:
    global _ENABLED
    if _ENABLED is None:
        raw = os.environ.get("XINFER_AA_RESAMPLE", "1").strip().lower()
        _ENABLED = raw in ("1", "true", "yes", "on")
    return _ENABLED


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
            _ENTRIES[key] = None
            return None
        pair = (call, error)
        _ENTRIES[key] = pair
    return pair


def _launch(name, tensors, scalars=()):
    if _native.launch_kernel(name, tensors, scalars):
        return True
    pair = _entry(name, len(tensors), len(scalars))
    if pair is None:
        return False
    call, error = pair
    stream = torch.npu.current_stream(tensors[0].device).npu_stream
    status = call(
        *[ctypes.c_void_p(t.data_ptr()) for t in tensors],
        *[ctypes.c_int64(s) for s in scalars],
        ctypes.c_void_p(stream),
    )
    if status:
        message = error()
        raise RuntimeError(
            f"{name} failed: {message.decode('utf-8', 'replace') if message else 'unknown'}"
        )
    return True


def _cubic(x):
    ax = np.abs(x)
    ax2 = ax * ax
    ax3 = ax2 * ax
    inner = (A + 2) * ax3 - (A + 3) * ax2 + 1
    outer = A * ax3 - 5 * A * ax2 + 8 * A * ax - 4 * A
    return np.where(ax <= 1, inner, np.where(ax < 2, outer, 0.0))


def _tables_1d(hin, hout, taps):
    """(w [hout,taps] fp32 normalized, base [hout] i32) of the extracted formula."""
    r = hin / hout
    fs = (1.0 / r) if r > 1.0 else 1.0
    support = 2.0 / fs
    src = r * (np.arange(hout, dtype=np.float64) + 0.5) - 0.5
    kmin = np.floor(src - support).astype(np.int64) + 1
    kmax = np.ceil(src + support).astype(np.int64) - 1
    assert int((kmax - kmin).max()) + 1 <= taps, (hin, hout, taps)
    ks = kmin[:, None] + np.arange(taps)[None, :]
    w = fs * _cubic(fs * (src[:, None] - ks))
    w = np.where((ks >= 0) & (ks < hin), w, 0.0)
    w = w / w.sum(axis=1, keepdims=True)
    return w.astype(np.float32), kmin.astype(np.int32)


def _shape_pack(device, hin, win, hout, wout, taps):
    device = torch.device(device)
    if device.index is None:
        device = torch.device(device.type, torch.npu.current_device())
    shape = (hin, win, hout, wout)
    key = (device, torch.npu.current_stream(device).npu_stream, *shape)
    pack = _SHAPES.get(key)
    if pack is None:
        vname, hname, taps = STEPS[shape]
        wy, basey = _tables_1d(hin, hout, taps)
        wx, basex = _tables_1d(win, wout, taps)
        woutp = (wout + 63) // 64 * 64
        # per-tap horizontal index table, clamped (zero weights pad the span)
        ksx = basex[:, None] + np.arange(taps)[None, :]
        ksx = np.clip(ksx, 0, win - 1)
        ix = np.zeros((taps, woutp), np.uint32)
        ix[:, :wout] = ksx.T
        wxp = np.zeros((taps, woutp), np.float32)
        wxp[:, :wout] = wx.T
        pack = dict(
            vname=vname, hname=hname, taps=taps, woutp=woutp,
            wy=torch.from_numpy(wy.reshape(-1)).to(device),
            basey=torch.from_numpy(basey).to(device),
            wx=torch.from_numpy(wxp.reshape(-1)).to(device),
            ix=torch.from_numpy(ix.reshape(-1)).to(device),
            mid=torch.empty(3 * hout * win, dtype=torch.float32, device=device),
        )
        _SHAPES[key] = pack
    return pack


def step(x, size):
    """One chain resize step: u8 [1,3,H,W] -> u8 [1,3,h,w], or None outside
    the envelope. ``x`` is uint8 and contiguous on the npu device."""
    if not enabled() or x is None:
        return None
    if x.dim() != 4 or x.shape[0] != 1 or x.shape[1] != 3 or x.dtype != torch.uint8:
        return None
    if not (x.is_contiguous() and x.device.type == "npu"):
        return None
    hin, win = int(x.shape[2]), int(x.shape[3])
    hout, wout = int(size[0]), int(size[1])
    key = (hin, win, hout, wout)
    if key not in STEPS:
        _STATS["declined"] += 1
        return None
    if any(_entry(name, 4, 0) is None for name in STEPS[key][:2]):
        return None
    pack = _shape_pack(x.device, hin, win, hout, wout, STEPS[key][2])
    out = torch.empty(3 * hout * wout, dtype=torch.uint8, device=x.device)
    # the released chain's own x.float() cast (exact u8->f32) feeds the
    # vertical kernel as fp32
    xf = x.view(1, 3, hin, win).float().view(-1)
    if not _launch(pack["vname"], (xf, pack["mid"], pack["wy"], pack["basey"])):
        _STATS["declined"] += 1
        return None
    if not _launch(pack["hname"], (out, pack["mid"], pack["wx"], pack["ix"])):
        raise RuntimeError("AA horizontal launcher disappeared after the vertical pass")
    _STATS["routed"] += 1
    return out.view(1, 3, hout, wout)
