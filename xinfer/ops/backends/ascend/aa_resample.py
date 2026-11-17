"""aa_resample mount: the separable AA-bicubic resize steps of the device image
chain through the AA-RESAMPLE TileLang AOT kernels.

Flag ``XINFER_AA_RESAMPLE`` (default on, the PUSH-FINAL composite adjudication -- gate 0.0259 <= 0.05 with full e2e dominance; =off restores the vendor): replaces each
``F.interpolate(bicubic, antialias=True, align_corners=False)`` +
``add_(0.5).floor_().clamp_(0,255).to(uint8)`` step of ``chain_view`` /
``device_images._resize`` on the two AA-down deployment shapes with two
kernel launches (vertical pass u8->fp32 mid, horizontal pass fp32->u8 with
the quantize law fused). Envelope = 720x1280->128x224 and 480x640->160x224
only: the two AA-up legs (128x224->224x352, 160x224->224x320) are NOT
routed -- their custom kernels lose to the aclnn vendor eager (73.5us vs
29.5us, unit-r6 receipt) and the WIN=224 truncation fix attempts (real9 /
real10 builds) deadlock the vector core through the TileLang
auto-scheduler's inserted flag stream; the vendor stays on the up legs and
this mount declines them (2026-11-16, AA-RESUME).

The knife shared object path is ``XINFER_AA_RESAMPLE_LIB``. An unset or
unloadable path, or a shape outside the envelope, returns None so the
caller keeps the released aclnn chain.

Weights are built host-side in fp64 with the extracted torch formula
(probe-validated 2026-11-16) and stored as fp32; the horizontal index tables
are per-tap uint32 windows clamped into range (zero-weight taps make the
clamped gathers exact). Tables, mid buffers and outputs are cached per shape
with stable addresses, so the NPUGraph capture of the chain replays cleanly.
"""

from __future__ import annotations

import atexit
import ctypes
import json
import os
import time
from threading import Lock

import numpy as np
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
# (HIN, WIN, HOUT, WOUT, TAPS) -- envelope: AA-down shapes only. The knife
# .so links only these four kernels (real6-validated objects); the up legs
# stay on the aclnn vendor -- see the module docstring for the receipt.
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
    global _LIB
    if _LIB is None:
        path = os.environ.get("XINFER_AA_RESAMPLE_LIB", "").strip()
        if not path:
            return None
        with _LOCK:
            if _LIB is None:
                try:
                    _LIB = ctypes.CDLL(path)
                except OSError:
                    return None
    return _LIB


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
    key = (hin, win, hout, wout)
    pack = _SHAPES.get(key)
    if pack is None:
        vname, hname, taps = STEPS[key]
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
    try:
        pack = _shape_pack(x.device, hin, win, hout, wout, STEPS[key][2])
        out = torch.empty(3 * hout * wout, dtype=torch.uint8, device=x.device)
        # the released chain's own x.float() cast (exact u8->f32) feeds the
        # vertical kernel as fp32
        xf = x.view(1, 3, hin, win).float().view(-1)
        if not _launch(pack["vname"], (xf, pack["mid"], pack["wy"], pack["basey"])):
            _STATS["declined"] += 1
            return None
        if not _launch(pack["hname"], (out, pack["mid"], pack["wx"], pack["ix"])):
            _STATS["declined"] += 1
            return None
    except (ValueError, RuntimeError):
        _STATS["declined"] += 1
        return None
    _STATS["routed"] += 1
    return out.view(1, 3, hout, wout)
