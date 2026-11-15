"""ADMISSION-GAP AG2: one batched host-to-device upload per admitted request (NPU/CUDA).

The admission handoff used to pay one ``tensor.to(device, non_blocking=True)`` per prepared CPU
tensor (12 calls, 181 us of host enqueue measured on the admission-gap P0 probe, card 2). This
module packs every CPU tensor of one request into a pinned flat byte slab (one small memcpy each,
byte-for-byte), issues a single stream-ordered H2D copy of the slab, and hands out device tensors
that are *views* of the slab's device copy -- the same bytes the per-tensor uploads would have
produced, so the stage inputs are bitwise unchanged.

Races and ownership (see R/admission-gap/RACE-ANALYSIS.md §2 AG2):

- One ring slot (pinned host slab + device slab + cached views) belongs to at most one in-flight
  request. The engine returns it when the request finishes, after its final event (or the failed
  dispatch's lane event) completed, so no stage reads a slab a later request overwrites -- the
  same lifetime the per-tensor device tensors had.
- The device views are cached per (ring slot, layout signature); ``copy_`` refreshes the device
  slab in place, so cached views always see the current request's bytes. A distinct shape builds a
  fresh signature once and then reuses its views.
- Everything runs on the engine thread inside the admission upload-stream context: one writer
  thread, the same stream and the same non-blocking copy primitive as before, just one call.
  No host-synchronous device memcpy is ever issued (DVPP-PARALLEL §4 discipline).
- Any failure (allocation, packing, dtype view) falls back to the per-tensor ``_to_device`` path;
  ``XINFER_UPLOAD_BATCH=0`` disables the batching outright.

Every method here runs in the engine process (never in a deviceless frontend), on the engine
thread.
"""

from __future__ import annotations

import os
import threading

import numpy as np
import torch

# Offsets are 8-byte aligned so every dtype (bool/int32/float32/int64) can view its slice.
_ALIGNMENT = 8

_ITEMSIZE = {torch.bool: 1, torch.uint8: 1, torch.int8: 1, torch.int16: 2, torch.int32: 4,
             torch.int64: 8, torch.float16: 2, torch.bfloat16: 2, torch.float32: 4,
             torch.float64: 8}


def upload_batch_enabled() -> bool:
    """Whether batched admission uploads are on (``XINFER_UPLOAD_BATCH``, default on)."""
    return os.environ.get("XINFER_UPLOAD_BATCH", "1") not in ("0", "false", "False")


class _RingSlot:
    """One pinned host slab, its device copy and the device views built over it."""

    __slots__ = ("host", "host_np", "device", "views", "signature")

    def __init__(self, slab_bytes: int, device: str) -> None:
        self.host = torch.empty(slab_bytes, dtype=torch.uint8).pin_memory()
        self.host_np = self.host.numpy()
        self.device = torch.empty(slab_bytes, dtype=torch.uint8, device=device)
        self.views: dict[str, torch.Tensor] = {}
        self.signature: tuple | None = None


class UploadBatcher:
    """Packs a request's CPU tensors into one pinned slab and uploads it as one copy.

    ``upload`` returns ``(inputs, keepalive)``: ``inputs`` maps every packed tensor name to its
    device view (byte-identical to a per-tensor upload); ``keepalive`` is the ring slot, which the
    request must hold until it finished -- the async H2D reads the host bytes and the stages read
    the device bytes.
    """

    def __init__(self, device: str, *, slots: int = 8, slab_bytes: int = 1 << 20) -> None:
        self._device = device
        self._slot_count = slots
        self._slab_bytes = slab_bytes
        self._free: list[_RingSlot] = []
        self._lock = threading.Lock()  # engine thread only in practice; guards release() callers
        self._outstanding = 0

    def release(self, slot: _RingSlot) -> None:
        """Return a ring slot after its request finished (all device readers done)."""
        with self._lock:
            self._outstanding -= 1
            self._free.append(slot)

    def upload(self, tensors: dict, names: list[str]) -> tuple[dict, _RingSlot] | None:
        """Pack and upload ``tensors[name] for name in names`` (CPU tensors); None to fall back.

        Runs on the engine thread inside the admission upload-stream context. The single
        ``copy_`` is stream-ordered on the caller's current stream, exactly like the per-tensor
        ``.to(non_blocking=True)`` calls it replaces.
        """
        sources = []
        layout = []
        offset = 0
        try:
            for name in names:
                tensor = tensors[name]
                flat = tensor.reshape(-1) if tensor.is_contiguous() else tensor.contiguous().reshape(-1)
                itemsize = _ITEMSIZE.get(tensor.dtype)
                if itemsize is None:
                    return None
                nbytes = flat.numel() * itemsize
                if offset % _ALIGNMENT:
                    offset += _ALIGNMENT - offset % _ALIGNMENT
                sources.append((offset, flat.view(torch.uint8).numpy(), nbytes))
                layout.append((name, offset, nbytes, tensor.dtype, tuple(tensor.shape),
                               nbytes // itemsize))
                offset += nbytes
        except Exception:  # noqa: BLE001 - non-viewable input: per-tensor path
            return None
        if offset > self._slab_bytes:
            return None
        with self._lock:
            slot = self._free.pop() if self._free else None
            if slot is None and self._outstanding >= self._slot_count:
                return None
            self._outstanding += 1  # in-use count: raised on reuse and on fresh allocation alike
        if slot is None:
            try:
                slot = _RingSlot(self._slab_bytes, self._device)
            except Exception:  # noqa: BLE001 - pinned or device allocation failed
                with self._lock:
                    self._outstanding -= 1
                return None
        try:
            host_np = slot.host_np
            for off, source, nbytes in sources:
                host_np[off:off + nbytes] = source
            signature = tuple(layout)
            if slot.signature != signature:
                device = slot.device
                views = {}
                for name, off, nbytes, dtype, shape, count in layout:
                    views[name] = device.narrow(0, off, nbytes).view(dtype).view(shape)
                slot.views, slot.signature = views, signature
            slot.device.copy_(slot.host, non_blocking=True)
            return dict(slot.views), slot
        except Exception:  # noqa: BLE001 - the caller falls back to the per-tensor path
            self.release(slot)
            return None
