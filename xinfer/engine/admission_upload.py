"""Pack prepared CPU tensors into one pinned slab for a stream-ordered admission upload.

Host slabs are reused only after their request's completion event. Each upload has a fresh
device allocation, retained by its tensor views, so later requests cannot overwrite graph inputs.
Unsupported packing falls back before enqueue; uncertain device completion never recycles a slab.
"""

from __future__ import annotations

import os
import threading
from typing import Any

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
    """Pinned source bytes; the request owns its separate device allocation."""

    __slots__ = ("host", "host_np")

    def __init__(self, slab_bytes: int) -> None:
        self.host = torch.empty(slab_bytes, dtype=torch.uint8).pin_memory()
        self.host_np = self.host.numpy()


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
        self._releasing: list[tuple[_RingSlot, Any]] = []  # (slot, event) awaiting the event's completion
        self._retired: list[_RingSlot] = []  # completion could not be established; never reuse these bytes
        self._lock = threading.Lock()  # engine thread only in practice; guards release() callers
        self._outstanding = 0

    def retire(self, slot: _RingSlot) -> None:
        """Keep source bytes alive when no completion event could be recorded."""
        with self._lock:
            self._retired.append(slot)

    def release(self, slot: _RingSlot, after: Any = None) -> None:
        """Return a ring slot once ``after`` (if given) completes; a later poll reclaims it.

        The event-gated form never blocks the caller: the engine thread must not wait on device
        work (a failed dispatch's lane event can stay pending a long while, and the request's
        future must resolve before it). The slot returns to the free list in this call when
        ``after`` is None or already complete, else when a later :meth:`poll` finds it complete.
        """
        with self._lock:
            try:
                done = after is None or after.query()
            except Exception:  # noqa: BLE001 - a failed query is not proof the copy stopped reading
                self._retired.append(slot)
                return
            if done:
                self._outstanding -= 1
                self._free.append(slot)
            else:
                self._releasing.append((slot, after))

    def poll(self) -> None:
        """Reclaim the slots whose release event has completed (the engine thread, each step)."""
        if not self._releasing:
            return
        with self._lock:
            pending = []
            for slot, after in self._releasing:
                try:
                    done = after.query()
                except Exception:  # noqa: BLE001 - retain uncertain allocations without reusing them
                    self._retired.append(slot)
                    continue
                if done:
                    self._outstanding -= 1
                    self._free.append(slot)
                else:
                    pending.append((slot, after))
            self._releasing = pending

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
                slot = _RingSlot(self._slab_bytes)
            except Exception:  # noqa: BLE001 - pinned or device allocation failed
                with self._lock:
                    self._outstanding -= 1
                return None
        try:
            host_np = slot.host_np
            for off, source, nbytes in sources:
                host_np[off:off + nbytes] = source
            # Build every view before enqueue, so a packing failure can safely return the host slab.
            device = torch.empty(self._slab_bytes, dtype=torch.uint8, device=self._device)
            views = {}
            for name, off, nbytes, dtype, shape, count in layout:
                views[name] = device.narrow(0, off, nbytes).view(dtype).view(shape)
        except Exception:  # noqa: BLE001 - no device copy has been enqueued; per-tensor upload is safe
            self.release(slot)
            return None
        try:
            device.copy_(slot.host, non_blocking=True)
        except Exception:
            # A failed enqueue may already have submitted work. Keep the source alive and propagate
            # the device failure instead of starting a second upload or overwriting its pinned bytes.
            self.retire(slot)
            raise
        return views, slot
