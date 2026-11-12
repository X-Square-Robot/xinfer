"""Per-view prefetch tickets from an InputProcessor into the engine's device-image thread.

NPU only: the DVPP JPEGD path decodes one JPEG per synchronous channel transaction, so handing
each view over as soon as its wire bytes are ready pipelines the processor's base64 decode and
host prepare work with the device decode. ``device_image`` claims a slot for the view and submits
the ticket when an engine bound this process's device-image queue; without a binding (frontend
processes under ``xinfer serve``, CPU-only tools, before the engine starts) it is a no-op and the
engine decodes the view at admission as before.

The slot is engine-internal and single-threaded by construction: tickets are put on the same
FIFO queue as the requests, and a request is only handed over (at admission) after its
processor finished, so its tickets are always ahead of it and complete first. A slot still
pending (the binding switched mid-flight, the engine shut down) or decoded for another device
falls back to the admission-time decode; a decode that raised fails the request when admitted,
exactly as an admission-time decode failure would.

Torch-free by design: the module is imported lazily from ``xinfer.models.processing``, which
also runs in frontend processes that must not initialize a device.
"""

from __future__ import annotations

import queue
import threading


class Ticket:
    """One ``DeviceImage`` handed over by its processor thread."""

    __slots__ = ("image",)

    def __init__(self, image) -> None:
        self.image = image


class Slot:
    """The mutable result box of one prefetched view, held by the frozen ``DeviceImage``."""

    __slots__ = ("view", "error", "device")

    def __init__(self) -> None:
        self.view = None    # the decoded+resized device tensor once done
        self.error = None   # the exception a failed decode raised
        self.device = None  # the device the view was decoded for


_bound: "queue.Queue | None" = None
_lock = threading.Lock()


def bound() -> bool:
    """Whether an engine bound this process's device-image queue."""
    return _bound is not None


def claim() -> "Slot | None":
    """A fresh slot for a view about to be submitted, or None when no engine is bound."""
    with _lock:
        return Slot() if _bound is not None else None


def submit(image) -> None:
    """Hand ``image`` (holding a claimed slot) to the bound engine's device-image thread."""
    with _lock:
        target = _bound
    if target is not None:
        target.put(Ticket(image))


def bind(target: "queue.Queue") -> "queue.Queue | None":
    """Bind this process's device-image queue; returns the previous binding to restore later."""
    global _bound
    with _lock:
        previous = _bound
        _bound = target
        return previous


def unbind(target: "queue.Queue") -> None:
    """Drop the binding if it still is ``target`` (another engine may have re-bound it)."""
    global _bound
    with _lock:
        if _bound is target:
            _bound = None


def hand_off(value):
    """The models-side hook: ``value`` (a frozen ``DeviceImage``) with a claimed slot, as a ticket.

    Installed into ``xinfer.models.processing._prefetch_hook`` by the engine when it binds this
    process's device-image queue (models never import the engine); the slot is claimed, the
    ticket carrying it is submitted, and the ticket value is returned to the processor. Called
    without a binding it returns ``value`` unchanged.
    """
    from xinfer.models.interface import DeviceImage

    slot = claim()
    if slot is None:
        return value
    ticket = DeviceImage(value.data, value.sizes, value.resize, prefetch=slot)
    submit(ticket)
    return ticket
