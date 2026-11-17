"""Processor-owned tickets for one engine's device-image queue.

Tickets precede the request's admission item on the same queue. Their decoded views and errors
stay on that request; closing another engine cannot redirect or disable this queue.
"""

from __future__ import annotations

import queue
import threading

from xinfer.models.interface import DeviceImage


class Ticket:
    """One view handed over by its processor thread."""

    __slots__ = ("image",)

    def __init__(self, image) -> None:
        self.image = image


class Boundary:
    """Marks the end of one request's views in the device-image queue."""

    __slots__ = ()


class Slot:
    """The mutable decode result of one request-owned view."""

    __slots__ = ("view", "error", "device", "txn", "owner")

    def __init__(self, owner) -> None:
        self.view = None
        self.error = None
        self.device = None
        self.txn = None
        self.owner = owner


class ImagePrefetch:
    """Hand views to one engine while it accepts prefetch work."""

    def __init__(self, target: queue.Queue) -> None:
        self._target = target
        self._lock = threading.Lock()

    def hand_off(self, image: DeviceImage) -> DeviceImage:
        with self._lock:
            if self._target is None:
                return image
            ticket = DeviceImage(image.data, image.sizes, image.resize, prefetch=Slot(self))
            self._target.put(Ticket(ticket))
            return ticket

    def flush(self) -> None:
        with self._lock:
            if self._target is not None:
                self._target.put(Boundary())

    def close(self) -> None:
        """Stop new tickets before the owning engine stops its image thread."""
        with self._lock:
            self._target = None
